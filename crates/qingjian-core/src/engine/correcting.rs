//! 整段拼写纠错：找一处编辑的纠正并按个人敲错表打折；规则无果时问神经模型要提名。
//!
//! 神经提名有两种问法：同步的（CLI 与评测，查询当场问）与异步的（壳里，后台线程；
//! 查询只记下作用域，壳在停稳后 [`Engine::request_correction`] 送去后台，
//! [`Engine::poll_correction`] 收到提名后重查一次，验证与噪声信道都在那次重查里）。

use super::*;

/// 一次查询最多试几个神经提名：现在的贪心解码只出一个，名额留给出多个候选的模型。
const NEURAL_CANDIDATES: usize = 3;

/// 后台给好的提名最多留几条（按作用域认领）：旧的对应已经过去的输入状态，攒多了是垃圾。
const NEURAL_READY_KEPT: usize = 8;

/// 神经提名的长度界：规则纠错的 4～24 是变体枚举成本卡出来的，神经没有枚举成本，
/// 只要求像个词的拼音（至少两个字母、不超过整句的常见长度）。
const NEURAL_MIN_LETTERS: usize = 2;

/// 见 [`NEURAL_MIN_LETTERS`]。
const NEURAL_MAX_LETTERS: usize = 32;

impl Engine {
    /// 这段作用域生效的拼写纠正（带缓存）：拼音不像话、用户没对它回车原样上屏过、
    /// 且一处编辑后能凑出至少一个两音节词时，取整句转换得分最高的那个纠正。
    pub(super) fn active_correction(&self, scope: &str) -> Option<Correction> {
        // 双拼敲错一个键换掉的是整个声母 / 韵母，全拼那套「一处编辑」的纠错模型不适用
        if self.shuangpin.is_some() {
            return None;
        }
        if let Some((cached_scope, cached)) = self.correction_cache.borrow().as_ref()
            && cached_scope == scope
        {
            return cached.clone();
        }
        let found = self.find_correction(scope);
        *self.correction_cache.borrow_mut() = Some((scope.to_owned(), found.clone()));
        found
    }

    pub(super) fn find_correction(&self, scope: &str) -> Option<Correction> {
        if !correction::eligible(scope) || self.learner.raw_count(scope) > 0 {
            return None;
        }
        // 整段就是个英文词（hello）：用户多半在打英文，别把它「纠」成 喝了哦
        if self
            .english
            .as_ref()
            .is_some_and(|english| english.get(scope).is_some())
        {
            return None;
        }
        let (segmentations, tail) = segment_longest_prefix(scope).ok()?;
        // 不像话的拼音试全部一处编辑；末尾单字母的只试相邻换位（`mingtain` → `mingtian`）；
        // 看似合法的也试相邻换位：跨音节的换位（`niaho` → `nihao`）切分照样成立，
        // 音节级敲错边按敲错的切分展开够不着，只能整段换回来比噪声信道。
        // 原样说得通时噪声信道会判原样赢，不会误纠。
        let candidates = if correction::unlikely_pinyin(segmentations.first(), tail) {
            correction::candidates(scope)
        } else {
            correction::transposition_candidates(scope)
        };
        // 噪声信道：原串按原样能转出的整句得分 vs 纠正后的整句得分扣掉一次编辑的代价，后者高才纠。
        // 原串切不干净（有尾巴）就没有原样得分，任何能转出整句的纠正都胜出
        let raw_score = if tail.is_empty() {
            segmentations
                .first()
                .and_then(|best| self.convert_sentence(&best.patterns(), true))
                .filter(|conversion| !conversion.has_placeholder())
                .map(|conversion| conversion.score)
        } else {
            None
        };
        // 每个纠正扣一次编辑的代价，接受过同样的 (敲的, 要的) 音节对越多次扣得越少（个人敲错表）
        let mut best = self.best_rule_correction(scope, candidates);
        // 规则只修一处编辑：两处以上的错拼它一个都凑不出来，这时才问神经模型。
        // 同步调用当场问，异步先记下作用域（壳停稳后送去后台）；提名同样走噪声信道验证，通不过就当没说。
        if best.is_none() {
            best = self.best_neural_correction(scope);
        }
        let (score, found) = best?;
        if raw_score.is_some_and(|raw| score <= raw) {
            tracing::debug!(
                original = %found.original,
                corrected = %found.corrected,
                score,
                raw = raw_score,
                "原样已经说得通，不纠"
            );
            return None;
        }
        tracing::debug!(original = %found.original, corrected = %found.corrected, score, raw = raw_score, "拼写纠正");
        Some(found)
    }

    /// 规则候选里按噪声信道挑：纠正后整句得分扣一次编辑代价（按个人敲错表打折）最高的那个。
    fn best_rule_correction(
        &self,
        scope: &str,
        candidates: Vec<Correction>,
    ) -> Option<(f64, Correction)> {
        let mut best: Option<(f64, Correction)> = None;
        for candidate in candidates {
            // 「删掉刚敲的最后一个字母」不算纠正：用户可能还没敲完，尾巴留着等下一键
            if matches!(candidate.edit, correction::Edit::Delete { index, .. } if index + 1 == scope.len())
            {
                continue;
            }
            let accepted = candidate
                .typo_pair(candidate.corrected.len())
                .map_or(0, |(typed, intended)| {
                    self.learner.typo_count(&typed, &intended)
                });
            let Some(score) = self.correction_score(&candidate) else {
                continue;
            };
            let score = score - self.typo_costs.correction_cost(accepted);
            if best.as_ref().is_none_or(|(best, _)| score > *best) {
                best = Some((score, candidate));
            }
        }
        best
    }

    /// 神经模型的提名：先看后台有没有给好的（异步），再看要不要当场问（同步）。
    /// 都没有（异步还没排到、同步没挂模型）就返回 `None`，本轮按无纠正出候选。
    fn best_neural_correction(&self, scope: &str) -> Option<(f64, Correction)> {
        if let Some(outputs) = self.ready_correction(scope) {
            // 后台给好的提名（含空提名）留着：同一作用域重查直接复验，不再送后台
            return self.validate_neural_candidates(scope, outputs);
        }
        if self
            .correction_worker
            .as_ref()
            .is_some_and(correction::CorrectionWorker::is_alive)
        {
            // 异步模式：记下作用域就走，按键回调不等模型；壳停稳后送去后台
            let mut wanted = self.correction_wanted.borrow_mut();
            if wanted.as_deref() != Some(scope) {
                *wanted = Some(scope.to_owned());
            }
            return None;
        }
        let corrector = self.neural_corrector.as_ref()?;
        self.validate_neural_candidates(scope, corrector.correct(scope))
    }

    /// 后台给好的、属于这个作用域的提名串；只看不取（连空提名一起留着），对不上的也不动，
    /// 攒多了按 [`NEURAL_READY_KEPT`] 丢旧的。
    fn ready_correction(&self, scope: &str) -> Option<Vec<String>> {
        self.correction_ready
            .borrow()
            .iter()
            .find(|(pending, _)| pending == scope)
            .map(|(_, outputs)| outputs.clone())
    }

    /// 最近一次查询里有作用域等着问模型：壳该在用户停稳后调 [`Self::request_correction`]。
    pub fn correction_pending(&self) -> bool {
        self.correction_worker
            .as_ref()
            .is_some_and(correction::CorrectionWorker::is_alive)
            && self.correction_wanted.borrow().is_some()
    }

    /// 把记下的作用域送去后台问模型。没接异步模型或没什么要问的返回 `false`。
    pub fn request_correction(&mut self) -> bool {
        let Some(worker) = &self.correction_worker else {
            return false;
        };
        let Some(scope) = self.correction_wanted.borrow_mut().take() else {
            return false;
        };
        tracing::debug!(scope, "神经纠错请求");
        worker.submit(scope);
        true
    }

    /// 收后台算好的提名。有新提名进了待验证就返回 `true`，壳该重新 [`Self::query`] 一次；
    /// 验证与噪声信道都在那次重查里。提名是空（模型认为不用改）也算数，免得下次重查又问一次。
    pub fn poll_correction(&mut self) -> bool {
        let Some(worker) = &self.correction_worker else {
            return false;
        };
        let mut updated = false;
        while let Some(done) = worker.poll() {
            let mut ready = self.correction_ready.borrow_mut();
            ready.retain(|(pending, _)| pending != &done.scope);
            ready.push((done.scope, done.outputs));
            while ready.len() > NEURAL_READY_KEPT {
                ready.remove(0);
            }
            updated = true;
        }
        if updated {
            // 待验证的提名变了：按作用域记的旧结论作废，下次查询重新验证
            *self.correction_cache.borrow_mut() = None;
        }
        updated
    }

    /// 神经模型的提名逐个验证：是像话的拼音、切得干净、整句得分扣一次编辑代价后仍胜出，才算纠正。
    /// 神经纠正错了不止一处，个人敲错表按单音节记的折扣用不上，按原价扣。
    fn validate_neural_candidates(
        &self,
        scope: &str,
        fixed: Vec<String>,
    ) -> Option<(f64, Correction)> {
        let mut best: Option<(f64, Correction)> = None;
        for fixed in fixed.into_iter().take(NEURAL_CANDIDATES) {
            if !is_neural_output(&fixed) || fixed == scope {
                continue;
            }
            // 同规则那边：只差一个末尾字母多半是还没敲完，不算纠正
            if fixed.len() + 1 == scope.len() && scope.starts_with(&fixed) {
                continue;
            }
            if !parser::is_fully_segmentable(&fixed) {
                continue;
            }
            let Some(segmentation) = correction::complete_segmentation(&fixed) else {
                continue;
            };
            let candidate = Correction {
                original: scope.to_owned(),
                corrected: fixed,
                edit: correction::Edit::Neural,
                segmentation,
            };
            let Some(score) = self.correction_score(&candidate) else {
                continue;
            };
            let score = score - self.typo_costs.correction_cost(0);
            if best.as_ref().is_none_or(|(best, _)| score > *best) {
                best = Some((score, candidate));
            }
        }
        if let Some((_, found)) = &best {
            tracing::debug!(original = %found.original, corrected = %found.corrected, "神经纠正提名");
        }
        best
    }

    /// 纠正后串的整句得分；转不出整句（词库里没词）或带占位时返回 `None`。
    /// 纠正后的拼音上不再猜第二处敲错：规则变体本来就是一处编辑之外的读法，神经提名则是模型自己的结论，
    /// 再叠一层既慢又几乎不会赢。
    fn correction_score(&self, candidate: &Correction) -> Option<f64> {
        let conversion = self.convert_sentence(&candidate.segmentation.patterns(), false)?;
        (!conversion.has_placeholder()).then_some(conversion.score)
    }
}

/// 神经提名像不像拼音：纯小写字母、长度在界内。切分与词库由调用方再验，这里只拦明显的胡话。
fn is_neural_output(text: &str) -> bool {
    (NEURAL_MIN_LETTERS..=NEURAL_MAX_LETTERS).contains(&text.len())
        && text.bytes().all(|b| b.is_ascii_lowercase())
}
