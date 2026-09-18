//! 上屏与刷新：键盘与鼠标点选共用的提交链路。
//!
//! 全是自由函数而不是 `Host` 的方法：函数体里要多次借 Host（`with`），读应用文本（等应用回话，阻塞）时必须先放开借用，
//! 否则重入的 IMK 回调借不到 Host 会被吞掉。调用方（控制器、候选窗）手里只有 client，调这里。

use qingjian_core::Candidate;

use super::with;
use crate::candidates::Preedit;
use crate::imk::TextClient;
use crate::imk::secure_input;

/// 给本地整句模型看的光标前文最多读多少字符（Engine 自己再按它的前文长度截）。
const RESCORE_LOOKBACK: usize = qingjian_core::RESCORE_CONTEXT_CHARS;

/// 按当前缓冲区重新查候选、更新 marked text，回到第一页并重画候选窗口。
pub fn refresh(client: TextClient<'_>) {
    // 本地整句模型要看光标前文：一段组句只在第一键读一次（组句中它不变；应用偶尔不回话也不至于让前文来回换），
    // 读应用文本要等应用回话，放在借 Host 之外（见 request_prediction）
    let wants_context = with(|h| {
        h.attach_loaded_model();
        h.attach_loaded_corrector();
        h.engine.has_sentence_scorer() && h.engine.composition().text().chars().count() == 1
    })
    .unwrap_or(false);
    let before = if wants_context && !secure_input::enabled() {
        Some(
            client
                .surrounding_text(RESCORE_LOOKBACK, 0)
                .map(|text| text.before),
        )
    } else {
        None
    };
    let Some((marked, cursor, inline)) = with(|h| {
        if let Some(before) = before {
            h.engine.set_rescoring_context(before);
        }
        // 查询失败（整段切不动）时退回显示原始字母
        let mut marked = h.engine.composition().text().to_owned();
        let mut cursor = h.engine.composition().cursor();
        let mut preedit = Preedit::plain(&marked, cursor);
        let candidates = h
            .engine
            .query()
            .map(|mut query| {
                h.engine.annotate(&mut query.candidates);
                marked = query.marked_text();
                cursor = query.marked_cursor();
                preedit = Preedit::from_marked(&query.marked_segments(), cursor);
                query.candidates.items
            })
            .unwrap_or_default();
        h.reset_session(preedit, candidates);
        h.schedule_rescoring();
        h.schedule_correction();
        (marked, cursor, h.preedit_mode.inline())
    }) else {
        return;
    };
    // 配置成只在候选窗口显示拼音时，应用里不放 marked text（光标位置仍按插入点取）
    if inline {
        client.set_marked_text(&marked, cursor);
    } else {
        client.set_marked_text("", 0);
    }
    // 先发联想再画：发出去就留好云端槽位，画出来的第一帧本地候选就已经在最终位置
    if !marked.is_empty() {
        let candidates = with(|h| h.session.layout.local().to_vec()).unwrap_or_default();
        request_prediction(client, &candidates);
    }
    render(client);
}

/// 记下光标位置并按会话状态重画候选窗口。
pub fn render(client: TextClient<'_>) {
    let anchor = client.caret_rect();
    with(|h| {
        h.anchor = anchor;
        h.render();
    });
}

/// 发一次联想请求。Secure Input 里绝不发；没接联想器时是空操作。
///
/// 读上下文要等应用回话，这段时间 IMK 可能把 `deactivateServer:` 之类的回调插进来，
/// 所以分两次借 Host：先拿策略、放开借用去读、再借回来发请求。
fn request_prediction(client: TextClient<'_>, candidates: &[Candidate]) {
    let policy = with(|h| {
        if !h.engine.prediction_enabled() {
            return None;
        }
        if secure_input::enabled() {
            tracing::debug!("Secure Input 中，不联想");
            h.cancel_prediction();
            return None;
        }
        Some(h.engine.prediction_policy())
    })
    .flatten();
    let Some(policy) = policy else {
        return;
    };
    let surrounding = client.surrounding_text(policy.before, policy.after);
    with(|h| {
        tracing::debug!(
            has_context = surrounding.is_some(),
            pinyin = h.engine.composition().scope(),
            "联想请求"
        );
        match h.engine.request_prediction(surrounding, candidates) {
            Some(_) => h.await_prediction(),
            None => h.cancel_prediction(),
        }
    });
}

/// 接受组句中的整句补全：作用域内的拼音作废，句子上屏。没有补全返回 false。
pub fn accept_sentence(client: TextClient<'_>) -> bool {
    let Some(text) = with(|h| h.sentence.take()).flatten() else {
        return false;
    };
    with(|h| h.engine.accept_prediction(&text));
    tracing::debug!(%text, "接受整句补全");
    client.insert_text(&text);
    refresh(client);
    true
}

/// 高亮上下移动，越过页边自动翻页。
pub fn move_highlight(delta: isize, client: TextClient<'_>) {
    if with(|h| h.session.move_highlight(delta)).unwrap_or(false) {
        render(client);
    }
}

/// 翻页，高亮落到新页第一项。已在首页 / 末页时不动。
pub fn turn_page(delta: isize, client: TextClient<'_>) -> bool {
    let turned = with(|h| {
        let turned = h.session.turn_page(delta);
        if turned {
            h.engine.note_page_turn();
        }
        turned
    })
    .unwrap_or(false);
    if turned {
        render(client);
    }
    true
}

pub fn commit_highlighted(client: TextClient<'_>) -> bool {
    let index = with(|h| h.session.highlighted).unwrap_or(0);
    commit_index(index, client)
}

/// 上屏第 `index` 个候选；没有候选时上屏拼音本身。上屏后剩余拼音继续组句。
pub fn commit_index(index: usize, client: TextClient<'_>) -> bool {
    let candidate = with(|h| h.session.candidate(index)).flatten();
    let Some(candidate) = candidate else {
        if with(|h| index < h.session.layout.len()).unwrap_or(false) {
            return true;
        }
        return commit_raw(client);
    };
    let Some(text) = with(|h| h.engine.commit(&candidate)) else {
        return false;
    };
    tracing::debug!(%text, "commit");
    client.insert_text(&text);
    refresh(client);
    true
}

/// 鼠标点选当前页第 `offset` 格：上屏那个候选，剩余拼音继续组句。点在空行（云端占位格）上什么都不做。
pub fn commit_clicked(client: TextClient<'_>, offset: usize) -> bool {
    let Some(index) = with(|h| h.session.index_on_page(offset)).flatten() else {
        return false;
    };
    tracing::debug!(offset, "鼠标点选候选");
    commit_index(index, client)
}

/// 把拼音原样上屏并清空。缓冲区为空时返回 false。
pub fn commit_raw(client: TextClient<'_>) -> bool {
    let Some(raw) = with(|h| h.engine.take_raw()) else {
        return false;
    };
    if raw.is_empty() {
        return false;
    }
    tracing::debug!(%raw, "commit raw");
    client.insert_text(&raw);
    refresh(client);
    true
}
