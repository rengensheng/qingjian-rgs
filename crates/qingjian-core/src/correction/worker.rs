//! 神经纠错的后台线程：模型前向要几十毫秒，不能放在按键回调里。
//!
//! 与整句重打分（[`crate::engine::rescoring`]）同一套用法：查询当场只记下作用域，
//! 壳在停稳后 [`Engine::request_correction`](crate::engine::Engine::request_correction) 送去后台，
//! [`Engine::poll_correction`](crate::engine::Engine::poll_correction) 收到提名后重查一次，
//! 验证与噪声信道都在那次重查里。任务排队时只算最新的一条。

use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use super::PinyinCorrector;

/// 后台线程：输入拼音作用域，输出模型的提名串。
pub(crate) struct CorrectionWorker {
    /// 发任务。
    jobs: Sender<String>,

    /// 收提名。
    results: Receiver<Corrected>,

    /// 线程句柄（只用来判活）。
    handle: Option<JoinHandle<()>>,
}

/// 算好的提名：作用域与模型给的串（可能为空，同样要记下来，免得下次重查又问一次）。
pub(crate) struct Corrected {
    /// 问模型时的作用域。
    pub scope: String,

    /// 模型的提名串。
    pub outputs: Vec<String>,
}

impl CorrectionWorker {
    /// 起后台线程；模型随本结构一起结束。
    pub fn spawn(corrector: Box<dyn PinyinCorrector>) -> Self {
        let (jobs, job_rx) = channel::<String>();
        let (result_tx, results) = channel::<Corrected>();
        let handle = std::thread::Builder::new()
            .name("qingjian-correct".to_owned())
            .spawn(move || {
                while let Ok(mut scope) = job_rx.recv() {
                    // 攒了好几个作用域只算最新的（旧的对应已经过去的输入状态）
                    while let Ok(newer) = job_rx.try_recv() {
                        scope = newer;
                    }
                    let started = std::time::Instant::now();
                    let outputs = corrector.correct(&scope);
                    tracing::debug!(
                        scope,
                        outputs = outputs.len(),
                        ms = started.elapsed().as_millis(),
                        "神经纠错完成"
                    );
                    let done = Corrected { scope, outputs };
                    if result_tx.send(done).is_err() {
                        break;
                    }
                }
            })
            .ok();
        if handle.is_none() {
            tracing::warn!("起不了神经纠错线程，本次不用模型");
        }
        Self {
            jobs,
            results,
            handle,
        }
    }

    /// 线程是否还在。
    pub fn is_alive(&self) -> bool {
        self.handle.is_some()
    }

    /// 把作用域送去后台；线程已退出就记一条日志。
    pub fn submit(&self, scope: String) {
        if self.jobs.send(scope).is_err() {
            tracing::warn!("神经纠错线程已退出");
        }
    }

    /// 取一条算好的提名；没有就 `None`。
    pub fn poll(&self) -> Option<Corrected> {
        self.results.try_recv().ok()
    }
}

impl Drop for CorrectionWorker {
    fn drop(&mut self) {
        // 关掉任务通道线程就会退出；不等它（模型可能正算到一半）
        let _ = self.handle.take();
    }
}
