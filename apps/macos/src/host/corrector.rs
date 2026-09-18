//! 神经拼音纠错模型：后台加载，停 1 秒后才问，提名到了重查一次。
//!
//! 与整句模型（[`super::model`]）同一套接法：后台线程加载 + 预热，加载完由 [`Host::attach_loaded_corrector`] 以异步
//! worker 接上。不同的是触发：查询当场不等模型，只记下规则修不动的作用域；`CorrectionMonitor` 停 1 秒防抖后送去后台，
//! 提名到了重查（验证与噪声信道都在这次重查里）再重画当前页。连着敲时一次都不问，平时一键都不碰模型。
//! 没有模型文件就什么都不做。

use std::sync::mpsc::{TryRecvError, channel};

use qingjian_core::correction::PinyinCorrector;
use qingjian_neural::{NeuralCorrector, NeuralError};

use super::*;

impl Host {
    /// 在后台线程加载纠错模型并预热（首个前向几十毫秒），加载完由 [`Self::attach_loaded_corrector`] 接上。
    /// 没有模型文件就什么都不做。
    pub(super) fn load_corrector(&mut self) {
        if self.corrector_loader.is_some() || self.engine.has_neural_corrector() {
            return;
        }
        let Some(path) = paths::corrector_path() else {
            tracing::info!("没有神经拼音纠错模型文件，只用规则纠错");
            return;
        };
        let (tx, rx) = channel::<Result<NeuralCorrector, NeuralError>>();
        let spawned = std::thread::Builder::new()
            .name("qingjian-corrector-load".to_owned())
            .spawn(move || {
                let started = std::time::Instant::now();
                let loaded = NeuralCorrector::load(&path).inspect(|corrector| {
                    // 预热一次， Metal 内核编译 / 首个前向的开销现在付，查询时只剩推理本身
                    let _ = corrector.correct("nihao");
                });
                if loaded.is_ok() {
                    tracing::info!(
                        path = %path.display(),
                        total_ms = started.elapsed().as_millis(),
                        "神经拼音纠错模型已加载并预热"
                    );
                }
                let _ = tx.send(loaded);
            });
        match spawned {
            Ok(_) => self.corrector_loader = Some(rx),
            Err(error) => tracing::warn!(%error, "起不了纠错模型加载线程，只用规则纠错"),
        }
    }

    /// 加载线程有结果了就接到 Engine 上；每次查询顺手看一眼，不阻塞。
    /// 接的是异步 worker：查询不等模型，停 1 秒后才问，提名到了重查一次。
    pub fn attach_loaded_corrector(&mut self) {
        let Some(rx) = &self.corrector_loader else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(corrector)) => {
                self.engine
                    .set_async_neural_corrector(Some(Box::new(corrector)));
                self.corrector_loader = None;
            }
            Ok(Err(error)) => {
                tracing::warn!(%error, "神经拼音纠错模型加载失败，只用规则纠错");
                self.corrector_loader = None;
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => self.corrector_loader = None,
        }
    }

    /// 卸掉纠错模型（配置关掉）。
    pub(super) fn unload_corrector(&mut self) {
        self.corrector_loader = None;
        self.engine.set_async_neural_corrector(None);
        self.correction.stop();
    }

    /// 每次查询之后：有作用域等着问模型就起防抖计时（停 1 秒才问，连着敲时一次都不问）。
    pub fn schedule_correction(&mut self) {
        if self.engine.correction_pending() {
            self.correction.schedule();
        }
    }

    /// 防抖到点：把记下的作用域送去后台，开始轮询。
    pub fn start_correction(&mut self) {
        if self.engine.composition().is_empty() {
            self.correction.stop();
            return;
        }
        if self.engine.request_correction() {
            self.correction.start_polling();
        }
    }

    /// 轮询到点：提名到了就重查一次、重画当前页；用户已翻页或动过高亮就只留着提名不动画面。
    /// 验证与噪声信道都在这次重查里，不过照样不纠。
    pub fn poll_correction(&mut self) {
        if self.engine.composition().is_empty() || self.translation.is_some() {
            self.correction.stop();
            return;
        }
        if !self.engine.poll_correction() {
            // 等太久多半是组句已经变了、提名作废；真卡住也只是这轮不纠
            if self.correction.expired() {
                tracing::debug!("等神经纠错超时，本轮不纠");
                self.correction.stop();
            }
            return;
        }
        self.correction.stop();
        if self.session.page != 0 || self.session.navigated {
            return;
        }
        let Ok(mut query) = self.engine.query() else {
            return;
        };
        self.engine.annotate(&mut query.candidates);
        let preedit = Preedit::from_marked(&query.marked_segments(), query.marked_cursor());
        let cloud = self.session.layout.cloud().to_vec();
        self.reset_session(preedit, query.candidates.items);
        if !cloud.is_empty() {
            self.session.layout.set_cloud(cloud);
        }
        self.render();
    }
}
