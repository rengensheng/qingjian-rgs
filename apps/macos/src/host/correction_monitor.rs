//! 神经纠错的两个定时器：停键后的防抖（到点才把作用域送去后台问模型）与结果轮询（到了就重查重画，平时不占 CPU）。
//!
//! 与整句重排（[`super::rescore_monitor`]）分开：纠错要等用户停稳（连着敲时一次都不问），防抖按秒计。

use std::time::{Duration, Instant};

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{MainThreadMarker, MainThreadOnly, define_class, msg_send, sel};
use objc2_foundation::{NSObject, NSObjectProtocol, NSTimer};

/// 停键多久才问模型：比一般的击键间隔长得多，连着敲时一次都不问；停稳了才花几十毫秒算一次。
const DEBOUNCE: f64 = 1.0;

/// 轮询间隔：模型一次几十毫秒。
const POLL_INTERVAL: f64 = 0.02;

/// 最长等多久；后台线程卡住时兜底。
const MAX_WAIT: Duration = Duration::from_secs(2);

/// 神经纠错的防抖与轮询定时器（与 [`RescoreMonitor`](super::rescore_monitor::RescoreMonitor) 同构，只是防抖更长、回调不同）。
pub struct CorrectionMonitor {
    /// 防抖定时器（一次性）；没在等为 `None`。
    debounce: Option<Retained<NSTimer>>,

    /// 轮询定时器；没在等结果为 `None`。
    poll: Option<Retained<NSTimer>>,

    /// 本轮开始等结果的时间。
    since: Option<Instant>,

    mtm: MainThreadMarker,
}

impl CorrectionMonitor {
    /// 新建（不启动任何定时器）。
    pub fn new(mtm: MainThreadMarker) -> Self {
        Self {
            debounce: None,
            poll: None,
            since: None,
            mtm,
        }
    }

    /// 又敲了一键：重新计时。
    pub fn schedule(&mut self) {
        if let Some(timer) = self.debounce.take() {
            timer.invalidate();
        }
        let target = CorrectionTicker::new(self.mtm);
        let timer = unsafe {
            NSTimer::scheduledTimerWithTimeInterval_target_selector_userInfo_repeats(
                DEBOUNCE,
                &target,
                sel!(fire:),
                None,
                false,
            )
        };
        self.debounce = Some(timer);
    }

    /// 请求已发出：开始轮询结果。
    pub fn start_polling(&mut self) {
        self.since = Some(Instant::now());
        if self.poll.is_some() {
            return;
        }
        let target = CorrectionTicker::new(self.mtm);
        let timer = unsafe {
            NSTimer::scheduledTimerWithTimeInterval_target_selector_userInfo_repeats(
                POLL_INTERVAL,
                &target,
                sel!(poll:),
                None,
                true,
            )
        };
        self.poll = Some(timer);
    }

    /// 停下两个定时器。
    pub fn stop(&mut self) {
        if let Some(timer) = self.debounce.take() {
            timer.invalidate();
        }
        if let Some(timer) = self.poll.take() {
            timer.invalidate();
        }
        self.since = None;
    }

    /// 结果等太久了。
    pub fn expired(&self) -> bool {
        self.since.is_some_and(|since| since.elapsed() > MAX_WAIT)
    }
}

define_class!(
    // SAFETY: NSObject 没有子类化要求；没有实现 Drop。
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[ivars = ()]
    struct CorrectionTicker;

    impl CorrectionTicker {
        #[unsafe(method(fire:))]
        fn fire(&self, _timer: Option<&AnyObject>) {
            crate::host::with(|h| h.start_correction());
        }

        #[unsafe(method(poll:))]
        fn poll(&self, _timer: Option<&AnyObject>) {
            crate::host::with(|h| h.poll_correction());
        }
    }

    unsafe impl NSObjectProtocol for CorrectionTicker {}
);

impl CorrectionTicker {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = mtm.alloc::<Self>().set_ivars(());
        unsafe { msg_send![super(this), init] }
    }
}
