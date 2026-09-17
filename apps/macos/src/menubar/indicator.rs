//! 菜单栏里的「中 / 英」状态项，常驻显示。
//!
//! 输入源图标（Info.plist 的 tsInputMethodIconFileKey）没法动态换，所以自己放一个 NSStatusItem。
//! 模式是单击 Shift 切出来的软件状态（见 [`crate::host::Host::english`]），切换时直接刷新标题；
//! 定时器只做兜底（云朵标识变化、极端情况下的状态对齐）。
//!
//! 状态项一旦创建就**常驻、不再收起**：`setVisible(false)` 再 `setVisible(true)` 会把它重新排到菜单栏最左边，用户 ⌘ 拖到输入法图标旁的位置就丢了
//! （固定 autosave 名也保不住），而焦点每进出一次输入框 IMK 就 deactivate / activate 一轮，所以之前收成零宽的逻辑已去掉。
//! 停用时只停掉轮询定时器，标题保留最后的中英状态；切到别的输入法它还占着位置，显示的是青简这边最后的模式。

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{MainThreadMarker, MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::{NSMenu, NSStatusBar, NSStatusItem, NSVariableStatusItemLength};
use objc2_foundation::{NSObject, NSObjectProtocol, NSString, NSTimer, ns_string};

/// 刷新状态项的兜底间隔。
const POLL_INTERVAL: f64 = 0.25;

pub struct ModeIndicator {
    /// 菜单栏状态项，常驻显示。
    item: Retained<NSStatusItem>,

    /// 轮询定时器；未激活时为 `None`。
    timer: Option<Retained<NSTimer>>,

    /// 上次显示的是否英文模式，避免每次轮询都重设标题。
    english: Option<bool>,

    /// 云联想开着：标题带云朵，让用户一眼知道上下文会发出去。
    cloud: bool,

    mtm: MainThreadMarker,
}

impl ModeIndicator {
    pub fn new(mtm: MainThreadMarker) -> Self {
        let item = NSStatusBar::systemStatusBar().statusItemWithLength(NSVariableStatusItemLength);
        item.setAutosaveName(Some(ns_string!("QingjianModeIndicator")));
        item.setVisible(true);
        Self {
            item,
            timer: None,
            english: None,
            cloud: false,
            mtm,
        }
    }

    /// 输入法激活：保证状态项展开并开始轮询。
    pub fn activate(&mut self, english: bool) {
        self.english = None;
        self.update(english);
        if self.timer.is_none() {
            let target = ModeMonitor::new(self.mtm);
            let timer = unsafe {
                NSTimer::scheduledTimerWithTimeInterval_target_selector_userInfo_repeats(
                    POLL_INTERVAL,
                    &target,
                    sel!(tick:),
                    None,
                    true,
                )
            };
            self.timer = Some(timer);
        }
    }

    /// 输入法停用：只停掉轮询，状态项常驻、标题保持最后的模式。
    pub fn deactivate(&mut self) {
        if let Some(timer) = self.timer.take() {
            timer.invalidate();
        }
    }

    /// 点状态项弹出的菜单。
    pub fn set_menu(&self, menu: &NSMenu) {
        self.item.setMenu(Some(menu));
    }

    pub fn set_cloud(&mut self, cloud: bool) {
        self.cloud = cloud;
        self.english = None;
    }

    /// 按中英模式刷新标题。模式变化时调用方主动调，定时器只做兜底。
    pub fn update(&mut self, english: bool) {
        if self.english == Some(english) {
            return;
        }
        self.english = Some(english);
        if let Some(button) = self.item.button(self.mtm) {
            let mode = if english { "英" } else { "中" };
            let title = if self.cloud {
                format!("{mode} ☁︎")
            } else {
                mode.to_owned()
            };
            button.setTitle(&NSString::from_str(&title));
        }
    }
}

define_class!(
    // SAFETY: NSObject 没有子类化要求；没有实现 Drop。
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[ivars = ()]
    struct ModeMonitor;

    impl ModeMonitor {
        #[unsafe(method(tick:))]
        fn tick(&self, _timer: Option<&AnyObject>) {
            crate::host::with(|h| {
                let english = h.english;
                h.indicator.update(english);
            });
        }
    }

    unsafe impl NSObjectProtocol for ModeMonitor {}
);

impl ModeMonitor {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = mtm.alloc::<Self>().set_ivars(());
        unsafe { msg_send![super(this), init] }
    }
}
