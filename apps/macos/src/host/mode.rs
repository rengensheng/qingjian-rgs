//! 中英模式：单击 Shift 在中 / 英之间翻转，Caps Lock 只管字母大小写（与 Windows 一致）。
//!
//! 单击的判定在 [`crate::imk::shift::ShiftTap`]，状态存在 [`super::Host`] 上是因为多个输入会话共用同一个 Host。

use super::Host;

impl Host {
    /// 翻转中英模式，返回组句中还没上屏的拼音（调用方负责插进应用）。
    ///
    /// 组句中切换按「先提交再切换」：拼音原样上屏，不当英文也不当拼音，避免半截拼音留在新模式里。
    pub fn toggle_english(&mut self) -> String {
        self.set_english(!self.english);
        self.engine.take_raw()
    }

    /// 直接指定中英模式。菜单入口：点菜单时没有 client 可上屏，组句中的拼音下一次按键时按新模式处理
    ///（切回中文的那次会先原样上屏，见 `controller::handle_text`）。
    pub fn set_english(&mut self, english: bool) {
        self.english = english;
        tracing::info!(english = self.english, "切换中英模式");
        self.indicator.update(self.english);
    }
}
