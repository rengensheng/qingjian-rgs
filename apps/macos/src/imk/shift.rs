//! 单击 Shift 的判定，喂的是控制器的 `FlagsChanged`（Shift 按下 / 抬起）与 `KeyDown`（别的键）。
//! 与 Windows 侧 `com::key::ShiftTap` 同一个规则：按下 Shift 到抬起之间没插进别的键，就是一次单击。

/// 单击 Shift 的判定状态。存在 Host 上，多个输入会话共用。
#[derive(Default)]
pub struct ShiftTap {
    /// Shift 按下后还没有别的键插进来。
    alone: bool,
}

impl ShiftTap {
    /// Shift 按下。
    pub fn shift_pressed(&mut self) {
        self.alone = true;
    }

    /// 任一普通键按下：Shift 正被当作组合键用，这次不算单击。
    pub fn key_down(&mut self) {
        self.alone = false;
    }

    /// Shift 抬起；单独抬起返回 `true`，一次抬起只算一次。
    pub fn shift_released(&mut self) -> bool {
        std::mem::replace(&mut self.alone, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tap_press_then_release_toggles_once() {
        let mut tap = ShiftTap::default();
        tap.shift_pressed();
        assert!(tap.shift_released());
        // 一次抬起只算一次
        assert!(!tap.shift_released());
    }

    #[test]
    fn key_between_press_and_release_cancels_tap() {
        let mut tap = ShiftTap::default();
        tap.shift_pressed();
        tap.key_down();
        assert!(!tap.shift_released());
    }

    #[test]
    fn release_without_press_does_nothing() {
        // 切输入源时按住 Shift 进来：没见过按下，抬起不切换
        let mut tap = ShiftTap::default();
        assert!(!tap.shift_released());
    }
}
