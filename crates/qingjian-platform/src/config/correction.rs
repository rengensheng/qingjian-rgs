use serde::{Deserialize, Serialize};

/// 配置文件 `[correction]` 分节：神经拼音纠错的开关。
///
/// 错了不止一处、规则修不动的拼音，问本地纠错模型要提名，全程离线、不联网，与整句模型互不影响
/// （一个管「敲错的拼音怎么修」，一个管「整句怎么排」）。模型文件不在包里（或用户目录 `corrector/` 里）时开关无效。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CorrectionConfig {
    /// 开着就加载模型、给规则修不动的拼音兜底。
    pub enabled: bool,
}

impl Default for CorrectionConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}
