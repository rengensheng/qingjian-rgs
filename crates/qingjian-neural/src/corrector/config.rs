use std::path::Path;

use serde::Deserialize;

use super::codec::VOCAB_SIZE;
use crate::NeuralError;

/// 纠错模型的结构，来自导出目录的 `config.json`（`tools/corrector/export.py` 从训练检查点写出）。
#[derive(Debug, Clone, Deserialize)]
pub struct CorrectorConfig {
    /// 隐层宽度。
    pub d_model: usize,

    /// 注意力头数。
    pub nhead: usize,

    /// 编码器层数。
    pub enc_layers: usize,

    /// 解码器层数。
    pub dec_layers: usize,

    /// 前馈层宽度。
    pub dim_ff: usize,

    /// 位置编码的行数（训练时的 `max_len`）。
    pub max_len: usize,

    /// 字表大小：固定 30（见 [`VOCAB_SIZE`](super::codec::VOCAB_SIZE)）。
    pub vocab_size: usize,
}

impl CorrectorConfig {
    /// 解析 `config.json` 的正文并校验；`path` 只用来报错。
    pub(crate) fn from_json(text: &str, path: &Path) -> Result<Self, NeuralError> {
        let cfg: Self = serde_json::from_str(text).map_err(|source| NeuralError::Json {
            path: path.to_owned(),
            source,
        })?;
        if cfg.vocab_size != VOCAB_SIZE as usize {
            return Err(NeuralError::Corrupt("corrector vocab must be 30"));
        }
        if cfg.d_model == 0 || cfg.nhead == 0 || !cfg.d_model.is_multiple_of(cfg.nhead) {
            return Err(NeuralError::Corrupt("d_model must be divisible by nhead"));
        }
        if cfg.max_len == 0 {
            return Err(NeuralError::Corrupt("max_len must be positive"));
        }
        Ok(cfg)
    }
}
