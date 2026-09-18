//! 神经拼音纠错：字符级 Transformer encoder-decoder 的本地推理（candle）。
//!
//! 输入连写拼音（可能含错字），贪心解码出纠正后的连写拼音，给 Core 当规则纠错的兜底
//! （[`qingjian_core::correction::PinyinCorrector`]）。训练与权重导出见 `tools/corrector/`。
//!
//! 目录形态是两件套：`config.json`（结构超参数）+ `model.safetensors`（权重）。
//! 与整句模型（[`qjm`]）不同，这里的字表固定为 26 个小写字母（见 [`codec`]），
//! 所以没有 `vocab.json`；加载只认显式给的目录，不做目录嗅探。

mod codec;
mod config;
mod model;

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

pub use codec::{BOS, EOS, PAD, UNK, VOCAB_SIZE, decode_row, encode, normalize};
pub use config::CorrectorConfig;
use model::CorrectorModel;

/// 导出目录里的结构配置。
pub const CONFIG_FILE: &str = "config.json";

/// 导出目录里的权重。
pub const WEIGHTS_FILE: &str = "model.safetensors";

/// 输入最多多少个字母：超过的直接拒掉，编码器没见过这么长的串，硬解也是胡话。
const MAX_INPUT_LETTERS: usize = 48;

/// 导出目录的落点：`config.json` + `model.safetensors` 都在的目录才算，返回目录本身；
/// 用户自己训的放用户目录，随包的放 Resources，两边都没有就是 `None`。
pub fn find_corrector(dir: &Path) -> Option<PathBuf> {
    (dir.join(CONFIG_FILE).is_file() && dir.join(WEIGHTS_FILE).is_file()).then(|| dir.to_path_buf())
}

/// 加载好的纠错模型：Core [`PinyinCorrector`](qingjian_core::correction::PinyinCorrector) 的实现。
pub struct NeuralCorrector {
    /// 推理用的 encoder-decoder。
    model: CorrectorModel,

    /// 推理设备（CPU，`metal` feature 下走 Apple GPU）。
    device: Device,
}

impl NeuralCorrector {
    /// 加载模型：`dir` 是 `tools/corrector/export.py` 导出的目录（`config.json` + `model.safetensors`）。
    pub fn load(dir: &Path) -> Result<Self, crate::NeuralError> {
        let config_path = dir.join(CONFIG_FILE);
        let text =
            std::fs::read_to_string(&config_path).map_err(|source| crate::NeuralError::Io {
                path: config_path.clone(),
                source,
            })?;
        let cfg = CorrectorConfig::from_json(&text, &config_path)?;
        let device = default_device()?;
        // SAFETY：mmap 的权重文件在模型存活期间不改动
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join(WEIGHTS_FILE)], DType::F32, &device)?
        };
        let model = CorrectorModel::load(vb, &cfg, device.clone())?;
        tracing::info!(
            source = %dir.display(),
            d_model = cfg.d_model,
            layers = cfg.enc_layers + cfg.dec_layers,
            "拼音纠错模型已加载"
        );
        Ok(Self { model, device })
    }

    /// 纠正一条连写拼音，出错返回 `Err`（调用方按 trait 约定吞掉，给空提名）。
    fn correct_one(&self, text: &str) -> Result<String, crate::NeuralError> {
        let cleaned = normalize(text);
        if cleaned.is_empty() || cleaned.len() > MAX_INPUT_LETTERS {
            return Err(crate::NeuralError::Corrupt("input is empty or too long"));
        }
        let mut ids = Vec::with_capacity(cleaned.len() + 2);
        ids.push(BOS);
        ids.extend(encode(&cleaned));
        ids.push(EOS);
        let src = Tensor::from_vec(ids.clone(), (1, ids.len()), &self.device)?;
        let rows = self.model.generate(&src, ids.len() + 8)?;
        Ok(decode_row(&rows[0]))
    }
}

impl qingjian_core::correction::PinyinCorrector for NeuralCorrector {
    /// 纠正 `input`：模型只出一个最可能的串；输入不像拼音或推理失败时给空提名。
    fn correct(&self, input: &str) -> Vec<String> {
        match self.correct_one(input) {
            Ok(fixed) if !fixed.is_empty() => vec![fixed],
            Ok(_) => Vec::new(),
            Err(error) => {
                tracing::warn!(%error, "拼音纠错模型推理失败");
                Vec::new()
            }
        }
    }
}

/// 推理设备：固定 CPU。纠错是单 batch 短序列（几十个 token），Metal 的 kernel 下发开销反而比计算本身贵；
/// `metal` feature 开着的壳里也一样，实测的延迟数（见 crate-notes）就是这条路径的。
/// 以后批量解码或模型变大时再按需切 Metal。
fn default_device() -> Result<Device, crate::NeuralError> {
    Ok(Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qingjian_core::correction::PinyinCorrector;

    /// 真模型冒烟：`QINGJIAN_CORRECTOR_DIR` 指向导出目录时才跑（`tools/corrector/export.py` 的用法）。
    /// 只断言结构（干净拼音、确定性），不对拍具体串：换权重后行为会变，对拍按导出脚本里的说明手动比。
    #[test]
    fn corrector_outputs_clean_pinyin() {
        let Some(dir) = std::env::var_os("QINGJIAN_CORRECTOR_DIR") else {
            eprintln!("没有纠错模型导出目录，跳过");
            return;
        };
        let corrector = NeuralCorrector::load(std::path::Path::new(&dir)).unwrap();
        for input in [
            "zhongguo",
            "zongguo",
            "zhoongguo",
            "shuurufa",
            "nihooma",
            "mingtain",
            "",
        ] {
            let out = corrector.correct(input);
            assert!(out.len() <= 1, "{input} -> {out:?}");
            for fixed in &out {
                assert!(
                    !fixed.is_empty() && fixed.bytes().all(|b| b.is_ascii_lowercase()),
                    "{input} -> {out:?}"
                );
            }
        }
        // 确定性：贪心解码同一串两次结果一致
        assert_eq!(
            corrector.correct("zhoongguo"),
            corrector.correct("zhoongguo")
        );
    }
}

#[cfg(test)]
mod latency {
    use super::*;
    use qingjian_core::correction::PinyinCorrector;

    /// 延迟探针：`QINGJIAN_CORRECTOR_DIR` 指向导出目录。
    /// `QINGJIAN_CORRECTOR_DIR=data/corrector cargo test --release -p qingjian-neural -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn corrector_latency() {
        let Some(dir) = std::env::var_os("QINGJIAN_CORRECTOR_DIR") else {
            eprintln!("没有纠错模型导出目录，跳过");
            return;
        };
        let start = std::time::Instant::now();
        let corrector = NeuralCorrector::load(std::path::Path::new(&dir)).unwrap();
        println!("加载: {:.0} ms", start.elapsed().as_secs_f64() * 1000.0);
        let _ = corrector.correct("zhoongguo");
        let start = std::time::Instant::now();
        let n = 10;
        for _ in 0..n {
            let _ = corrector.correct("zhoongguo");
        }
        println!(
            "单次纠错（已预热）: {:.2} ms",
            start.elapsed().as_secs_f64() * 1000.0 / f64::from(n)
        );
    }
}
