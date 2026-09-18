//! 字符级纠错 Transformer 的 candle 推理：与训练侧 `model.py::PinyinCorrector` 同构。
//!
//! 权重名直接沿用 torch 的层名（`encoder.layers.0.self_attn.in_proj_weight` 等），
//! 导出脚本原样转存 safetensors，这里按名取张量。语义对齐点：
//! pre-LN（`norm_first=True`）、前馈是 ReLU（torch 缺省）、注意力缩放 `1/√d_h`、
//! 掩码（因果上三角 + padding 位）以极小值加进 logits。

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, layer_norm, ops};

use super::codec::PAD;
use super::config::CorrectorConfig;

/// 掩码里忽略位置加的值：够大到 softmax 后为 0，又在 f16 范围内（与整句模型同值）。
const MASKED: f32 = -1.0e4;

/// 多头注意力：Q/K/V 三个投影存成一个 `in_proj`（与 torch 的打包方式一致），用时再切。
struct Attention {
    /// Q 投影（`d → d`）。
    q_proj: Linear,

    /// K 投影（`d → d`）。
    k_proj: Linear,

    /// V 投影（`d → d`）。
    v_proj: Linear,

    /// 输出投影（`d → d`）。
    out_proj: Linear,

    /// 头数。
    n_head: usize,
}

impl Attention {
    /// 从 `self_attn` / `multihead_attn` 前缀下载入（torch 的 `in_proj_weight` 是 Q/K/V 三段拼起来的）。
    fn load(vb: VarBuilder, n_head: usize, d_model: usize) -> Result<Self> {
        let weight = vb.get((3 * d_model, d_model), "in_proj_weight")?;
        let bias = vb.get(3 * d_model, "in_proj_bias")?;
        let split = |row: usize| -> Result<Linear> {
            Ok(Linear::new(
                weight.narrow(0, row * d_model, d_model)?,
                Some(bias.narrow(0, row * d_model, d_model)?),
            ))
        };
        Ok(Self {
            q_proj: split(0)?,
            k_proj: split(1)?,
            v_proj: split(2)?,
            out_proj: Linear::new(
                vb.get((d_model, d_model), "out_proj.weight")?,
                Some(vb.get(d_model, "out_proj.bias")?),
            ),
            n_head,
        })
    }

    /// `queries` 形状 `[b, t, d]`，`keys` / `values` 形状 `[b, s, d]`（自注意力时三者相同）；
    /// `masks` 是要加进 logits 的 `[b, h, t, s]` / `[t, s]` / `[b, 1, 1, s]` 加性掩码，可多个叠加。
    fn forward(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        masks: &[&Tensor],
    ) -> Result<Tensor> {
        let (b, t, d) = queries.dims3()?;
        let s = keys.dim(1)?;
        let (h, dh) = (self.n_head, d / self.n_head);
        let project = |proj: &Linear, x: &Tensor, len: usize| {
            proj.forward(x)?
                .reshape((b, len, h, dh))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = project(&self.q_proj, queries, t)?;
        let k = project(&self.k_proj, keys, s)?;
        let v = project(&self.v_proj, values, s)?;
        let scale = 1.0 / (dh as f64).sqrt();
        let mut att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        for mask in masks {
            att = att.broadcast_add(mask)?;
        }
        let att = ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v)?;
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, d))?;
        self.out_proj.forward(&y)
    }
}

/// 前馈：`dim_ff → ReLU → d_model`（torch 缺省激活就是 ReLU，不是 GELU）。
struct FeedForward {
    /// 升维（`d → dim_ff`）。
    up: Linear,

    /// 降维（`dim_ff → d`）。
    down: Linear,
}

impl FeedForward {
    fn load(vb: VarBuilder, d_model: usize, dim_ff: usize) -> Result<Self> {
        Ok(Self {
            up: Linear::new(
                vb.get((dim_ff, d_model), "linear1.weight")?,
                Some(vb.get(dim_ff, "linear1.bias")?),
            ),
            down: Linear::new(
                vb.get((d_model, dim_ff), "linear2.weight")?,
                Some(vb.get(d_model, "linear2.bias")?),
            ),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down.forward(&self.up.forward(x)?.relu()?)
    }
}

/// 编码器一层：pre-LN 自注意力 + pre-LN 前馈，都带残差。
struct EncoderLayer {
    /// 自注意力。
    attn: Attention,

    /// 前馈。
    ff: FeedForward,

    /// 注意力前的归一化。
    norm1: LayerNorm,

    /// 前馈前的归一化。
    norm2: LayerNorm,
}

impl EncoderLayer {
    fn load(vb: VarBuilder, cfg: &CorrectorConfig) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(vb.pp("self_attn"), cfg.nhead, cfg.d_model)?,
            ff: FeedForward::load(vb.clone(), cfg.d_model, cfg.dim_ff)?,
            norm1: layer_norm(cfg.d_model, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(cfg.d_model, 1e-5, vb.pp("norm2"))?,
        })
    }

    /// `pad` 是 `[b, 1, 1, s]` 加性掩码（padding 位为极小值）。
    fn forward(&self, x: &Tensor, pad: &Tensor) -> Result<Tensor> {
        let att = self.attn.forward(&self.norm1.forward(x)?, x, x, &[pad])?;
        let x = (x + att)?;
        let ff = self.ff.forward(&self.norm2.forward(&x)?)?;
        x + ff
    }
}

/// 解码器一层：pre-LN 自注意力（因果）+ pre-LN 交叉注意力 + pre-LN 前馈，都带残差。
struct DecoderLayer {
    /// 自注意力。
    self_attn: Attention,

    /// 交叉注意力（K/V 来自编码器）。
    cross_attn: Attention,

    /// 前馈。
    ff: FeedForward,

    /// 自注意力前的归一化。
    norm1: LayerNorm,

    /// 交叉注意力前的归一化。
    norm2: LayerNorm,

    /// 前馈前的归一化。
    norm3: LayerNorm,
}

impl DecoderLayer {
    fn load(vb: VarBuilder, cfg: &CorrectorConfig) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::load(vb.pp("self_attn"), cfg.nhead, cfg.d_model)?,
            cross_attn: Attention::load(vb.pp("multihead_attn"), cfg.nhead, cfg.d_model)?,
            ff: FeedForward::load(vb.clone(), cfg.d_model, cfg.dim_ff)?,
            norm1: layer_norm(cfg.d_model, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(cfg.d_model, 1e-5, vb.pp("norm2"))?,
            norm3: layer_norm(cfg.d_model, 1e-5, vb.pp("norm3"))?,
        })
    }

    /// `causal` 是 `[t, t]` 因果掩码，`tgt_pad` / `mem_pad` 是 `[b, 1, 1, t/s]` padding 掩码。
    fn forward(
        &self,
        x: &Tensor,
        memory: &Tensor,
        causal: &Tensor,
        tgt_pad: &Tensor,
        mem_pad: &Tensor,
    ) -> Result<Tensor> {
        let att = self
            .self_attn
            .forward(&self.norm1.forward(x)?, x, x, &[causal, tgt_pad])?;
        let x = (x + att)?;
        let normed = self.norm2.forward(&x)?;
        let cross = self
            .cross_attn
            .forward(&normed, memory, memory, &[mem_pad])?;
        let x = (x + cross)?;
        let ff = self.ff.forward(&self.norm3.forward(&x)?)?;
        x + ff
    }
}

/// 字符级 encoder-decoder 纠错模型。张量名见 `tools/corrector/export.py`（训练侧原样转存）。
pub(crate) struct CorrectorModel {
    /// 错拼序列的嵌入。
    src_emb: Embedding,

    /// 已生成序列的嵌入。
    tgt_emb: Embedding,

    /// 正弦位置编码 `[max_len, d_model]`（训练侧 `PositionalEncoding` 的 buffer，无参数、这里现算）。
    pos: Tensor,

    /// 编码器层。
    enc_layers: Vec<EncoderLayer>,

    /// 解码器层。
    dec_layers: Vec<DecoderLayer>,

    /// 输出层（`d_model → 30`）。
    out: Linear,

    /// 设备。
    device: Device,
}

impl CorrectorModel {
    pub(crate) fn load(vb: VarBuilder, cfg: &CorrectorConfig, device: Device) -> Result<Self> {
        let d = cfg.d_model;
        let mut enc_layers = Vec::with_capacity(cfg.enc_layers);
        for i in 0..cfg.enc_layers {
            enc_layers.push(EncoderLayer::load(
                vb.pp(format!("encoder.layers.{i}")),
                cfg,
            )?);
        }
        let mut dec_layers = Vec::with_capacity(cfg.dec_layers);
        for i in 0..cfg.dec_layers {
            dec_layers.push(DecoderLayer::load(
                vb.pp(format!("decoder.layers.{i}")),
                cfg,
            )?);
        }
        Ok(Self {
            src_emb: Embedding::new(vb.get((cfg.vocab_size, d), "src_emb.weight")?, d),
            tgt_emb: Embedding::new(vb.get((cfg.vocab_size, d), "tgt_emb.weight")?, d),
            pos: sinusoidal(cfg.max_len, d, &device)?,
            enc_layers,
            dec_layers,
            out: Linear::new(
                vb.get((cfg.vocab_size, d), "out.weight")?,
                Some(vb.get(cfg.vocab_size, "out.bias")?),
            ),
            device,
        })
    }

    /// 编码错拼序列：`src_ids` 是每行的 id（含 BOS/EOS 与 PAD 填充）。
    fn encode(&self, src_ids: &[Vec<u32>]) -> Result<Tensor> {
        let (b, s) = (src_ids.len(), src_ids[0].len());
        let flat: Vec<u32> = src_ids.iter().flatten().copied().collect();
        let src = Tensor::from_vec(flat, (b, s), &self.device)?;
        let pad = pad_mask(src_ids, &self.device)?;
        let pos = self.pos.narrow(0, 0, s)?;
        let mut x = self.src_emb.forward(&src)?.broadcast_add(&pos)?;
        for layer in &self.enc_layers {
            x = layer.forward(&x, &pad)?;
        }
        Ok(x)
    }

    /// 解码一步：已生成的 `tgt`（`[b, t]`）每个位置的输出分布 `[b, t, 30]`。
    fn decode(
        &self,
        tgt: &Tensor,
        tgt_ids: &[Vec<u32>],
        memory: &Tensor,
        mem_pad: &Tensor,
    ) -> Result<Tensor> {
        let t = tgt.dim(1)?;
        let causal = causal_mask(t, &self.device)?;
        let tgt_pad = pad_mask(tgt_ids, &self.device)?;
        let pos = self.pos.narrow(0, 0, t)?;
        let mut x = self.tgt_emb.forward(tgt)?.broadcast_add(&pos)?;
        for layer in &self.dec_layers {
            x = layer.forward(&x, memory, &causal, &tgt_pad, mem_pad)?;
        }
        self.out.forward(&x)
    }

    /// 贪心解码：从 BOS 开始逐个取最可能的 id，遇到 EOS 停；返回每行的 id（含 EOS，后面截掉）。
    /// `max_len` 是解码步数上限（训练侧按 `src 宽 + 8` 给）。
    pub(crate) fn generate(&self, src: &Tensor, max_len: usize) -> Result<Vec<Vec<u32>>> {
        let (b, _) = src.dims2()?;
        let src_ids = src.to_vec2::<u32>()?;
        let memory = self.encode(&src_ids)?;
        let mem_pad = pad_mask(&src_ids, &self.device)?;
        let mut tgt_ids = vec![vec![super::codec::BOS]; b];
        let mut finished = vec![false; b];
        for _ in 0..max_len {
            let width = tgt_ids[0].len();
            let flat: Vec<u32> = tgt_ids.iter().flatten().copied().collect();
            let tgt = Tensor::from_vec(flat, (b, width), &self.device)?;
            let logits = self.decode(&tgt, &tgt_ids, &memory, &mem_pad)?;
            let last = logits.narrow(1, width - 1, 1)?.squeeze(1)?;
            let next = last.argmax(D::Minus1)?.to_vec1::<u32>()?;
            let mut all_done = true;
            for (row, &id) in tgt_ids.iter_mut().zip(&next) {
                row.push(id);
            }
            for (done, &id) in finished.iter_mut().zip(&next) {
                *done |= id == super::codec::EOS;
                all_done &= *done;
            }
            if all_done {
                break;
            }
        }
        Ok(tgt_ids
            .into_iter()
            .map(|row| row.into_iter().skip(1).collect())
            .collect())
    }
}

/// 正弦位置编码 `[len, d]`（与训练侧 `PositionalEncoding` 同公式），f32。
fn sinusoidal(len: usize, d: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0.0f32; len * d];
    for pos in 0..len {
        for i in 0..d / 2 {
            let div = (-(i as f32 * 2.0) * (10000.0f32.ln()) / d as f32).exp();
            data[pos * d + i * 2] = (pos as f32 * div).sin();
            data[pos * d + i * 2 + 1] = (pos as f32 * div).cos();
        }
    }
    Tensor::from_vec(data, (len, d), device)
}

/// 因果掩码 `[t, t]`：未来位置为极小值（与训练侧 `causal_mask` 同含义）。
fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0.0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            data[i * t + j] = MASKED;
        }
    }
    Tensor::from_vec(data, (t, t), device)?.to_dtype(DType::F32)
}

/// padding 加性掩码 `[b, 1, 1, s]`：PAD 位为极小值（torch 的 `key_padding_mask=True` 即忽略）。
fn pad_mask(ids: &[Vec<u32>], device: &Device) -> Result<Tensor> {
    let (b, s) = (ids.len(), ids[0].len());
    let mut data = vec![0.0f32; b * s];
    for (row, ids) in ids.iter().enumerate() {
        for (col, &id) in ids.iter().enumerate() {
            if id == PAD {
                data[row * s + col] = MASKED;
            }
        }
    }
    Tensor::from_vec(data, (b, 1, 1, s), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// 随机小模型上跑通贪心解码：验证权重名映射与掩码形状不断链（数值正确性靠与 Python 的对拍，见 `tools/corrector/`）。
    #[test]
    fn greedy_decode_runs_on_random_weights() {
        let device = Device::Cpu;
        let cfg = CorrectorConfig {
            d_model: 8,
            nhead: 2,
            enc_layers: 1,
            dec_layers: 1,
            dim_ff: 16,
            max_len: 16,
            vocab_size: crate::corrector::VOCAB_SIZE as usize,
        };
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let mut put = |name: &str, shape: &[usize]| {
            tensors.insert(
                name.to_owned(),
                Tensor::randn(0.0f32, 0.5, shape, &device).unwrap(),
            );
        };
        put("src_emb.weight", &[30, 8]);
        put("tgt_emb.weight", &[30, 8]);
        put("out.weight", &[30, 8]);
        put("out.bias", &[30]);
        for side in ["encoder", "decoder"] {
            for i in 0..1 {
                let base = format!("{side}.layers.{i}");
                let attns: &[&str] = if side == "encoder" {
                    &["self_attn"]
                } else {
                    &["self_attn", "multihead_attn"]
                };
                for attn in attns {
                    put(&format!("{base}.{attn}.in_proj_weight"), &[24, 8]);
                    put(&format!("{base}.{attn}.in_proj_bias"), &[24]);
                    put(&format!("{base}.{attn}.out_proj.weight"), &[8, 8]);
                    put(&format!("{base}.{attn}.out_proj.bias"), &[8]);
                }
                put(&format!("{base}.linear1.weight"), &[16, 8]);
                put(&format!("{base}.linear1.bias"), &[16]);
                put(&format!("{base}.linear2.weight"), &[8, 16]);
                put(&format!("{base}.linear2.bias"), &[8]);
                put(&format!("{base}.norm1.weight"), &[8]);
                put(&format!("{base}.norm1.bias"), &[8]);
                put(&format!("{base}.norm2.weight"), &[8]);
                put(&format!("{base}.norm2.bias"), &[8]);
                if side == "decoder" {
                    put(&format!("{base}.norm3.weight"), &[8]);
                    put(&format!("{base}.norm3.bias"), &[8]);
                }
            }
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = CorrectorModel::load(vb, &cfg, device.clone()).unwrap();
        let src = Tensor::from_vec(
            vec![crate::corrector::BOS, 11, 18, crate::corrector::EOS],
            (1, 4),
            &device,
        )
        .unwrap();
        let rows = model.generate(&src, 6).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].len() <= 6, "{:?}", rows[0]);
        assert!(rows[0].iter().all(|&id| id < 30), "{:?}", rows[0]);
    }
}
