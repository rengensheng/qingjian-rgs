//! 神经拼音纠错的注入点：把「敲的拼音串」变成「纠正后的拼音串」的模型，只认这个 trait。
//!
//! 规则纠错（[`candidates`](super::candidates)）只修一处编辑；两处以上的错拼（`zongguoo` → `zhongguo`）
//! 它够不着，这时 Engine 才问这里挂上的模型（见 [`Engine::find_correction`](crate::engine::Engine::find_correction)）。
//! 模型只管提名：返回的每个串仍要过「完整切分 + 整句得分扣编辑代价仍高于原样」这道噪声信道，
//! 通不过就当没说，原样优先的原则不变。
//!
//! 实现在 `qingjian-neural`（字符级 Transformer encoder-decoder 的 candle 推理）。
//! 按键回调永远不等模型：没挂模型时这条路不存在，挂了也只在规则纠错无果时跑一次，结果按作用域缓存。

/// 神经拼音纠错模型：输入作用域里的拼音（纯小写字母、不含 `'`），返回纠正后的拼音串。
///
/// 调用方只取前几个；算不了（模型出错）返回空 `Vec`，调用方就当没有这个提名。
pub trait PinyinCorrector: Send {
    /// 纠正 `input`，返回候选的纠正后拼音串（连写、无空格），按可信度从高到低。
    fn correct(&self, input: &str) -> Vec<String>;
}
