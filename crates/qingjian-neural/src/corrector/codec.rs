//! 字符级字表：与训练侧 `vocab.py` 同一套编号（`a-z` 26 个字母 + 4 个特殊标记），写死在代码里。
//!
//! 推理输入输出都是连写拼音（无空格），`normalize` 与训练侧 `infer.normalize` 同步：
//! 去掉所有空白并小写。

/// 填充位。
pub const PAD: u32 = 0;

/// 解码起始位。
pub const BOS: u32 = 1;

/// 解码结束位。
pub const EOS: u32 = 2;

/// 未登录字符（推理输入里出现非字母时落到这里，`correct` 入口处已拦）。
pub const UNK: u32 = 3;

/// 字母 `a` 的编号。
pub const OFFSET: u32 = 4;

/// 字表大小：4 个特殊标记 + 26 个字母。
pub const VOCAB_SIZE: u32 = 30;

/// 文本 -> id 列表（非小写字母记为 [`UNK`]）。
pub fn encode(text: &str) -> Vec<u32> {
    text.bytes()
        .map(|b| {
            if b.is_ascii_lowercase() {
                u32::from(b - b'a') + OFFSET
            } else {
                UNK
            }
        })
        .collect()
}

/// id 行 -> 文本：遇到 [`EOS`] 截断，跳过特殊标记。
pub fn decode_row(ids: &[u32]) -> String {
    let mut out = String::with_capacity(ids.len());
    for &id in ids {
        if id == EOS {
            break;
        }
        if (OFFSET..VOCAB_SIZE).contains(&id) {
            out.push((b'a' + (id - OFFSET) as u8) as char);
        }
    }
    out
}

/// 规整输入：去掉所有空白并小写（连写拼音）。
pub fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<String>()
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_round_trips_and_stops_at_eos() {
        assert_eq!(encode("zhong"), vec![29, 11, 18, 17, 10]);
        assert_eq!(decode_row(&encode("zhongguo")), "zhongguo");
        assert_eq!(decode_row(&[BOS, 11, EOS, 12]), "h");
        assert_eq!(decode_row(&[PAD, BOS, UNK]), "");
        assert_eq!(normalize(" Zhong Guo\n"), "zhongguo");
    }
}
