//! 候选窗的鼠标命中测试：点击位置落在当前页第几格（从 0 数）。
//!
//! 纯函数：调用方（`view.rs`）按当前布局算好几何量传进来。竖排行高统一、横排逐项累加，与绘制用同一套数字，
//! 所以点中的格子与画出来的格子一致。拼音行 / 页码 / 空白处返回 `None`。

/// 竖排命中：点在主体顶部之下第几行；落在拼音行 / 下方页码处返回 `None`。
/// 行高是统一的（`view` 按列里最高的算），与绘制逐行累加的算法一致。
pub fn vertical_row(y: f64, top: f64, row_height: f64, count: usize) -> Option<usize> {
    if row_height <= 0.0 || y < top {
        return None;
    }
    let offset = ((y - top) / row_height) as usize;
    (offset < count).then_some(offset)
}

/// 横排命中：逐项比 x 范围；落在拼音行（含高亮译文行）返回 `None`。
pub fn horizontal_item(
    x: f64,
    y: f64,
    top: f64,
    start_x: f64,
    widths: &[f64],
    gap: f64,
) -> Option<usize> {
    if y < top {
        return None;
    }
    let mut x0 = start_x;
    for (offset, width) in widths.iter().enumerate() {
        if x >= x0 && x < x0 + width {
            return Some(offset);
        }
        x0 += width + gap;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_hit_counts_rows_from_body_top() {
        // 主体从 y=30 起，每行 20 高，共 3 行
        assert_eq!(vertical_row(10.0, 30.0, 20.0, 3), None);
        assert_eq!(vertical_row(30.0, 30.0, 20.0, 3), Some(0));
        assert_eq!(vertical_row(49.9, 30.0, 20.0, 3), Some(0));
        assert_eq!(vertical_row(50.0, 30.0, 20.0, 3), Some(1));
        assert_eq!(vertical_row(89.9, 30.0, 20.0, 3), Some(2));
        // 页码行与再往下：不管
        assert_eq!(vertical_row(90.0, 30.0, 20.0, 3), None);
        assert_eq!(vertical_row(200.0, 30.0, 20.0, 3), None);
        // 行高非法：不管
        assert_eq!(vertical_row(40.0, 30.0, 0.0, 3), None);
    }

    #[test]
    fn horizontal_hit_walks_items_with_gaps() {
        // 两项：[10, 50) 与 [60, 90)，拼音行在 y=30 之上
        let widths = [40.0, 30.0];
        assert_eq!(horizontal_item(20.0, 10.0, 30.0, 10.0, &widths, 10.0), None);
        assert_eq!(
            horizontal_item(10.0, 30.0, 30.0, 10.0, &widths, 10.0),
            Some(0)
        );
        assert_eq!(
            horizontal_item(49.9, 40.0, 30.0, 10.0, &widths, 10.0),
            Some(0)
        );
        // 两项之间的缝：不管
        assert_eq!(horizontal_item(55.0, 40.0, 30.0, 10.0, &widths, 10.0), None);
        assert_eq!(
            horizontal_item(60.0, 40.0, 30.0, 10.0, &widths, 10.0),
            Some(1)
        );
        assert_eq!(
            horizontal_item(200.0, 40.0, 30.0, 10.0, &widths, 10.0),
            None
        );
    }
}
