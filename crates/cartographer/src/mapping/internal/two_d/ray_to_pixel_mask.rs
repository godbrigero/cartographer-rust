//! Port of `cartographer/mapping/internal/2d/ray_to_pixel_mask` (.h, .cc).

use crate::mapping::two_d::xy_index::Array2i;

/// Pushes `current` unless it equals the last pixel.
#[inline]
fn push_if_new(pixel_mask: &mut Vec<Array2i>, current: Array2i) {
    if pixel_mask.last() != Some(&current) {
        pixel_mask.push(current);
    }
}

/// Computes all pixels that contain some part of the line segment connecting
/// `scaled_begin` and `scaled_end`. `scaled_begin` and `scaled_end` are
/// scaled by `subpixel_scale`. `scaled_begin` and `scaled_end` are expected
/// to be greater than zero. Return values are in pixels and not scaled, and
/// ordered by ascending x.
///
/// The accumulator arithmetic uses `i64` (as in C++) since the products
/// overflow `i32` for `subpixel_scale = 1000`.
pub fn ray_to_pixel_mask(
    scaled_begin: &Array2i,
    scaled_end: &Array2i,
    subpixel_scale: i32,
) -> Vec<Array2i> {
    let mut pixel_mask = Vec::new();
    ray_to_pixel_mask_into(scaled_begin, scaled_end, subpixel_scale, &mut pixel_mask);
    pixel_mask
}

/// Same as [`ray_to_pixel_mask`], but writes the pixels into `pixel_mask`
/// (cleared first), so that callers casting many rays can reuse one buffer.
///
/// RUST PORT NOTE: added for performance (the C++ returns a fresh
/// `std::vector` per ray); the produced pixels are identical.
pub fn ray_to_pixel_mask_into(
    scaled_begin: &Array2i,
    scaled_end: &Array2i,
    subpixel_scale: i32,
    pixel_mask: &mut Vec<Array2i>,
) {
    // For simplicity, we order 'scaled_begin' and 'scaled_end' by their x
    // coordinate.
    if scaled_begin.x > scaled_end.x {
        return ray_to_pixel_mask_into(scaled_end, scaled_begin, subpixel_scale, pixel_mask);
    }

    assert!(
        scaled_begin.x >= 0,
        "Check failed: scaled_begin.x() >= 0 ({} vs. 0)",
        scaled_begin.x
    );
    assert!(
        scaled_begin.y >= 0,
        "Check failed: scaled_begin.y() >= 0 ({} vs. 0)",
        scaled_begin.y
    );
    assert!(
        scaled_end.y >= 0,
        "Check failed: scaled_end.y() >= 0 ({} vs. 0)",
        scaled_end.y
    );
    pixel_mask.clear();
    // Special case: We have to draw a vertical line in full pixels, as
    // 'scaled_begin' and 'scaled_end' have the same full pixel x coordinate.
    if scaled_begin.x / subpixel_scale == scaled_end.x / subpixel_scale {
        let mut current = Array2i::new(
            scaled_begin.x / subpixel_scale,
            scaled_begin.y.min(scaled_end.y) / subpixel_scale,
        );
        pixel_mask.push(current);
        let end_y = scaled_begin.y.max(scaled_end.y) / subpixel_scale;
        while current.y <= end_y {
            push_if_new(pixel_mask, current);
            current.y += 1;
        }
        return;
    }

    let dx = i64::from(scaled_end.x - scaled_begin.x);
    let dy = i64::from(scaled_end.y - scaled_begin.y);
    let denominator = 2 * i64::from(subpixel_scale) * dx;

    // The current full pixel coordinates. We scaled_begin at 'scaled_begin'.
    let mut current = scaled_begin / subpixel_scale;
    pixel_mask.push(current);

    // To represent subpixel centers, we use a factor of 2 * 'subpixel_scale'
    // in the denominator.
    // +-+-+-+ -- 1 = (2 * subpixel_scale) / (2 * subpixel_scale)
    // | | | |
    // +-+-+-+
    // | | | |
    // +-+-+-+ -- top edge of first subpixel = 2 / (2 * subpixel_scale)
    // | | | | -- center of first subpixel = 1 / (2 * subpixel_scale)
    // +-+-+-+ -- 0 = 0 / (2 * subpixel_scale)

    // The center of the subpixel part of 'scaled_begin.y()' assuming the
    // 'denominator', i.e., sub_y / denominator is in (0, 1).
    let mut sub_y = i64::from(2 * (scaled_begin.y % subpixel_scale) + 1) * dx;

    // The distance from the from 'scaled_begin' to the right pixel border, to
    // be divided by 2 * 'subpixel_scale'.
    let first_pixel = 2 * subpixel_scale - 2 * (scaled_begin.x % subpixel_scale) - 1;
    // The same from the left pixel border to 'scaled_end'.
    let last_pixel = 2 * (scaled_end.x % subpixel_scale) + 1;

    // The full pixel x coordinate of 'scaled_end'.
    let end_x = scaled_begin.x.max(scaled_end.x) / subpixel_scale;

    // Move from 'scaled_begin' to the next pixel border to the right.
    sub_y += dy * i64::from(first_pixel);
    if dy > 0 {
        loop {
            push_if_new(pixel_mask, current);
            while sub_y > denominator {
                sub_y -= denominator;
                current.y += 1;
                push_if_new(pixel_mask, current);
            }
            current.x += 1;
            if sub_y == denominator {
                sub_y -= denominator;
                current.y += 1;
            }
            if current.x == end_x {
                break;
            }
            // Move from one pixel border to the next.
            sub_y += dy * 2 * i64::from(subpixel_scale);
        }
        // Move from the pixel border on the right to 'scaled_end'.
        sub_y += dy * i64::from(last_pixel);
        push_if_new(pixel_mask, current);
        while sub_y > denominator {
            sub_y -= denominator;
            current.y += 1;
            push_if_new(pixel_mask, current);
        }
        assert_ne!(sub_y, denominator, "Check failed: sub_y != denominator");
        assert_eq!(
            current.y,
            scaled_end.y / subpixel_scale,
            "Check failed: current.y() == scaled_end.y() / subpixel_scale"
        );
        return;
    }

    // Same for lines non-ascending in y coordinates.
    loop {
        push_if_new(pixel_mask, current);
        while sub_y < 0 {
            sub_y += denominator;
            current.y -= 1;
            push_if_new(pixel_mask, current);
        }
        current.x += 1;
        if sub_y == 0 {
            sub_y += denominator;
            current.y -= 1;
        }
        if current.x == end_x {
            break;
        }
        sub_y += dy * 2 * i64::from(subpixel_scale);
    }
    sub_y += dy * i64::from(last_pixel);
    push_if_new(pixel_mask, current);
    while sub_y < 0 {
        sub_y += denominator;
        current.y -= 1;
        push_if_new(pixel_mask, current);
    }
    assert_ne!(sub_y, 0, "Check failed: sub_y != 0");
    assert_eq!(
        current.y,
        scaled_end.y / subpixel_scale,
        "Check failed: current.y() == scaled_end.y() / subpixel_scale"
    );
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector2;

    use super::{ray_to_pixel_mask, ray_to_pixel_mask_into};
    use crate::mapping::two_d::map_limits::MapLimits;
    use crate::mapping::two_d::xy_index::{Array2i, CellLimits};

    fn pixels(list: &[(i32, i32)]) -> Vec<Array2i> {
        list.iter().map(|&(x, y)| Array2i::new(x, y)).collect()
    }

    mod ray_to_pixel_mask_test {
        use super::*;

        #[test]
        fn single_cell() {
            let begin = Array2i::new(1, 1);
            let end = Array2i::new(1, 1);
            let ray = ray_to_pixel_mask(&begin, &end, 1);
            assert_eq!(ray, vec![begin]);
        }

        #[test]
        fn axis_aligned_x() {
            let begin = Array2i::new(1, 1);
            let end = Array2i::new(3, 1);
            let expected = pixels(&[(1, 1), (2, 1), (3, 1)]);
            assert_eq!(ray_to_pixel_mask(&begin, &end, 1), expected);
            assert_eq!(ray_to_pixel_mask(&end, &begin, 1), expected);
        }

        #[test]
        fn axis_aligned_y() {
            let begin = Array2i::new(1, 1);
            let end = Array2i::new(1, 3);
            let expected = pixels(&[(1, 1), (1, 2), (1, 3)]);
            assert_eq!(ray_to_pixel_mask(&begin, &end, 1), expected);
            assert_eq!(ray_to_pixel_mask(&end, &begin, 1), expected);
        }

        #[test]
        fn diagonal() {
            let begin = Array2i::new(1, 1);
            let end = Array2i::new(3, 3);
            let expected = pixels(&[(1, 1), (2, 2), (3, 3)]);
            assert_eq!(ray_to_pixel_mask(&begin, &end, 1), expected);
            assert_eq!(ray_to_pixel_mask(&end, &begin, 1), expected);

            let begin = Array2i::new(1, 3);
            let end = Array2i::new(3, 1);
            let expected = pixels(&[(1, 3), (2, 2), (3, 1)]);
            assert_eq!(ray_to_pixel_mask(&begin, &end, 1), expected);
            assert_eq!(ray_to_pixel_mask(&end, &begin, 1), expected);
        }

        #[test]
        fn steep_line() {
            let begin = Array2i::new(1, 1);
            let end = Array2i::new(2, 5);
            assert_eq!(
                ray_to_pixel_mask(&begin, &end, 1),
                pixels(&[(1, 1), (1, 2), (1, 3), (2, 3), (2, 4), (2, 5)])
            );
            let begin = Array2i::new(1, 1);
            let end = Array2i::new(2, 4);
            assert_eq!(
                ray_to_pixel_mask(&begin, &end, 1),
                pixels(&[(1, 1), (1, 2), (2, 3), (2, 4)])
            );
        }

        #[test]
        fn flat_line() {
            let begin = Array2i::new(1, 1);
            let end = Array2i::new(5, 2);
            assert_eq!(
                ray_to_pixel_mask(&begin, &end, 1),
                pixels(&[(1, 1), (2, 1), (3, 1), (3, 2), (4, 2), (5, 2)])
            );
            let begin = Array2i::new(1, 1);
            let end = Array2i::new(4, 2);
            assert_eq!(
                ray_to_pixel_mask(&begin, &end, 1),
                pixels(&[(1, 1), (2, 1), (3, 2), (4, 2)])
            );
        }

        #[test]
        fn multi_scale_axis_aligned_x() {
            let num_cells_x = 20;
            let num_cells_y = 20;
            let resolution = 0.1;
            let max = Vector2::new(1.0, 1.0);
            let mut subpixel_scale = 1;
            while subpixel_scale < 10000 {
                let superscaled_resolution = resolution / f64::from(subpixel_scale);
                let superscaled_limits = MapLimits::new(
                    superscaled_resolution,
                    max,
                    CellLimits::new(num_cells_x * subpixel_scale, num_cells_y * subpixel_scale),
                );
                let begin = superscaled_limits.get_cell_index(&Vector2::new(0.05, 0.05));
                let end = superscaled_limits.get_cell_index(&Vector2::new(0.35, 0.05));
                let ray = ray_to_pixel_mask(&begin, &end, subpixel_scale);
                assert_eq!(
                    ray,
                    pixels(&[(9, 6), (9, 7), (9, 8), (9, 9)]),
                    "subpixel_scale {subpixel_scale}"
                );
                subpixel_scale *= 2;
            }
        }

        #[test]
        fn multi_scale_skewed_line() {
            let num_cells_x = 20;
            let num_cells_y = 20;
            let resolution = 0.1;
            let max = Vector2::new(1.0, 1.0);

            let mut subpixel_scale = 1;
            let mut superscaled_resolution = resolution / f64::from(subpixel_scale);
            let mut superscaled_limits = MapLimits::new(
                superscaled_resolution,
                max,
                CellLimits::new(num_cells_x * subpixel_scale, num_cells_y * subpixel_scale),
            );
            let mut begin = superscaled_limits.get_cell_index(&Vector2::new(0.01, 0.09));
            let mut end = superscaled_limits.get_cell_index(&Vector2::new(0.21, 0.19));
            let mut ray = ray_to_pixel_mask(&begin, &end, subpixel_scale);
            assert_eq!(ray, pixels(&[(8, 7), (8, 8), (9, 8), (9, 9)]));

            subpixel_scale = 20;
            superscaled_resolution = resolution / f64::from(subpixel_scale);
            superscaled_limits = MapLimits::new(
                superscaled_resolution,
                max,
                CellLimits::new(num_cells_x * subpixel_scale, num_cells_y * subpixel_scale),
            );
            begin = superscaled_limits.get_cell_index(&Vector2::new(0.01, 0.09));
            end = superscaled_limits.get_cell_index(&Vector2::new(0.21, 0.19));
            ray = ray_to_pixel_mask(&begin, &end, subpixel_scale);
            assert_eq!(ray, pixels(&[(8, 7), (8, 8), (8, 9), (9, 9)]));
        }
    }

    mod proptests {
        use proptest::prelude::*;

        use super::*;

        proptest! {
            // Structural properties of the mask: endpoints are the pixels of
            // begin/end, consecutive pixels are 4- or diagonally-connected
            // with x non-decreasing, and there are no duplicates.
            #[test]
            fn mask_is_connected_and_ends_at_endpoints(
                bx in 0i32..20_000, by in 0i32..20_000,
                ex in 0i32..20_000, ey in 0i32..20_000,
                scale in prop::sample::select(vec![1, 2, 7, 1000])) {
                let begin = Array2i::new(bx, by);
                let end = Array2i::new(ex, ey);
                let mask = ray_to_pixel_mask(&begin, &end, scale);
                let (first, last) = if bx > ex { (end, begin) } else { (begin, end) };
                prop_assert!(!mask.is_empty());
                if first.x / scale == last.x / scale {
                    // Vertical special case: ascending y.
                    prop_assert_eq!(mask[0].y, first.y.min(last.y) / scale);
                    prop_assert_eq!(mask.last().unwrap().y, first.y.max(last.y) / scale);
                } else {
                    prop_assert_eq!(mask[0], first / scale);
                    prop_assert_eq!(*mask.last().unwrap(), last / scale);
                }
                for w in mask.windows(2) {
                    let d = w[1] - w[0];
                    prop_assert!(d.x == 0 || d.x == 1);
                    prop_assert!(d.y.abs() <= 1);
                    prop_assert!(d != Array2i::zeros());
                }
            }

            // Reusing a (non-empty) buffer gives exactly the fresh result.
            #[test]
            fn into_reused_buffer_matches_fresh_mask(
                bx in 0i32..20_000, by in 0i32..20_000,
                ex in 0i32..20_000, ey in 0i32..20_000,
                scale in prop::sample::select(vec![1, 2, 7, 1000])) {
                let begin = Array2i::new(bx, by);
                let end = Array2i::new(ex, ey);
                let mut buffer = vec![Array2i::new(-1, -1); 3];
                ray_to_pixel_mask_into(&begin, &end, scale, &mut buffer);
                prop_assert_eq!(buffer, ray_to_pixel_mask(&begin, &end, scale));
            }
        }
    }
}
