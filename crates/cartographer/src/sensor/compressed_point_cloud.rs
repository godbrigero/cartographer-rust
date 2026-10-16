//! Port of `cartographer/sensor/compressed_point_cloud` (.h, .cc).

use std::collections::BTreeMap;

use nalgebra::Vector3;

use crate::common::port::round_to_int;
use crate::sensor::point_cloud::PointCloud;
use crate::sensor::proto;
use crate::sensor::rangefinder_point::RangefinderPoint;

// Points are encoded on a fixed grid with a grid spacing of 'kPrecision' with
// integers. Points are organized in blocks, where each point is encoded
// relative to the block's origin in an int32 with 'kBitsPerCoordinate' bits per
// coordinate.
const PRECISION: f32 = 0.001; // in meters.
const BITS_PER_COORDINATE: i32 = 10;
const COORDINATE_MASK: i32 = (1 << BITS_PER_COORDINATE) - 1;
const MAX_BITS_PER_DIRECTION: i32 = 23;

/// A compressed representation of a point cloud consisting of a collection of
/// points (`Vector3<f32>`) without time information (and without
/// intensities).
///
/// Internally, points are grouped by blocks. Each block encodes a bit of meta
/// data (number of points in block, coordinates of the block) and encodes each
/// point with a fixed bit rate in relation to the block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompressedPointCloud {
    point_data: Vec<i32>,
    num_points: usize,
}

/// Iteration order of the blocks of the C++ `mapping::HybridGridBase` used
/// for encoding: `DynamicGrid<NestedGrid<FlatGrid<_, 3>, 3>>`, each level
/// iterating z-major. Because the `DynamicGrid` centering offset is always a
/// multiple of 64, this equals lexicographic order of this key.
///
/// RUST PORT NOTE: replicates the HybridGrid iteration order with a
/// `BTreeMap` (instead of depending on `mapping::three_d::HybridGrid`), so
/// `point_data` is bit-identical to C++.
fn block_order_key(block: &[i32; 3]) -> [i32; 9] {
    let [x, y, z] = *block;
    // `>>` on i32 is an arithmetic shift (floor division) and `&` on two's
    // complement is the Euclidean modulo.
    [
        z >> 6,
        y >> 6,
        x >> 6,
        (z >> 3) & 7,
        (y >> 3) & 7,
        (x >> 3) & 7,
        z & 7,
        y & 7,
        x & 7,
    ]
}

impl CompressedPointCloud {
    /// Creates an empty compressed point cloud.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compresses `point_cloud` (C++ `CompressedPointCloud(const PointCloud&)`).
    /// Panics if a coordinate's absolute value is `>= 2^23 mm` (8388.608 m).
    pub fn from_point_cloud(point_cloud: &PointCloud) -> Self {
        // Distribute points into blocks.
        struct RasterPoint {
            point: [i32; 3],
        }
        let mut blocks: BTreeMap<[i32; 9], ([i32; 3], Vec<RasterPoint>)> = BTreeMap::new();
        assert!(i32::try_from(point_cloud.size()).is_ok());
        for point in point_cloud.points() {
            let max_abs = point.position.abs().max();
            assert!(
                max_abs / PRECISION < (1 << MAX_BITS_PER_DIRECTION) as f32,
                "Point out of bounds: {:?}",
                point.position
            );
            let mut raster_point = [0i32; 3];
            let mut block_coordinate = [0i32; 3];
            for i in 0..3 {
                raster_point[i] = round_to_int(point.position[i] / PRECISION);
                block_coordinate[i] = raster_point[i] >> BITS_PER_COORDINATE;
                raster_point[i] &= COORDINATE_MASK;
            }
            blocks
                .entry(block_order_key(&block_coordinate))
                .or_insert_with(|| (block_coordinate, Vec::new()))
                .1
                .push(RasterPoint {
                    point: raster_point,
                });
        }

        // Encode blocks.
        let mut point_data = Vec::with_capacity(4 * blocks.len() + point_cloud.size());
        for (block_coordinate, raster_points) in blocks.values() {
            point_data.push(
                i32::try_from(raster_points.len()).expect("too many points in a single block"),
            );
            point_data.extend_from_slice(block_coordinate);
            for raster_point in raster_points {
                let [x, y, z] = raster_point.point;
                point_data.push((((z << BITS_PER_COORDINATE) + y) << BITS_PER_COORDINATE) + x);
            }
        }
        CompressedPointCloud {
            point_data,
            num_points: point_cloud.size(),
        }
    }

    /// Creates a compressed point cloud from its proto representation.
    ///
    /// Like in C++ the data is not validated here; malformed data makes
    /// iteration panic (instead of reading out of bounds as in C++).
    pub fn from_proto(proto: &proto::CompressedPointCloud) -> Self {
        CompressedPointCloud {
            // RUST PORT NOTE: a negative `num_points` (invalid) becomes 0
            // instead of a huge `size_t`.
            num_points: usize::try_from(proto.num_points).unwrap_or(0),
            point_data: proto.point_data.clone(),
        }
    }

    /// Returns the decompressed point cloud (grouped by block, i.e. not in
    /// input order; no intensities).
    pub fn decompress(&self) -> PointCloud {
        PointCloud::from_points(self.iter().collect())
    }

    /// True if there are no points.
    pub fn empty(&self) -> bool {
        self.num_points == 0
    }

    /// Same as [`CompressedPointCloud::empty`].
    pub fn is_empty(&self) -> bool {
        self.num_points == 0
    }

    /// Number of points.
    pub fn size(&self) -> usize {
        self.num_points
    }

    /// Same as [`CompressedPointCloud::size`].
    pub fn len(&self) -> usize {
        self.num_points
    }

    /// Forward iterator over the decompressed points (yielded by value).
    pub fn iter(&self) -> ConstIterator<'_> {
        ConstIterator {
            compressed_point_cloud: self,
            remaining_points: self.num_points,
            remaining_points_in_current_block: 0,
            current_block_coordinates: [0; 3],
            input: 0,
        }
    }

    /// Converts to the proto representation (C++ `ToProto`).
    pub fn to_proto(&self) -> proto::CompressedPointCloud {
        proto::CompressedPointCloud {
            num_points: self.num_points as i32,
            point_data: self.point_data.clone(),
        }
    }
}

impl<'a> IntoIterator for &'a CompressedPointCloud {
    type Item = RangefinderPoint;
    type IntoIter = ConstIterator<'a>;
    fn into_iter(self) -> ConstIterator<'a> {
        self.iter()
    }
}

/// Forward iterator for compressed point clouds.
///
/// RUST PORT NOTE: the C++ iterator eagerly reads the next point; this one
/// reads it lazily in `next()`, which yields the same sequence.
#[derive(Clone, Debug)]
pub struct ConstIterator<'a> {
    compressed_point_cloud: &'a CompressedPointCloud,
    remaining_points: usize,
    remaining_points_in_current_block: i32,
    current_block_coordinates: [i32; 3],
    input: usize,
}

impl ConstIterator<'_> {
    fn read_input(&mut self) -> i32 {
        let value = *self
            .compressed_point_cloud
            .point_data
            .get(self.input)
            .expect("Malformed CompressedPointCloud: point data too short.");
        self.input += 1;
        value
    }

    // Reads next point from buffer. Also handles reading the meta data of the
    // next block, if the current block is depleted.
    fn read_next_point(&mut self) -> RangefinderPoint {
        if self.remaining_points_in_current_block == 0 {
            self.remaining_points_in_current_block = self.read_input();
            for i in 0..3 {
                self.current_block_coordinates[i] =
                    self.read_input().wrapping_shl(BITS_PER_COORDINATE as u32);
            }
        }
        self.remaining_points_in_current_block -= 1;
        let point = self.read_input();
        const MASK: i32 = (1 << BITS_PER_COORDINATE) - 1;
        let b = &self.current_block_coordinates;
        RangefinderPoint {
            position: Vector3::new(
                b[0].wrapping_add(point & MASK) as f32 * PRECISION,
                b[1].wrapping_add((point >> BITS_PER_COORDINATE) & MASK) as f32 * PRECISION,
                b[2].wrapping_add(point >> (2 * BITS_PER_COORDINATE)) as f32 * PRECISION,
            ),
        }
    }
}

impl Iterator for ConstIterator<'_> {
    type Item = RangefinderPoint;

    fn next(&mut self) -> Option<RangefinderPoint> {
        if self.remaining_points == 0 {
            return None;
        }
        let point = self.read_next_point();
        self.remaining_points -= 1;
        Some(point)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining_points, Some(self.remaining_points))
    }
}

impl ExactSizeIterator for ConstIterator<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use proptest::prelude::*;

    const PRECISION: f32 = 0.001;

    // Matcher for 3-d vectors w.r.t. to the target precision (Eigen
    // `isZero(prec)`: every coefficient's absolute value <= prec).
    fn approximately_equals(arg: &RangefinderPoint, expected: &Vector3<f32>) -> bool {
        (arg.position - expected)
            .iter()
            .all(|c| c.abs() <= PRECISION)
    }

    fn cloud(points: &[Vector3<f32>]) -> PointCloud {
        PointCloud::from_points(
            points
                .iter()
                .map(|p| RangefinderPoint { position: *p })
                .collect(),
        )
    }

    // Helper function to test the mapping of a single point. Includes test for
    // recompressing the same point again.
    fn test_point(p: Vector3<f32>) {
        let compressed = CompressedPointCloud::from_point_cloud(&cloud(&[p]));
        assert_eq!(1, compressed.size());
        let first = compressed.iter().next().unwrap();
        assert!(approximately_equals(&first, &p), "{first:?} vs {p:?}");
        let recompressed =
            CompressedPointCloud::from_point_cloud(&PointCloud::from_points(vec![first]));
        let first = recompressed.iter().next().unwrap();
        assert!(approximately_equals(&first, &p), "{first:?} vs {p:?}");
    }

    mod compress_point_cloud_test {
        use super::*;

        #[test]
        fn compresses_points_correctly() {
            test_point(Vector3::new(8000., 7500., 5000.));
            test_point(Vector3::new(1000., 2000., 3000.));
            test_point(Vector3::new(100., 200., 300.));
            test_point(Vector3::new(10., 20., 30.));
            test_point(Vector3::new(-0.00049, -0.0005, -0.0015));
            test_point(Vector3::new(0.05119, 0.0512, 0.05121));
            test_point(Vector3::new(-0.05119, -0.0512, -0.05121));
            test_point(Vector3::new(0.8405, 0.84, 0.8396));
            test_point(Vector3::new(0.8395, 0.8394, 0.8393));
            test_point(Vector3::new(0.839, 0.8391, 0.8392));
            test_point(Vector3::new(0.8389, 0.8388, 0.83985));
        }

        #[test]
        fn compresses() {
            let compressed = CompressedPointCloud::from_point_cloud(&cloud(&[
                Vector3::new(0.838, 0., 0.),
                Vector3::new(0.839, 0., 0.),
                Vector3::new(0.840, 0., 0.),
            ]));
            assert!(!compressed.empty());
            assert_eq!(3, compressed.size());
            let decompressed = compressed.decompress();
            assert_eq!(3, decompressed.size());
            for expected in [
                Vector3::new(0.838, 0., 0.),
                Vector3::new(0.839, 0., 0.),
                Vector3::new(0.840, 0., 0.),
            ] {
                assert!(
                    decompressed
                        .iter()
                        .any(|p| approximately_equals(p, &expected))
                );
            }
        }

        #[test]
        fn compresses_empty_point_cloud() {
            let compressed = CompressedPointCloud::new();
            assert!(compressed.empty());
            assert_eq!(0, compressed.size());
        }

        // Test for gaps.
        // Produces a series of points densly packed along the x axis,
        // compresses these points (twice), and tests, whether there are gaps
        // between two consecutive points.
        #[test]
        fn compresses_no_gaps() {
            let mut point_cloud = PointCloud::new();
            for i in 0..3000 {
                point_cloud.push_back(RangefinderPoint {
                    position: Vector3::new(PRECISION * i as f32 - 1.5, 0., 0.),
                });
            }
            let compressed = CompressedPointCloud::from_point_cloud(&point_cloud);
            let decompressed = compressed.decompress();
            let recompressed = CompressedPointCloud::from_point_cloud(&decompressed);
            assert_eq!(decompressed.size(), recompressed.size());

            let mut x_coord: Vec<f32> = compressed.iter().map(|p| p.position[0]).collect();
            x_coord.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for i in 1..x_coord.len() {
                let diff = (x_coord[i] - x_coord[i - 1]).abs();
                assert!((diff - PRECISION).abs() <= 1e-7, "{diff}");
            }
        }
    }

    #[test]
    fn proto_round_trip_and_equality() {
        let compressed = CompressedPointCloud::from_point_cloud(&cloud(&[
            Vector3::new(1., 2., 3.),
            Vector3::new(-1., -2., -3.),
            Vector3::new(0.5, 0.5, 0.5),
        ]));
        let decoded = CompressedPointCloud::from_proto(&compressed.to_proto());
        assert_eq!(decoded, compressed);
        assert_eq!(decoded.decompress(), compressed.decompress());
    }

    #[test]
    fn block_layout_matches_hybrid_grid_order() {
        // Blocks (0,0,0), (-1,0,0), (0,0,-1). HybridGrid order is z-major on
        // the outer (64-block) level, so (0,0,-1) comes first; (-1,0,0) is in
        // outer cell x=-1 and therefore precedes (0,0,0).
        let compressed = CompressedPointCloud::from_point_cloud(&cloud(&[
            Vector3::new(0.0, 0., 0.),
            Vector3::new(-0.5, 0., 0.),
            Vector3::new(0.0, 0., -0.5),
        ]));
        let data = compressed.to_proto().point_data;
        assert_eq!(data.len(), 3 * 4 + 3);
        assert_eq!(&data[1..4], &[0, 0, -1]);
        assert_eq!(&data[6..9], &[-1, 0, 0]);
        assert_eq!(&data[11..14], &[0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "Point out of bounds")]
    fn out_of_bounds_panics() {
        let _ = CompressedPointCloud::from_point_cloud(&cloud(&[Vector3::new(9000., 0., 0.)]));
    }

    proptest! {
        // Round-trip error bound: every point is reproduced within half a
        // grid cell (plus float error) and the multiset of points is
        // preserved (up to order).
        #[test]
        fn round_trip_error_bound(
            points in prop::collection::vec(prop::array::uniform3(-100f32..100f32), 0..200)
        ) {
            let input: Vec<Vector3<f32>> = points.iter().map(|a| Vector3::from(*a)).collect();
            let compressed = CompressedPointCloud::from_point_cloud(&cloud(&input));
            prop_assert_eq!(compressed.size(), input.len());
            let mut decoded: Vec<Vector3<f32>> = compressed.iter().map(|p| p.position).collect();
            prop_assert_eq!(decoded.len(), input.len());
            // Match each input point to a distinct decoded point.
            for p in &input {
                let idx = decoded.iter().position(|q| {
                    (q - p).iter().all(|c| c.abs() <= 0.5 * PRECISION + 1e-5 * (1. + p.abs().max()))
                });
                prop_assert!(idx.is_some(), "no match for {:?}", p);
                decoded.swap_remove(idx.unwrap());
            }
            let reencoded = CompressedPointCloud::from_proto(&compressed.to_proto());
            prop_assert_eq!(reencoded, compressed);
        }
    }
}
