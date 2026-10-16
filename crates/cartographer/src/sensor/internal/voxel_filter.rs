//! Port of `cartographer/sensor/internal/voxel_filter` (.h, .cc).
//!
//! RUST PORT NOTE: the C++ `VoxelFilter` overloads are disambiguated by name:
//!
//! | C++ overload | Rust |
//! |---|---|
//! | `VoxelFilter(const PointCloud&, float)` | [`voxel_filter`] |
//! | `VoxelFilter(const std::vector<RangefinderPoint>&, float)` | [`voxel_filter_points`] |
//! | `VoxelFilter(const TimedPointCloud&, float)` | [`voxel_filter_timed_point_cloud`] |
//! | `VoxelFilter(const std::vector<RangeMeasurement>&, float)` | [`voxel_filter_range_measurements`] |

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use nalgebra::Vector3;

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::port::round_to_int;
use crate::sensor::point_cloud::{PointCloud, TimedPointCloud};
use crate::sensor::proto;
use crate::sensor::rangefinder_point::{RangefinderPoint, TimedRangefinderPoint};
use crate::sensor::timed_point_cloud_data::RangeMeasurement;

/// `std::minstd_rand0`: the Lehmer LCG `x = 16807 * x mod (2^31 - 1)` with
/// the default seed 1. Outputs are in `[1, 2^31 - 2]`.
#[derive(Clone, Debug)]
pub(crate) struct MinStdRand0 {
    state: u64,
}

impl MinStdRand0 {
    const MULTIPLIER: u64 = 16807;
    const MODULUS: u64 = 2_147_483_647;
    const MIN: u64 = 1;
    const MAX: u64 = Self::MODULUS - 1;

    /// Creates the generator with the default seed (1).
    pub(crate) fn new() -> Self {
        MinStdRand0 { state: 1 }
    }

    /// Returns the next value.
    pub(crate) fn next_u32(&mut self) -> u32 {
        self.state = Self::MULTIPLIER * self.state % Self::MODULUS;
        self.state as u32
    }

    /// `std::uniform_int_distribution<int>(a, b)(*this)` with the libstdc++
    /// (GCC) algorithm: downscaling with rejection.
    ///
    /// RUST PORT NOTE: the distribution algorithm is implementation-defined
    /// in C++; this mimics libstdc++ (the reference Linux build). libc++
    /// (macOS) builds of the C++ pick different points.
    pub(crate) fn uniform_int(&mut self, a: i32, b: i32) -> i32 {
        debug_assert!(a <= b);
        let urngrange = Self::MAX - Self::MIN;
        let urange = (i64::from(b) - i64::from(a)) as u64;
        debug_assert!(urngrange > urange);
        // Downscaling.
        let uerange = urange + 1;
        let scaling = urngrange / uerange;
        let past = uerange * scaling;
        let mut ret;
        loop {
            ret = u64::from(self.next_u32()) - Self::MIN;
            if ret < past {
                break;
            }
        }
        ret /= scaling;
        (ret as i64 + i64::from(a)) as i32
    }
}

fn filter_by_max_range(point_cloud: &PointCloud, max_range: f32) -> PointCloud {
    point_cloud.copy_if(|point| point.position.norm() <= max_range)
}

fn adaptively_voxel_filtered(
    options: &proto::AdaptiveVoxelFilterOptions,
    point_cloud: &PointCloud,
) -> PointCloud {
    // `size_t` is compared with the `float` option, like in C++.
    if point_cloud.size() as f32 <= options.min_num_points {
        // 'point_cloud' is already sparse enough.
        return point_cloud.clone();
    }
    let mut result = voxel_filter(point_cloud, options.max_length);
    if result.size() as f32 >= options.min_num_points {
        // Filtering with 'max_length' resulted in a sufficiently dense point
        // cloud.
        return result;
    }
    // Search for a 'low_length' that is known to result in a sufficiently
    // dense point cloud. We give up and use the full 'point_cloud' if reducing
    // the edge length by a factor of 1e-2 is not enough.
    // NOTE (port): like the C++ code, the fall-through actually returns the
    // result of the last (max_length / 128) filtering, not 'point_cloud'.
    let mut high_length = options.max_length;
    while high_length > 1e-2 * options.max_length {
        let mut low_length = high_length / 2.;
        result = voxel_filter(point_cloud, low_length);
        if result.size() as f32 >= options.min_num_points {
            // Binary search to find the right amount of filtering. 'low_length'
            // gave a sufficiently dense 'result', 'high_length' did not. We
            // stop when the edge length is at most 10% off.
            while (high_length - low_length) / low_length > 1e-1 {
                let mid_length = (low_length + high_length) / 2.;
                let candidate = voxel_filter(point_cloud, mid_length);
                if candidate.size() as f32 >= options.min_num_points {
                    low_length = mid_length;
                    result = candidate;
                } else {
                    high_length = mid_length;
                }
            }
            return result;
        }
        high_length /= 2.;
    }
    result
}

type VoxelKeyType = u64;

fn get_voxel_cell_index(point: &Vector3<f32>, resolution: f32) -> VoxelKeyType {
    let index = point / resolution;
    // Sign-extended like the C++ `int` -> `uint64_t` conversion; the bit
    // fields are not masked (negative indices bleed into neighbouring fields),
    // exactly like in C++.
    let x = round_to_int(index.x) as u64;
    let y = round_to_int(index.y) as u64;
    let z = round_to_int(index.z) as u64;
    (x << 42).wrapping_add(y << 21).wrapping_add(z)
}

/// Hasher for voxel keys: the splitmix64 finalizer of the `u64` key.
///
/// RUST PORT NOTE: performance only. The default SipHash is several times
/// slower for these integer keys; the hash function does not influence which
/// points are kept (only the per-voxel reservoir sampling in input order
/// does), so results are identical.
#[derive(Default)]
struct VoxelKeyHasher(u64);

impl Hasher for VoxelKeyHasher {
    #[inline]
    fn finish(&self) -> u64 {
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Only used for non-`u64` keys (never by the voxel filter).
        for &byte in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

fn randomized_voxel_filter_indices<T>(
    point_cloud: &[T],
    resolution: f32,
    point_function: impl Fn(&T) -> &Vector3<f32>,
) -> Vec<bool> {
    // According to https://en.wikipedia.org/wiki/Reservoir_sampling
    let mut generator = MinStdRand0::new();
    let mut voxel_count_and_point_index: HashMap<
        VoxelKeyType,
        (i32, usize),
        BuildHasherDefault<VoxelKeyHasher>,
    > = HashMap::with_capacity_and_hasher(point_cloud.len(), Default::default());
    for (i, point) in point_cloud.iter().enumerate() {
        let voxel = voxel_count_and_point_index
            .entry(get_voxel_cell_index(point_function(point), resolution))
            .or_insert((0, 0));
        voxel.0 += 1;
        // The first point of a voxel is taken without a random draw; every
        // further point replaces it with probability 1 / count.
        if voxel.0 == 1 || generator.uniform_int(1, voxel.0) == voxel.0 {
            voxel.1 = i;
        }
    }
    let mut points_used = vec![false; point_cloud.len()];
    for (_, index) in voxel_count_and_point_index.values() {
        points_used[*index] = true;
    }
    points_used
}

fn randomized_voxel_filter<T: Clone>(
    point_cloud: &[T],
    resolution: f32,
    point_function: impl Fn(&T) -> &Vector3<f32>,
) -> Vec<T> {
    let points_used = randomized_voxel_filter_indices(point_cloud, resolution, point_function);
    point_cloud
        .iter()
        .zip(&points_used)
        .filter(|(_, used)| **used)
        .map(|(point, _)| point.clone())
        .collect()
}

/// Keeps one (randomly chosen, deterministic per input) point per voxel of
/// edge length `resolution`, in input order.
pub fn voxel_filter_points(points: &[RangefinderPoint], resolution: f32) -> Vec<RangefinderPoint> {
    randomized_voxel_filter(points, resolution, |point| &point.position)
}

/// Voxel filters a point cloud, keeping the intensities of the kept points.
pub fn voxel_filter(point_cloud: &PointCloud, resolution: f32) -> PointCloud {
    let points_used =
        randomized_voxel_filter_indices(point_cloud.points(), resolution, |point| &point.position);

    let filtered_points: Vec<RangefinderPoint> = point_cloud
        .points()
        .iter()
        .zip(&points_used)
        .filter(|(_, used)| **used)
        .map(|(point, _)| *point)
        .collect();
    assert!(point_cloud.intensities().len() <= point_cloud.points().len());
    let filtered_intensities: Vec<f32> = point_cloud
        .intensities()
        .iter()
        .zip(&points_used)
        .filter(|(_, used)| **used)
        .map(|(intensity, _)| *intensity)
        .collect();
    PointCloud::with_intensities(filtered_points, filtered_intensities)
}

/// Voxel filters a timed point cloud (time is not part of the voxel key).
pub fn voxel_filter_timed_point_cloud(
    timed_point_cloud: &[TimedRangefinderPoint],
    resolution: f32,
) -> TimedPointCloud {
    randomized_voxel_filter(timed_point_cloud, resolution, |point| &point.position)
}

/// Voxel filters range measurements by `point_time.position`.
pub fn voxel_filter_range_measurements(
    range_measurements: &[RangeMeasurement],
    resolution: f32,
) -> Vec<RangeMeasurement> {
    randomized_voxel_filter(range_measurements, resolution, |range_measurement| {
        &range_measurement.point_time.position
    })
}

/// Reads `AdaptiveVoxelFilterOptions` from Lua (`max_length`,
/// `min_num_points`, `max_range`).
pub fn create_adaptive_voxel_filter_options(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::AdaptiveVoxelFilterOptions {
    proto::AdaptiveVoxelFilterOptions {
        max_length: parameter_dictionary.get_double("max_length") as f32,
        min_num_points: parameter_dictionary.get_non_negative_int("min_num_points") as f32,
        max_range: parameter_dictionary.get_double("max_range") as f32,
    }
}

/// Removes points farther than `max_range` and voxel filters with the largest
/// edge length (searched down from `max_length`) that keeps at least
/// `min_num_points` points.
pub fn adaptive_voxel_filter(
    point_cloud: &PointCloud,
    options: &proto::AdaptiveVoxelFilterOptions,
) -> PointCloud {
    adaptively_voxel_filtered(
        options,
        &filter_by_max_range(point_cloud, options.max_range),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
    use proptest::prelude::*;

    fn p(x: f32, y: f32, z: f32) -> RangefinderPoint {
        RangefinderPoint {
            position: Vector3::new(x, y, z),
        }
    }

    mod voxel_filter_test {
        use super::*;

        #[test]
        fn returns_one_point_in_each_voxel() {
            let point_cloud = PointCloud::from_points(vec![
                p(0., 0., 0.),
                p(0.1, -0.1, 0.1),
                p(0.3, -0.1, 0.),
                p(0., 0., 0.1),
            ]);
            let result = voxel_filter(&point_cloud, 0.3);
            assert_eq!(result.size(), 2);
            assert!(result.intensities().is_empty());
            assert!(point_cloud.points().contains(&result[0]));
            assert!(point_cloud.points().contains(&result[1]));
            assert!(result.points().contains(&point_cloud[2]));
        }

        #[test]
        fn correct_intensities() {
            let mut points = Vec::new();
            let mut intensities = Vec::new();
            for i in 0..100 {
                let value = 0.1f32 * i as f32;
                // We add points with intensity equal to the z coordinate, so
                // we can later verify that the resulting intensities are
                // corresponding to the filtered points.
                points.push(p(-100., 0.3, value));
                intensities.push(value);
            }
            let point_cloud = PointCloud::with_intensities(points, intensities);
            let result = voxel_filter(&point_cloud, 0.3);

            assert_eq!(result.intensities().len(), result.points().len());
            for i in 0..result.size() {
                assert!((result[i].position.z - result.intensities()[i]).abs() <= 1e-6);
            }
        }

        #[test]
        fn handles_large_coordinates() {
            let point_cloud = PointCloud::from_points(vec![
                p(100000., 0., 0.),
                p(100000.001, -0.0001, 0.0001),
                p(100000.003, -0.0001, 0.),
                p(-200000., 0., 0.),
            ]);
            let result = voxel_filter(&point_cloud, 0.01);
            assert_eq!(result.size(), 2);
            assert!(result.intensities().is_empty());
            assert!(result.points().contains(&point_cloud[3]));
        }

        #[test]
        fn ignores_time() {
            let mut timed_point_cloud = TimedPointCloud::new();
            for i in 0..100 {
                timed_point_cloud.push(TimedRangefinderPoint {
                    position: Vector3::new(-100., 0.3, 0.4),
                    time: 1. * i as f32,
                });
            }
            let result = voxel_filter_timed_point_cloud(&timed_point_cloud, 0.3);
            assert_eq!(result.len(), 1);
            assert!(timed_point_cloud.contains(&result[0]));
        }
    }

    #[test]
    fn minstd_rand0_matches_cpp_standard() {
        let mut generator = MinStdRand0::new();
        assert_eq!(generator.next_u32(), 16807);
        assert_eq!(generator.next_u32(), 282_475_249);
        assert_eq!(generator.next_u32(), 1_622_650_073);
        // [rand.predef]: the 10000th consecutive invocation of a
        // default-constructed minstd_rand0 produces 1043618065.
        let mut generator = MinStdRand0::new();
        let mut value = 0;
        for _ in 0..10000 {
            value = generator.next_u32();
        }
        assert_eq!(value, 1_043_618_065);
    }

    #[test]
    fn uniform_int_matches_libstdcxx() {
        // libstdc++: uniform_int_distribution<int>(1, 2) with a fresh
        // minstd_rand0: scaling = (2^31 - 3) / 2, ret = (16807 - 1) / scaling.
        let mut generator = MinStdRand0::new();
        assert_eq!(generator.uniform_int(1, 2), 1);
        let mut generator = MinStdRand0::new();
        for _ in 0..1000 {
            let v = generator.uniform_int(1, 7);
            assert!((1..=7).contains(&v));
        }
    }

    #[test]
    fn exact_selection_is_deterministic() {
        // Three points in one voxel: draws (1..=2) -> 1 (keep index 0), then
        // (1..=3) on the second value 282475249: scaling = 715827881,
        // (282475249 - 1) / 715827881 = 0 -> 1, keep index 0.
        let point_cloud =
            PointCloud::from_points(vec![p(0., 0., 0.), p(0.01, 0., 0.), p(0.02, 0., 0.)]);
        let result = voxel_filter(&point_cloud, 1.);
        assert_eq!(result.points(), &[p(0., 0., 0.)]);
        assert_eq!(voxel_filter(&point_cloud, 1.), result);
    }

    #[test]
    fn voxel_key_rounds_to_nearest() {
        // Voxels are centred on multiples of the resolution.
        assert_eq!(
            get_voxel_cell_index(&Vector3::new(0.14, 0., 0.), 0.3),
            get_voxel_cell_index(&Vector3::new(-0.14, 0., 0.), 0.3)
        );
        assert_ne!(
            get_voxel_cell_index(&Vector3::new(0.16, 0., 0.), 0.3),
            get_voxel_cell_index(&Vector3::new(0.14, 0., 0.), 0.3)
        );
    }

    #[test]
    fn create_options_from_lua() {
        let dictionary =
            make_dictionary("return { max_length = 0.5, min_num_points = 200, max_range = 50. }");
        let options = create_adaptive_voxel_filter_options(&dictionary);
        assert_eq!(options.max_length, 0.5);
        assert_eq!(options.min_num_points, 200.);
        assert_eq!(options.max_range, 50.);
    }

    fn grid_cloud(n: usize, spacing: f32) -> PointCloud {
        let mut points = Vec::new();
        for i in 0..n {
            for j in 0..n {
                points.push(p(i as f32 * spacing, j as f32 * spacing, 0.));
            }
        }
        PointCloud::from_points(points)
    }

    #[test]
    fn adaptive_voxel_filter_sparse_cloud_is_unchanged() {
        let cloud = grid_cloud(3, 0.01);
        let options = proto::AdaptiveVoxelFilterOptions {
            max_length: 1.,
            min_num_points: 100.,
            max_range: 100.,
        };
        assert_eq!(adaptive_voxel_filter(&cloud, &options), cloud);
    }

    #[test]
    fn adaptive_voxel_filter_max_range_and_density() {
        // 20x20 grid with 0.1 spacing, plus a far point that is removed.
        let mut points = grid_cloud(20, 0.1).into_parts().0;
        points.push(p(100., 0., 0.));
        let cloud = PointCloud::from_points(points);
        let options = proto::AdaptiveVoxelFilterOptions {
            max_length: 1.,
            min_num_points: 50.,
            max_range: 10.,
        };
        let result = adaptive_voxel_filter(&cloud, &options);
        assert!(result.size() >= 50);
        assert!(result.size() < 400);
        assert!(!result.points().contains(&p(100., 0., 0.)));
        // max_length alone already suffices.
        let options = proto::AdaptiveVoxelFilterOptions {
            max_length: 0.05,
            min_num_points: 50.,
            max_range: 10.,
        };
        assert_eq!(adaptive_voxel_filter(&cloud, &options).size(), 400);
    }

    #[test]
    fn adaptive_voxel_filter_falls_through_to_smallest_length() {
        // All points in a tiny region: even max_length / 128 is too coarse,
        // so (like C++) the last filtered result is returned, not the input.
        let mut points = Vec::new();
        for i in 0..10 {
            points.push(p(i as f32 * 1e-4, 0., 0.));
        }
        let cloud = PointCloud::from_points(points);
        let options = proto::AdaptiveVoxelFilterOptions {
            max_length: 1.,
            min_num_points: 5.,
            max_range: 10.,
        };
        let result = adaptive_voxel_filter(&cloud, &options);
        assert_eq!(result, voxel_filter(&cloud, 1. / 128.));
        assert!(result.size() < cloud.size());
    }

    proptest! {
        // Invariants: the result is a subset of the input (in input order),
        // with exactly one point per occupied voxel.
        #[test]
        fn voxel_filter_invariants(
            points in prop::collection::vec(prop::array::uniform3(-5f32..5f32), 0..300),
            resolution in 0.05f32..2.,
        ) {
            let input: Vec<RangefinderPoint> =
                points.iter().map(|a| RangefinderPoint { position: Vector3::from(*a) }).collect();
            let result = voxel_filter_points(&input, resolution);
            // Subset in input order.
            let mut it = input.iter();
            for r in &result {
                prop_assert!(it.any(|q| q == r));
            }
            // At most one point per voxel and every occupied voxel is kept.
            let keys: std::collections::HashSet<u64> =
                input.iter().map(|q| get_voxel_cell_index(&q.position, resolution)).collect();
            let result_keys: Vec<u64> =
                result.iter().map(|q| get_voxel_cell_index(&q.position, resolution)).collect();
            let unique: std::collections::HashSet<u64> = result_keys.iter().copied().collect();
            prop_assert_eq!(unique.len(), result_keys.len());
            prop_assert_eq!(unique, keys);
        }
    }
}
