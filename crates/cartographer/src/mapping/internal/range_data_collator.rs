//! Port of `cartographer/mapping/internal/range_data_collator` (.h, .cc).

use std::collections::{BTreeMap, BTreeSet};

use crate::common::time::{Time, from_seconds, to_seconds};
use crate::sensor::timed_point_cloud_data::{
    RangeMeasurement, TimedPointCloudData, TimedPointCloudOriginData,
};

/// Synchronizes `TimedPointCloudData` from different sensors. Input needs
/// only be monotonous in `TimedPointCloudData::time`, output is monotonous in
/// per-point timing. Up to one message per sensor is buffered, so a delay of
/// the period of the slowest sensor may be introduced, which can be alleviated
/// by passing subdivisions.
#[derive(Clone, Debug)]
pub struct RangeDataCollator {
    expected_sensor_ids: BTreeSet<String>,
    // Store at most one message for each sensor.
    id_to_pending_data: BTreeMap<String, TimedPointCloudData>,
    current_start: Time,
    current_end: Time,
}

impl RangeDataCollator {
    const DEFAULT_INTENSITY_VALUE: f32 = 0.;

    /// Creates a collator for the given range sensor ids.
    pub fn new<S: AsRef<str>>(expected_range_sensor_ids: &[S]) -> Self {
        RangeDataCollator {
            expected_sensor_ids: expected_range_sensor_ids
                .iter()
                .map(|s| s.as_ref().to_string())
                .collect(),
            id_to_pending_data: BTreeMap::new(),
            current_start: Time::min(),
            current_end: Time::min(),
        }
    }

    /// Adds range data of `sensor_id` and returns the merged data that is
    /// complete now (possibly empty, with time `Time::default()`). If
    /// `timed_point_cloud_data` has incomplete intensity data, the missing
    /// intensities are filled with 0.
    pub fn add_range_data(
        &mut self,
        sensor_id: &str,
        mut timed_point_cloud_data: TimedPointCloudData,
    ) -> TimedPointCloudOriginData {
        assert!(
            self.expected_sensor_ids.contains(sensor_id),
            "Check failed: unexpected range sensor id '{sensor_id}'"
        );
        timed_point_cloud_data.intensities.resize(
            timed_point_cloud_data.ranges.len(),
            Self::DEFAULT_INTENSITY_VALUE,
        );
        // TODO(gaschler): These two cases can probably be one.
        if let Some(pending) = self.id_to_pending_data.get(sensor_id) {
            self.current_start = self.current_end;
            // When we have two messages of the same sensor, move forward the
            // older of the two (do not send out current).
            self.current_end = pending.time;
            let result = self.crop_and_merge();
            // Like C++ `emplace`: does not overwrite a still pending message.
            self.id_to_pending_data
                .entry(sensor_id.to_string())
                .or_insert(timed_point_cloud_data);
            return result;
        }
        self.id_to_pending_data
            .insert(sensor_id.to_string(), timed_point_cloud_data);
        if self.expected_sensor_ids.len() != self.id_to_pending_data.len() {
            return TimedPointCloudOriginData::default();
        }
        self.current_start = self.current_end;
        // We have messages from all sensors, move forward to oldest.
        let oldest_timestamp = self
            .id_to_pending_data
            .values()
            .map(|data| data.time)
            .fold(Time::max(), std::cmp::min);
        self.current_end = oldest_timestamp;
        self.crop_and_merge()
    }

    fn crop_and_merge(&mut self) -> TimedPointCloudOriginData {
        let mut result = TimedPointCloudOriginData {
            time: self.current_end,
            origins: Vec::new(),
            ranges: Vec::new(),
        };
        let mut warned_for_dropped_points = false;
        let current_start = self.current_start;
        let current_end = self.current_end;
        self.id_to_pending_data.retain(|_, data| {
            let ranges = &data.ranges;
            let intensities = &data.intensities;

            let mut overlap_begin = 0;
            while overlap_begin < ranges.len()
                && data.time + from_seconds(f64::from(ranges[overlap_begin].time)) < current_start
            {
                overlap_begin += 1;
            }
            let mut overlap_end = overlap_begin;
            while overlap_end < ranges.len()
                && data.time + from_seconds(f64::from(ranges[overlap_end].time)) <= current_end
            {
                overlap_end += 1;
            }
            if overlap_begin > 0 && !warned_for_dropped_points {
                log::warn!("Dropped {overlap_begin} earlier points.");
                warned_for_dropped_points = true;
            }

            // Copy overlapping range.
            if overlap_begin < overlap_end {
                let origin_index = result.origins.len();
                result.origins.push(data.origin);
                let time_correction = to_seconds(data.time - current_end) as f32;
                result.ranges.reserve(overlap_end - overlap_begin);
                for i in overlap_begin..overlap_end {
                    let mut point = RangeMeasurement {
                        point_time: ranges[i],
                        intensity: intensities[i],
                        origin_index,
                    };
                    // current_end_ + point_time[3]_after == in_timestamp +
                    // point_time[3]_before
                    point.point_time.time += time_correction;
                    result.ranges.push(point);
                }
            }

            // Drop buffered points until overlap_end.
            if overlap_end == ranges.len() {
                false
            } else if overlap_end == 0 {
                true
            } else {
                data.ranges.drain(..overlap_end);
                data.intensities.drain(..overlap_end);
                true
            }
        });

        // RUST PORT NOTE: C++ uses `std::sort` (unstable, unspecified order of
        // equal times); this stable sort keeps equal-time points in merge
        // order.
        result.ranges.sort_by(|a, b| {
            a.point_time
                .time
                .partial_cmp(&b.point_time.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::time::{from_universal, to_universal};
    use crate::sensor::rangefinder_point::TimedRangefinderPoint;
    use nalgebra::Vector3;

    const NUM_SAMPLES: usize = 10;

    fn create_fake_range_data(from: i64, to: i64, fake_intensities: bool) -> TimedPointCloudData {
        let duration = to_seconds(from_universal(to) - from_universal(from));
        let mut result = TimedPointCloudData {
            time: from_universal(to),
            origin: Vector3::new(0., 1., 2.),
            ranges: Vec::with_capacity(NUM_SAMPLES),
            intensities: Vec::new(),
        };
        for i in 0..NUM_SAMPLES {
            let fraction = i as f64 / (NUM_SAMPLES - 1) as f64;
            let relative_time = ((1. - fraction) * -duration) as f32;
            result.ranges.push(TimedRangefinderPoint {
                position: Vector3::new(1., 2., fraction as f32),
                time: relative_time,
            });
            if fake_intensities {
                result
                    .intensities
                    .push(result.ranges.last().unwrap().position.z);
            }
        }
        result
    }

    fn are_point_timestamps_sorted(data: &TimedPointCloudOriginData) -> bool {
        data.ranges
            .windows(2)
            .all(|w| w[0].point_time.time <= w[1].point_time.time)
    }

    fn intensities_are_consistent(data: &TimedPointCloudOriginData) {
        for range in &data.ranges {
            assert!((range.point_time.position.z - range.intensity).abs() <= 1e-6);
        }
    }

    fn absolute_time(data: &TimedPointCloudOriginData, index: usize) -> i64 {
        to_universal(data.time + from_seconds(f64::from(data.ranges[index].point_time.time)))
    }

    fn assert_near(actual: i64, expected: i64, tolerance: i64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} vs {expected}"
        );
    }

    mod range_data_collator_test {
        use super::*;

        #[test]
        fn single_sensor() {
            let sensor_id = "single_sensor";
            let mut collator = RangeDataCollator::new(&[sensor_id]);
            let output_0 =
                collator.add_range_data(sensor_id, create_fake_range_data(200, 300, false));
            assert_eq!(to_universal(output_0.time), 300);
            assert_eq!(output_0.origins.len(), 1);
            assert_eq!(output_0.ranges.len(), NUM_SAMPLES);
            assert!(are_point_timestamps_sorted(&output_0));
            let output_1 =
                collator.add_range_data(sensor_id, create_fake_range_data(300, 500, false));
            assert_eq!(to_universal(output_1.time), 500);
            assert_eq!(output_1.origins.len(), 1);
            assert_eq!(output_1.ranges.len(), NUM_SAMPLES);
            assert!(are_point_timestamps_sorted(&output_1));
            assert_near(absolute_time(&output_1, 0), 300, 2);
            let output_2 =
                collator.add_range_data(sensor_id, create_fake_range_data(-1000, 510, false));
            assert_eq!(to_universal(output_2.time), 510);
            assert_eq!(output_2.origins.len(), 1);
            assert_eq!(output_2.ranges.len(), 1);
            assert_eq!(output_2.ranges[0].point_time.time, 0.);
            assert!(are_point_timestamps_sorted(&output_2));
        }

        #[test]
        fn single_sensor_empty_data() {
            let sensor_id = "single_sensor";
            let mut collator = RangeDataCollator::new(&[sensor_id]);
            let empty_data = TimedPointCloudData {
                time: from_universal(300),
                ..Default::default()
            };
            let output_0 = collator.add_range_data(sensor_id, empty_data.clone());
            assert_eq!(output_0.time, empty_data.time);
            assert_eq!(output_0.ranges.len(), empty_data.ranges.len());
            assert!(are_point_timestamps_sorted(&output_0));
            let output_1 =
                collator.add_range_data(sensor_id, create_fake_range_data(300, 500, false));
            assert_eq!(to_universal(output_1.time), 500);
            assert_eq!(output_1.origins.len(), 1);
            assert_eq!(output_1.ranges.len(), NUM_SAMPLES);
            assert!(are_point_timestamps_sorted(&output_1));
            assert_near(absolute_time(&output_1, 0), 300, 2);
            let output_2 =
                collator.add_range_data(sensor_id, create_fake_range_data(-1000, 510, false));
            assert_eq!(to_universal(output_2.time), 510);
            assert_eq!(output_2.origins.len(), 1);
            assert_eq!(output_2.ranges.len(), 1);
            assert_eq!(output_2.ranges[0].point_time.time, 0.);
            assert!(are_point_timestamps_sorted(&output_2));
        }

        #[test]
        fn two_sensors() {
            let sensor_0 = "sensor_0";
            let sensor_1 = "sensor_1";
            let mut collator = RangeDataCollator::new(&[sensor_0, sensor_1]);
            let output_0 =
                collator.add_range_data(sensor_0, create_fake_range_data(200, 300, false));
            assert_eq!(output_0.ranges.len(), 0);
            let output_1 =
                collator.add_range_data(sensor_1, create_fake_range_data(-1000, 310, false));
            assert_eq!(output_1.origins.len(), 2);
            assert_eq!(to_universal(output_1.time), 300);
            assert_eq!(output_1.ranges.len(), 2 * NUM_SAMPLES - 1);
            assert_near(absolute_time(&output_1, 0), -1000, 2);
            assert_eq!(output_1.ranges.last().unwrap().point_time.time, 0.);
            assert!(are_point_timestamps_sorted(&output_1));
            let output_2 =
                collator.add_range_data(sensor_0, create_fake_range_data(300, 500, false));
            assert_eq!(output_2.origins.len(), 2);
            assert_eq!(to_universal(output_2.time), 310);
            assert_eq!(output_2.ranges.len(), 2);
            assert_near(absolute_time(&output_2, 0), 300, 2);
            assert_eq!(output_2.ranges.last().unwrap().point_time.time, 0.);
            assert!(are_point_timestamps_sorted(&output_2));
            // Sending the same sensor will flush everything before.
            let output_3 =
                collator.add_range_data(sensor_0, create_fake_range_data(600, 700, false));
            assert_eq!(to_universal(output_3.time), 500);
            assert_eq!(
                output_1.ranges.len() + output_2.ranges.len() + output_3.ranges.len(),
                3 * NUM_SAMPLES
            );
            assert_eq!(output_3.ranges.last().unwrap().point_time.time, 0.);
            assert!(are_point_timestamps_sorted(&output_3));
        }

        fn run_three_sensors(fake_intensities: bool) {
            let sensor_0 = "sensor_0";
            let sensor_1 = "sensor_1";
            let sensor_2 = "sensor_2";
            let mut collator = RangeDataCollator::new(&[sensor_0, sensor_1, sensor_2]);
            let output_0 = collator
                .add_range_data(sensor_0, create_fake_range_data(100, 200, fake_intensities));
            assert_eq!(output_0.ranges.len(), 0);
            let output_1 = collator
                .add_range_data(sensor_1, create_fake_range_data(199, 250, fake_intensities));
            assert_eq!(output_1.ranges.len(), 0);
            let output_2 = collator
                .add_range_data(sensor_2, create_fake_range_data(210, 300, fake_intensities));
            assert_eq!(output_2.ranges.len(), NUM_SAMPLES + 1);
            assert!(are_point_timestamps_sorted(&output_2));
            if fake_intensities {
                intensities_are_consistent(&output_2);
            }
            let output_3 = collator
                .add_range_data(sensor_2, create_fake_range_data(400, 500, fake_intensities));
            assert_eq!(
                output_2.ranges.len() + output_3.ranges.len(),
                3 * NUM_SAMPLES
            );
            assert!(are_point_timestamps_sorted(&output_3));
            if fake_intensities {
                intensities_are_consistent(&output_3);
            }
        }

        #[test]
        fn three_sensors() {
            run_three_sensors(false);
        }

        #[test]
        fn three_sensors_with_intensities() {
            run_three_sensors(true);
        }
    }

    #[test]
    #[should_panic(expected = "unexpected range sensor id")]
    fn unexpected_sensor_panics() {
        let mut collator = RangeDataCollator::new(&["a"]);
        let _ = collator.add_range_data("b", TimedPointCloudData::default());
    }
}
