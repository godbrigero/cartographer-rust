//! Port of `cartographer/mapping/detect_floors` (.h, .cc).

use std::collections::BTreeMap;

use crate::common::time::{Time, from_universal};
use crate::mapping::proto;
use crate::transform::transform::ToEigen;

/// A span of time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timespan {
    /// Start of the span.
    pub start: Time,
    /// End of the span.
    pub end: Time,
}

/// A detected floor.
#[derive(Clone, Debug, PartialEq)]
pub struct Floor {
    /// The spans of time we spent on this floor. Since we might have walked up
    /// and down many times in this place, there can be many spans of time we
    /// spent on a particular floor.
    pub timespans: Vec<Timespan>,
    /// The median z-value of this floor.
    pub z: f64,
}

type Levels = BTreeMap<usize, usize>;

const MAX_SHORT_SPAN_LENGTH_METERS: f64 = 25.;
const LEVEL_HEIGHT_METERS: f64 = 2.5;
const MIN_LEVEL_SEPARATION_METERS: f64 = 1.;

/// A continuous span of nodes `[start_index, end_index)` at roughly the same
/// altitude, with the sorted z values of its nodes.
#[derive(Clone, Debug)]
struct Span {
    start_index: usize,
    end_index: usize,
    z_values: Vec<f64>,
}

fn node_z(node: &proto::trajectory::Node) -> f64 {
    node.pose
        .and_then(|pose| pose.translation)
        .unwrap_or_default()
        .z
}

/// Union-find "find" over `levels` (C++ `LevelFind`, written iteratively).
fn level_find(i: usize, levels: &Levels) -> usize {
    let mut i = i;
    loop {
        let parent = *levels.get(&i).expect("Check failed: it != levels.end()");
        if parent == i {
            return parent;
        }
        i = parent;
    }
}

fn level_union(i: usize, j: usize, levels: &mut Levels) {
    let repr_i = level_find(i, levels);
    let repr_j = level_find(j, levels);
    levels.insert(repr_i, repr_j);
}

fn insert_sorted(val: f64, vals: &mut Vec<f64>) {
    // C++ `std::upper_bound`.
    let index = vals.partition_point(|&v| v <= val);
    vals.insert(index, val);
}

fn median(sorted: &[f64]) -> f64 {
    assert!(!sorted.is_empty(), "Check failed: !sorted.empty()");
    sorted[sorted.len() / 2]
}

fn slice_by_altitude_change(trajectory: &proto::Trajectory) -> Vec<Span> {
    assert!(
        !trajectory.node.is_empty(),
        "Check failed: trajectory.node_size() > 0"
    );
    let mut spans = vec![Span {
        start_index: 0,
        end_index: 0,
        z_values: vec![node_z(&trajectory.node[0])],
    }];
    for (i, node) in trajectory.node.iter().enumerate().skip(1) {
        let z = node_z(node);
        if (median(&spans.last().expect("non-empty").z_values) - z).abs() > LEVEL_HEIGHT_METERS {
            spans.push(Span {
                start_index: i,
                end_index: i,
                z_values: Vec::new(),
            });
        }
        let back = spans.last_mut().expect("non-empty");
        insert_sorted(z, &mut back.z_values);
        back.end_index = i + 1;
    }
    spans
}

fn span_length(trajectory: &proto::Trajectory, span: &Span) -> f64 {
    let mut length = 0.;
    for i in (span.start_index + 1)..span.end_index {
        let translation = |index: usize| {
            trajectory.node[index]
                .pose
                .and_then(|pose| pose.translation)
                .unwrap_or_default()
                .to_eigen()
        };
        let a = translation(i - 1);
        let b = translation(i);
        length += (a - b).xy().norm();
    }
    length
}

fn is_short(trajectory: &proto::Trajectory, span: &Span) -> bool {
    span_length(trajectory, span) < MAX_SHORT_SPAN_LENGTH_METERS
}

fn group_segments_by_altitude(spans: &[Span], levels: &mut Levels) {
    for i in 0..spans.len() {
        for j in (i + 1)..spans.len() {
            if (median(&spans[i].z_values) - median(&spans[j].z_values)).abs()
                < MIN_LEVEL_SEPARATION_METERS
            {
                level_union(i, j, levels);
            }
        }
    }
}

fn find_floors(trajectory: &proto::Trajectory, spans: &[Span], levels: &Levels) -> Vec<Floor> {
    let mut level_spans: BTreeMap<usize, Vec<Span>> = BTreeMap::new();

    // Initialize the levels to start out with only long spans.
    for (i, span) in spans.iter().enumerate() {
        if !is_short(trajectory, span) {
            level_spans
                .entry(level_find(i, levels))
                .or_default()
                .push(span.clone());
        }
    }

    for (i, span) in spans.iter().enumerate() {
        if !is_short(trajectory, span) {
            continue;
        }

        // If we have a long piece on this floor already, merge this short piece
        // into it.
        let level = level_find(i, levels);
        let level_entry = level_spans.entry(level).or_default();
        if !level_entry.is_empty() {
            level_entry.push(span.clone());
            continue;
        }

        // Otherwise, add this short piece to the level before and after it. It
        // is likely some intermediate level on stairs.
        // (The C++ relies on unsigned wrap-around of `i - 1` for `i == 0`.)
        if let Some(index) = i.checked_sub(1) {
            level_spans
                .entry(level_find(index, levels))
                .or_default()
                .push(span.clone());
        }
        let index = i + 1;
        if index < spans.len() {
            level_spans
                .entry(level_find(index, levels))
                .or_default()
                .push(span.clone());
        }
    }

    // Convert the level_spans structure to 'Floor'.
    let mut floors = Vec::new();
    for level in level_spans.values_mut() {
        if level.is_empty() {
            continue;
        }

        let mut z_values = Vec::new();
        level.sort_by_key(|span| (span.start_index, span.end_index));
        let mut floor = Floor {
            timespans: Vec::new(),
            z: 0.,
        };
        for span in level.iter() {
            if !is_short(trajectory, span) {
                // To figure out the median height of this floor, we only care
                // for the long pieces that are guaranteed to be in the
                // structure. This is a heuristic to leave out intermediate
                // (short) levels.
                z_values.extend_from_slice(&span.z_values);
            }
            floor.timespans.push(Timespan {
                start: from_universal(trajectory.node[span.start_index].timestamp),
                end: from_universal(trajectory.node[span.end_index - 1].timestamp),
            });
        }
        if !z_values.is_empty() {
            z_values.sort_by(f64::total_cmp);
            floor.z = median(&z_values);
            floors.push(floor);
        } else {
            log::error!("All spans in level are short");
        }
    }
    floors
}

/// Uses a heuristic which looks at z-values of the poses to detect individual
/// floors of a building. This requires that floors are *mostly* on the same
/// z-height and that level changes happen *relatively* abrubtly, e.g. by
/// taking the stairs.
pub fn detect_floors(trajectory: &proto::Trajectory) -> Vec<Floor> {
    let spans = slice_by_altitude_change(trajectory);
    let mut levels: Levels = (0..spans.len()).map(|i| (i, i)).collect();
    group_segments_by_altitude(&spans, &mut levels);

    let mut floors = find_floors(trajectory, &spans, &levels);
    // RUST PORT NOTE: `std::sort` is not stable; a stable sort is used here
    // (floors with identical z may come out in a different order than C++).
    floors.sort_by(|a, b| a.z.total_cmp(&b.z));
    floors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::proto as tproto;

    /// Builds a trajectory from `(x, y, z)` positions, one node per 1 s.
    fn trajectory(positions: &[(f64, f64, f64)]) -> proto::Trajectory {
        proto::Trajectory {
            trajectory_id: 0,
            node: positions
                .iter()
                .enumerate()
                .map(|(i, &(x, y, z))| proto::trajectory::Node {
                    node_index: i as i32,
                    timestamp: i as i64 * 10_000_000,
                    pose: Some(tproto::Rigid3d {
                        translation: Some(tproto::Vector3d { x, y, z }),
                        rotation: Some(tproto::Quaterniond {
                            x: 0.,
                            y: 0.,
                            z: 0.,
                            w: 1.,
                        }),
                    }),
                })
                .collect(),
            submap: Vec::new(),
        }
    }

    /// `n` nodes walking 1 m per node along x at height `z`.
    fn walk(start_x: f64, n: usize, z: f64) -> Vec<(f64, f64, f64)> {
        (0..n).map(|i| (start_x + i as f64, 0., z)).collect()
    }

    #[test]
    fn single_floor() {
        let floors = detect_floors(&trajectory(&walk(0., 40, 1.)));
        assert_eq!(floors.len(), 1);
        assert_eq!(floors[0].z, 1.);
        assert_eq!(
            floors[0].timespans,
            vec![Timespan {
                start: from_universal(0),
                end: from_universal(39 * 10_000_000),
            }]
        );
    }

    #[test]
    fn two_floors_with_return_and_stairs() {
        // Ground floor, a short intermediate landing, first floor, back to the
        // ground floor.
        let mut positions = walk(0., 30, 0.);
        positions.extend(walk(30., 3, 3.));
        positions.extend(walk(33., 30, 6.));
        positions.extend(walk(63., 30, 0.2));
        let floors = detect_floors(&trajectory(&positions));
        assert_eq!(floors.len(), 2);
        // Sorted by z. The ground floor's median uses the z values of both
        // long ground spans.
        assert_eq!(floors[0].z, 0.2);
        assert_eq!(floors[1].z, 6.);
        // The short landing (nodes 30..33) is attached to the levels before
        // and after it.
        let landing = Timespan {
            start: from_universal(30 * 10_000_000),
            end: from_universal(32 * 10_000_000),
        };
        assert_eq!(floors[0].timespans.len(), 3);
        assert_eq!(floors[0].timespans[1], landing);
        assert_eq!(floors[1].timespans.len(), 2);
        assert_eq!(floors[1].timespans[0], landing);
    }

    #[test]
    fn only_short_spans_yield_no_floor() {
        let floors = detect_floors(&trajectory(&walk(0., 5, 0.)));
        assert!(floors.is_empty());
    }

    #[test]
    #[should_panic(expected = "node_size() > 0")]
    fn empty_trajectory_panics() {
        detect_floors(&trajectory(&[]));
    }

    #[test]
    fn insert_sorted_is_upper_bound() {
        let mut vals = vec![1., 2., 2., 3.];
        insert_sorted(2., &mut vals);
        assert_eq!(vals, vec![1., 2., 2., 2., 3.]);
        insert_sorted(0., &mut vals);
        insert_sorted(4., &mut vals);
        assert_eq!(vals, vec![0., 1., 2., 2., 2., 3., 4.]);
    }
}
