//! Port of `cartographer/mapping/submaps` (.h).
//!
//! RUST PORT NOTE: sharing model. In C++ the local trajectory builder keeps a
//! `std::shared_ptr<Submap2D>` (mutated on every insertion) while the pose
//! graph holds `std::shared_ptr<const Submap2D>` to the same object. In Rust,
//! submaps are shared as `Arc<RwLock<Submap2D>>` / `Arc<RwLock<Submap3D>>`
//! (see [`SharedSubmap`]); the builder takes a write lock to insert, readers
//! take a read lock. Such a handle coerces to `Arc<RwLock<dyn Submap>>` for
//! code that only needs the polymorphic C++ `Submap` interface.

use std::any::Any;
use std::sync::LazyLock;

use std::sync::{Arc, RwLock};

use crate::common::port::round_to_int;
use crate::mapping::probability_values::{MAX_PROBABILITY, MIN_PROBABILITY};
use crate::mapping::proto;
use crate::transform::rigid_transform::Rigid3d;

/// Converts the given probability to log odds.
#[inline]
pub fn logit(probability: f32) -> f32 {
    (probability / (1.0 - probability)).ln()
}

/// `Logit(kMaxProbability)` (C++ `kMaxLogOdds`, dynamically initialized).
pub static MAX_LOG_ODDS: LazyLock<f32> = LazyLock::new(|| logit(MAX_PROBABILITY));
/// `Logit(kMinProbability)` (C++ `kMinLogOdds`, dynamically initialized).
pub static MIN_LOG_ODDS: LazyLock<f32> = LazyLock::new(|| logit(MIN_PROBABILITY));

/// Converts a probability to a log odds integer. 0 means unknown,
/// [`MIN_LOG_ODDS`, `MAX_LOG_ODDS`] is mapped to [1, 255].
#[inline]
pub fn probability_to_log_odds_integer(probability: f32) -> u8 {
    let value = round_to_int(
        (logit(probability) - *MIN_LOG_ODDS) * 254.0_f32 / (*MAX_LOG_ODDS - *MIN_LOG_ODDS),
    ) + 1;
    assert!(1 <= value, "Check failed: 1 <= value ({value})");
    assert!(255 >= value, "Check failed: 255 >= value ({value})");
    value as u8
}

/// Shared, lockable submap handle (C++ `std::shared_ptr<SubmapT>`).
pub type SharedSubmap<S> = Arc<RwLock<S>>;

/// The state common to all submaps (the non-virtual part of the C++
/// `Submap` base class). Concrete submaps embed it and expose it through
/// [`Submap::base`] / [`Submap::base_mut`].
#[derive(Debug, Clone)]
pub struct SubmapBase {
    local_pose: Rigid3d,
    num_range_data: i32,
    insertion_finished: bool,
}

impl SubmapBase {
    /// Creates the base state for a submap at `local_submap_pose`.
    pub fn new(local_submap_pose: Rigid3d) -> Self {
        Self {
            local_pose: local_submap_pose,
            num_range_data: 0,
            insertion_finished: false,
        }
    }
}

/// An individual submap, which has a 'local_pose' in the local map frame,
/// keeps track of how many range data were inserted into it, and sets
/// 'insertion_finished' when the map no longer changes and is ready for loop
/// closing.
///
/// RUST PORT NOTE: `Any` is a supertrait so that `&dyn Submap` can be upcast
/// to `&dyn Any` and downcast to the concrete type (C++
/// `std::static_pointer_cast<const Submap2D>`).
pub trait Submap: Any + Send + Sync {
    /// Access to the shared base state.
    fn base(&self) -> &SubmapBase;
    /// Mutable access to the shared base state.
    fn base_mut(&mut self) -> &mut SubmapBase;

    /// Serializes the submap, optionally including the grid data.
    fn to_proto(&self, include_grid_data: bool) -> proto::Submap;
    /// Updates the submap from `proto`.
    fn update_from_proto(&mut self, proto: &proto::Submap);
    /// Fills data into the 'response'.
    fn to_response_proto(
        &self,
        global_submap_pose: &Rigid3d,
        response: &mut proto::submap_query::Response,
    );

    /// Pose of this submap in the local map frame.
    fn local_pose(&self) -> Rigid3d {
        self.base().local_pose
    }
    /// Number of RangeData inserted.
    fn num_range_data(&self) -> i32 {
        self.base().num_range_data
    }
    /// Sets the number of RangeData inserted.
    fn set_num_range_data(&mut self, num_range_data: i32) {
        self.base_mut().num_range_data = num_range_data;
    }
    /// Whether the submap is finished (no further insertions).
    fn insertion_finished(&self) -> bool {
        self.base().insertion_finished
    }
    /// Marks the submap as finished or not.
    fn set_insertion_finished(&mut self, insertion_finished: bool) {
        self.base_mut().insertion_finished = insertion_finished;
    }
}

#[cfg(test)]
mod tests {
    mod submaps_test {
        use crate::mapping::submaps::*;

        // Converts the given log odds to a probability. This function is
        // known to be very slow, because expf is incredibly slow.
        fn expit(log_odds: f32) -> f32 {
            let exp_log_odds = log_odds.exp();
            exp_log_odds / (1.0 + exp_log_odds)
        }

        #[test]
        fn log_odds_conversions() {
            assert!((expit(logit(MIN_PROBABILITY)) - MIN_PROBABILITY).abs() <= 1e-6);
            assert!((expit(logit(MAX_PROBABILITY)) - MAX_PROBABILITY).abs() <= 1e-6);
            assert!((expit(logit(0.5)) - 0.5).abs() <= 1e-6);
        }

        #[test]
        fn log_odds_integer_bounds() {
            assert_eq!(probability_to_log_odds_integer(MIN_PROBABILITY), 1);
            assert_eq!(probability_to_log_odds_integer(MAX_PROBABILITY), 255);
            assert_eq!(probability_to_log_odds_integer(0.5), 128);
        }
    }
}
