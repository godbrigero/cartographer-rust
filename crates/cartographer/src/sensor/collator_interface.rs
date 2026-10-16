//! Port of `cartographer/sensor/collator_interface` (.h).

use std::collections::HashSet;

use crate::sensor::data::Data;

/// Callback receiving collated data: `(sensor_id, data)` (C++
/// `CollatorInterface::Callback`).
pub type Callback = Box<dyn FnMut(&str, Box<dyn Data>) + Send>;

/// Produces time-sorted sensor output per trajectory.
pub trait CollatorInterface {
    /// Adds a trajectory to produce sorted sensor output for. Calls `callback`
    /// for each collated sensor data.
    fn add_trajectory(
        &mut self,
        trajectory_id: i32,
        expected_sensor_ids: &HashSet<String>,
        callback: Callback,
    );

    /// Marks `trajectory_id` as finished.
    fn finish_trajectory(&mut self, trajectory_id: i32);

    /// Adds `data` for `trajectory_id` to be collated. `data` must contain
    /// valid sensor data. Sensor packets with matching sensor id must be added
    /// in time order.
    fn add_sensor_data(&mut self, trajectory_id: i32, data: Box<dyn Data>);

    /// Dispatches all queued sensor packets. May only be called once.
    /// `add_sensor_data` may not be called after `flush`.
    fn flush(&mut self);

    /// Must only be called if at least one unfinished trajectory exists.
    /// Returns the ID of the trajectory that needs more data before the
    /// collator is unblocked. Returns `None` for implementations that do not
    /// wait for a particular trajectory.
    fn get_blocking_trajectory_id(&self) -> Option<i32>;
}
