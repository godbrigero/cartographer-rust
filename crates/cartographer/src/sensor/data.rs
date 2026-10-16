//! Port of `cartographer/sensor/data` (.h).

use crate::common::time::Time;
use crate::mapping::trajectory_builder_interface::TrajectoryBuilderInterface;

/// A polymorphic sensor packet, owned as `Box<dyn Data>` while it travels
/// through the collators (C++ abstract class `sensor::Data`).
///
/// Implemented by [`crate::sensor::internal::dispatchable::Dispatchable`] for
/// every sensor data type and by `mapping::LocalSlamResultData`.
pub trait Data: Send {
    /// Time of the packet (C++ `GetTime`).
    fn get_time(&self) -> Time;

    /// The sensor id (C++ `GetSensorId`).
    fn get_sensor_id(&self) -> &str;

    /// Double dispatch into the matching `add_*` method of
    /// `trajectory_builder` (C++ `AddToTrajectoryBuilder`).
    ///
    /// RUST PORT NOTE: consumes the boxed packet (the C++ passes the stored
    /// payload by const reference; the packet is destroyed right afterwards
    /// in all callers).
    fn add_to_trajectory_builder(
        self: Box<Self>,
        trajectory_builder: &mut dyn TrajectoryBuilderInterface,
    );
}

/// Data types with a `time` member (C++ templates such as `MapByTime` and
/// `Dispatchable` require a `common::Time time` member).
pub trait HasTime {
    /// The time of the data.
    fn time(&self) -> Time;
}
