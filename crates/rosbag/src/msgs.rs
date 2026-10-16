//! Plain-struct decoders for common ROS1 messages.
//!
//! Field names and layouts follow the ROS1 `.msg` definitions exactly.

use crate::decode::{Decoder, RosMsg};
use crate::{Error, Result, Time};

/// `std_msgs/Header`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Header {
    /// Sequence number.
    pub seq: u32,
    /// Acquisition time.
    pub stamp: Time,
    /// Coordinate frame.
    pub frame_id: String,
}

impl RosMsg for Header {
    const MSG_TYPE: &'static str = "std_msgs/Header";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        Ok(Self {
            seq: d.u32()?,
            stamp: d.time()?,
            frame_id: d.string()?,
        })
    }
}

/// `geometry_msgs/Vector3` (also used for `geometry_msgs/Point`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vector3 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

impl Vector3 {
    /// Components as `[x, y, z]`.
    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

impl RosMsg for Vector3 {
    const MSG_TYPE: &'static str = "geometry_msgs/Vector3";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        Ok(Self {
            x: d.f64()?,
            y: d.f64()?,
            z: d.f64()?,
        })
    }
}

/// `geometry_msgs/Point` has the same layout as `Vector3`.
pub type Point = Vector3;

/// `geometry_msgs/Quaternion` (ROS order: x, y, z, w).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
    /// W (scalar) component.
    pub w: f64,
}

impl Default for Quaternion {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
}

impl RosMsg for Quaternion {
    const MSG_TYPE: &'static str = "geometry_msgs/Quaternion";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        Ok(Self {
            x: d.f64()?,
            y: d.f64()?,
            z: d.f64()?,
            w: d.f64()?,
        })
    }
}

/// `geometry_msgs/Pose`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Pose {
    /// Position.
    pub position: Point,
    /// Orientation.
    pub orientation: Quaternion,
}

impl RosMsg for Pose {
    const MSG_TYPE: &'static str = "geometry_msgs/Pose";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        Ok(Self {
            position: Point::decode_from(d)?,
            orientation: Quaternion::decode_from(d)?,
        })
    }
}

/// `geometry_msgs/PoseStamped`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PoseStamped {
    /// Header.
    pub header: Header,
    /// Pose.
    pub pose: Pose,
}

impl RosMsg for PoseStamped {
    const MSG_TYPE: &'static str = "geometry_msgs/PoseStamped";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        Ok(Self {
            header: Header::decode_from(d)?,
            pose: Pose::decode_from(d)?,
        })
    }
}

/// `geometry_msgs/PoseWithCovariance`.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseWithCovariance {
    /// Pose.
    pub pose: Pose,
    /// Row-major 6x6 covariance (x, y, z, rot x, rot y, rot z).
    pub covariance: [f64; 36],
}

impl RosMsg for PoseWithCovariance {
    const MSG_TYPE: &'static str = "geometry_msgs/PoseWithCovariance";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        Ok(Self {
            pose: Pose::decode_from(d)?,
            covariance: d.f64_array()?,
        })
    }
}

/// `geometry_msgs/Twist`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Twist {
    /// Linear velocity.
    pub linear: Vector3,
    /// Angular velocity.
    pub angular: Vector3,
}

impl RosMsg for Twist {
    const MSG_TYPE: &'static str = "geometry_msgs/Twist";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        Ok(Self {
            linear: Vector3::decode_from(d)?,
            angular: Vector3::decode_from(d)?,
        })
    }
}

/// `geometry_msgs/TwistWithCovariance`.
#[derive(Debug, Clone, PartialEq)]
pub struct TwistWithCovariance {
    /// Twist.
    pub twist: Twist,
    /// Row-major 6x6 covariance.
    pub covariance: [f64; 36],
}

impl RosMsg for TwistWithCovariance {
    const MSG_TYPE: &'static str = "geometry_msgs/TwistWithCovariance";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        Ok(Self {
            twist: Twist::decode_from(d)?,
            covariance: d.f64_array()?,
        })
    }
}

/// `nav_msgs/Odometry`.
#[derive(Debug, Clone, PartialEq)]
pub struct Odometry {
    /// Header (`frame_id` is the pose's frame).
    pub header: Header,
    /// Frame of the twist.
    pub child_frame_id: String,
    /// Pose estimate.
    pub pose: PoseWithCovariance,
    /// Velocity estimate (in `child_frame_id`).
    pub twist: TwistWithCovariance,
}

impl RosMsg for Odometry {
    const MSG_TYPE: &'static str = "nav_msgs/Odometry";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        Ok(Self {
            header: Header::decode_from(d)?,
            child_frame_id: d.string()?,
            pose: PoseWithCovariance::decode_from(d)?,
            twist: TwistWithCovariance::decode_from(d)?,
        })
    }
}

/// `sensor_msgs/Imu`.
#[derive(Debug, Clone, PartialEq)]
pub struct Imu {
    /// Header.
    pub header: Header,
    /// Orientation estimate.
    pub orientation: Quaternion,
    /// Row-major 3x3 covariance; element 0 == -1 means "no orientation".
    pub orientation_covariance: [f64; 9],
    /// Angular velocity, rad/s.
    pub angular_velocity: Vector3,
    /// Row-major 3x3 covariance.
    pub angular_velocity_covariance: [f64; 9],
    /// Linear acceleration, m/s^2.
    pub linear_acceleration: Vector3,
    /// Row-major 3x3 covariance.
    pub linear_acceleration_covariance: [f64; 9],
}

impl RosMsg for Imu {
    const MSG_TYPE: &'static str = "sensor_msgs/Imu";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        Ok(Self {
            header: Header::decode_from(d)?,
            orientation: Quaternion::decode_from(d)?,
            orientation_covariance: d.f64_array()?,
            angular_velocity: Vector3::decode_from(d)?,
            angular_velocity_covariance: d.f64_array()?,
            linear_acceleration: Vector3::decode_from(d)?,
            linear_acceleration_covariance: d.f64_array()?,
        })
    }
}

/// `sensor_msgs/LaserScan`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LaserScan {
    /// Header.
    pub header: Header,
    /// Start angle, rad.
    pub angle_min: f32,
    /// End angle, rad.
    pub angle_max: f32,
    /// Angular step, rad.
    pub angle_increment: f32,
    /// Time between measurements, s.
    pub time_increment: f32,
    /// Time between scans, s.
    pub scan_time: f32,
    /// Minimum valid range, m.
    pub range_min: f32,
    /// Maximum valid range, m.
    pub range_max: f32,
    /// Ranges, m.
    pub ranges: Vec<f32>,
    /// Intensities (may be empty).
    pub intensities: Vec<f32>,
}

impl RosMsg for LaserScan {
    const MSG_TYPE: &'static str = "sensor_msgs/LaserScan";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        Ok(Self {
            header: Header::decode_from(d)?,
            angle_min: d.f32()?,
            angle_max: d.f32()?,
            angle_increment: d.f32()?,
            time_increment: d.f32()?,
            scan_time: d.f32()?,
            range_min: d.f32()?,
            range_max: d.f32()?,
            ranges: d.f32_vec()?,
            intensities: d.f32_vec()?,
        })
    }
}

/// `geometry_msgs/Transform`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Transform {
    /// Translation.
    pub translation: Vector3,
    /// Rotation.
    pub rotation: Quaternion,
}

impl RosMsg for Transform {
    const MSG_TYPE: &'static str = "geometry_msgs/Transform";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        Ok(Self {
            translation: Vector3::decode_from(d)?,
            rotation: Quaternion::decode_from(d)?,
        })
    }
}

/// `geometry_msgs/TransformStamped`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TransformStamped {
    /// Header (`frame_id` is the parent frame).
    pub header: Header,
    /// Child frame.
    pub child_frame_id: String,
    /// Transform from child to parent frame.
    pub transform: Transform,
}

impl RosMsg for TransformStamped {
    const MSG_TYPE: &'static str = "geometry_msgs/TransformStamped";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        Ok(Self {
            header: Header::decode_from(d)?,
            child_frame_id: d.string()?,
            transform: Transform::decode_from(d)?,
        })
    }
}

/// `tf2_msgs/TFMessage` (also published on `/tf` and `/tf_static`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TFMessage {
    /// Transforms.
    pub transforms: Vec<TransformStamped>,
}

impl RosMsg for TFMessage {
    const MSG_TYPE: &'static str = "tf2_msgs/TFMessage";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        // Minimum TransformStamped: header (4+8+4) + string (4) + 7 f64.
        Ok(Self {
            transforms: d.msg_vec(4 + 8 + 4 + 4 + 56)?,
        })
    }
}

/// `sensor_msgs/PointField` datatype constants.
pub mod point_field {
    /// `int8`.
    pub const INT8: u8 = 1;
    /// `uint8`.
    pub const UINT8: u8 = 2;
    /// `int16`.
    pub const INT16: u8 = 3;
    /// `uint16`.
    pub const UINT16: u8 = 4;
    /// `int32`.
    pub const INT32: u8 = 5;
    /// `uint32`.
    pub const UINT32: u8 = 6;
    /// `float32`.
    pub const FLOAT32: u8 = 7;
    /// `float64`.
    pub const FLOAT64: u8 = 8;

    /// Size in bytes of a datatype, or `None` if unknown.
    pub fn size_of(datatype: u8) -> Option<usize> {
        match datatype {
            INT8 | UINT8 => Some(1),
            INT16 | UINT16 => Some(2),
            INT32 | UINT32 | FLOAT32 => Some(4),
            FLOAT64 => Some(8),
            _ => None,
        }
    }
}

/// `sensor_msgs/PointField`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PointField {
    /// Field name, e.g. `x`, `intensity`.
    pub name: String,
    /// Byte offset within a point.
    pub offset: u32,
    /// One of the [`point_field`] constants.
    pub datatype: u8,
    /// Number of elements.
    pub count: u32,
}

impl RosMsg for PointField {
    const MSG_TYPE: &'static str = "sensor_msgs/PointField";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        Ok(Self {
            name: d.string()?,
            offset: d.u32()?,
            datatype: d.u8()?,
            count: d.u32()?,
        })
    }
}

/// `sensor_msgs/PointCloud2`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PointCloud2 {
    /// Header.
    pub header: Header,
    /// Number of rows (1 for unordered clouds).
    pub height: u32,
    /// Points per row.
    pub width: u32,
    /// Field layout of each point.
    pub fields: Vec<PointField>,
    /// Whether multi-byte fields are big-endian.
    pub is_bigendian: bool,
    /// Bytes per point.
    pub point_step: u32,
    /// Bytes per row.
    pub row_step: u32,
    /// Raw point data, `row_step * height` bytes.
    pub data: Vec<u8>,
    /// True if there are no invalid points.
    pub is_dense: bool,
}

impl RosMsg for PointCloud2 {
    const MSG_TYPE: &'static str = "sensor_msgs/PointCloud2";
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self> {
        Ok(Self {
            header: Header::decode_from(d)?,
            height: d.u32()?,
            width: d.u32()?,
            fields: d.msg_vec(4 + 4 + 1 + 4)?,
            is_bigendian: d.bool()?,
            point_step: d.u32()?,
            row_step: d.u32()?,
            data: d.byte_vec()?,
            is_dense: d.bool()?,
        })
    }
}

/// Reads one named field from every point of a [`PointCloud2`], converting
/// any datatype to `f64`. Obtain one with [`PointCloud2::field_reader`].
#[derive(Debug, Clone, Copy)]
pub struct FieldReader {
    offset: usize,
    datatype: u8,
    big_endian: bool,
}

impl FieldReader {
    /// Reads the field from the point starting at byte `point_start`.
    /// Returns `None` if the point lies outside `data`.
    pub fn read(&self, data: &[u8], point_start: usize) -> Option<f64> {
        use point_field::*;
        let start = point_start.checked_add(self.offset)?;
        let len = size_of(self.datatype)?;
        let b = data.get(start..start.checked_add(len)?)?;
        macro_rules! num {
            ($t:ty) => {{
                let a = b.try_into().ok()?;
                if self.big_endian {
                    <$t>::from_be_bytes(a)
                } else {
                    <$t>::from_le_bytes(a)
                }
            }};
        }
        Some(match self.datatype {
            INT8 => f64::from(b[0] as i8),
            UINT8 => f64::from(b[0]),
            INT16 => f64::from(num!(i16)),
            UINT16 => f64::from(num!(u16)),
            INT32 => f64::from(num!(i32)),
            UINT32 => f64::from(num!(u32)),
            FLOAT32 => f64::from(num!(f32)),
            FLOAT64 => num!(f64),
            _ => return None,
        })
    }
}

/// One point extracted by [`PointCloud2::read_points`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CloudPoint {
    /// Position `[x, y, z]` in the cloud's frame.
    pub position: [f32; 3],
    /// `intensity` field, if present.
    pub intensity: Option<f32>,
    /// `ring` field, if present.
    pub ring: Option<u16>,
    /// Per-point `time` field, if present (driver-specific units, typically
    /// seconds relative to the header stamp).
    pub time: Option<f64>,
}

impl PointCloud2 {
    /// Number of points (`width * height`).
    pub fn num_points(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Looks up a field by name.
    pub fn field(&self, name: &str) -> Option<&PointField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Returns a reader for the first element of field `name`, or `Ok(None)`
    /// if the cloud has no such field. Errors if the datatype is unknown or
    /// the field does not fit within `point_step`.
    pub fn field_reader(&self, name: &str) -> Result<Option<FieldReader>> {
        let Some(f) = self.field(name) else {
            return Ok(None);
        };
        let size = point_field::size_of(f.datatype).ok_or_else(|| {
            Error::BadPointField(format!(
                "field `{name}` has unknown datatype {}",
                f.datatype
            ))
        })?;
        if f.offset as usize + size > self.point_step as usize {
            return Err(Error::BadPointField(format!(
                "field `{name}` (offset {}, size {size}) exceeds point_step {}",
                f.offset, self.point_step
            )));
        }
        Ok(Some(FieldReader {
            offset: f.offset as usize,
            datatype: f.datatype,
            big_endian: self.is_bigendian,
        }))
    }

    /// Byte offsets of each point's start, honoring `row_step`.
    pub fn point_offsets(&self) -> impl Iterator<Item = usize> + '_ {
        let (w, ps, rs) = (
            self.width as usize,
            self.point_step as usize,
            self.row_step as usize,
        );
        (0..self.height as usize).flat_map(move |r| (0..w).map(move |c| r * rs + c * ps))
    }

    /// Reads every value of field `name` as `f64`, or `Ok(None)` if absent.
    pub fn read_field(&self, name: &str) -> Result<Option<Vec<f64>>> {
        let Some(reader) = self.field_reader(name)? else {
            return Ok(None);
        };
        self.check_data_len()?;
        Ok(Some(
            self.point_offsets()
                .map(|o| reader.read(&self.data, o).expect("bounds checked"))
                .collect(),
        ))
    }

    fn check_data_len(&self) -> Result<()> {
        if self.num_points() == 0 {
            return Ok(());
        }
        let needed = (self.height as usize - 1) * self.row_step as usize
            + self.width as usize * self.point_step as usize;
        if self.data.len() < needed {
            return Err(Error::BadPointField(format!(
                "data has {} bytes, layout needs {needed}",
                self.data.len()
            )));
        }
        Ok(())
    }

    /// Extracts `x`, `y`, `z` (required) and `intensity`, `ring`, `time`
    /// (optional) from every point, whatever their datatypes and offsets.
    pub fn read_points(&self) -> Result<Vec<CloudPoint>> {
        let need = |n: &str| {
            self.field_reader(n)?
                .ok_or_else(|| Error::BadPointField(format!("cloud has no `{n}` field")))
        };
        let (x, y, z) = (need("x")?, need("y")?, need("z")?);
        let intensity = self.field_reader("intensity")?;
        let ring = self.field_reader("ring")?;
        let time = self.field_reader("time")?;
        self.check_data_len()?;
        let d = &self.data;
        Ok(self
            .point_offsets()
            .map(|o| {
                let rd = |r: &FieldReader| r.read(d, o).expect("bounds checked");
                CloudPoint {
                    position: [rd(&x) as f32, rd(&y) as f32, rd(&z) as f32],
                    intensity: intensity.as_ref().map(|r| rd(r) as f32),
                    ring: ring.as_ref().map(|r| rd(r) as u16),
                    time: time.as_ref().map(rd),
                }
            })
            .collect())
    }
}
