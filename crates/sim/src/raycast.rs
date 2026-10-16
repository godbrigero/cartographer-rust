//! A common 3D ray-casting interface implemented by both world types.

use nalgebra::{Point3, Unit, Vector3};

/// Result of a 3D ray cast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit3 {
    /// Distance along the (unit) ray direction, in metres.
    pub range: f64,
    /// Hit point in world coordinates.
    pub point: Point3<f64>,
    /// Unit surface normal at the hit, oriented against the ray direction.
    pub normal: Vector3<f64>,
}

/// Anything that can be ray-cast in 3D world coordinates.
pub trait RayCast {
    /// Returns the closest hit with `0 <= range <= max_range`, if any.
    fn raycast(
        &self,
        origin: &Point3<f64>,
        direction: &Unit<Vector3<f64>>,
        max_range: f64,
    ) -> Option<Hit3>;

    /// Whether the straight segment from `a` to `b` is unobstructed (ignoring
    /// hits within `1e-6` m of `b`).
    fn line_of_sight(&self, a: &Point3<f64>, b: &Point3<f64>) -> bool {
        let d = b - a;
        let len = d.norm();
        if len < 1e-9 {
            return true;
        }
        let dir = Unit::new_unchecked(d / len);
        self.raycast(a, &dir, len - 1e-6).is_none()
    }
}
