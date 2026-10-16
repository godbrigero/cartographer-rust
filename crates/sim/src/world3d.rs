//! 3D worlds made of axis-aligned boxes and infinite planes.

use nalgebra::{Point3, Unit, Vector3};

use crate::raycast::{Hit3, RayCast};
use crate::world2d::World2d;

/// Axis-aligned box `[min, max]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Min corner.
    pub min: Point3<f64>,
    /// Max corner.
    pub max: Point3<f64>,
}

impl Aabb {
    /// Creates a box from two opposite corners (any order).
    pub fn new(a: Point3<f64>, b: Point3<f64>) -> Self {
        Self {
            min: a.inf(&b),
            max: a.sup(&b),
        }
    }

    /// Creates a box from raw min/max coordinates.
    pub fn from_bounds(x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64) -> Self {
        Self::new(Point3::new(x0, y0, z0), Point3::new(x1, y1, z1))
    }

    /// Whether `p` is inside or on the box.
    pub fn contains(&self, p: &Point3<f64>) -> bool {
        (0..3).all(|i| p[i] >= self.min[i] && p[i] <= self.max[i])
    }

    /// Slab-method ray intersection. Returns the entry distance and the
    /// outward normal of the entry face. Rays starting inside the box
    /// (entry distance `< 0`) report no hit.
    pub fn intersect(
        &self,
        origin: &Point3<f64>,
        dir: &Vector3<f64>,
        max_range: f64,
    ) -> Option<(f64, Vector3<f64>)> {
        let mut t_enter = f64::NEG_INFINITY;
        let mut t_exit = f64::INFINITY;
        let mut enter_axis = 0;
        let mut enter_sign = 0.0;
        for i in 0..3 {
            if dir[i].abs() < 1e-15 {
                if origin[i] < self.min[i] || origin[i] > self.max[i] {
                    return None;
                }
                continue;
            }
            let inv = 1.0 / dir[i];
            let (t0, t1) = (
                (self.min[i] - origin[i]) * inv,
                (self.max[i] - origin[i]) * inv,
            );
            let (near, far, sign) = if t0 <= t1 {
                (t0, t1, -1.0)
            } else {
                (t1, t0, 1.0)
            };
            if near > t_enter {
                t_enter = near;
                enter_axis = i;
                enter_sign = sign;
            }
            t_exit = t_exit.min(far);
        }
        if t_enter > t_exit || t_enter < 0.0 || t_enter > max_range {
            return None;
        }
        let mut n = Vector3::zeros();
        n[enter_axis] = enter_sign;
        Some((t_enter, n))
    }
}

/// Infinite plane `{x : normal . x = offset}`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// Unit normal.
    pub normal: Unit<Vector3<f64>>,
    /// Signed offset along the normal.
    pub offset: f64,
}

impl Plane {
    /// Horizontal plane at height `z`.
    pub fn horizontal(z: f64) -> Self {
        Self {
            normal: Vector3::z_axis(),
            offset: z,
        }
    }

    /// Ray intersection; returns distance and normal facing the ray.
    pub fn intersect(
        &self,
        origin: &Point3<f64>,
        dir: &Vector3<f64>,
        max_range: f64,
    ) -> Option<(f64, Vector3<f64>)> {
        let denom = self.normal.dot(dir);
        if denom.abs() < 1e-12 {
            return None;
        }
        let t = (self.offset - self.normal.dot(&origin.coords)) / denom;
        if t < 0.0 || t > max_range {
            return None;
        }
        let n = if denom > 0.0 {
            -self.normal.into_inner()
        } else {
            self.normal.into_inner()
        };
        Some((t, n))
    }
}

/// A 3D world of boxes and planes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct World3d {
    /// Solid axis-aligned boxes (walls, pillars, clutter).
    pub boxes: Vec<Aabb>,
    /// Infinite planes (typically floor and ceiling).
    pub planes: Vec<Plane>,
}

impl World3d {
    /// Creates an empty world.
    pub fn new() -> Self {
        Self::default()
    }

    /// Extrudes every wall of a 2D world into a box of the given `thickness`
    /// (centred on the wall line) from `z = 0` to `z = height`, and adds a
    /// floor plane at `z = 0` and a ceiling plane at `z = height`.
    ///
    /// Axis-aligned walls become one box each; other walls are approximated by
    /// a chain of `thickness`-sized boxes. Note that the visible wall faces
    /// are therefore offset by `thickness / 2` from the 2D wall lines.
    pub fn extrude_2d(world: &World2d, height: f64, thickness: f64) -> Self {
        let h = thickness * 0.5;
        let mut w = Self::new();
        for s in &world.segments {
            let (a, b) = (s.a, s.b);
            if (a.x - b.x).abs() < 1e-12 || (a.y - b.y).abs() < 1e-12 {
                w.boxes.push(Aabb::from_bounds(
                    a.x.min(b.x) - h,
                    a.y.min(b.y) - h,
                    0.0,
                    a.x.max(b.x) + h,
                    a.y.max(b.y) + h,
                    height,
                ));
            } else {
                let n = (s.length() / thickness).ceil() as usize;
                for k in 0..=n {
                    let p = a + (b - a) * (k as f64 / n as f64);
                    w.boxes.push(Aabb::from_bounds(
                        p.x - h,
                        p.y - h,
                        0.0,
                        p.x + h,
                        p.y + h,
                        height,
                    ));
                }
            }
        }
        w.planes.push(Plane::horizontal(0.0));
        w.planes.push(Plane::horizontal(height));
        w
    }

    /// [`World2d::rect_room`] extruded to 3 m (0.1 m thick walls) plus
    /// clutter at several heights: a table top, a wall shelf, a hanging lamp
    /// and a floor crate.
    pub fn room() -> Self {
        let mut w = Self::extrude_2d(&World2d::rect_room(), 3.0, 0.1);
        w.boxes.extend([
            Aabb::from_bounds(-1.0, 2.0, 0.70, 0.6, 2.9, 0.75), // table top
            Aabb::from_bounds(-1.0, 2.0, 0.0, -0.95, 2.05, 0.70), // table leg
            Aabb::from_bounds(0.55, 2.85, 0.0, 0.6, 2.9, 0.70), // table leg
            Aabb::from_bounds(-5.95, -1.0, 1.5, -5.6, 1.0, 1.55), // wall shelf
            Aabb::from_bounds(1.0, 0.5, 2.5, 1.4, 0.9, 2.95),   // hanging lamp
            Aabb::from_bounds(4.6, -3.9, 0.0, 5.6, -3.2, 0.5),  // floor crate
        ]);
        w
    }

    /// [`World2d::loop_corridor`] extruded to 3 m (0.1 m thick walls) plus
    /// ceiling beams, wall-mounted pipes/boxes and floor clutter at irregular
    /// positions.
    pub fn loop_corridor() -> Self {
        let mut w = Self::extrude_2d(&World2d::loop_corridor(), 3.0, 0.1);
        w.boxes.extend([
            // Ceiling beams across the north and south corridors.
            Aabb::from_bounds(-12.0, 7.5, 2.6, -11.7, 10.0, 3.0),
            Aabb::from_bounds(2.0, 7.5, 2.7, 2.4, 10.0, 3.0),
            Aabb::from_bounds(-3.0, -10.0, 2.5, -2.8, -7.5, 3.0),
            // Ceiling beam across the east corridor.
            Aabb::from_bounds(17.5, -6.0, 2.6, 20.0, -5.6, 3.0),
            // Wall-mounted pipe along part of the west wall.
            Aabb::from_bounds(-20.0, -6.0, 2.1, -19.8, 1.0, 2.3),
            // Electrical box on the north outer wall.
            Aabb::from_bounds(12.0, 9.7, 1.2, 12.6, 10.0, 1.8),
            // Floor clutter.
            Aabb::from_bounds(-15.0, -9.95, 0.0, -14.4, -9.5, 0.6),
            Aabb::from_bounds(19.4, 6.0, 0.0, 19.95, 7.0, 1.0),
            Aabb::from_bounds(-19.95, 7.5, 0.0, -19.5, 8.0, 0.4),
        ]);
        w
    }
}

impl RayCast for World3d {
    fn raycast(
        &self,
        origin: &Point3<f64>,
        direction: &Unit<Vector3<f64>>,
        max_range: f64,
    ) -> Option<Hit3> {
        let d = direction.as_ref();
        let mut best: Option<(f64, Vector3<f64>)> = None;
        let mut limit = max_range;
        for b in &self.boxes {
            if let Some(h) = b.intersect(origin, d, limit) {
                limit = h.0;
                best = Some(h);
            }
        }
        for p in &self.planes {
            if let Some(h) = p.intersect(origin, d, limit) {
                limit = h.0;
                best = Some(h);
            }
        }
        best.map(|(range, normal)| Hit3 {
            range,
            point: origin + d * range,
            normal,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(x: f64, y: f64, z: f64) -> Unit<Vector3<f64>> {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    #[test]
    fn ray_box() {
        let b = Aabb::from_bounds(1.0, -1.0, -1.0, 2.0, 1.0, 1.0);
        let o = Point3::origin();
        let (t, n) = b.intersect(&o, &Vector3::x(), 10.0).unwrap();
        assert!((t - 1.0).abs() < 1e-12);
        assert_eq!(n, Vector3::new(-1.0, 0.0, 0.0));
        assert!(b.intersect(&o, &Vector3::y(), 10.0).is_none());
        assert!(b.intersect(&o, &-Vector3::x(), 10.0).is_none());
        assert!(b.intersect(&o, &Vector3::x(), 0.5).is_none());
        // Inside: no hit.
        assert!(
            b.intersect(&Point3::new(1.5, 0.0, 0.0), &Vector3::x(), 10.0)
                .is_none()
        );
        // Diagonal hitting the top face.
        let d = Vector3::new(1.0, 0.0, 1.0).normalize();
        let o2 = Point3::new(0.0, 0.0, 0.0);
        let b2 = Aabb::from_bounds(-5.0, -5.0, 2.0, 5.0, 5.0, 3.0);
        let (t, n) = b2.intersect(&o2, &d, 10.0).unwrap();
        assert!((t - 2.0 * 2f64.sqrt()).abs() < 1e-12);
        assert_eq!(n, Vector3::new(0.0, 0.0, -1.0));
    }

    #[test]
    fn ray_plane() {
        let p = Plane::horizontal(3.0);
        let (t, n) = p
            .intersect(&Point3::new(0.0, 0.0, 1.0), &Vector3::z(), 10.0)
            .unwrap();
        assert!((t - 2.0).abs() < 1e-12);
        assert_eq!(n, -Vector3::z());
        assert!(
            p.intersect(&Point3::new(0.0, 0.0, 1.0), &-Vector3::z(), 10.0)
                .is_none()
        );
        assert!(
            p.intersect(&Point3::new(0.0, 0.0, 1.0), &Vector3::x(), 10.0)
                .is_none()
        );
    }

    #[test]
    fn room_exact_ranges() {
        let w = World3d::room();
        let o = Point3::new(0.0, 0.0, 1.2);
        let r = |d: Unit<Vector3<f64>>| w.raycast(&o, &d, 100.0).unwrap().range;
        // Walls are 0.1 m thick boxes centred on x = +-6, y = +-4.
        assert!((r(dir(1.0, 0.0, 0.0)) - 5.95).abs() < 1e-12);
        assert!((r(dir(-1.0, 0.0, 0.0)) - 5.95).abs() < 1e-12);
        assert!((r(dir(0.0, -1.0, 0.0)) - 3.95).abs() < 1e-12);
        assert!((r(dir(0.0, 0.0, -1.0)) - 1.2).abs() < 1e-12); // floor
        assert!((r(dir(0.0, 0.0, 1.0)) - 1.8).abs() < 1e-12); // ceiling
        // Upwards at 45 deg towards +x: ceiling at 1.8 m height difference.
        assert!((r(dir(1.0, 0.0, 1.0)) - 1.8 * 2f64.sqrt()).abs() < 1e-12);
        // Hanging lamp straight above (1.2, 0.7).
        let h = w
            .raycast(&Point3::new(1.2, 0.7, 1.0), &dir(0.0, 0.0, 1.0), 10.0)
            .unwrap();
        assert!((h.range - 1.5).abs() < 1e-12);
    }

    #[test]
    fn line_of_sight() {
        let w = World3d::room();
        assert!(w.line_of_sight(&Point3::new(0.0, 0.0, 1.0), &Point3::new(1.0, 0.0, 1.0)));
        // Through the pillar at (-3, 1.5).
        assert!(!w.line_of_sight(&Point3::new(-5.0, 1.5, 1.0), &Point3::new(-1.0, 1.5, 1.0)));
    }

    #[test]
    fn corridor_has_floor_and_ceiling() {
        let w = World3d::loop_corridor();
        let o = Point3::new(0.0, 8.75, 1.0);
        let down = w.raycast(&o, &dir(0.0, 0.0, -1.0), 10.0).unwrap();
        assert!((down.range - 1.0).abs() < 1e-12);
        let north = w.raycast(&o, &dir(0.0, 1.0, 0.0), 10.0).unwrap();
        assert!((north.range - 1.2).abs() < 1e-12);
    }
}
