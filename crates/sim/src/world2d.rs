//! 2D worlds made of line-segment walls, with ray casting, a free-space test
//! and ground-truth rasterisation (see [`crate::grid`]).
//!
//! All coordinates are metres in the world (= ground-truth map) frame. All
//! preset worlds are made of axis-aligned walls so they can also be extruded
//! into 3D (see [`crate::world3d::World3d::extrude_2d`]).

use nalgebra::{Point2, Point3, Unit, Vector2, Vector3};

use crate::grid::GroundTruthGrid;
use crate::raycast::{Hit3, RayCast};

/// A wall segment from `a` to `b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment2 {
    /// First end point.
    pub a: Point2<f64>,
    /// Second end point.
    pub b: Point2<f64>,
}

impl Segment2 {
    /// Creates a segment from raw coordinates.
    pub fn new(ax: f64, ay: f64, bx: f64, by: f64) -> Self {
        Self {
            a: Point2::new(ax, ay),
            b: Point2::new(bx, by),
        }
    }

    /// Segment length in metres.
    pub fn length(&self) -> f64 {
        (self.b - self.a).norm()
    }

    /// Euclidean distance from `p` to the closest point of the segment.
    pub fn distance_to(&self, p: &Point2<f64>) -> f64 {
        let e = self.b - self.a;
        let len2 = e.norm_squared();
        if len2 == 0.0 {
            return (p - self.a).norm();
        }
        let s = ((p - self.a).dot(&e) / len2).clamp(0.0, 1.0);
        (p - (self.a + e * s)).norm()
    }
}

/// A simple polygon (closed implicitly from the last vertex to the first).
#[derive(Clone, Debug, PartialEq)]
pub struct Polygon2 {
    /// Vertices in order (either winding).
    pub vertices: Vec<Point2<f64>>,
}

impl Polygon2 {
    /// Creates a polygon from `(x, y)` tuples.
    pub fn from_xy(vertices: &[(f64, f64)]) -> Self {
        Self {
            vertices: vertices.iter().map(|&(x, y)| Point2::new(x, y)).collect(),
        }
    }

    /// Axis-aligned rectangle `[x0, x1] x [y0, y1]`.
    pub fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        Self::from_xy(&[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    }

    /// The closed boundary as segments.
    pub fn edges(&self) -> Vec<Segment2> {
        let n = self.vertices.len();
        (0..n)
            .map(|i| Segment2 {
                a: self.vertices[i],
                b: self.vertices[(i + 1) % n],
            })
            .collect()
    }

    /// Even-odd point-in-polygon test (boundary points may go either way).
    pub fn contains(&self, p: &Point2<f64>) -> bool {
        let v = &self.vertices;
        let n = v.len();
        let mut inside = false;
        let mut j = n.wrapping_sub(1);
        for i in 0..n {
            let (vi, vj) = (v[i], v[j]);
            if (vi.y > p.y) != (vj.y > p.y) {
                let x_cross = vj.x + (p.y - vj.y) * (vi.x - vj.x) / (vi.y - vj.y);
                if p.x < x_cross {
                    inside = !inside;
                }
            }
            j = i;
        }
        inside
    }
}

/// Result of a 2D ray cast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit2 {
    /// Distance from the ray origin to the hit point, in metres.
    pub range: f64,
    /// Hit point in world coordinates.
    pub point: Point2<f64>,
    /// Unit surface normal, oriented against the ray direction.
    pub normal: Vector2<f64>,
    /// Index into [`World2d::segments`] of the wall that was hit.
    pub segment: usize,
}

/// A 2D world: wall segments plus a description of the walkable area.
///
/// The walkable area is the union of [`World2d::walkable`] polygons minus the
/// union of [`World2d::obstacles`] polygons. Wall segments are what sensors
/// see; obstacle polygon edges are added to the walls automatically by
/// [`World2d::add_obstacle`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct World2d {
    /// All wall segments (what the LiDAR sees).
    pub segments: Vec<Segment2>,
    /// Polygons whose union is the walkable region (before obstacles).
    pub walkable: Vec<Polygon2>,
    /// Solid obstacles subtracted from the walkable region.
    pub obstacles: Vec<Polygon2>,
}

impl World2d {
    /// Creates an empty world.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a single wall segment.
    pub fn add_segment(&mut self, segment: Segment2) -> &mut Self {
        self.segments.push(segment);
        self
    }

    /// Adds a polygon whose edges are walls and whose interior is walkable
    /// (e.g. the outer boundary of a room).
    pub fn add_enclosure(&mut self, polygon: Polygon2) -> &mut Self {
        self.segments.extend(polygon.edges());
        self.walkable.push(polygon);
        self
    }

    /// Adds a solid obstacle: its edges become walls and its interior is not
    /// walkable.
    pub fn add_obstacle(&mut self, polygon: Polygon2) -> &mut Self {
        self.segments.extend(polygon.edges());
        self.obstacles.push(polygon);
        self
    }

    /// Adds a walkable region without walls (e.g. to union rooms).
    pub fn add_walkable(&mut self, polygon: Polygon2) -> &mut Self {
        self.walkable.push(polygon);
        self
    }

    /// A 12 m x 8 m room spanning `[-6, 6] x [-4, 4]` with four asymmetric
    /// interior obstacles (two pillars and two boxes), so that scan matching
    /// is unambiguous.
    pub fn rect_room() -> Self {
        let mut w = Self::new();
        w.add_enclosure(Polygon2::rect(-6.0, -4.0, 6.0, 4.0));
        w.add_obstacle(Polygon2::rect(-3.3, 1.2, -2.7, 1.8)); // pillar
        w.add_obstacle(Polygon2::rect(1.9, -1.9, 3.1, -1.1)); // box
        w.add_obstacle(Polygon2::rect(3.8, 2.4, 4.2, 2.8)); // small pillar
        w.add_obstacle(Polygon2::rect(-2.5, -2.75, -1.5, -2.25)); // box
        w
    }

    /// A rectangular ring corridor: outer boundary `[-20, 20] x [-10, 10]`
    /// (40 m x 20 m), central island `[-17.5, 17.5] x [-7.5, 7.5]`, i.e. a
    /// 2.5 m wide corridor whose centre line is the rectangle
    /// `x = +-18.75, y = +-8.75`.
    ///
    /// Distinctive features make every stretch locally unique: one alcove of a
    /// different size on each outer wall, three 0.4 m bumps protruding from the
    /// island, and a small pillar inside the east alcove.
    pub fn loop_corridor() -> Self {
        let mut w = Self::new();
        w.add_enclosure(Polygon2::from_xy(&[
            (-20.0, -10.0),
            (-10.0, -10.0),
            (-10.0, -11.5),
            (-8.0, -11.5),
            (-8.0, -10.0),
            (20.0, -10.0),
            (20.0, -2.0),
            (22.0, -2.0),
            (22.0, 1.0),
            (20.0, 1.0),
            (20.0, 10.0),
            (8.0, 10.0),
            (8.0, 11.0),
            (5.0, 11.0),
            (5.0, 10.0),
            (-5.5, 10.0),
            (-5.5, 10.5),
            (-6.0, 10.5),
            (-6.0, 10.0),
            (-20.0, 10.0),
            (-20.0, 6.0),
            (-21.0, 6.0),
            (-21.0, 3.0),
            (-20.0, 3.0),
        ]));
        w.add_obstacle(Polygon2::from_xy(&[
            (-17.5, -7.5),
            (11.0, -7.5),
            (11.0, -7.9),
            (12.5, -7.9),
            (12.5, -7.5),
            (17.5, -7.5),
            (17.5, 3.0),
            (17.9, 3.0),
            (17.9, 4.0),
            (17.5, 4.0),
            (17.5, 7.5),
            (-4.0, 7.5),
            (-4.0, 7.9),
            (-5.0, 7.9),
            (-5.0, 7.5),
            (-17.5, 7.5),
        ]));
        w.add_obstacle(Polygon2::rect(21.2, -0.7, 21.5, -0.4)); // pillar in east alcove
        w
    }

    /// An office-like layout inside `[-10, 10] x [-7, 7]`: a 2 m wide hallway
    /// along `y in [-1, 1]`, three rooms to the north, two to the south,
    /// 1 m door gaps, and furniture blocks. Walls are zero-thickness lines.
    pub fn office() -> Self {
        let mut w = Self::new();
        w.add_enclosure(Polygon2::rect(-10.0, -7.0, 10.0, 7.0));
        // North hallway wall (y = 1) with doors at [-7.5,-6.5], [-1,0], [5,6].
        for &(x0, x1) in &[(-10.0, -7.5), (-6.5, -1.0), (0.0, 5.0), (6.0, 10.0)] {
            w.add_segment(Segment2::new(x0, 1.0, x1, 1.0));
        }
        // South hallway wall (y = -1) with doors at [-5,-4], [4,5].
        for &(x0, x1) in &[(-10.0, -5.0), (-4.0, 4.0), (5.0, 10.0)] {
            w.add_segment(Segment2::new(x0, -1.0, x1, -1.0));
        }
        // Room dividers.
        w.add_segment(Segment2::new(-4.0, 1.0, -4.0, 7.0));
        w.add_segment(Segment2::new(3.0, 1.0, 3.0, 7.0));
        w.add_segment(Segment2::new(0.0, -7.0, 0.0, -1.0));
        // Furniture.
        w.add_obstacle(Polygon2::rect(-9.0, 5.0, -7.0, 6.0));
        w.add_obstacle(Polygon2::rect(0.0, 6.3, 2.5, 6.9));
        w.add_obstacle(Polygon2::rect(5.0, 3.0, 7.0, 4.5));
        w.add_obstacle(Polygon2::rect(-9.5, -6.0, -9.0, -3.0));
        w.add_obstacle(Polygon2::rect(6.0, -6.0, 8.5, -5.0));
        w.add_obstacle(Polygon2::rect(-2.0, -4.5, -1.2, -3.7)); // pillar
        w
    }

    /// Axis-aligned bounding box `(min, max)` of all walls.
    pub fn bounds(&self) -> (Point2<f64>, Point2<f64>) {
        let mut min = Point2::new(f64::INFINITY, f64::INFINITY);
        let mut max = Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        for s in &self.segments {
            for p in [s.a, s.b] {
                min = min.inf(&p);
                max = max.sup(&p);
            }
        }
        (min, max)
    }

    /// Whether `p` lies in the walkable area (inside some walkable polygon and
    /// outside every obstacle).
    pub fn is_free(&self, p: &Point2<f64>) -> bool {
        self.walkable.iter().any(|poly| poly.contains(p))
            && !self.obstacles.iter().any(|poly| poly.contains(p))
    }

    /// Like [`World2d::is_free`] but additionally requires a clearance of at
    /// least `margin` metres to every wall.
    pub fn is_free_with_margin(&self, p: &Point2<f64>, margin: f64) -> bool {
        self.is_free(p) && self.distance_to_nearest_wall(p) >= margin
    }

    /// Distance from `p` to the closest wall segment.
    pub fn distance_to_nearest_wall(&self, p: &Point2<f64>) -> f64 {
        self.segments
            .iter()
            .map(|s| s.distance_to(p))
            .fold(f64::INFINITY, f64::min)
    }

    /// Casts a ray from `origin` along `direction` (need not be normalised)
    /// and returns the first wall hit with `range <= max_range`.
    pub fn raycast(
        &self,
        origin: &Point2<f64>,
        direction: &Vector2<f64>,
        max_range: f64,
    ) -> Option<Hit2> {
        let d = direction.normalize();
        let mut best: Option<Hit2> = None;
        for (i, s) in self.segments.iter().enumerate() {
            let e = s.b - s.a;
            let denom = cross(&d, &e);
            if denom.abs() < 1e-12 {
                continue; // parallel
            }
            let ao = s.a - origin;
            let t = cross(&ao, &e) / denom;
            let u = cross(&ao, &d) / denom;
            if !(0.0..=1.0).contains(&u) || t < 0.0 || t > max_range {
                continue;
            }
            if best.is_none_or(|b| t < b.range) {
                let mut n = Vector2::new(-e.y, e.x).normalize();
                if n.dot(&d) > 0.0 {
                    n = -n;
                }
                best = Some(Hit2 {
                    range: t,
                    point: origin + d * t,
                    normal: n,
                    segment: i,
                });
            }
        }
        best
    }

    /// Rasterises the world into a ground-truth grid covering the wall
    /// bounding box plus `padding` metres on every side.
    pub fn rasterize(&self, resolution: f64, padding: f64) -> GroundTruthGrid {
        let (min, max) = self.bounds();
        let origin = Point2::new(min.x - padding, min.y - padding);
        let width = ((max.x - min.x + 2.0 * padding) / resolution).ceil() as usize;
        let height = ((max.y - min.y + 2.0 * padding) / resolution).ceil() as usize;
        GroundTruthGrid::from_world(self, origin, resolution, width, height)
    }
}

impl RayCast for World2d {
    /// Walls are treated as infinitely tall vertical surfaces; only the
    /// horizontal component of the ray matters.
    fn raycast(
        &self,
        origin: &Point3<f64>,
        direction: &Unit<Vector3<f64>>,
        max_range: f64,
    ) -> Option<Hit3> {
        let dxy = Vector2::new(direction.x, direction.y);
        let horiz = dxy.norm();
        if horiz < 1e-9 {
            return None;
        }
        let hit = World2d::raycast(self, &origin.xy(), &dxy, max_range * horiz)?;
        let range = hit.range / horiz;
        Some(Hit3 {
            range,
            point: origin + direction.into_inner() * range,
            normal: Vector3::new(hit.normal.x, hit.normal.y, 0.0),
        })
    }
}

fn cross(a: &Vector2<f64>, b: &Vector2<f64>) -> f64 {
    a.x * b.y - a.y * b.x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_ranges_in_empty_room() {
        let mut w = World2d::new();
        w.add_enclosure(Polygon2::rect(-6.0, -4.0, 6.0, 4.0));
        let o = Point2::new(1.0, 0.5);
        let r = |dx: f64, dy: f64| w.raycast(&o, &Vector2::new(dx, dy), 100.0).unwrap();
        assert!((r(1.0, 0.0).range - 5.0).abs() < 1e-12);
        assert!((r(-1.0, 0.0).range - 7.0).abs() < 1e-12);
        assert!((r(0.0, 1.0).range - 3.5).abs() < 1e-12);
        assert!((r(0.0, -1.0).range - 4.5).abs() < 1e-12);
        // Diagonal (1, 1): reaches y = 4 after 3.5 m along y, at x = 4.5 < 6.
        let h = r(1.0, 1.0);
        assert!((h.range - 3.5 * 2f64.sqrt()).abs() < 1e-12);
        assert!((h.point - Point2::new(4.5, 4.0)).norm() < 1e-12);
        assert!((h.normal - Vector2::new(0.0, -1.0)).norm() < 1e-12);
        // Out of range.
        assert!(w.raycast(&o, &Vector2::new(1.0, 0.0), 4.99).is_none());
    }

    #[test]
    fn rect_room_obstacle_ranges() {
        let w = World2d::rect_room();
        // From (0, 1.5) towards -x: pillar east face at x = -2.7.
        let h = w
            .raycast(&Point2::new(0.0, 1.5), &Vector2::new(-1.0, 0.0), 30.0)
            .unwrap();
        assert!((h.range - 2.7).abs() < 1e-12);
        // From (2.5, 0) towards -y: box top face at y = -1.1.
        let h = w
            .raycast(&Point2::new(2.5, 0.0), &Vector2::new(0.0, -1.0), 30.0)
            .unwrap();
        assert!((h.range - 1.1).abs() < 1e-12);
    }

    #[test]
    fn free_space() {
        let w = World2d::loop_corridor();
        assert!(w.is_free(&Point2::new(0.0, 8.75)));
        assert!(w.is_free(&Point2::new(-18.75, 0.0)));
        assert!(w.is_free(&Point2::new(21.0, 0.5))); // east alcove
        assert!(!w.is_free(&Point2::new(0.0, 0.0))); // island
        assert!(!w.is_free(&Point2::new(30.0, 0.0))); // outside
        assert!(!w.is_free(&Point2::new(21.3, -0.5))); // pillar
        assert!(w.is_free_with_margin(&Point2::new(0.0, 8.75), 1.0));
        assert!(!w.is_free_with_margin(&Point2::new(0.0, 9.5), 1.0));

        let o = World2d::office();
        assert!(o.is_free(&Point2::new(0.0, 0.0)));
        assert!(!o.is_free(&Point2::new(6.0, 3.5)));
        let r = World2d::rect_room();
        assert!(r.is_free(&Point2::new(0.0, 0.0)));
        assert!(!r.is_free(&Point2::new(-3.0, 1.5)));
    }

    #[test]
    fn segment_distance() {
        let s = Segment2::new(0.0, 0.0, 2.0, 0.0);
        assert!((s.distance_to(&Point2::new(1.0, 1.0)) - 1.0).abs() < 1e-12);
        assert!((s.distance_to(&Point2::new(3.0, 0.0)) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn raycast_trait_projects_to_plane() {
        let w = World2d::rect_room();
        let d = Unit::new_normalize(Vector3::new(1.0, 0.0, 1.0));
        let h = RayCast::raycast(&w, &Point3::new(0.0, 0.0, 0.0), &d, 100.0).unwrap();
        assert!((h.range - 6.0 * 2f64.sqrt()).abs() < 1e-9);
    }
}
