//! Map export: 2D occupancy grids (PNG, ROS map_server PGM + YAML) and 3D
//! point maps (PLY), composed from the submaps at their optimized global
//! poses.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use nalgebra::{Isometry3, Vector2, Vector3};

use super::error::SlamError;
use super::types::to_isometry;
use crate::mapping::id::MapById;
use crate::mapping::id::SubmapId;
use crate::mapping::pose_graph_interface::SubmapData;
use crate::mapping::submaps::Submap;
use crate::mapping::three_d::submap_3d::Submap3D;
use crate::mapping::two_d::grid_2d::Grid2DTrait;
use crate::mapping::two_d::probability_grid::ProbabilityGrid;
use crate::mapping::two_d::submap_2d::Submap2D;
use crate::mapping::two_d::xy_index::Array2i;

/// A 2D occupancy grid in the map frame (ROS `nav_msgs/OccupancyGrid`
/// layout).
///
/// Cell `(x, y)` (column `x`, row `y`) is `cells[y * width + x]`; row 0 is
/// the **bottom** row (smallest map y). Its lower-left corner is at
/// `origin + (x, y) * resolution`.
#[derive(Clone, Debug, PartialEq)]
pub struct OccupancyGrid {
    /// Cell size (m).
    pub resolution: f64,
    /// Map-frame position (m) of the lower-left corner of cell `(0, 0)`.
    pub origin: Vector2<f64>,
    /// Number of columns.
    pub width: usize,
    /// Number of rows.
    pub height: usize,
    /// Occupancy probability per cell in `[0, 1]`, or
    /// [`OccupancyGrid::UNKNOWN`] for cells no submap observed.
    pub cells: Vec<f32>,
}

impl OccupancyGrid {
    /// Value of cells that were never observed.
    pub const UNKNOWN: f32 = -1.0;
    /// `occupied_thresh` written to ROS map YAML files (map_server default).
    pub const ROS_OCCUPIED_THRESHOLD: f64 = 0.65;
    /// `free_thresh` written to ROS map YAML files (map_server default).
    pub const ROS_FREE_THRESHOLD: f64 = 0.196;

    /// The occupancy probability of cell `(x, y)`, `None` if unknown or out
    /// of bounds.
    pub fn get(&self, x: usize, y: usize) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let value = self.cells[y * self.width + x];
        (value >= 0.0).then_some(value)
    }

    /// The cell containing the map-frame point `(px, py)`, if inside.
    pub fn cell_at(&self, px: f64, py: f64) -> Option<(usize, usize)> {
        let x = ((px - self.origin.x) / self.resolution).floor();
        let y = ((py - self.origin.y) / self.resolution).floor();
        (x >= 0.0 && y >= 0.0 && (x as usize) < self.width && (y as usize) < self.height)
            .then_some((x as usize, y as usize))
    }

    /// The occupancy probability at a map-frame point, `None` if unknown or
    /// outside.
    pub fn probability_at(&self, px: f64, py: f64) -> Option<f32> {
        let (x, y) = self.cell_at(px, py)?;
        self.get(x, y)
    }

    /// Cells as ROS `OccupancyGrid.data`: `-1` unknown, else `0..=100`.
    pub fn to_ros_occupancy(&self) -> Vec<i8> {
        self.cells
            .iter()
            .map(|&p| {
                if p < 0.0 {
                    -1
                } else {
                    (f64::from(p) * 100.0).round() as i8
                }
            })
            .collect()
    }

    /// Grayscale image rows from top (largest y) to bottom; occupied is
    /// dark: `round(255 * (1 - p))`, unknown is `unknown_value`.
    fn grayscale_rows_top_down(&self, unknown_value: u8) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(self.width * self.height);
        for y in (0..self.height).rev() {
            for x in 0..self.width {
                let p = self.cells[y * self.width + x];
                pixels.push(if p < 0.0 {
                    unknown_value
                } else {
                    (255.0 * (1.0 - f64::from(p).clamp(0.0, 1.0))).round() as u8
                });
            }
        }
        pixels
    }

    /// Writes an 8-bit grayscale PNG (north up: the top row is the largest
    /// map y). Occupied cells are dark, free cells light, unknown cells
    /// mid-gray (128).
    pub fn to_png(&self, path: impl AsRef<Path>) -> Result<(), SlamError> {
        let path = path.as_ref();
        let file = std::fs::File::create(path).map_err(|e| SlamError::io(path, e))?;
        let mut encoder = png::Encoder::new(
            std::io::BufWriter::new(file),
            self.width as u32,
            self.height as u32,
        );
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        let to_io = |e: png::EncodingError| SlamError::io(path, std::io::Error::other(e));
        let mut writer = encoder.write_header().map_err(to_io)?;
        writer
            .write_image_data(&self.grayscale_rows_top_down(128))
            .map_err(to_io)?;
        writer.finish().map_err(to_io)
    }

    /// Writes a ROS `map_server` map: `<base>.pgm` (binary P5, grayscale
    /// like `map_saver`/cartographer_ros: `255 * (1 - p)`, unknown = 205)
    /// and `<base>.yaml` (image, resolution, origin, negate 0,
    /// occupied_thresh 0.65, free_thresh 0.196). A `.pgm`/`.yaml`
    /// extension on `base` is replaced. Returns the YAML path.
    pub fn to_ros_map(&self, base: impl AsRef<Path>) -> Result<PathBuf, SlamError> {
        let base = base.as_ref();
        let base = match base.extension().and_then(|e| e.to_str()) {
            Some("pgm" | "yaml" | "yml") => base.with_extension(""),
            _ => base.to_path_buf(),
        };
        let pgm_path = base.with_extension("pgm");
        let yaml_path = base.with_extension("yaml");

        let mut pgm = format!(
            "P5\n# CREATOR: cartographer-rust {:.3} m/pix\n{} {}\n255\n",
            self.resolution, self.width, self.height
        )
        .into_bytes();
        pgm.extend(self.grayscale_rows_top_down(205));
        std::fs::write(&pgm_path, pgm).map_err(|e| SlamError::io(&pgm_path, e))?;

        let image_name = pgm_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let yaml = format!(
            "image: {image_name}\nresolution: {}\norigin: [{}, {}, 0.0]\nnegate: 0\n\
             occupied_thresh: {}\nfree_thresh: {}\n",
            self.resolution,
            self.origin.x,
            self.origin.y,
            Self::ROS_OCCUPIED_THRESHOLD,
            Self::ROS_FREE_THRESHOLD
        );
        std::fs::write(&yaml_path, yaml).map_err(|e| SlamError::io(&yaml_path, e))?;
        Ok(yaml_path)
    }
}

/// One occupied voxel of a [`PointMap`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MapPoint {
    /// Voxel center in the map frame (m).
    pub position: Vector3<f32>,
    /// Occupancy probability.
    pub probability: f32,
}

/// Occupied voxels of the high-resolution 3D submaps in the map frame,
/// de-duplicated on a global voxel grid of `resolution`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PointMap {
    /// Voxel size (m).
    pub resolution: f64,
    /// The occupied voxels.
    pub points: Vec<MapPoint>,
}

impl PointMap {
    /// Writes a binary little-endian PLY with `float x, y, z, probability`
    /// per vertex (readable by CloudCompare, MeshLab, Open3D, PCL).
    pub fn to_ply(&self, path: impl AsRef<Path>) -> Result<(), SlamError> {
        let path = path.as_ref();
        let file = std::fs::File::create(path).map_err(|e| SlamError::io(path, e))?;
        let mut out = std::io::BufWriter::new(file);
        let write = |out: &mut std::io::BufWriter<std::fs::File>| -> std::io::Result<()> {
            write!(
                out,
                "ply\nformat binary_little_endian 1.0\ncomment generated by cartographer-rust\n\
                 element vertex {}\nproperty float x\nproperty float y\nproperty float z\n\
                 property float probability\nend_header\n",
                self.points.len()
            )?;
            for point in &self.points {
                for value in [
                    point.position.x,
                    point.position.y,
                    point.position.z,
                    point.probability,
                ] {
                    out.write_all(&value.to_le_bytes())?;
                }
            }
            out.flush()
        };
        write(&mut out).map_err(|e| SlamError::io(path, e))
    }
}

fn logit(p: f32) -> f64 {
    let p = f64::from(p).clamp(1e-3, 1.0 - 1e-3);
    (p / (1.0 - p)).ln()
}

/// Composes the 2D submaps (probability grids) at their global poses.
///
/// For every output cell, the log-odds of all submaps that observed it
/// (nearest-cell lookup in the submap's grid) are summed; the result is
/// converted back to a probability.
pub(crate) fn compose_occupancy_grid(
    submaps: &MapById<SubmapId, SubmapData>,
) -> Result<OccupancyGrid, SlamError> {
    struct Placed<'a> {
        grid: &'a ProbabilityGrid,
        // map_from_local in 2D: rotation (cos, sin) and translation.
        cos: f64,
        sin: f64,
        translation: Vector2<f64>,
        offset: Array2i,
        num_x: i32,
        num_y: i32,
    }
    let guards: Vec<_> = submaps
        .iter()
        .filter_map(|submap| {
            let handle = submap.data.submap.as_ref()?;
            Some((
                submap.data.pose,
                handle
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            ))
        })
        .collect();
    let mut placed = Vec::new();
    let mut resolution: Option<f64> = None;
    for (global_pose, guard) in &guards {
        let any: &dyn std::any::Any = &**guard;
        let Some(submap) = any.downcast_ref::<Submap2D>() else {
            return Err(SlamError::Unsupported(
                "2D occupancy grids need 2D submaps".to_string(),
            ));
        };
        let Some(grid) = submap.try_grid() else {
            continue;
        };
        let Some(grid) = grid.downcast_ref::<ProbabilityGrid>() else {
            return Err(SlamError::Unsupported(
                "only probability grids (not TSDF) can be exported".to_string(),
            ));
        };
        let (offset, limits) = grid.compute_cropped_limits();
        if limits.num_x_cells <= 0 || limits.num_y_cells <= 0 {
            continue;
        }
        resolution.get_or_insert(grid.limits().resolution());
        // Grid cells are in the local SLAM frame; the pose graph gives the
        // global pose of the submap frame, so map_from_local =
        // global_submap_pose * local_submap_pose^-1.
        let map_from_local = to_isometry(global_pose) * to_isometry(&submap.local_pose()).inverse();
        let yaw = map_from_local.rotation.euler_angles().2;
        placed.push(Placed {
            grid,
            cos: yaw.cos(),
            sin: yaw.sin(),
            translation: map_from_local.translation.vector.xy(),
            offset,
            num_x: limits.num_x_cells,
            num_y: limits.num_y_cells,
        });
    }
    let Some(resolution) = resolution else {
        return Ok(OccupancyGrid {
            resolution: 0.0,
            origin: Vector2::zeros(),
            width: 0,
            height: 0,
            cells: Vec::new(),
        });
    };

    // Bounds of all known cells in the map frame.
    let mut min = Vector2::repeat(f64::INFINITY);
    let mut max = Vector2::repeat(f64::NEG_INFINITY);
    let to_map = |p: &Placed, local: Vector2<f64>| {
        Vector2::new(
            p.cos * local.x - p.sin * local.y,
            p.sin * local.x + p.cos * local.y,
        ) + p.translation
    };
    for p in &placed {
        let limits = p.grid.limits();
        let half = resolution / 2.0;
        for (ix, iy) in [
            (p.offset.x, p.offset.y),
            (p.offset.x + p.num_x - 1, p.offset.y),
            (p.offset.x, p.offset.y + p.num_y - 1),
            (p.offset.x + p.num_x - 1, p.offset.y + p.num_y - 1),
        ] {
            let center = limits.get_cell_center(&Array2i::new(ix, iy)).cast::<f64>();
            for dx in [-half, half] {
                for dy in [-half, half] {
                    let corner = to_map(p, center + Vector2::new(dx, dy));
                    min = min.inf(&corner);
                    max = max.sup(&corner);
                }
            }
        }
    }
    let width = ((max.x - min.x) / resolution).ceil().max(1.0) as usize;
    let height = ((max.y - min.y) / resolution).ceil().max(1.0) as usize;
    if width.saturating_mul(height) > 1 << 30 {
        return Err(SlamError::Unsupported(format!(
            "the map would have {width} x {height} cells"
        )));
    }
    let mut log_odds = vec![0.0_f64; width * height];
    let mut observed = vec![false; width * height];
    for p in &placed {
        let limits = p.grid.limits();
        // Output cells covered by this submap.
        let mut sub_min = Vector2::repeat(f64::INFINITY);
        let mut sub_max = Vector2::repeat(f64::NEG_INFINITY);
        for (ix, iy) in [
            (p.offset.x, p.offset.y),
            (p.offset.x + p.num_x - 1, p.offset.y),
            (p.offset.x, p.offset.y + p.num_y - 1),
            (p.offset.x + p.num_x - 1, p.offset.y + p.num_y - 1),
        ] {
            let center = to_map(
                p,
                limits.get_cell_center(&Array2i::new(ix, iy)).cast::<f64>(),
            );
            sub_min = sub_min.inf(&center);
            sub_max = sub_max.sup(&center);
        }
        let x0 = (((sub_min.x - min.x) / resolution).floor() as i64 - 1).max(0) as usize;
        let y0 = (((sub_min.y - min.y) / resolution).floor() as i64 - 1).max(0) as usize;
        let x1 =
            ((((sub_max.x - min.x) / resolution).ceil() as i64 + 1).max(0) as usize).min(width);
        let y1 =
            ((((sub_max.y - min.y) / resolution).ceil() as i64 + 1).max(0) as usize).min(height);
        for y in y0..y1 {
            for x in x0..x1 {
                let map_point = Vector2::new(
                    min.x + (x as f64 + 0.5) * resolution,
                    min.y + (y as f64 + 0.5) * resolution,
                );
                // local = R^T (map - t).
                let d = map_point - p.translation;
                let local = Vector2::new(p.cos * d.x + p.sin * d.y, -p.sin * d.x + p.cos * d.y);
                let index = limits.get_cell_index(&local.cast::<f32>());
                if index.x < p.offset.x
                    || index.y < p.offset.y
                    || index.x >= p.offset.x + p.num_x
                    || index.y >= p.offset.y + p.num_y
                    || !p.grid.is_known(&index)
                {
                    continue;
                }
                let cell = y * width + x;
                log_odds[cell] += logit(p.grid.get_probability(&index));
                observed[cell] = true;
            }
        }
    }
    let cells = log_odds
        .iter()
        .zip(&observed)
        .map(|(&l, &seen)| {
            if seen {
                (1.0 / (1.0 + (-l).exp())) as f32
            } else {
                OccupancyGrid::UNKNOWN
            }
        })
        .collect();
    Ok(OccupancyGrid {
        resolution,
        origin: min,
        width,
        height,
        cells,
    })
}

/// Collects the occupied voxels (`probability >= min_probability`) of the
/// high-resolution grids of 3D submaps, transformed by the optimized global
/// submap poses and de-duplicated on a global voxel grid (keeping the
/// highest probability).
pub(crate) fn compose_point_map(
    submaps: &MapById<SubmapId, SubmapData>,
    min_probability: f32,
) -> Result<PointMap, SlamError> {
    let mut voxels: HashMap<[i64; 3], f32> = HashMap::new();
    let mut resolution: Option<f64> = None;
    for submap in submaps.iter() {
        let Some(handle) = submap.data.submap.as_ref() else {
            continue;
        };
        let guard = handle
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let any: &dyn std::any::Any = &*guard;
        let Some(submap_3d) = any.downcast_ref::<Submap3D>() else {
            return Err(SlamError::Unsupported(
                "3D point maps need 3D submaps".to_string(),
            ));
        };
        let grid = submap_3d.high_resolution_hybrid_grid();
        let voxel = f64::from(grid.resolution());
        let resolution = *resolution.get_or_insert(voxel);
        // The 3D grids are in the submap frame.
        let map_from_submap: Isometry3<f64> = to_isometry(&submap.data.pose);
        for (index, _) in grid {
            let probability = grid.get_probability(&index);
            if probability < min_probability {
                continue;
            }
            let center = grid.get_center_of_cell(&index).cast::<f64>();
            let in_map = map_from_submap * nalgebra::Point3::from(center);
            let key = [
                (in_map.x / resolution).round() as i64,
                (in_map.y / resolution).round() as i64,
                (in_map.z / resolution).round() as i64,
            ];
            let entry = voxels.entry(key).or_insert(probability);
            *entry = entry.max(probability);
        }
    }
    let resolution = resolution.unwrap_or(0.0);
    let mut points: Vec<MapPoint> = voxels
        .into_iter()
        .map(|(key, probability)| MapPoint {
            position: Vector3::new(
                (key[0] as f64 * resolution) as f32,
                (key[1] as f64 * resolution) as f32,
                (key[2] as f64 * resolution) as f32,
            ),
            probability,
        })
        .collect();
    // Deterministic output order.
    points.sort_by(|a, b| {
        (a.position.z, a.position.y, a.position.x)
            .partial_cmp(&(b.position.z, b.position.y, b.position.x))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(PointMap { resolution, points })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> OccupancyGrid {
        // 3 x 2: bottom row free/unknown/occupied, top row all free.
        OccupancyGrid {
            resolution: 0.5,
            origin: Vector2::new(-1.0, 2.0),
            width: 3,
            height: 2,
            cells: vec![0.1, OccupancyGrid::UNKNOWN, 0.9, 0.2, 0.2, 0.2],
        }
    }

    #[test]
    fn accessors() {
        let g = grid();
        assert_eq!(g.get(0, 0), Some(0.1));
        assert_eq!(g.get(1, 0), None);
        assert_eq!(g.get(3, 0), None);
        assert_eq!(g.cell_at(-0.9, 2.1), Some((0, 0)));
        assert_eq!(g.cell_at(0.4, 2.9), Some((2, 1)));
        assert_eq!(g.cell_at(0.6, 2.1), None);
        assert_eq!(g.probability_at(0.2, 2.2), Some(0.9));
        assert_eq!(g.to_ros_occupancy(), vec![10, -1, 90, 20, 20, 20]);
    }

    #[test]
    fn png_is_north_up() {
        let g = grid();
        let path = std::env::temp_dir().join(format!("slam_map_test_{}.png", std::process::id()));
        g.to_png(&path).unwrap();
        let decoder =
            png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let mut buffer = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buffer).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!((info.width, info.height), (3, 2));
        assert_eq!(info.color_type, png::ColorType::Grayscale);
        // Top row = cells row 1 (all 0.2), bottom row = row 0.
        assert_eq!(&buffer[..6], &[204, 204, 204, 229, 128, 26]);
    }

    #[test]
    fn ros_map_files() {
        let g = grid();
        let base = std::env::temp_dir().join(format!("slam_map_test_{}", std::process::id()));
        let yaml_path = g.to_ros_map(base.with_extension("yaml")).unwrap();
        assert_eq!(yaml_path, base.with_extension("yaml"));
        let pgm = std::fs::read(base.with_extension("pgm")).unwrap();
        let yaml = std::fs::read_to_string(&yaml_path).unwrap();
        std::fs::remove_file(base.with_extension("pgm")).unwrap();
        std::fs::remove_file(&yaml_path).unwrap();
        let header_end = pgm.windows(4).position(|w| w == b"255\n").unwrap() + 4;
        assert!(pgm.starts_with(b"P5\n"));
        assert!(String::from_utf8_lossy(&pgm[..header_end]).contains("\n3 2\n"));
        assert_eq!(&pgm[header_end..], &[204, 204, 204, 229, 205, 26]);
        let name = base.with_extension("pgm");
        let name = name.file_name().unwrap().to_string_lossy();
        assert!(yaml.contains(&format!("image: {name}\n")), "{yaml}");
        assert!(yaml.contains("resolution: 0.5\n"));
        assert!(yaml.contains("origin: [-1, 2, 0.0]\n"));
        assert!(yaml.contains("occupied_thresh: 0.65\n"));
        assert!(yaml.contains("free_thresh: 0.196\n"));
        assert!(yaml.contains("negate: 0\n"));
    }

    #[test]
    fn ply_header_and_payload() {
        let map = PointMap {
            resolution: 0.1,
            points: vec![
                MapPoint {
                    position: Vector3::new(1.0, 2.0, 3.0),
                    probability: 0.75,
                },
                MapPoint {
                    position: Vector3::new(-1.0, 0.5, 0.0),
                    probability: 0.6,
                },
            ],
        };
        let path = std::env::temp_dir().join(format!("slam_map_test_{}.ply", std::process::id()));
        map.to_ply(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let header_end = bytes
            .windows(11)
            .position(|w| w == b"end_header\n")
            .unwrap()
            + 11;
        let header = String::from_utf8_lossy(&bytes[..header_end]);
        assert!(header.contains("element vertex 2\n"));
        assert!(header.contains("format binary_little_endian 1.0\n"));
        let payload = &bytes[header_end..];
        assert_eq!(payload.len(), 2 * 4 * 4);
        let first: Vec<f32> = payload[..16]
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(first, [1.0, 2.0, 3.0, 0.75]);
    }

    #[test]
    fn io_errors_carry_the_path() {
        let g = grid();
        match g.to_png("/nonexistent_dir/x/map.png") {
            Err(SlamError::Io { path, .. }) => assert!(path.ends_with("map.png")),
            other => panic!("{other:?}"),
        }
    }
}
