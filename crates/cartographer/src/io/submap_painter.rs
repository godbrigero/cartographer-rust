//! Port of `cartographer/io/submap_painter` (.h, .cc).
//!
//! RUST PORT NOTE: Cairo surfaces are the pure-Rust [`Surface`] of
//! [`crate::io::cairo`]; a `SubmapSlice` owns its surface (C++ keeps the
//! pixel data in a separate `cairo_data` vector that the surface aliases).

use std::collections::BTreeMap;
use std::sync::Arc;

use nalgebra::Vector2;

use crate::common::port::fast_gunzip_string;
use crate::io::cairo::{Context, Matrix, Surface, format_stride_for_width};
use crate::io::proto_stream_deserializer::ProtoStreamDeserializer;
use crate::mapping::id::SubmapId;
use crate::mapping::proto;
use crate::mapping::submaps::Submap;
use crate::mapping::three_d::submap_3d::Submap3D;
use crate::mapping::two_d::submap_2d::Submap2D;
use crate::mapping::value_conversion_tables::ValueConversionTables;
use crate::transform::rigid_transform::{Rigid3d, quaternion_to_rotation_matrix};
use crate::transform::transform::to_rigid3;

/// Result of [`paint_submap_slices`].
#[derive(Clone, Debug, PartialEq)]
pub struct PaintSubmapSlicesResult {
    /// The painted image.
    pub surface: Surface,
    /// Top left pixel of `surface` in map frame.
    pub origin: Vector2<f32>,
}

/// A submap texture placed in the map.
#[derive(Clone, Debug, PartialEq)]
pub struct SubmapSlice {
    // Texture data.
    /// Texture width in cells.
    pub width: i32,
    /// Texture height in cells.
    pub height: i32,
    /// Submap version.
    pub version: i32,
    /// Texture resolution in meters per cell.
    pub resolution: f64,
    /// Pose of the texture relative to the submap.
    pub slice_pose: Rigid3d,
    /// The texture (C++ `nullptr` -> `None`).
    pub surface: Option<Surface>,
    // Metadata.
    /// Global pose of the submap.
    pub pose: Rigid3d,
    /// Metadata version (-1 if unknown).
    pub metadata_version: i32,
}

impl Default for SubmapSlice {
    fn default() -> Self {
        // RUST PORT NOTE: `width`, `height`, `version` and `resolution` are
        // uninitialized in C++; they are zero here.
        Self {
            width: 0,
            height: 0,
            version: 0,
            resolution: 0.,
            slice_pose: Rigid3d::identity(),
            surface: None,
            pose: Rigid3d::identity(),
            metadata_version: -1,
        }
    }
}

/// Unpacked texture cells (C++ `SubmapTexture::Pixels`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SubmapTexturePixels {
    /// Intensity per cell, row-major.
    pub intensity: Vec<u8>,
    /// Alpha per cell, row-major.
    pub alpha: Vec<u8>,
}

/// An unpacked submap texture.
#[derive(Clone, Debug, PartialEq)]
pub struct SubmapTexture {
    /// The cells.
    pub pixels: SubmapTexturePixels,
    /// Width in cells.
    pub width: i32,
    /// Height in cells.
    pub height: i32,
    /// Resolution in meters per cell.
    pub resolution: f64,
    /// Pose of the texture relative to the submap.
    pub slice_pose: Rigid3d,
}

/// All textures of a submap.
#[derive(Clone, Debug, PartialEq)]
pub struct SubmapTextures {
    /// Submap version.
    pub version: i32,
    /// The textures.
    pub textures: Vec<SubmapTexture>,
}

/// Axis aligned float box (C++ `Eigen::AlignedBox2f`), initially empty.
#[derive(Clone, Copy, Debug)]
struct AlignedBox2f {
    min: Vector2<f32>,
    max: Vector2<f32>,
}

impl AlignedBox2f {
    fn empty() -> Self {
        Self {
            min: Vector2::repeat(f32::MAX),
            max: Vector2::repeat(f32::MIN),
        }
    }

    fn extend(&mut self, p: &Vector2<f32>) {
        self.min = self.min.inf(p);
        self.max = self.max.sup(p);
    }

    fn sizes(&self) -> Vector2<f32> {
        self.max - self.min
    }
}

fn cairo_paint_submap_slices<'s, 'p>(
    scale: f64,
    submaps: &'p BTreeMap<SubmapId, SubmapSlice>,
    cr: &mut Context<'s, 'p>,
    mut draw_callback: impl FnMut(&mut Context<'s, 'p>, &'p SubmapSlice),
) {
    cr.scale(scale, scale);

    for submap_slice in submaps.values() {
        // Note: like the C++ code, this stops at the first slice without a
        // surface and skips all following slices.
        if submap_slice.surface.is_none() {
            return;
        }
        // `Eigen::Translation3d(t) * rotation` as a 4x4 matrix.
        let pose = submap_slice.pose * submap_slice.slice_pose;
        let rotation = quaternion_to_rotation_matrix(pose.rotation());
        let translation = pose.translation();
        let homo = |row: usize, col: usize| -> f64 {
            if col == 3 {
                translation[row]
            } else {
                rotation[(row, col)]
            }
        };
        cr.save();
        let matrix = Matrix::new(
            homo(1, 0),
            homo(0, 0),
            -homo(1, 1),
            -homo(0, 1),
            homo(0, 3),
            -homo(1, 3),
        );
        cr.transform(&matrix);

        let submap_resolution = submap_slice.resolution;
        cr.scale(submap_resolution, submap_resolution);

        // Invokes caller's callback to utilize slice data in global
        // cooridnate frame. e.g. finds bounding box, paints slices.
        draw_callback(cr, submap_slice);
        cr.restore();
    }
}

fn has_2d_grid(submap: &proto::Submap) -> bool {
    submap.submap_2d.as_ref().is_some_and(|s| s.grid.is_some())
}

fn has_3d_grids(submap: &proto::Submap) -> bool {
    submap.submap_3d.as_ref().is_some_and(|s| {
        s.low_resolution_hybrid_grid.is_some() && s.high_resolution_hybrid_grid.is_some()
    })
}

/// Paints all `submaps` (in `SubmapId` order) into one image with
/// `resolution` meters per pixel, on a dark red (0.5, 0, 0) background with 5
/// pixels of padding.
///
/// RUST PORT NOTE: if no slice is painted (no slices, or the first slice has
/// no surface) the bounding box is empty; C++ then converts infinite sizes
/// to int (undefined behavior) and Cairo returns an error surface. We return
/// a 0x0 surface instead.
pub fn paint_submap_slices(
    submaps: &BTreeMap<SubmapId, SubmapSlice>,
    resolution: f64,
) -> PaintSubmapSlicesResult {
    let mut bounding_box = AlignedBox2f::empty();
    {
        let mut surface = Surface::new(1, 1);
        let mut cr = Context::new(&mut surface);
        let mut update_bounding_box = |cr: &Context<'_, '_>, x: f64, y: f64| {
            let (x, y) = cr.user_to_device(x, y);
            bounding_box.extend(&Vector2::new(x as f32, y as f32));
        };

        cairo_paint_submap_slices(1. / resolution, submaps, &mut cr, |cr, submap_slice| {
            let (w, h) = (
                f64::from(submap_slice.width),
                f64::from(submap_slice.height),
            );
            update_bounding_box(cr, 0., 0.);
            update_bounding_box(cr, w, 0.);
            update_bounding_box(cr, 0., h);
            update_bounding_box(cr, w, h);
        });
    }

    const PADDING_PIXEL: i32 = 5;
    let sizes = bounding_box.sizes();
    // C++: `std::ceil(float) + int`, converted to int.
    let size = Vector2::new(
        (sizes.x.ceil() + (2 * PADDING_PIXEL) as f32) as i32,
        (sizes.y.ceil() + (2 * PADDING_PIXEL) as f32) as i32,
    );
    let origin = Vector2::new(
        -bounding_box.min.x + PADDING_PIXEL as f32,
        -bounding_box.min.y + PADDING_PIXEL as f32,
    );

    let mut surface = Surface::new(size.x.max(0), size.y.max(0));
    {
        let mut cr = Context::new(&mut surface);
        cr.set_source_rgba(0.5, 0.0, 0.0, 1.);
        cr.paint();
        cr.translate(f64::from(origin.x), f64::from(origin.y));
        cairo_paint_submap_slices(1. / resolution, submaps, &mut cr, |cr, submap_slice| {
            let slice_surface = submap_slice
                .surface
                .as_ref()
                .expect("checked by cairo_paint_submap_slices");
            cr.set_source_surface(slice_surface, 0., 0.);
            cr.paint();
        });
    }
    PaintSubmapSlicesResult { surface, origin }
}

/// Fills `submap_slice` with the texture of the submap `proto` at
/// `global_submap_pose` (for 3D submaps the high resolution slice).
pub fn fill_submap_slice(
    global_submap_pose: &Rigid3d,
    proto: &proto::Submap,
    submap_slice: &mut SubmapSlice,
    conversion_tables: &Arc<ValueConversionTables>,
) {
    let mut response = proto::submap_query::Response::default();
    // The C++ code also computes the (unused) local pose of the submap.
    if let Some(submap_3d) = &proto.submap_3d {
        let submap = Submap3D::from_proto(submap_3d);
        submap.to_response_proto(global_submap_pose, &mut response);
    } else {
        let submap = Submap2D::from_proto(
            &proto.submap_2d.clone().unwrap_or_default(),
            conversion_tables,
        );
        submap.to_response_proto(global_submap_pose, &mut response);
    }
    submap_slice.pose = *global_submap_pose;

    let texture_proto = &response.textures[0];
    let pixels = unpack_texture_data(
        &texture_proto.cells,
        texture_proto.width,
        texture_proto.height,
    );
    submap_slice.width = texture_proto.width;
    submap_slice.height = texture_proto.height;
    submap_slice.resolution = texture_proto.resolution;
    submap_slice.slice_pose = to_rigid3(&texture_proto.slice_pose.unwrap_or_default());
    submap_slice.surface = Some(draw_texture(
        &pixels.intensity,
        &pixels.alpha,
        texture_proto.width,
        texture_proto.height,
    ));
}

/// Reads all remaining `SerializedData` of `deserializer` and fills a slice
/// for every submap with grid data, placed at its pose from the pose graph.
pub fn deserialize_and_fill_submap_slices(
    deserializer: &mut ProtoStreamDeserializer<'_>,
    submap_slices: &mut BTreeMap<SubmapId, SubmapSlice>,
    conversion_tables: &Arc<ValueConversionTables>,
) {
    let mut submap_poses: BTreeMap<SubmapId, Rigid3d> = BTreeMap::new();
    for trajectory in &deserializer.pose_graph().trajectory {
        for submap in &trajectory.submap {
            submap_poses.insert(
                SubmapId::new(trajectory.trajectory_id, submap.submap_index),
                to_rigid3(&submap.pose.unwrap_or_default()),
            );
        }
    }
    let mut proto = proto::SerializedData::default();
    while deserializer.read_next_serialized_data(&mut proto) {
        if let Some(proto::serialized_data::Data::Submap(submap)) = &proto.data
            && (has_2d_grid(submap) || has_3d_grids(submap))
        {
            let submap_id_proto = submap.submap_id.unwrap_or_default();
            let id = SubmapId::new(submap_id_proto.trajectory_id, submap_id_proto.submap_index);
            let pose = submap_poses
                .get(&id)
                .unwrap_or_else(|| panic!("map::at: no pose for submap {id}"));
            fill_submap_slice(
                pose,
                submap,
                submap_slices.entry(id).or_default(),
                conversion_tables,
            );
        }
    }
}

/// Unpacks cell data as provided by the backend into `intensity` and
/// `alpha`.
pub fn unpack_texture_data(
    compressed_cells: &[u8],
    width: i32,
    height: i32,
) -> SubmapTexturePixels {
    let mut pixels = SubmapTexturePixels::default();
    let mut cells = Vec::new();
    fast_gunzip_string(compressed_cells, &mut cells);
    let num_pixels = (width * height) as usize;
    assert_eq!(
        cells.len(),
        2 * num_pixels,
        "Check failed: cells.size() == 2 * num_pixels"
    );
    pixels.intensity.reserve(num_pixels);
    pixels.alpha.reserve(num_pixels);
    for i in 0..height as usize {
        for j in 0..width as usize {
            pixels.intensity.push(cells[(i * width as usize + j) * 2]);
            pixels.alpha.push(cells[(i * width as usize + j) * 2 + 1]);
        }
    }
    pixels
}

/// Draws a texture into a Cairo surface: per cell
/// `(alpha << 24) | (intensity << 16) | (observed << 8)`, i.e. the red
/// channel tracks intensity and the green channel whether the cell was ever
/// observed (`observed = 0` iff intensity and alpha are both 0).
///
/// Note: these pixels are generally not valid premultiplied colors (red may
/// exceed alpha); like Cairo they are stored and composited verbatim.
///
/// RUST PORT NOTE: the C++ out-parameter `cairo_data` (the pixel storage the
/// surface aliases) is replaced by the returned surface owning its pixels.
pub fn draw_texture(intensity: &[u8], alpha: &[u8], width: i32, height: i32) -> Surface {
    // Properly dealing with a non-common stride would make this code much
    // more complicated. Let's check that it is not needed.
    let expected_stride = 4 * width;
    assert_eq!(
        expected_stride,
        format_stride_for_width(width),
        "Check failed: expected_stride == cairo_format_stride_for_width(kCairoFormat, width)"
    );
    assert_eq!(
        intensity.len(),
        alpha.len(),
        "intensity and alpha must have the same length"
    );
    let cairo_data: Vec<u32> = intensity
        .iter()
        .zip(alpha)
        .map(|(&intensity_value, &alpha_value)| {
            // We use the red channel to track intensity information. The
            // green channel we use to track if a cell was ever observed.
            let observed: u32 = if intensity_value == 0 && alpha_value == 0 {
                0
            } else {
                255
            };
            (u32::from(alpha_value) << 24) | (u32::from(intensity_value) << 16) | (observed << 8)
        })
        .collect();
    Surface::from_data(cairo_data, width, height)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use nalgebra::Vector3;

    use super::*;
    use crate::common::port::fast_gzip_string;
    use crate::io::internal::testing::test_helpers::proto_reader_from_protos;
    use crate::io::probability_grid_points_processor::create_probability_grid;
    use crate::mapping::two_d::grid_2d::Grid2DTrait;
    use crate::mapping::two_d::probability_grid_range_data_inserter_2d::ProbabilityGridRangeDataInserter2D;
    use crate::sensor::point_cloud::PointCloud;
    use crate::sensor::range_data::RangeData;
    use crate::sensor::rangefinder_point::RangefinderPoint;
    use crate::transform::rigid_transform::AngleAxis;
    use crate::transform::transform::to_proto;

    fn texture_slice(intensity: &[u8], alpha: &[u8], width: i32, height: i32) -> SubmapSlice {
        SubmapSlice {
            width,
            height,
            resolution: 0.05,
            surface: Some(draw_texture(intensity, alpha, width, height)),
            ..Default::default()
        }
    }

    #[test]
    fn unpack_texture_data_deinterleaves() {
        let cells: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut compressed = Vec::new();
        fast_gzip_string(&cells, &mut compressed);
        let pixels = unpack_texture_data(&compressed, 3, 2);
        assert_eq!(pixels.intensity, vec![1, 3, 5, 7, 9, 11]);
        assert_eq!(pixels.alpha, vec![2, 4, 6, 8, 10, 12]);
    }

    #[test]
    #[should_panic(expected = "2 * num_pixels")]
    fn unpack_texture_data_checks_size() {
        let mut compressed = Vec::new();
        fast_gzip_string(&[1, 2, 3], &mut compressed);
        unpack_texture_data(&compressed, 1, 1);
    }

    #[test]
    fn draw_texture_encodes_intensity_and_observed() {
        let surface = draw_texture(&[0, 200, 0], &[0, 50, 7], 3, 1);
        assert_eq!(surface.data(), &[0x0000_0000, 0x32c8_ff00, 0x0700_ff00]);
    }

    #[test]
    fn paint_single_slice_with_identity_pose() {
        // A 3x2 texture painted at its own resolution: texel (u, v) lands on
        // device pixel (h + 4 - v, u + 5), i.e. map -y is image x.
        let intensity: Vec<u8> = (1..=6).collect();
        let alpha = vec![255u8; 6];
        let mut submaps = BTreeMap::new();
        submaps.insert(SubmapId::new(0, 0), texture_slice(&intensity, &alpha, 3, 2));
        let result = paint_submap_slices(&submaps, 0.05);
        let (w, h) = (3, 2);
        assert_eq!(result.surface.width(), h + 10);
        assert_eq!(result.surface.height(), w + 10);
        assert_eq!(result.origin, Vector2::new((h + 5) as f32, 5.));
        for v in 0..h {
            for u in 0..w {
                let expected = 0xff00_ff00 | (u32::from(intensity[(v * w + u) as usize]) << 16);
                assert_eq!(
                    result.surface.pixel(h + 4 - v, u + 5),
                    expected,
                    "({u}, {v})"
                );
            }
        }
        // Background.
        assert_eq!(result.surface.pixel(0, 0), 0xff80_0000);
        assert_eq!(result.surface.pixel(h + 5, 5), 0xff80_0000);
        let painted = result
            .surface
            .data()
            .iter()
            .filter(|&&p| p != 0xff80_0000)
            .count();
        assert_eq!(painted, 6);
    }

    #[test]
    fn translucent_texture_composites_over_background() {
        let mut submaps = BTreeMap::new();
        submaps.insert(SubmapId::new(0, 0), texture_slice(&[100], &[50], 1, 1));
        let result = paint_submap_slices(&submaps, 0.05);
        // (50, 100, 255, 0) OVER (255, 128, 0, 0): a = 50 + 205 = 255,
        // r = 100 + round(128 * 205 / 255) = 203, g = 255 + 0, b = 0.
        assert_eq!(result.surface.pixel(5, 5), 0xffcb_ff00);
    }

    #[test]
    fn slice_pose_rotation_and_translation() {
        // Rotating the submap by 90 degrees about z maps texture u to image x.
        let mut slice = texture_slice(&[10, 20], &[255, 255], 2, 1);
        slice.pose = Rigid3d::new(
            Vector3::new(1.0, 0.0, 0.0),
            AngleAxis::new(FRAC_PI_2, Vector3::z()).to_quaternion(),
        );
        let mut submaps = BTreeMap::new();
        submaps.insert(SubmapId::new(0, 0), slice);
        let result = paint_submap_slices(&submaps, 0.05);
        assert_eq!(result.surface.width(), 2 + 10);
        assert_eq!(result.surface.height(), 1 + 10);
        // Corners: (0,0) -> (0, -20), (2,0) -> (2, -20) up to rounding.
        let find = |value: u32| {
            (0..result.surface.height())
                .flat_map(|y| (0..result.surface.width()).map(move |x| (x, y)))
                .find(|&(x, y)| result.surface.pixel(x, y) == value)
        };
        let a = find(0xff0a_ff00).expect("texel 0 painted");
        let b = find(0xff14_ff00).expect("texel 1 painted");
        assert_eq!(a.1, b.1);
        assert_eq!(b.0, a.0 + 1);
    }

    #[test]
    fn stops_at_first_slice_without_surface() {
        let mut submaps = BTreeMap::new();
        submaps.insert(SubmapId::new(0, 0), texture_slice(&[10], &[255], 1, 1));
        submaps.insert(SubmapId::new(0, 1), SubmapSlice::default());
        let mut far_away = texture_slice(&[20], &[255], 1, 1);
        far_away.pose = Rigid3d::from_translation(Vector3::new(10., 10., 0.));
        submaps.insert(SubmapId::new(0, 2), far_away);
        let result = paint_submap_slices(&submaps, 0.05);
        // Only the first slice contributes to size and pixels.
        assert_eq!(result.surface.width(), 11);
        assert_eq!(result.surface.height(), 11);
        assert!(
            result
                .surface
                .data()
                .iter()
                .all(|&p| (p >> 16) & 0xff != 20)
        );
    }

    #[test]
    fn empty_input_gives_empty_surface() {
        let result = paint_submap_slices(&BTreeMap::new(), 0.05);
        assert_eq!(result.surface.width(), 0);
        assert_eq!(result.surface.height(), 0);
    }

    fn submap_2d_proto() -> proto::Submap {
        let conversion_tables = Arc::new(ValueConversionTables::new());
        let mut grid = create_probability_grid(0.05, &conversion_tables);
        let inserter = ProbabilityGridRangeDataInserter2D::new(
            &proto::ProbabilityGridRangeDataInserterOptions2D {
                hit_probability: 0.55,
                miss_probability: 0.49,
                insert_free_space: true,
            },
        );
        inserter.insert_into_probability_grid(
            &RangeData {
                origin: Vector3::zeros(),
                returns: PointCloud::from_points(vec![
                    RangefinderPoint {
                        position: Vector3::new(1.0, 0.5, 0.0),
                    },
                    RangefinderPoint {
                        position: Vector3::new(-0.5, 1.0, 0.0),
                    },
                ]),
                misses: PointCloud::new(),
            },
            &mut grid,
        );
        proto::Submap {
            submap_id: Some(proto::SubmapId {
                trajectory_id: 0,
                submap_index: 0,
            }),
            submap_2d: Some(proto::Submap2D {
                local_pose: Some(to_proto(&Rigid3d::identity())),
                num_range_data: 1,
                finished: true,
                grid: Some(grid.to_proto()),
            }),
            ..Default::default()
        }
    }

    #[test]
    fn deserialize_and_fill_submap_slices_2d() {
        use proto::serialized_data::Data;
        let pose_graph = proto::PoseGraph {
            trajectory: vec![proto::Trajectory {
                trajectory_id: 0,
                submap: vec![proto::trajectory::Submap {
                    submap_index: 0,
                    pose: Some(to_proto(&Rigid3d::from_translation(Vector3::new(
                        2., 0., 0.,
                    )))),
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut reader = proto_reader_from_protos(
            proto::SerializationHeader { format_version: 2 },
            vec![
                proto::SerializedData {
                    data: Some(Data::PoseGraph(pose_graph)),
                },
                proto::SerializedData {
                    data: Some(Data::AllTrajectoryBuilderOptions(
                        proto::AllTrajectoryBuilderOptions {
                            options_with_sensor_ids: vec![Default::default()],
                        },
                    )),
                },
                proto::SerializedData {
                    data: Some(Data::Submap(submap_2d_proto())),
                },
                // A submap without grid is skipped.
                proto::SerializedData {
                    data: Some(Data::Submap(proto::Submap {
                        submap_id: Some(proto::SubmapId {
                            trajectory_id: 0,
                            submap_index: 7,
                        }),
                        submap_2d: Some(proto::Submap2D::default()),
                        ..Default::default()
                    })),
                },
            ],
        );
        let mut deserializer = ProtoStreamDeserializer::new(reader.as_mut());
        let mut slices = BTreeMap::new();
        deserialize_and_fill_submap_slices(
            &mut deserializer,
            &mut slices,
            &Arc::new(ValueConversionTables::new()),
        );
        assert_eq!(slices.len(), 1);
        let slice = &slices[&SubmapId::new(0, 0)];
        assert!(slice.width > 0 && slice.height > 0);
        assert_eq!(slice.resolution, 0.05);
        assert_eq!(slice.pose.translation(), &Vector3::new(2., 0., 0.));
        let surface = slice.surface.as_ref().unwrap();
        assert_eq!(
            (surface.width(), surface.height()),
            (slice.width, slice.height)
        );
        // Some cells were observed.
        assert!(surface.data().iter().any(|&p| (p >> 8) & 0xff == 255));

        let result = paint_submap_slices(&slices, 0.05);
        assert_eq!(result.surface.width(), slice.height + 10);
        assert_eq!(result.surface.height(), slice.width + 10);
        assert!(result.surface.data().iter().any(|&p| p != 0xff80_0000));
    }

    #[test]
    fn fill_submap_slice_3d_uses_high_resolution_texture() {
        let mut high = crate::mapping::three_d::hybrid_grid::HybridGrid::new(0.1);
        let mut low = crate::mapping::three_d::hybrid_grid::HybridGrid::new(0.5);
        for x in 0..5 {
            high.set_probability(&Vector3::new(x, 0, 0), 0.9);
            low.set_probability(&Vector3::new(x / 5, 0, 0), 0.9);
        }
        let submap = proto::Submap {
            submap_3d: Some(proto::Submap3D {
                local_pose: Some(to_proto(&Rigid3d::identity())),
                num_range_data: 1,
                finished: true,
                high_resolution_hybrid_grid: Some(high.to_proto()),
                low_resolution_hybrid_grid: Some(low.to_proto()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut slice = SubmapSlice::default();
        fill_submap_slice(
            &Rigid3d::identity(),
            &submap,
            &mut slice,
            &Arc::new(ValueConversionTables::new()),
        );
        assert!((slice.resolution - 0.1).abs() < 1e-6);
        assert!(slice.surface.is_some());
    }
}
