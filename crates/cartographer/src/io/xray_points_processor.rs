//! Port of `cartographer/io/xray_points_processor` (.h, .cc).

use std::collections::BTreeMap;
use std::sync::Arc;

use nalgebra::{Vector2, Vector3};

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::time::Time;
use crate::io::color::{FloatColor, get_color, to_uint8_color};
use crate::io::draw_trajectories::draw_trajectory;
use crate::io::file_writer::{FileWriter, FileWriterFactory};
use crate::io::image::Image;
use crate::io::points_batch::PointsBatch;
use crate::io::points_processor::{FlushResult, PointsProcessor};
use crate::mapping::detect_floors::{Floor, Timespan, detect_floors};
use crate::mapping::proto;
use crate::mapping::three_d::hybrid_grid::{Array3i, HybridGridBase};
use crate::transform::rigid_transform::{Rigid3d, Rigid3f, from_dictionary};

/// An axis aligned integer box (C++ `Eigen::AlignedBox3i`). Initially empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlignedBox3i {
    min: Vector3<i32>,
    max: Vector3<i32>,
}

impl Default for AlignedBox3i {
    fn default() -> Self {
        Self {
            min: Vector3::repeat(i32::MAX),
            max: Vector3::repeat(i32::MIN),
        }
    }
}

impl AlignedBox3i {
    /// Minimum corner.
    pub fn min(&self) -> &Vector3<i32> {
        &self.min
    }

    /// Maximum corner.
    pub fn max(&self) -> &Vector3<i32> {
        &self.max
    }

    /// Whether the box is empty.
    pub fn is_empty(&self) -> bool {
        (0..3).any(|i| self.min[i] > self.max[i])
    }

    /// Extends the box to contain `p`.
    pub fn extend(&mut self, p: &Vector3<i32>) {
        self.min = self.min.inf(p);
        self.max = self.max.sup(p);
    }

    /// `max - min`.
    pub fn sizes(&self) -> Vector3<i32> {
        self.max - self.min
    }
}

/// Whether to draw the trajectories into the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawTrajectories {
    /// Do not draw trajectories.
    No,
    /// Draw trajectories.
    Yes,
}

#[derive(Clone, Copy, Debug, Default)]
struct ColumnData {
    sum_r: f32,
    sum_g: f32,
    sum_b: f32,
    count: u32,
}

struct Aggregation {
    voxels: HybridGridBase<bool>,
    column_data: BTreeMap<(i32, i32), ColumnData>,
}

#[derive(Clone, Copy, Debug, Default)]
struct PixelData {
    num_occupied_cells_in_column: usize,
    mean_r: f32,
    mean_g: f32,
    mean_b: f32,
}

struct PixelDataMatrix {
    width: i32,
    data: Vec<PixelData>,
}

impl PixelDataMatrix {
    fn new(width: i32, height: i32) -> Self {
        Self {
            width,
            data: vec![PixelData::default(); (width * height) as usize],
        }
    }

    fn width(&self) -> i32 {
        self.width
    }

    fn height(&self) -> i32 {
        self.data.len() as i32 / self.width
    }

    fn at(&self, x: i32, y: i32) -> &PixelData {
        &self.data[(x + y * self.width) as usize]
    }

    fn at_mut(&mut self, x: i32, y: i32) -> &mut PixelData {
        &mut self.data[(x + y * self.width) as usize]
    }
}

// C++: `a * (1. - t) + t * b` with a double literal, i.e. computed in double
// and narrowed to float on return.
fn mix(a: f32, b: f32, t: f32) -> f32 {
    (f64::from(a) * (1. - f64::from(t)) + f64::from(t * b)) as f32
}

// Convert 'matrix' into a pleasing-to-look-at image.
fn into_image(matrix: &PixelDataMatrix, saturation_factor: f64) -> Image {
    let mut image = Image::new(matrix.width(), matrix.height());
    // `std::numeric_limits<float>::min()`: the smallest positive normal float.
    let mut max = f32::MIN_POSITIVE;
    for y in 0..matrix.height() {
        for x in 0..matrix.width() {
            let cell = matrix.at(x, y);
            if cell.num_occupied_cells_in_column == 0 {
                continue;
            }
            // `std::log` of an integer is computed in double.
            max = max.max((cell.num_occupied_cells_in_column as f64).ln() as f32);
        }
    }

    for y in 0..matrix.height() {
        for x in 0..matrix.width() {
            let cell = matrix.at(x, y);
            if cell.num_occupied_cells_in_column == 0 {
                image.set_pixel(x, y, &[255, 255, 255]);
                continue;
            }

            // We use a logarithmic weighting for how saturated a pixel will
            // be. The basic idea here was that walls (full height) are fully
            // saturated, but details like chairs and tables are still well
            // visible.
            let saturation = (1.0f64).min(
                (cell.num_occupied_cells_in_column as f64).ln() / f64::from(max)
                    * saturation_factor,
            ) as f32;
            let color: FloatColor = [
                mix(1.0, cell.mean_r, saturation),
                mix(1.0, cell.mean_g, saturation),
                mix(1.0, cell.mean_b, saturation),
            ];
            image.set_pixel(x, y, &to_uint8_color(&color));
        }
    }
    image
}

fn contained_in(time: Time, timespans: &[Timespan]) -> bool {
    timespans
        .iter()
        .any(|timespan| timespan.start <= time && time <= timespan.end)
}

/// Creates X-ray cuts through the points with pixels being `voxel_size` big.
pub struct XRayPointsProcessor {
    draw_trajectories: DrawTrajectories,
    trajectories: Arc<Vec<proto::Trajectory>>,
    file_writer_factory: FileWriterFactory,
    next: Box<dyn PointsProcessor>,
    // If empty, we do not separate into floors.
    floors: Vec<Floor>,
    output_filename: String,
    transform: Rigid3f,
    // Only has one entry if we do not separate into floors.
    aggregations: Vec<Aggregation>,
    // Bounding box containing all cells with data in all 'aggregations_'.
    bounding_box: AlignedBox3i,
    // Scale the saturation of the point color. If saturation_factor_ > 1, the
    // point has darker color, otherwise it has lighter color.
    saturation_factor: f64,
}

impl XRayPointsProcessor {
    /// Name of this processor in the Lua pipeline configuration.
    pub const CONFIGURATION_FILE_ACTION_NAME: &'static str = "write_xray_image";

    /// Creates the processor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        voxel_size: f64,
        saturation_factor: f64,
        transform: &Rigid3f,
        floors: &[Floor],
        draw_trajectories: DrawTrajectories,
        output_filename: &str,
        trajectories: Arc<Vec<proto::Trajectory>>,
        file_writer_factory: FileWriterFactory,
        next: Box<dyn PointsProcessor>,
    ) -> Self {
        let num_aggregations = if floors.is_empty() { 1 } else { floors.len() };
        let aggregations = (0..num_aggregations)
            .map(|_| Aggregation {
                voxels: HybridGridBase::new(voxel_size as f32),
                column_data: BTreeMap::new(),
            })
            .collect();
        Self {
            draw_trajectories,
            trajectories,
            file_writer_factory,
            next,
            floors: floors.to_vec(),
            output_filename: output_filename.to_string(),
            transform: *transform,
            aggregations,
            bounding_box: AlignedBox3i::default(),
            saturation_factor,
        }
    }

    /// Creates the processor from `voxel_size`, `filename`, `transform` and
    /// the optional `separate_floors` (default false), `draw_trajectories`
    /// (default true) and `saturation_factor` (default 1).
    ///
    /// RUST PORT NOTE: the C++ registration captures `trajectories` by
    /// reference; here they are shared through an `Arc`.
    pub fn from_dictionary(
        trajectories: Arc<Vec<proto::Trajectory>>,
        file_writer_factory: FileWriterFactory,
        dictionary: &LuaParameterDictionary,
        next: Box<dyn PointsProcessor>,
    ) -> Box<Self> {
        let mut floors = Vec::new();
        let separate_floor =
            dictionary.has_key("separate_floors") && dictionary.get_bool("separate_floors");
        let draw_trajectories = if !dictionary.has_key("draw_trajectories")
            || dictionary.get_bool("draw_trajectories")
        {
            DrawTrajectories::Yes
        } else {
            DrawTrajectories::No
        };
        let saturation_factor = if dictionary.has_key("saturation_factor") {
            dictionary.get_double("saturation_factor")
        } else {
            1.
        };
        if separate_floor {
            assert_eq!(
                trajectories.len(),
                1,
                "Can only detect floors with a single trajectory."
            );
            floors = detect_floors(&trajectories[0]);
        }
        Box::new(Self::new(
            dictionary.get_double("voxel_size"),
            saturation_factor,
            &from_dictionary(&dictionary.get_dictionary("transform")).cast::<f32>(),
            &floors,
            draw_trajectories,
            &dictionary.get_string("filename"),
            trajectories,
            file_writer_factory,
            next,
        ))
    }

    /// Bounding box of all occupied cells in all aggregations.
    pub fn bounding_box(&self) -> AlignedBox3i {
        self.bounding_box
    }

    fn write_voxels(&self, aggregation: &Aggregation, file_writer: &mut dyn FileWriter) {
        if self.bounding_box.is_empty() {
            log::warn!("Not writing output: bounding box is empty.");
            return;
        }

        // Returns the (x, y) pixel of the given 'index'.
        let bounding_box = self.bounding_box;
        let voxel_index_to_pixel = |index: &Array3i| -> Vector2<i32> {
            // We flip the y axis, since matrices rows are counted from the
            // top.
            Vector2::new(
                bounding_box.max()[1] - index[1],
                bounding_box.max()[2] - index[2],
            )
        };

        // Hybrid grid uses X: forward, Y: left, Z: up.
        // For the screen we are using. X: right, Y: up
        let xsize = bounding_box.sizes()[1] + 1;
        let ysize = bounding_box.sizes()[2] + 1;
        let mut pixel_data_matrix = PixelDataMatrix::new(xsize, ysize);
        for (cell_index, _) in &aggregation.voxels {
            let pixel = voxel_index_to_pixel(&cell_index);
            let column_data = aggregation
                .column_data
                .get(&(cell_index[1], cell_index[2]))
                .expect("column data must exist for every occupied voxel");
            let pixel_data = pixel_data_matrix.at_mut(pixel.x, pixel.y);
            pixel_data.mean_r = column_data.sum_r / column_data.count as f32;
            pixel_data.mean_g = column_data.sum_g / column_data.count as f32;
            pixel_data.mean_b = column_data.sum_b / column_data.count as f32;
            pixel_data.num_occupied_cells_in_column += 1;
        }

        let mut image = into_image(&pixel_data_matrix, self.saturation_factor);
        if self.draw_trajectories == DrawTrajectories::Yes {
            for (i, trajectory) in self.trajectories.iter().enumerate() {
                draw_trajectory(
                    trajectory,
                    &get_color(i as i32),
                    &|pose: &Rigid3d| -> Vector2<i32> {
                        voxel_index_to_pixel(
                            &aggregation.voxels.get_cell_index(
                                (self.transform * pose.cast::<f32>()).translation(),
                            ),
                        )
                    },
                    image.get_cairo_surface(),
                );
            }
        }
        image.write_png(file_writer);
        assert!(file_writer.close(), "Check failed: file_writer->Close()");
    }

    fn insert(
        transform: &Rigid3f,
        bounding_box: &mut AlignedBox3i,
        batch: &PointsBatch,
        aggregation: &mut Aggregation,
    ) {
        const DEFAULT_COLOR: FloatColor = [0., 0., 0.];
        for (i, point) in batch.points.iter().enumerate() {
            let camera_point = transform * point;
            let cell_index = aggregation.voxels.get_cell_index(&camera_point.position);
            *aggregation.voxels.mutable_value(&cell_index) = true;
            bounding_box.extend(&cell_index);
            let column_data = aggregation
                .column_data
                .entry((cell_index[1], cell_index[2]))
                .or_default();
            let color = if batch.colors.is_empty() {
                &DEFAULT_COLOR
            } else {
                &batch.colors[i]
            };
            column_data.sum_r += color[0];
            column_data.sum_g += color[1];
            column_data.sum_b += color[2];
            column_data.count += 1;
        }
    }
}

impl PointsProcessor for XRayPointsProcessor {
    fn process(&mut self, batch: Box<PointsBatch>) {
        if self.floors.is_empty() {
            assert_eq!(self.aggregations.len(), 1);
            Self::insert(
                &self.transform,
                &mut self.bounding_box,
                &batch,
                &mut self.aggregations[0],
            );
        } else {
            for i in 0..self.floors.len() {
                if !contained_in(batch.start_time, &self.floors[i].timespans) {
                    continue;
                }
                Self::insert(
                    &self.transform,
                    &mut self.bounding_box,
                    &batch,
                    &mut self.aggregations[i],
                );
            }
        }
        self.next.process(batch);
    }

    fn flush(&mut self) -> FlushResult {
        if self.floors.is_empty() {
            assert_eq!(self.aggregations.len(), 1);
            let mut file_writer =
                (self.file_writer_factory)(&format!("{}.png", self.output_filename));
            self.write_voxels(&self.aggregations[0], file_writer.as_mut());
        } else {
            for i in 0..self.floors.len() {
                let mut file_writer =
                    (self.file_writer_factory)(&format!("{}{}.png", self.output_filename, i));
                self.write_voxels(&self.aggregations[i], file_writer.as_mut());
            }
        }

        match self.next.flush() {
            FlushResult::RestartStream => panic!(
                "X-Ray generation must be configured to occur after any stages that require \
                 multiple passes."
            ),
            FlushResult::Finished => FlushResult::Finished,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
    use crate::io::fake_file_writer::FakeFileWriter;
    use crate::io::null_points_processor::NullPointsProcessor;
    use crate::sensor::rangefinder_point::RangefinderPoint;

    fn decode_png(bytes: Vec<u8>) -> (u32, u32, Vec<u8>) {
        let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        buf.truncate(info.buffer_size());
        (info.width, info.height, buf)
    }

    #[test]
    fn writes_side_view() {
        let content = Arc::new(Mutex::new(Vec::new()));
        let dictionary = make_dictionary(
            r#"return {
              voxel_size = 1.,
              filename = "xray",
              transform = {
                translation = { 0., 0., 0. },
                rotation = { 0., 0., 0. },
              },
              draw_trajectories = false,
            }"#,
        );
        let filenames = Arc::new(Mutex::new(Vec::new()));
        let factory: FileWriterFactory = {
            let content = Arc::clone(&content);
            let filenames = Arc::clone(&filenames);
            Arc::new(move |filename: &str| {
                filenames.lock().unwrap().push(filename.to_string());
                Box::new(FakeFileWriter::new(filename, Arc::clone(&content))) as Box<dyn FileWriter>
            })
        };
        let mut processor = XRayPointsProcessor::from_dictionary(
            Arc::new(Vec::new()),
            factory,
            &dictionary,
            Box::new(NullPointsProcessor::new()),
        );
        // A column of 3 voxels at (y, z) = (0, 0..3) (red) and a single voxel
        // at (y, z) = (2, 0) (no color -> black).
        processor.process(Box::new(PointsBatch {
            points: (0..3)
                .map(|z| RangefinderPoint {
                    position: Vector3::new(0., 0., z as f32),
                })
                .chain((0..3).map(|x| RangefinderPoint {
                    position: Vector3::new(x as f32, 0., 0.),
                }))
                .collect(),
            colors: vec![[1., 0., 0.]; 6],
            ..Default::default()
        }));
        processor.process(Box::new(PointsBatch {
            points: vec![RangefinderPoint {
                position: Vector3::new(0., 2., 0.),
            }],
            ..Default::default()
        }));
        assert_eq!(processor.bounding_box().min(), &Vector3::new(0, 0, 0));
        assert_eq!(processor.bounding_box().max(), &Vector3::new(2, 2, 2));
        assert_eq!(processor.flush(), FlushResult::Finished);
        assert_eq!(*filenames.lock().unwrap(), vec!["xray.png".to_string()]);

        let (width, height, pixels) = decode_png(content.lock().unwrap().clone());
        assert_eq!((width, height), (3, 3));
        let pixel = |x: usize, y: usize| {
            let i = (y * 3 + x) * 3;
            [pixels[i], pixels[i + 1], pixels[i + 2]]
        };
        // Pixel (x, y) = (max_y - y_index, max_z - z_index).
        // Column (y=0, z=0) has 3 occupied voxels (x = 0, 1, 2), the maximum:
        // saturation 1 -> mean color of its 4 (red) points -> red.
        assert_eq!(pixel(2, 2), [255, 0, 0]);
        // Column (y=0, z=1): 1 voxel -> log(1) = 0 -> white.
        assert_eq!(pixel(2, 1), [255, 255, 255]);
        // Column (y=2, z=0): 1 voxel -> white as well.
        assert_eq!(pixel(0, 2), [255, 255, 255]);
        // Empty columns are white.
        assert_eq!(pixel(1, 1), [255, 255, 255]);
    }

    #[test]
    fn mix_and_saturation_follow_cpp_precision() {
        assert_eq!(mix(1.0, 0.0, 1.0), 0.0);
        assert_eq!(mix(1.0, 0.0, 0.0), 1.0);
        assert!((mix(1.0, 0.5, 0.5) - 0.75).abs() < 1e-7);
    }
}
