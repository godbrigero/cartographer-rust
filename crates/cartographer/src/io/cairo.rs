//! A small pure-Rust re-implementation of the subset of the Cairo image
//! backend (and the pixman pixel math underneath it) that Cartographer uses.
//!
//! RUST PORT NOTE: this file has no C++ counterpart. The C++ code links the
//! Cairo C library; per the project's dependency policy (see
//! `DEPENDENCIES.md`) we re-implement the operations used by `io/image`,
//! `io/draw_trajectories` and `io/submap_painter` on raw ARGB32 buffers:
//!
//! * [`Surface`]: `CAIRO_FORMAT_ARGB32` image surfaces (native-endian `u32`
//!   pixels `0xAARRGGBB`, **premultiplied** alpha, stride `4 * width`). Like
//!   Cairo, pixel values are stored verbatim even if they are not valid
//!   premultiplied colors (`DrawTexture` relies on that).
//! * [`Matrix`]: `cairo_matrix_t` with Cairo's multiplication order and
//!   inversion.
//! * [`Context`]: `cairo_t` with `save`/`restore`, `translate`/`scale`/
//!   `transform`, `user_to_device`, solid and surface sources, `paint`,
//!   `line_to`/`move_to`/`arc`, `stroke` (butt caps, miter joins with miter
//!   limit 10) and `fill` (non-zero winding), operator `OVER` only.
//! * PNG output ([`Surface::write_to_png_stream`]) like
//!   `cairo_surface_write_to_png_stream`: RGB if every pixel is opaque,
//!   otherwise un-premultiplied RGBA (with Cairo's integer rounding, including
//!   the wrap-around for invalid premultiplied pixels), 8 bits per channel,
//!   with a white `bKGD` chunk.
//!
//! Fidelity:
//!
//! * Exact: matrix math, solid color conversion (`double` -> 16 bit -> 8 bit),
//!   pixman's `OVER` compositing (`MUL_UN8` rounding, saturating add),
//!   pixman's bilinear interpolation (7-bit weights, `EXTEND_NONE`, 16.16
//!   fixed point sample positions), arc construction (Bezier segments as in
//!   `cairo-arc.c`, flattened with Cairo's spline decomposition at tolerance
//!   0.1 in 24.8 fixed point), PNG pixel values.
//! * Approximate: anti-aliased coverage. Cairo's default rasterizer
//!   (`cairo-tor-scan-converter`) samples 15 sub-scanlines per pixel with
//!   1/256 pixel horizontal resolution; we do the same (sub-scanline centers,
//!   edge endpoints quantized to the grid like tor), but do not reproduce its
//!   incremental edge stepping and full-row fast paths bit for bit, and the
//!   stroke outline is built from per-segment quads plus join wedges rather
//!   than Cairo's contour stroker. Edge pixels of strokes and arcs may
//!   therefore differ by a few alpha levels from real Cairo output.
//! * Not implemented: Cairo >= 1.14 uses a box-filtered separable
//!   convolution instead of bilinear filtering for `CAIRO_FILTER_GOOD` when a
//!   source surface is *down-scaled* by more than 1/0.75; we always use
//!   bilinear filtering (exact for the common case of painting submaps at
//!   their own resolution).

use std::f64::consts::PI;

/// Cairo status codes used by Cartographer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// `CAIRO_STATUS_SUCCESS`.
    Success,
    /// `CAIRO_STATUS_WRITE_ERROR`.
    WriteError,
    /// `CAIRO_STATUS_INVALID_SIZE`.
    InvalidSize,
    /// `CAIRO_STATUS_INVALID_MATRIX`.
    InvalidMatrix,
}

/// Returns the stride of an ARGB32 surface of `width` pixels
/// (`cairo_format_stride_for_width`).
pub fn format_stride_for_width(width: i32) -> i32 {
    width * 4
}

// ---------------------------------------------------------------------------
// Surface
// ---------------------------------------------------------------------------

/// An ARGB32 image surface (`cairo_image_surface_t` with
/// `CAIRO_FORMAT_ARGB32`). Pixels are row-major, index `y * width + x`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Surface {
    width: i32,
    height: i32,
    data: Vec<u32>,
}

impl Surface {
    /// Creates a surface with all pixels transparent black
    /// (`cairo_image_surface_create`). Panics on negative sizes (Cairo
    /// returns an error surface).
    pub fn new(width: i32, height: i32) -> Self {
        assert!(
            width >= 0 && height >= 0,
            "cairo_image_surface_create: invalid size {width}x{height}"
        );
        Self {
            width,
            height,
            data: vec![0; width as usize * height as usize],
        }
    }

    /// Creates a surface over existing pixel data
    /// (`cairo_image_surface_create_for_data`). RUST PORT NOTE: the surface
    /// owns the data instead of aliasing it.
    pub fn from_data(data: Vec<u32>, width: i32, height: i32) -> Self {
        assert!(width >= 0 && height >= 0);
        assert_eq!(
            data.len(),
            width as usize * height as usize,
            "pixel data does not match the surface size"
        );
        Self {
            width,
            height,
            data,
        }
    }

    /// Width in pixels.
    pub fn width(&self) -> i32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> i32 {
        self.height
    }

    /// Stride in bytes (always `4 * width`).
    pub fn stride(&self) -> i32 {
        format_stride_for_width(self.width)
    }

    /// The pixel data (`cairo_image_surface_get_data`).
    pub fn data(&self) -> &[u32] {
        &self.data
    }

    /// Mutable pixel data.
    pub fn data_mut(&mut self) -> &mut [u32] {
        &mut self.data
    }

    /// Consumes the surface, returning its pixel data.
    pub fn into_data(self) -> Vec<u32> {
        self.data
    }

    /// Returns the pixel at `(x, y)`.
    pub fn pixel(&self, x: i32, y: i32) -> u32 {
        self.data[(y * self.width + x) as usize]
    }

    /// Returns the pixel at `(x, y)` or 0 outside of the surface
    /// (`PIXMAN_REPEAT_NONE`).
    #[inline]
    fn pixel_or_zero(&self, x: i32, y: i32) -> u32 {
        if x < 0 || y < 0 || x >= self.width || y >= self.height {
            0
        } else {
            self.data[(y * self.width + x) as usize]
        }
    }

    /// Encodes the surface as PNG like `cairo_surface_write_to_png_stream`
    /// and passes the bytes to `write_func`. Returns
    /// [`Status::WriteError`] if `write_func` returns false.
    pub fn write_to_png_stream(&self, write_func: &mut dyn FnMut(&[u8]) -> bool) -> Status {
        match self.encode_png() {
            Ok(bytes) => {
                if write_func(&bytes) {
                    Status::Success
                } else {
                    Status::WriteError
                }
            }
            Err(status) => status,
        }
    }

    /// Encodes the surface as PNG (see [`Self::write_to_png_stream`]).
    pub fn encode_png(&self) -> Result<Vec<u8>, Status> {
        if self.width <= 0 || self.height <= 0 {
            // libpng refuses zero-sized images; Cairo reports this as an
            // invalid size.
            return Err(Status::InvalidSize);
        }
        let opaque = self.data.iter().all(|&p| p >> 24 == 0xff);
        let mut bytes = Vec::with_capacity(self.data.len() * if opaque { 3 } else { 4 });
        for &pixel in &self.data {
            let alpha = pixel >> 24;
            let r = (pixel >> 16) & 0xff;
            let g = (pixel >> 8) & 0xff;
            let b = pixel & 0xff;
            if opaque {
                // convert_data_to_bytes()
                bytes.extend_from_slice(&[r as u8, g as u8, b as u8]);
            } else if alpha == 0 {
                // unpremultiply_data()
                bytes.extend_from_slice(&[0, 0, 0, 0]);
            } else {
                // Cairo stores the quotient in a uint8_t: values of invalid
                // premultiplied pixels (component > alpha) wrap around.
                let unpremultiply = |c: u32| ((c * 255 + alpha / 2) / alpha) as u8;
                bytes.extend_from_slice(&[
                    unpremultiply(r),
                    unpremultiply(g),
                    unpremultiply(b),
                    alpha as u8,
                ]);
            }
        }
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, self.width as u32, self.height as u32);
            encoder.set_color(if opaque {
                png::ColorType::Rgb
            } else {
                png::ColorType::Rgba
            });
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().map_err(|_| Status::WriteError)?;
            // Cairo always sets a white background color.
            let white: [u8; 6] = [0, 0xff, 0, 0xff, 0, 0xff];
            writer
                .write_chunk(png::chunk::bKGD, &white)
                .map_err(|_| Status::WriteError)?;
            writer
                .write_image_data(&bytes)
                .map_err(|_| Status::WriteError)?;
            writer.finish().map_err(|_| Status::WriteError)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------

/// An affine transformation (`cairo_matrix_t`): `x' = xx * x + xy * y + x0`,
/// `y' = yx * x + yy * y + y0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix {
    /// xx component.
    pub xx: f64,
    /// yx component.
    pub yx: f64,
    /// xy component.
    pub xy: f64,
    /// yy component.
    pub yy: f64,
    /// x translation.
    pub x0: f64,
    /// y translation.
    pub y0: f64,
}

impl Default for Matrix {
    fn default() -> Self {
        Self::identity()
    }
}

impl Matrix {
    /// `cairo_matrix_init` (same argument order).
    pub fn new(xx: f64, yx: f64, xy: f64, yy: f64, x0: f64, y0: f64) -> Self {
        Self {
            xx,
            yx,
            xy,
            yy,
            x0,
            y0,
        }
    }

    /// `cairo_matrix_init_identity`.
    pub fn identity() -> Self {
        Self::new(1., 0., 0., 1., 0., 0.)
    }

    /// `cairo_matrix_init_translate`.
    pub fn translation(tx: f64, ty: f64) -> Self {
        Self::new(1., 0., 0., 1., tx, ty)
    }

    /// `cairo_matrix_init_scale`.
    pub fn scaling(sx: f64, sy: f64) -> Self {
        Self::new(sx, 0., 0., sy, 0., 0.)
    }

    /// `cairo_matrix_multiply(&result, a, b)`: the transformation that first
    /// applies `a`, then `b`.
    pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
        Matrix {
            xx: a.xx * b.xx + a.yx * b.xy,
            yx: a.xx * b.yx + a.yx * b.yy,
            xy: a.xy * b.xx + a.yy * b.xy,
            yy: a.xy * b.yx + a.yy * b.yy,
            x0: a.x0 * b.xx + a.y0 * b.xy + b.x0,
            y0: a.x0 * b.yx + a.y0 * b.yy + b.y0,
        }
    }

    /// `cairo_matrix_transform_point`.
    pub fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = self.transform_distance(x, y);
        (dx + self.x0, dy + self.y0)
    }

    /// `cairo_matrix_transform_distance`.
    pub fn transform_distance(&self, x: f64, y: f64) -> (f64, f64) {
        (self.xx * x + self.xy * y, self.yx * x + self.yy * y)
    }

    /// `cairo_matrix_invert`; `None` if the matrix is not invertible.
    pub fn invert(&self) -> Option<Matrix> {
        let mut m = *self;
        // Simple scaling|translation matrices are quite common...
        if m.xy == 0. && m.yx == 0. {
            m.x0 = -m.x0;
            m.y0 = -m.y0;
            if m.xx != 1. {
                if m.xx == 0. {
                    return None;
                }
                m.xx = 1. / m.xx;
                m.x0 *= m.xx;
            }
            if m.yy != 1. {
                if m.yy == 0. {
                    return None;
                }
                m.yy = 1. / m.yy;
                m.y0 *= m.yy;
            }
            return Some(m);
        }
        // inv (A) = 1/det (A) * adj (A)
        let det = m.xx * m.yy - m.yx * m.xy;
        if det == 0. || !det.is_finite() {
            return None;
        }
        let (a, b, c, d, tx, ty) = (m.xx, m.yx, m.xy, m.yy, m.x0, m.y0);
        let adjoint = Matrix::new(d, -b, -c, a, c * ty - d * tx, b * tx - a * ty);
        let s = 1. / det;
        Some(Matrix::new(
            adjoint.xx * s,
            adjoint.yx * s,
            adjoint.xy * s,
            adjoint.yy * s,
            adjoint.x0 * s,
            adjoint.y0 * s,
        ))
    }
}

// ---------------------------------------------------------------------------
// Pixel math (pixman)
// ---------------------------------------------------------------------------

/// pixman `MUL_UN8`: `round(a * b / 255)`.
#[inline]
fn mul_un8(a: u32, b: u32) -> u32 {
    let t = a * b + 0x80;
    (t + (t >> 8)) >> 8
}

/// Multiplies all four channels of `x` by `a` (`UN8x4_MUL_UN8`).
#[inline]
fn mul_un8x4(x: u32, a: u32) -> u32 {
    let mut out = 0;
    for shift in [0, 8, 16, 24] {
        out |= mul_un8((x >> shift) & 0xff, a) << shift;
    }
    out
}

/// `OVER`: `dest * (255 - alpha(src)) / 255 + src` with saturating add
/// (`UN8x4_MUL_UN8_ADD_UN8x4`).
#[inline]
pub(crate) fn over(src: u32, dest: u32) -> u32 {
    let ia = 255 - (src >> 24);
    let mut out = 0;
    for shift in [0, 8, 16, 24] {
        let d = mul_un8((dest >> shift) & 0xff, ia);
        let s = (src >> shift) & 0xff;
        out |= (d + s).min(255) << shift;
    }
    out
}

/// Number of bits of the bilinear interpolation weights in pixman.
const BILINEAR_INTERPOLATION_BITS: i32 = 7;

/// pixman `bilinear_interpolation` (generic 64-bit version).
#[inline]
fn bilinear_interpolation(tl: u32, tr: u32, bl: u32, br: u32, distx: i32, disty: i32) -> u32 {
    let distx = (distx << (8 - BILINEAR_INTERPOLATION_BITS)) as u64;
    let disty = (disty << (8 - BILINEAR_INTERPOLATION_BITS)) as u64;
    let distxy = distx * disty;
    let distxiy = distx * (256 - disty);
    let distixy = (256 - distx) * disty;
    let distixiy = (256 - distx) * (256 - disty);
    let mut out = 0u32;
    for shift in [0, 8, 16, 24] {
        let c = |p: u32| u64::from((p >> shift) & 0xff);
        let f = c(tl) * distixiy + c(tr) * distxiy + c(bl) * distixy + c(br) * distxy;
        out |= ((f >> 16) as u32 & 0xff) << shift;
    }
    out
}

/// `pixman_fixed_t`: 16.16 fixed point.
type Fixed16 = i64;

/// `_cairo_fixed_16_16_from_double`: round to nearest.
fn fixed_16_16_from_double(d: f64) -> Fixed16 {
    (d * 65536.).round_ties_even() as Fixed16
}

/// An affine pixman transform in 16.16 fixed point.
#[derive(Clone, Copy, Debug)]
struct PixmanTransform {
    m: [[Fixed16; 3]; 2],
}

impl PixmanTransform {
    fn from_matrix(matrix: &Matrix) -> Self {
        Self {
            m: [
                [
                    fixed_16_16_from_double(matrix.xx),
                    fixed_16_16_from_double(matrix.xy),
                    fixed_16_16_from_double(matrix.x0),
                ],
                [
                    fixed_16_16_from_double(matrix.yx),
                    fixed_16_16_from_double(matrix.yy),
                    fixed_16_16_from_double(matrix.y0),
                ],
            ],
        }
    }

    /// `pixman_transform_point_3d` for an affine transform.
    fn transform_point(&self, x: Fixed16, y: Fixed16) -> (Fixed16, Fixed16) {
        let apply = |row: &[Fixed16; 3]| {
            let tmp = row[0] * x + row[1] * y + row[2] * 0x10000;
            (tmp + 0x8000) >> 16
        };
        (apply(&self.m[0]), apply(&self.m[1]))
    }
}

/// Samples `surface` with pixman's bilinear filter at the pattern-space
/// position `(x, y)` (16.16 fixed point, already transformed).
#[inline]
fn fetch_bilinear(surface: &Surface, x: Fixed16, y: Fixed16) -> u32 {
    let x1 = x - 0x8000;
    let y1 = y - 0x8000;
    let distx = ((x1 >> (16 - BILINEAR_INTERPOLATION_BITS))
        & ((1 << BILINEAR_INTERPOLATION_BITS) - 1)) as i32;
    let disty = ((y1 >> (16 - BILINEAR_INTERPOLATION_BITS))
        & ((1 << BILINEAR_INTERPOLATION_BITS) - 1)) as i32;
    let x1 = (x1 >> 16) as i32;
    let y1 = (y1 >> 16) as i32;
    let tl = surface.pixel_or_zero(x1, y1);
    let tr = surface.pixel_or_zero(x1 + 1, y1);
    let bl = surface.pixel_or_zero(x1, y1 + 1);
    let br = surface.pixel_or_zero(x1 + 1, y1 + 1);
    bilinear_interpolation(tl, tr, bl, br, distx, disty)
}

/// `_cairo_color_double_to_short`.
fn color_double_to_short(d: f64) -> u16 {
    (d * 65535.0 + 0.5) as u16
}

/// Converts a non-premultiplied RGBA color in `[0, 1]` to a premultiplied
/// ARGB32 pixel like Cairo (via 16-bit premultiplied shorts, `>> 8`).
pub fn color_to_pixel(red: f64, green: f64, blue: f64, alpha: f64) -> u32 {
    let clamp = |v: f64| v.clamp(0., 1.);
    let (red, green, blue, alpha) = (clamp(red), clamp(green), clamp(blue), clamp(alpha));
    let a = u32::from(color_double_to_short(alpha) >> 8);
    let r = u32::from(color_double_to_short(red * alpha) >> 8);
    let g = u32::from(color_double_to_short(green * alpha) >> 8);
    let b = u32::from(color_double_to_short(blue * alpha) >> 8);
    (a << 24) | (r << 16) | (g << 8) | b
}

// ---------------------------------------------------------------------------
// Path construction (24.8 fixed point device coordinates)
// ---------------------------------------------------------------------------

/// `cairo_fixed_t`: 24.8 fixed point.
type Fixed = i64;
const FIXED_ONE: f64 = 256.;

fn fixed_from_double(d: f64) -> Fixed {
    (d * FIXED_ONE).round_ties_even() as Fixed
}

fn fixed_to_double(f: Fixed) -> f64 {
    f as f64 / FIXED_ONE
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Point {
    x: Fixed,
    y: Fixed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PathOp {
    Move(Point),
    Line(Point),
    Curve(Point, Point, Point),
}

/// The current path, in device space (like `cairo_path_fixed_t`).
#[derive(Clone, Debug, Default)]
struct Path {
    ops: Vec<PathOp>,
    current_point: Option<Point>,
}

fn slope_equal_and_forward(p: &Point, q: &Point, r: &Point) -> bool {
    let (dx1, dy1) = (q.x - p.x, q.y - p.y);
    let (dx2, dy2) = (r.x - q.x, r.y - q.y);
    // _cairo_slope_equal && !_cairo_slope_backwards
    dx1 * dy2 == dx2 * dy1 && !(dx1 * dx2 < 0 || dy1 * dy2 < 0)
}

impl Path {
    fn move_to(&mut self, point: Point) {
        if let Some(PathOp::Move(last)) = self.ops.last_mut() {
            *last = point;
        } else {
            self.ops.push(PathOp::Move(point));
        }
        self.current_point = Some(point);
    }

    /// `_cairo_path_fixed_line_to`, including its removal of degenerate and
    /// collinear segments.
    fn line_to(&mut self, point: Point) {
        let Some(current) = self.current_point else {
            self.move_to(point);
            return;
        };
        let last_is_move_to = matches!(self.ops.last(), Some(PathOp::Move(_)));
        // If the previous op was but the initial MOVE_TO and this segment is
        // degenerate, then we can simply skip this point.
        if !last_is_move_to && point == current {
            return;
        }
        // If the previous op was also a LINE_TO with the same gradient, then
        // just change its end-point rather than adding a new op.
        if let Some(PathOp::Line(_)) = self.ops.last() {
            let penultimate = match self.ops.get(self.ops.len().wrapping_sub(2)) {
                Some(PathOp::Move(p)) | Some(PathOp::Line(p)) => Some(*p),
                Some(PathOp::Curve(_, _, p)) => Some(*p),
                None => None,
            };
            if let Some(p) = penultimate
                && (p == current || slope_equal_and_forward(&p, &current, &point))
            {
                self.ops.pop();
            }
        }
        self.ops.push(PathOp::Line(point));
        self.current_point = Some(point);
    }

    fn curve_to(&mut self, p1: Point, p2: Point, p3: Point) {
        if self.current_point.is_none() {
            self.move_to(p1);
        }
        self.ops.push(PathOp::Curve(p1, p2, p3));
        self.current_point = Some(p3);
    }

    fn clear(&mut self) {
        self.ops.clear();
        self.current_point = None;
    }

    /// Splits the path into flattened sub-paths (curves decomposed with
    /// `tolerance`).
    fn flatten(&self, tolerance: f64) -> Vec<Vec<Point>> {
        let mut subpaths: Vec<Vec<Point>> = Vec::new();
        for op in &self.ops {
            match op {
                PathOp::Move(p) => subpaths.push(vec![*p]),
                PathOp::Line(p) => subpaths
                    .last_mut()
                    .expect("line_to without move_to")
                    .push(*p),
                PathOp::Curve(b, c, d) => {
                    let subpath = subpaths.last_mut().expect("curve_to without move_to");
                    let a = *subpath.last().unwrap();
                    decompose_spline(&[a, *b, *c, *d], tolerance, subpath);
                }
            }
        }
        subpaths
    }
}

/// `_cairo_spline_error_squared`.
fn spline_error_squared(k: &[Point; 4]) -> f64 {
    let (a, b, c, d) = (k[0], k[1], k[2], k[3]);
    let mut bdx = fixed_to_double(b.x - a.x);
    let mut bdy = fixed_to_double(b.y - a.y);
    let mut cdx = fixed_to_double(c.x - a.x);
    let mut cdy = fixed_to_double(c.y - a.y);
    if a != d {
        let dx = fixed_to_double(d.x - a.x);
        let dy = fixed_to_double(d.y - a.y);
        let v = dx * dx + dy * dy;
        let u = bdx * dx + bdy * dy;
        if u <= 0. {
        } else if u >= v {
            bdx -= dx;
            bdy -= dy;
        } else {
            bdx -= u / v * dx;
            bdy -= u / v * dy;
        }
        let u = cdx * dx + cdy * dy;
        if u <= 0. {
        } else if u >= v {
            cdx -= dx;
            cdy -= dy;
        } else {
            cdx -= u / v * dx;
            cdy -= u / v * dy;
        }
    }
    let berr = bdx * bdx + bdy * bdy;
    let cerr = cdx * cdx + cdy * cdy;
    berr.max(cerr)
}

fn lerp_half(a: Point, b: Point) -> Point {
    Point {
        x: a.x + ((b.x - a.x) >> 1),
        y: a.y + ((b.y - a.y) >> 1),
    }
}

/// `_cairo_spline_decompose`: appends the flattened curve (without its start
/// point) to `out`.
fn decompose_spline(knots: &[Point; 4], tolerance: f64, out: &mut Vec<Point>) {
    fn decompose_into(s1: [Point; 4], tolerance_squared: f64, out: &mut Vec<Point>) {
        if spline_error_squared(&s1) < tolerance_squared {
            add_point(out, s1[0]);
            return;
        }
        // _de_casteljau
        let ab = lerp_half(s1[0], s1[1]);
        let bc = lerp_half(s1[1], s1[2]);
        let cd = lerp_half(s1[2], s1[3]);
        let abbc = lerp_half(ab, bc);
        let bccd = lerp_half(bc, cd);
        let fin = lerp_half(abbc, bccd);
        let first = [s1[0], ab, abbc, fin];
        let second = [fin, bccd, cd, s1[3]];
        decompose_into(first, tolerance_squared, out);
        decompose_into(second, tolerance_squared, out);
    }
    fn add_point(out: &mut Vec<Point>, p: Point) {
        if out.last() != Some(&p) {
            out.push(p);
        }
    }
    decompose_into(*knots, tolerance * tolerance, out);
    add_point(out, knots[3]);
}

// ---------------------------------------------------------------------------
// Rasterization
// ---------------------------------------------------------------------------

/// Sub-scanlines per pixel of Cairo's tor scan converter.
const GRID_Y: i64 = 15;
/// Horizontal resolution (equals the 24.8 fixed point resolution).
const GRID_X: i64 = 256;

#[derive(Clone, Copy, Debug)]
struct Edge {
    // Endpoints in 24.8 fixed point, y0 < y1 after normalization.
    x0: Fixed,
    y0: Fixed,
    x1: Fixed,
    y1: Fixed,
    direction: i32,
}

/// A coverage mask: alpha values for the pixels of a rectangle.
struct Mask {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    alpha: Vec<u8>,
}

/// Rasterizes closed polygons (non-zero winding) into a coverage mask
/// clipped to `[0, clip_width) x [0, clip_height)`.
fn rasterize(polygons: &[Vec<Point>], clip_width: i32, clip_height: i32) -> Option<Mask> {
    let mut edges = Vec::new();
    let (mut min_x, mut min_y, mut max_x, mut max_y) =
        (Fixed::MAX, Fixed::MAX, Fixed::MIN, Fixed::MIN);
    for polygon in polygons {
        if polygon.len() < 3 {
            continue;
        }
        for i in 0..polygon.len() {
            let a = polygon[i];
            let b = polygon[(i + 1) % polygon.len()];
            min_x = min_x.min(a.x);
            max_x = max_x.max(a.x);
            min_y = min_y.min(a.y);
            max_y = max_y.max(a.y);
            if a.y == b.y {
                continue;
            }
            edges.push(if a.y < b.y {
                Edge {
                    x0: a.x,
                    y0: a.y,
                    x1: b.x,
                    y1: b.y,
                    direction: 1,
                }
            } else {
                Edge {
                    x0: b.x,
                    y0: b.y,
                    x1: a.x,
                    y1: a.y,
                    direction: -1,
                }
            });
        }
    }
    if edges.is_empty() {
        return None;
    }
    let px0 = (min_x.div_euclid(256) as i32).max(0);
    let py0 = (min_y.div_euclid(256) as i32).max(0);
    let px1 = ((max_x + 255).div_euclid(256) as i32).min(clip_width);
    let py1 = ((max_y + 255).div_euclid(256) as i32).min(clip_height);
    if px0 >= px1 || py0 >= py1 {
        return None;
    }
    let width = px1 - px0;
    let height = py1 - py0;
    // Edge endpoints in grid rows (tor: `(y * GRID_Y) >> 8`, i.e. floor).
    struct GridEdge {
        top: i64,
        bottom: i64,
        x0: Fixed,
        dx: Fixed,
        dy: i64,
        direction: i32,
    }
    let grid_edges: Vec<GridEdge> = edges
        .iter()
        .filter_map(|e| {
            let top = (e.y0 * GRID_Y).div_euclid(256);
            let bottom = (e.y1 * GRID_Y).div_euclid(256);
            if top == bottom {
                return None;
            }
            Some(GridEdge {
                top,
                bottom,
                x0: e.x0,
                dx: e.x1 - e.x0,
                dy: bottom - top,
                direction: e.direction,
            })
        })
        .collect();

    let full = GRID_X * GRID_Y;
    let mut alpha = vec![0u8; (width * height) as usize];
    let mut coverage = vec![0i64; width as usize + 2];
    let mut crossings: Vec<(Fixed, i32)> = Vec::new();
    for row in 0..height {
        let py = i64::from(py0 + row);
        coverage.iter_mut().for_each(|c| *c = 0);
        let mut any = false;
        for sub in 0..GRID_Y {
            let gy = py * GRID_Y + sub;
            crossings.clear();
            for e in &grid_edges {
                if gy < e.top || gy >= e.bottom {
                    continue;
                }
                // x at the centre of the sub-row.
                let num = (2 * (gy - e.top) + 1) * e.dx;
                let x = e.x0 + num.div_euclid(2 * e.dy);
                crossings.push((x, e.direction));
            }
            if crossings.is_empty() {
                continue;
            }
            crossings.sort_by_key(|c| c.0);
            // Accumulate the spans with non-zero winding.
            let mut winding = 0;
            let mut span_start = 0;
            for &(x, dir) in &crossings {
                let was_inside = winding != 0;
                winding += dir;
                let inside = winding != 0;
                if !was_inside && inside {
                    span_start = x;
                } else if was_inside && !inside {
                    any = true;
                    add_span(
                        &mut coverage,
                        span_start - i64::from(px0) * GRID_X,
                        x - i64::from(px0) * GRID_X,
                        width,
                    );
                }
            }
        }
        if !any {
            continue;
        }
        // `coverage` holds per-pixel partial coverage plus a running
        // difference for fully covered pixels in its upper bits; see
        // `add_span`.
        let mut run = 0i64;
        for x in 0..width as usize {
            run += coverage[x] >> 32;
            let c = (coverage[x] & 0xffff_ffff) + run * GRID_X;
            let a = (c * 255 + full / 2) / full;
            alpha[row as usize * width as usize + x] = a.clamp(0, 255) as u8;
        }
    }
    Some(Mask {
        x: px0,
        y: py0,
        width,
        height,
        alpha,
    })
}

/// Adds the horizontal span `[a, b)` (1/256 pixel units relative to the
/// mask's left edge) to `coverage`. Partial pixel coverage is stored in the
/// low 32 bits, a difference array of fully covered pixels in the high bits.
fn add_span(coverage: &mut [i64], a: i64, b: i64, width: i32) {
    let limit = i64::from(width) * GRID_X;
    let a = a.clamp(0, limit);
    let b = b.clamp(0, limit);
    if a >= b {
        return;
    }
    let pa = (a / GRID_X) as usize;
    let pb = (b / GRID_X) as usize;
    if pa == pb {
        coverage[pa] += b - a;
        return;
    }
    coverage[pa] += GRID_X - (a % GRID_X);
    // Pixels pa + 1 .. pb are fully covered.
    coverage[pa + 1] += 1 << 32;
    coverage[pb] -= 1 << 32;
    coverage[pb] += b % GRID_X;
}

// ---------------------------------------------------------------------------
// Stroking
// ---------------------------------------------------------------------------

/// Default miter limit of Cairo.
const MITER_LIMIT: f64 = 10.;
/// Default tolerance of Cairo.
const TOLERANCE: f64 = 0.1;

fn signed_area(polygon: &[(f64, f64)]) -> f64 {
    let mut area = 0.;
    for i in 0..polygon.len() {
        let (x0, y0) = polygon[i];
        let (x1, y1) = polygon[(i + 1) % polygon.len()];
        area += x0 * y1 - x1 * y0;
    }
    area / 2.
}

/// Builds the stroke outline of a polyline (device space points) as a set of
/// positively oriented polygons whose non-zero union is the stroke. The pen
/// is `half_width` in user space, transformed by `ctm`.
fn stroke_polygons(points: &[(f64, f64)], half_width: f64, ctm: &Matrix) -> Vec<Vec<Point>> {
    let mut polygons: Vec<Vec<(f64, f64)>> = Vec::new();
    // Work in user space so that the pen is circular there, then map the
    // outline back to device space.
    let Some(ctm_inverse) = ctm.invert() else {
        return Vec::new();
    };
    let user: Vec<(f64, f64)> = points
        .iter()
        .map(|&(x, y)| ctm_inverse.transform_point(x, y))
        .collect();
    let mut directions = Vec::new();
    for w in user.windows(2) {
        let (dx, dy) = (w[1].0 - w[0].0, w[1].1 - w[0].1);
        let len = (dx * dx + dy * dy).sqrt();
        directions.push((dx / len, dy / len));
    }
    for (i, w) in user.windows(2).enumerate() {
        let (ux, uy) = directions[i];
        let (nx, ny) = (-uy * half_width, ux * half_width);
        polygons.push(vec![
            (w[0].0 + nx, w[0].1 + ny),
            (w[1].0 + nx, w[1].1 + ny),
            (w[1].0 - nx, w[1].1 - ny),
            (w[0].0 - nx, w[0].1 - ny),
        ]);
    }
    for i in 1..user.len() - 1 {
        let p = user[i];
        let d_in = directions[i - 1];
        let d_out = directions[i];
        let cross = d_in.0 * d_out.1 - d_in.1 * d_out.0;
        if cross == 0. {
            continue;
        }
        // The outer side is to the right of the turn for a left turn.
        let sign = if cross > 0. { -1. } else { 1. };
        let n_in = (-d_in.1 * sign, d_in.0 * sign);
        let n_out = (-d_out.1 * sign, d_out.0 * sign);
        let a = (p.0 + n_in.0 * half_width, p.1 + n_in.1 * half_width);
        let b = (p.0 + n_out.0 * half_width, p.1 + n_out.1 * half_width);
        let dot = d_in.0 * d_out.0 + d_in.1 * d_out.1;
        if MITER_LIMIT * MITER_LIMIT * (1. + dot) >= 2. {
            let denom = 1. + (n_in.0 * n_out.0 + n_in.1 * n_out.1);
            let m = (
                p.0 + (n_in.0 + n_out.0) * half_width / denom,
                p.1 + (n_in.1 + n_out.1) * half_width / denom,
            );
            polygons.push(vec![p, a, m, b]);
        } else {
            polygons.push(vec![p, a, b]);
        }
    }
    polygons
        .into_iter()
        .map(|polygon| {
            let device: Vec<(f64, f64)> = polygon
                .iter()
                .map(|&(x, y)| ctm.transform_point(x, y))
                .collect();
            let mut fixed: Vec<Point> = device
                .iter()
                .map(|&(x, y)| Point {
                    x: fixed_from_double(x),
                    y: fixed_from_double(y),
                })
                .collect();
            if signed_area(&device) < 0. {
                fixed.reverse();
            }
            fixed
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// The source pattern of a context.
#[derive(Clone, Copy, Debug)]
enum Source<'p> {
    /// Premultiplied ARGB32 pixel of a solid color.
    Solid(u32),
    /// A surface pattern; `device_to_pattern` maps device space to the
    /// surface's pixel space.
    Surface {
        surface: &'p Surface,
        device_to_pattern: Matrix,
    },
}

#[derive(Clone, Copy, Debug)]
struct GState<'p> {
    ctm: Matrix,
    source: Source<'p>,
    line_width: f64,
}

/// A drawing context on a [`Surface`] (`cairo_t`).
///
/// `'p` is the lifetime of surfaces used as sources.
pub struct Context<'s, 'p> {
    target: &'s mut Surface,
    gstate: GState<'p>,
    stack: Vec<GState<'p>>,
    path: Path,
}

impl<'s, 'p> Context<'s, 'p> {
    /// `cairo_create`: identity transformation, opaque black source, line
    /// width 2.
    pub fn new(target: &'s mut Surface) -> Self {
        Self {
            target,
            gstate: GState {
                ctm: Matrix::identity(),
                source: Source::Solid(0xff00_0000),
                line_width: 2.,
            },
            stack: Vec::new(),
            path: Path::default(),
        }
    }

    /// The target surface.
    pub fn target(&mut self) -> &mut Surface {
        self.target
    }

    /// `cairo_save`.
    pub fn save(&mut self) {
        self.stack.push(self.gstate);
    }

    /// `cairo_restore`.
    pub fn restore(&mut self) {
        self.gstate = self
            .stack
            .pop()
            .expect("cairo_restore() without matching cairo_save()");
    }

    /// The current transformation matrix (`cairo_get_matrix`).
    pub fn matrix(&self) -> Matrix {
        self.gstate.ctm
    }

    /// `cairo_transform`: applies `matrix` before the current transformation.
    pub fn transform(&mut self, matrix: &Matrix) {
        self.gstate.ctm = Matrix::multiply(matrix, &self.gstate.ctm);
    }

    /// `cairo_translate`.
    pub fn translate(&mut self, tx: f64, ty: f64) {
        self.transform(&Matrix::translation(tx, ty));
    }

    /// `cairo_scale`.
    pub fn scale(&mut self, sx: f64, sy: f64) {
        self.transform(&Matrix::scaling(sx, sy));
    }

    /// `cairo_user_to_device`.
    pub fn user_to_device(&self, x: f64, y: f64) -> (f64, f64) {
        self.gstate.ctm.transform_point(x, y)
    }

    /// `cairo_set_source_rgba`.
    pub fn set_source_rgba(&mut self, red: f64, green: f64, blue: f64, alpha: f64) {
        self.gstate.source = Source::Solid(color_to_pixel(red, green, blue, alpha));
    }

    /// `cairo_set_source_surface`: user space point `(x, y)` maps to the
    /// surface origin; the pattern is locked to the current user space.
    pub fn set_source_surface(&mut self, surface: &'p Surface, x: f64, y: f64) {
        let ctm_inverse = self
            .gstate
            .ctm
            .invert()
            .expect("cairo: non-invertible transformation matrix");
        self.gstate.source = Source::Surface {
            surface,
            device_to_pattern: Matrix::multiply(&ctm_inverse, &Matrix::translation(-x, -y)),
        };
    }

    /// `cairo_set_line_width` (in user space units).
    pub fn set_line_width(&mut self, width: f64) {
        self.gstate.line_width = width;
    }

    fn to_device_fixed(&self, x: f64, y: f64) -> Point {
        let (dx, dy) = self.gstate.ctm.transform_point(x, y);
        Point {
            x: fixed_from_double(dx),
            y: fixed_from_double(dy),
        }
    }

    /// `cairo_move_to`.
    pub fn move_to(&mut self, x: f64, y: f64) {
        let p = self.to_device_fixed(x, y);
        self.path.move_to(p);
    }

    /// `cairo_line_to`; acts as `move_to` if there is no current point.
    pub fn line_to(&mut self, x: f64, y: f64) {
        let p = self.to_device_fixed(x, y);
        self.path.line_to(p);
    }

    /// `cairo_curve_to`.
    pub fn curve_to(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64) {
        let p1 = self.to_device_fixed(x1, y1);
        let p2 = self.to_device_fixed(x2, y2);
        let p3 = self.to_device_fixed(x3, y3);
        self.path.curve_to(p1, p2, p3);
    }

    /// `cairo_arc` (positive direction), approximated by Bezier splines
    /// exactly like `cairo-arc.c`.
    pub fn arc(&mut self, xc: f64, yc: f64, radius: f64, angle1: f64, mut angle2: f64) {
        if angle2 < angle1 {
            // Increase angle2 by multiples of full circle until it satisfies
            // angle2 >= angle1.
            angle2 = (angle2 - angle1) % (2. * PI);
            if angle2 < 0. {
                angle2 += 2. * PI;
            }
            angle2 += angle1;
        }
        self.line_to(xc + radius * angle1.cos(), yc + radius * angle1.sin());
        if radius <= 0. {
            return;
        }
        self.arc_in_direction(xc, yc, radius, angle1, angle2);
    }

    fn arc_in_direction(&mut self, xc: f64, yc: f64, radius: f64, angle_min: f64, angle_max: f64) {
        if angle_max - angle_min > PI {
            let angle_mid = angle_min + (angle_max - angle_min) / 2.;
            self.arc_in_direction(xc, yc, radius, angle_min, angle_mid);
            self.arc_in_direction(xc, yc, radius, angle_mid, angle_max);
            return;
        }
        if angle_max == angle_min {
            return;
        }
        let mut segments = self.arc_segments_needed(angle_max - angle_min, radius);
        let step = (angle_max - angle_min) / f64::from(segments);
        segments -= 1;
        self.line_to(xc + radius * angle_min.cos(), yc + radius * angle_min.sin());
        let mut angle = angle_min;
        for _ in 0..segments {
            self.arc_segment(xc, yc, radius, angle, angle + step);
            angle += step;
        }
        self.arc_segment(xc, yc, radius, angle, angle_max);
    }

    fn arc_segments_needed(&self, angle: f64, radius: f64) -> i32 {
        // `_cairo_matrix_transformed_circle_major_axis`.
        let m = &self.gstate.ctm;
        let (a, b, c, d) = (m.xx, m.yx, m.xy, m.yy);
        let i = a * a + b * b;
        let j = c * c + d * d;
        let f = 0.5 * (i + j);
        let g = 0.5 * (i - j);
        let h = a * c + b * d;
        let major_axis = radius * (f + g.hypot(h)).sqrt();
        let max_angle = arc_max_angle_for_tolerance_normalized(TOLERANCE / major_axis);
        (angle.abs() / max_angle).ceil() as i32
    }

    fn arc_segment(&mut self, xc: f64, yc: f64, radius: f64, angle_a: f64, angle_b: f64) {
        let r_sin_a = radius * angle_a.sin();
        let r_cos_a = radius * angle_a.cos();
        let r_sin_b = radius * angle_b.sin();
        let r_cos_b = radius * angle_b.cos();
        let h = 4.0 / 3.0 * ((angle_b - angle_a) / 4.0).tan();
        self.curve_to(
            xc + r_cos_a - h * r_sin_a,
            yc + r_sin_a + h * r_cos_a,
            xc + r_cos_b + h * r_sin_b,
            yc + r_sin_b - h * r_cos_b,
            xc + r_cos_b,
            yc + r_sin_b,
        );
    }

    /// `cairo_stroke`: strokes the current path with the current line width
    /// (butt caps, miter joins) and clears the path.
    pub fn stroke(&mut self) {
        let subpaths = self.path.flatten(TOLERANCE);
        self.path.clear();
        let half_width = self.gstate.line_width / 2.;
        let mut polygons = Vec::new();
        for subpath in subpaths {
            // Remove degenerate segments.
            let mut points: Vec<Point> = Vec::with_capacity(subpath.len());
            for p in subpath {
                if points.last() != Some(&p) {
                    points.push(p);
                }
            }
            if points.len() < 2 {
                // Butt caps: degenerate sub-paths draw nothing.
                continue;
            }
            let device: Vec<(f64, f64)> = points
                .iter()
                .map(|p| (fixed_to_double(p.x), fixed_to_double(p.y)))
                .collect();
            polygons.extend(stroke_polygons(&device, half_width, &self.gstate.ctm));
        }
        if let Some(mask) = rasterize(&polygons, self.target.width, self.target.height) {
            self.composite_with_mask(&mask);
        }
    }

    /// `cairo_fill`: fills the current path (non-zero winding) and clears the
    /// path.
    pub fn fill(&mut self) {
        let subpaths = self.path.flatten(TOLERANCE);
        self.path.clear();
        if let Some(mask) = rasterize(&subpaths, self.target.width, self.target.height) {
            self.composite_with_mask(&mask);
        }
    }

    /// `cairo_paint`: composites the source over the whole surface.
    pub fn paint(&mut self) {
        let (width, height) = (self.target.width, self.target.height);
        match self.gstate.source {
            Source::Solid(pixel) => {
                for d in &mut self.target.data {
                    *d = over(pixel, *d);
                }
            }
            Source::Surface {
                surface,
                device_to_pattern,
            } => {
                // Restrict to the device-space extents of the source (it is
                // transparent outside, EXTEND_NONE), padded for filtering.
                let Some(pattern_to_device) = device_to_pattern.invert() else {
                    return;
                };
                let (sw, sh) = (f64::from(surface.width), f64::from(surface.height));
                let corners = [
                    (-1., -1.),
                    (sw + 1., -1.),
                    (-1., sh + 1.),
                    (sw + 1., sh + 1.),
                ];
                let (mut x_min, mut y_min, mut x_max, mut y_max) = (
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                );
                for (x, y) in corners {
                    let (dx, dy) = pattern_to_device.transform_point(x, y);
                    x_min = x_min.min(dx);
                    y_min = y_min.min(dy);
                    x_max = x_max.max(dx);
                    y_max = y_max.max(dy);
                }
                let x0 = (x_min.floor().max(0.)) as i32;
                let y0 = (y_min.floor().max(0.)) as i32;
                let x1 = (x_max.ceil().min(f64::from(width))) as i32;
                let y1 = (y_max.ceil().min(f64::from(height))) as i32;
                let transform = PixmanTransform::from_matrix(&device_to_pattern);
                for y in y0..y1 {
                    for x in x0..x1 {
                        let (px, py) = transform.transform_point(
                            (i64::from(x) << 16) + 0x8000,
                            (i64::from(y) << 16) + 0x8000,
                        );
                        let s = fetch_bilinear(surface, px, py);
                        if s == 0 {
                            continue;
                        }
                        let index = (y * width + x) as usize;
                        self.target.data[index] = over(s, self.target.data[index]);
                    }
                }
            }
        }
    }

    fn composite_with_mask(&mut self, mask: &Mask) {
        let width = self.target.width;
        let source = self.gstate.source;
        let transform = match source {
            Source::Surface {
                device_to_pattern, ..
            } => Some(PixmanTransform::from_matrix(&device_to_pattern)),
            Source::Solid(_) => None,
        };
        for my in 0..mask.height {
            for mx in 0..mask.width {
                let m = u32::from(mask.alpha[(my * mask.width + mx) as usize]);
                if m == 0 {
                    continue;
                }
                let (x, y) = (mask.x + mx, mask.y + my);
                let s = match source {
                    Source::Solid(pixel) => pixel,
                    Source::Surface { surface, .. } => {
                        let (px, py) = transform.unwrap().transform_point(
                            (i64::from(x) << 16) + 0x8000,
                            (i64::from(y) << 16) + 0x8000,
                        );
                        fetch_bilinear(surface, px, py)
                    }
                };
                let s = if m == 255 { s } else { mul_un8x4(s, m) };
                let index = (y * width + x) as usize;
                self.target.data[index] = over(s, self.target.data[index]);
            }
        }
    }
}

/// `_arc_max_angle_for_tolerance_normalized` (table from `cairo-arc.c`).
fn arc_max_angle_for_tolerance_normalized(tolerance: f64) -> f64 {
    // The literals are copied verbatim from cairo-arc.c.
    #[allow(clippy::excessive_precision)]
    const TABLE: [(f64, f64); 11] = [
        (PI / 1.0, 0.0185185185185185036127),
        (PI / 2.0, 0.000272567143730179811158),
        (PI / 3.0, 2.38647043651461047433e-05),
        (PI / 4.0, 4.2455377443222443279e-06),
        (PI / 5.0, 1.11281001494389081528e-06),
        (PI / 6.0, 3.72662000942734705475e-07),
        (PI / 7.0, 1.47783685574284411325e-07),
        (PI / 8.0, 6.63240432022601149057e-08),
        (PI / 9.0, 3.2715520137536980553e-08),
        (PI / 10.0, 1.73863223499021216974e-08),
        (PI / 11.0, 9.81410988043554039085e-09),
    ];
    for &(angle, error) in &TABLE {
        if error < tolerance {
            return angle;
        }
    }
    // Fall back to searching (C++: `do { angle = M_PI / i++; error = ...; }
    // while (error > tolerance && i < 100)`).
    let mut i = TABLE.len() + 1;
    loop {
        let angle = PI / i as f64;
        i += 1;
        if arc_error_normalized(angle) <= tolerance || i >= 100 {
            return angle;
        }
    }
}

/// `_arc_error_normalized`.
fn arc_error_normalized(angle: f64) -> f64 {
    2.0 / 27.0 * (angle / 4.0).sin().powi(6) / (angle / 4.0).cos().powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_multiply_order_and_invert() {
        // scale then translate: (1, 1) -> (2, 2) -> (12, 2)
        let m = Matrix::multiply(&Matrix::scaling(2., 2.), &Matrix::translation(10., 0.));
        assert_eq!(m.transform_point(1., 1.), (12., 2.));
        let inv = m.invert().unwrap();
        assert_eq!(inv.transform_point(12., 2.), (1., 1.));
        let r = Matrix::new(0., 1., -1., 0., 3., 4.);
        let (x, y) = r
            .invert()
            .unwrap()
            .transform_point(r.transform_point(5., 7.).0, r.transform_point(5., 7.).1);
        assert!((x - 5.).abs() < 1e-12 && (y - 7.).abs() < 1e-12);
        assert!(Matrix::scaling(0., 1.).invert().is_none());
    }

    #[test]
    fn context_transform_composition_matches_cairo() {
        let mut surface = Surface::new(1, 1);
        let mut cr = Context::new(&mut surface);
        cr.translate(5., 6.);
        cr.scale(2., 3.);
        // user (1, 1) -> scaled (2, 3) -> translated (7, 9)
        assert_eq!(cr.user_to_device(1., 1.), (7., 9.));
        cr.save();
        cr.scale(10., 10.);
        assert_eq!(cr.user_to_device(1., 1.), (25., 36.));
        cr.restore();
        assert_eq!(cr.user_to_device(1., 1.), (7., 9.));
    }

    #[test]
    fn solid_colors_are_converted_like_cairo() {
        assert_eq!(color_to_pixel(0.5, 0., 0., 1.), 0xff80_0000);
        // 0.7 * 65535 + 0.5 = 45875 -> 179.
        assert_eq!(color_to_pixel(0., 1., 0., 0.7) >> 24, 179);
        assert_eq!((color_to_pixel(0., 1., 0., 0.7) >> 8) & 0xff, 179);
    }

    #[test]
    fn over_operator() {
        assert_eq!(over(0xff12_3456, 0x8000_0000), 0xff12_3456);
        assert_eq!(over(0, 0x8011_2233), 0x8011_2233);
        // 50% black over opaque white.
        assert_eq!(over(0x8000_0000, 0xffff_ffff), 0xff7f_7f7f);
        // Invalid premultiplied source saturates.
        assert_eq!(over(0x10ff_0000, 0xff80_0000), 0xffff_0000);
    }

    #[test]
    fn paint_solid_fills_everything() {
        let mut surface = Surface::new(3, 2);
        let mut cr = Context::new(&mut surface);
        cr.set_source_rgba(0.5, 0.0, 0.0, 1.);
        cr.paint();
        assert!(surface.data().iter().all(|&p| p == 0xff80_0000));
    }

    #[test]
    fn fill_axis_aligned_rectangle() {
        let mut surface = Surface::new(6, 6);
        let mut cr = Context::new(&mut surface);
        cr.set_source_rgba(1., 1., 1., 1.);
        cr.move_to(1., 1.);
        cr.line_to(4., 1.);
        cr.line_to(4., 3.5);
        cr.line_to(1., 3.5);
        cr.fill();
        for y in 0..6 {
            for x in 0..6 {
                let p = surface.pixel(x, y);
                let expected = if (1..4).contains(&x) && (1..3).contains(&y) {
                    0xffff_ffff
                } else if (1..4).contains(&x) && y == 3 {
                    // Half covered: 7 or 8 of 15 sub-scanlines.
                    p
                } else {
                    0
                };
                assert_eq!(p, expected, "pixel ({x}, {y})");
            }
        }
        let half = surface.pixel(2, 3) >> 24;
        assert!((119..=137).contains(&half), "{half}");
    }

    #[test]
    fn stroke_horizontal_line() {
        let mut surface = Surface::new(10, 10);
        let mut cr = Context::new(&mut surface);
        cr.set_source_rgba(0., 0., 1., 1.);
        cr.set_line_width(4.);
        cr.line_to(2., 5.);
        cr.line_to(8., 5.);
        cr.stroke();
        for y in 0..10 {
            for x in 0..10 {
                let inside = (2..8).contains(&x) && (3..7).contains(&y);
                assert_eq!(
                    surface.pixel(x, y),
                    if inside { 0xff00_00ff } else { 0 },
                    "pixel ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn stroke_of_single_point_draws_nothing() {
        let mut surface = Surface::new(5, 5);
        let mut cr = Context::new(&mut surface);
        cr.set_line_width(4.);
        cr.line_to(2., 2.);
        cr.line_to(2., 2.);
        cr.stroke();
        assert!(surface.data().iter().all(|&p| p == 0));
    }

    #[test]
    fn stroke_overlap_is_not_double_blended() {
        // A path going back and forth over itself must be composited once.
        let mut once = Surface::new(12, 12);
        {
            let mut cr = Context::new(&mut once);
            cr.set_source_rgba(1., 0., 0., 0.5);
            cr.set_line_width(4.);
            cr.line_to(2., 6.);
            cr.line_to(10., 6.);
            cr.stroke();
        }
        let mut twice = Surface::new(12, 12);
        {
            let mut cr = Context::new(&mut twice);
            cr.set_source_rgba(1., 0., 0., 0.5);
            cr.set_line_width(4.);
            cr.line_to(2., 6.);
            cr.line_to(10., 6.);
            cr.line_to(2., 6.);
            cr.stroke();
        }
        assert_eq!(once, twice);
    }

    #[test]
    fn arc_fill_is_round_and_symmetric() {
        let mut surface = Surface::new(20, 20);
        let mut cr = Context::new(&mut surface);
        cr.set_source_rgba(0., 1., 0., 1.);
        cr.arc(10., 10., 6., 0., 2. * PI);
        cr.fill();
        let alpha = |x: i32, y: i32| surface.pixel(x, y) >> 24;
        // Fully covered centre, empty corners.
        assert_eq!(alpha(10, 10), 255);
        assert_eq!(alpha(9, 9), 255);
        assert_eq!(alpha(4, 4), 0);
        assert_eq!(alpha(0, 0), 0);
        // Total coverage ~ pi r^2.
        let area: f64 = (0..20)
            .flat_map(|y| (0..20).map(move |x| (x, y)))
            .map(|(x, y)| f64::from(alpha(x, y)) / 255.)
            .sum();
        assert!((area - PI * 36.).abs() < 1.0, "{area}");
        // Mirror symmetry around the centre (x -> 19 - x).
        for y in 0..20 {
            for x in 0..20 {
                let a = alpha(x, y) as i32;
                let b = alpha(19 - x, y) as i32;
                assert!((a - b).abs() <= 18, "({x},{y}) {a} vs {b}");
            }
        }
    }

    #[test]
    fn paint_surface_identity_and_translation() {
        let source = Surface::from_data(vec![0xff11_2233, 0x8040_0000, 0, 0xffff_ffff], 2, 2);
        let mut target = Surface::new(4, 4);
        {
            let mut cr = Context::new(&mut target);
            cr.translate(1., 2.);
            cr.set_source_surface(&source, 0., 0.);
            cr.paint();
        }
        assert_eq!(target.pixel(1, 2), 0xff11_2233);
        assert_eq!(target.pixel(2, 2), 0x8040_0000);
        assert_eq!(target.pixel(1, 3), 0);
        assert_eq!(target.pixel(2, 3), 0xffff_ffff);
        assert_eq!(target.pixel(0, 0), 0);
        assert_eq!(target.pixel(3, 2), 0);
    }

    #[test]
    fn paint_surface_bilinear_half_pixel_shift() {
        // Shifting by half a pixel averages neighbours with pixman weights.
        let source = Surface::from_data(vec![0xff00_0000, 0xffff_ffff], 2, 1);
        let mut target = Surface::new(3, 1);
        {
            let mut cr = Context::new(&mut target);
            cr.translate(0.5, 0.);
            cr.set_source_surface(&source, 0., 0.);
            cr.paint();
        }
        // Left: 50% black (edge, transparent outside). Middle: 50/50 mix.
        // pixman truncates: 255 * 128 * 256 >> 16 = 127.
        assert_eq!(target.pixel(0, 0), 0x7f00_0000);
        assert_eq!(target.pixel(1, 0), 0xff7f_7f7f);
        assert_eq!(target.pixel(2, 0), 0x7f7f_7f7f);
    }

    #[test]
    fn png_round_trip_opaque_and_transparent() {
        let opaque = Surface::from_data(vec![0xff01_0203, 0xffff_ffff], 2, 1);
        let bytes = opaque.encode_png().unwrap();
        let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        let mut reader = decoder.read_info().unwrap();
        assert_eq!(reader.info().color_type, png::ColorType::Rgb);
        assert!(reader.info().bkgd.is_some());
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!(&buf[..info.buffer_size()], &[1, 2, 3, 255, 255, 255]);

        // Premultiplied (128, 64, 0, 0) at alpha 128 -> (128 * 255 + 64) / 128.
        let translucent = Surface::from_data(vec![0x8080_4000, 0, 0x10ff_0000], 3, 1);
        let bytes = translucent.encode_png().unwrap();
        let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        let mut reader = decoder.read_info().unwrap();
        assert_eq!(reader.info().color_type, png::ColorType::Rgba);
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        // Invalid premultiplied pixel: (255 * 255 + 8) / 16 = 4064 -> wraps to 224.
        assert_eq!(
            &buf[..info.buffer_size()],
            &[255, 128, 0, 128, 0, 0, 0, 0, 224, 0, 0, 16]
        );
    }

    #[test]
    fn png_write_error_is_reported() {
        let surface = Surface::new(1, 1);
        assert_eq!(
            surface.write_to_png_stream(&mut |_| false),
            Status::WriteError
        );
        assert_eq!(surface.write_to_png_stream(&mut |_| true), Status::Success);
    }
}
