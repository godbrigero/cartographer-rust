//! Port of `cartographer/io/image` (.h, .cc).
//!
//! RUST PORT NOTE: the C++ `UniqueCairoSurfacePtr`/`UniqueCairoPtr` wrappers
//! are replaced by the owned [`Surface`] and the borrowing
//! [`crate::io::cairo::Context`] of the pure-Rust Cairo emulation in
//! [`crate::io::cairo`].

use crate::io::cairo::{Status, Surface, format_stride_for_width};
use crate::io::color::Uint8Color;
use crate::io::file_writer::FileWriter;

fn uint8_color_to_cairo(color: &Uint8Color) -> u32 {
    (255u32 << 24) | (u32::from(color[0]) << 16) | (u32::from(color[1]) << 8) | u32::from(color[2])
}

fn cairo_to_uint8_color(color: u32) -> Uint8Color {
    [(color >> 16) as u8, (color >> 8) as u8, color as u8]
}

fn check_stride_is_as_expected(width: i32) {
    let stride = format_stride_for_width(width);
    assert_eq!(stride, width * 4, "Check failed: stride == width * 4");
}

/// An ARGB32 image (Cairo pixel layout, premultiplied alpha).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    // RUST PORT NOTE: C++ stores `width_`, `height_` and
    // `std::vector<uint32> pixels_` and creates aliasing Cairo surfaces on
    // demand; here the image owns a `Surface` holding exactly that data.
    surface: Surface,
}

impl Image {
    /// Creates an image with all pixels set to 0 (transparent black).
    pub fn new(width: i32, height: i32) -> Self {
        Self {
            surface: Surface::new(width, height),
        }
    }

    /// Creates an image copying the pixels of an ARGB32 `surface`.
    pub fn from_surface(surface: Surface) -> Self {
        check_stride_is_as_expected(surface.width());
        Self { surface }
    }

    /// Returns the RGB color of the pixel at `(x, y)` (alpha is ignored).
    pub fn get_pixel(&self, x: i32, y: i32) -> Uint8Color {
        cairo_to_uint8_color(self.surface.pixel(x, y))
    }

    /// Sets the pixel at `(x, y)` to the opaque `color`.
    pub fn set_pixel(&mut self, x: i32, y: i32, color: &Uint8Color) {
        let width = self.surface.width();
        self.surface.data_mut()[(y * width + x) as usize] = uint8_color_to_cairo(color);
    }

    /// Writes the image as PNG to `file_writer`.
    pub fn write_png(&self, file_writer: &mut dyn FileWriter) {
        let status = self
            .surface
            .write_to_png_stream(&mut |data: &[u8]| file_writer.write(data));
        assert_eq!(
            status,
            Status::Success,
            "Check failed: cairo_surface_write_to_png_stream(...) == CAIRO_STATUS_SUCCESS"
        );
    }

    /// Rotates the image in place.
    pub fn rotate_90_degrees_clockwise(&mut self) {
        let (width, height) = (self.surface.width(), self.surface.height());
        let old_pixels = self.surface.data();
        let mut pixels = Vec::with_capacity(old_pixels.len());
        for x in 0..width {
            for y in (0..height).rev() {
                pixels.push(old_pixels[(y * width + x) as usize]);
            }
        }
        self.surface = Surface::from_data(pixels, height, width);
    }

    /// Returns the Cairo surface holding the pixel data. Drawing into it
    /// modifies the image (like the aliasing surface returned in C++).
    pub fn get_cairo_surface(&mut self) -> &mut Surface {
        &mut self.surface
    }

    /// Read-only access to the surface.
    pub fn surface(&self) -> &Surface {
        &self.surface
    }

    /// Width in pixels.
    pub fn width(&self) -> i32 {
        self.surface.width()
    }

    /// Height in pixels.
    pub fn height(&self) -> i32 {
        self.surface.height()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::io::fake_file_writer::FakeFileWriter;

    #[test]
    fn set_and_get_pixel() {
        let mut image = Image::new(3, 2);
        assert_eq!(image.get_pixel(1, 1), [0, 0, 0]);
        image.set_pixel(1, 1, &[10, 20, 30]);
        assert_eq!(image.get_pixel(1, 1), [10, 20, 30]);
        assert_eq!(image.surface().pixel(1, 1), 0xff0a_141e);
        assert_eq!(image.surface().pixel(0, 0), 0);
    }

    #[test]
    fn rotate_90_degrees_clockwise() {
        // 3x2 image:
        //   a b c
        //   d e f
        let mut image = Image::new(3, 2);
        for (i, (x, y)) in [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
            .into_iter()
            .enumerate()
        {
            image.set_pixel(x, y, &[i as u8, 0, 0]);
        }
        image.rotate_90_degrees_clockwise();
        // Rotated 2x3:
        //   d a
        //   e b
        //   f c
        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 3);
        let r = |x, y| image.get_pixel(x, y)[0];
        assert_eq!([r(0, 0), r(1, 0)], [3, 0]);
        assert_eq!([r(0, 1), r(1, 1)], [4, 1]);
        assert_eq!([r(0, 2), r(1, 2)], [5, 2]);
    }

    #[test]
    fn write_png_decodes_to_same_pixels() {
        let mut image = Image::new(2, 2);
        image.set_pixel(0, 0, &[255, 0, 0]);
        image.set_pixel(1, 0, &[0, 255, 0]);
        image.set_pixel(0, 1, &[0, 0, 255]);
        image.set_pixel(1, 1, &[1, 2, 3]);
        let content = Arc::new(Mutex::new(Vec::new()));
        let mut writer = FakeFileWriter::new("image.png", Arc::clone(&content));
        image.write_png(&mut writer);
        let bytes = content.lock().unwrap().clone();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!((info.width, info.height), (2, 2));
        assert_eq!(info.color_type, png::ColorType::Rgb);
        assert_eq!(
            &buf[..info.buffer_size()],
            &[255, 0, 0, 0, 255, 0, 0, 0, 255, 1, 2, 3]
        );
    }
}
