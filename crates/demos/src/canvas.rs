//! Minimal RGB raster with line drawing, discs and a 3×5 bitmap font,
//! saved as 8-bit RGB PNG.

use std::io::BufWriter;
use std::path::Path;

/// An RGB color.
pub type Color = [u8; 3];

/// Common colors.
pub mod colors {
    use super::Color;
    /// White.
    pub const WHITE: Color = [255, 255, 255];
    /// Black.
    pub const BLACK: Color = [0, 0, 0];
    /// Light gray (grid lines).
    pub const GRID: Color = [225, 225, 225];
    /// Ground truth.
    pub const GREEN: Color = [0, 150, 60];
    /// Optimized estimate.
    pub const RED: Color = [220, 30, 30];
    /// Local-SLAM-only estimate.
    pub const BLUE: Color = [40, 90, 220];
    /// Loop closures.
    pub const MAGENTA: Color = [200, 0, 200];
    /// Other inter-submap constraints.
    pub const ORANGE: Color = [240, 150, 30];
}

/// An RGB image.
#[derive(Clone, Debug)]
pub struct Canvas {
    width: usize,
    height: usize,
    pixels: Vec<Color>,
}

impl Canvas {
    /// A `width × height` canvas filled with `background`.
    pub fn new(width: usize, height: usize, background: Color) -> Self {
        Canvas {
            width,
            height,
            pixels: vec![background; width * height],
        }
    }

    /// Width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Sets a pixel (ignored outside the canvas).
    pub fn set(&mut self, x: i64, y: i64, color: Color) {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            self.pixels[y as usize * self.width + x as usize] = color;
        }
    }

    /// Fills the axis-aligned rectangle `[x0, x1) × [y0, y1)`.
    pub fn fill_rect(&mut self, x0: i64, y0: i64, x1: i64, y1: i64, color: Color) {
        for y in y0.max(0)..y1.min(self.height as i64) {
            for x in x0.max(0)..x1.min(self.width as i64) {
                self.set(x, y, color);
            }
        }
    }

    /// A one-pixel line (Bresenham).
    pub fn line(&mut self, from: (f64, f64), to: (f64, f64), color: Color) {
        let (mut x0, mut y0) = (from.0.round() as i64, from.1.round() as i64);
        let (x1, y1) = (to.0.round() as i64, to.1.round() as i64);
        // Skip absurd lines (e.g. a pose far outside the view).
        if (x1 - x0).abs().max((y1 - y0).abs()) > 100_000 {
            return;
        }
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut error = dx + dy;
        loop {
            self.set(x0, y0, color);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * error;
            if e2 >= dy {
                error += dy;
                x0 += sx;
            }
            if e2 <= dx {
                error += dx;
                y0 += sy;
            }
        }
    }

    /// A line `width` pixels thick (a square brush along a Bresenham line).
    pub fn thick_line(&mut self, from: (f64, f64), to: (f64, f64), width: u32, color: Color) {
        let r = f64::from(width.saturating_sub(1)) / 2.0;
        let steps = width.max(1);
        for i in 0..steps {
            for j in 0..steps {
                let ox = f64::from(i) - r;
                let oy = f64::from(j) - r;
                self.line((from.0 + ox, from.1 + oy), (to.0 + ox, to.1 + oy), color);
            }
        }
    }

    /// A polyline through `points`.
    pub fn polyline(&mut self, points: &[(f64, f64)], width: u32, color: Color) {
        for pair in points.windows(2) {
            self.thick_line(pair[0], pair[1], width, color);
        }
    }

    /// A filled disc.
    pub fn disc(&mut self, center: (f64, f64), radius: f64, color: Color) {
        let r = radius.ceil() as i64;
        let (cx, cy) = (center.0.round() as i64, center.1.round() as i64);
        for dy in -r..=r {
            for dx in -r..=r {
                if ((dx * dx + dy * dy) as f64) <= radius * radius {
                    self.set(cx + dx, cy + dy, color);
                }
            }
        }
    }

    /// Draws `text` with the top-left corner at `(x, y)` using the 3×5 font
    /// magnified `scale` times. Lowercase letters are drawn as uppercase;
    /// unknown characters as blanks. Returns the x after the text.
    pub fn text(&mut self, x: i64, y: i64, text: &str, scale: i64, color: Color) -> i64 {
        let mut cursor = x;
        for c in text.chars() {
            if let Some(rows) = glyph(c.to_ascii_uppercase()) {
                for (row, bits) in rows.iter().enumerate() {
                    for col in 0..3 {
                        if bits & (0b100 >> col) != 0 {
                            let px = cursor + col * scale;
                            let py = y + row as i64 * scale;
                            self.fill_rect(px, py, px + scale, py + scale, color);
                        }
                    }
                }
            }
            cursor += 4 * scale;
        }
        cursor
    }

    /// Width in pixels of `text` drawn with [`text`](Self::text).
    pub fn text_width(text: &str, scale: i64) -> i64 {
        (text.chars().count() as i64 * 4 - 1).max(0) * scale
    }

    /// Writes the canvas as an 8-bit RGB PNG.
    pub fn save_png(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut encoder =
            png::Encoder::new(BufWriter::new(file), self.width as u32, self.height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(std::io::Error::other)?;
        let data: Vec<u8> = self.pixels.iter().flatten().copied().collect();
        writer
            .write_image_data(&data)
            .map_err(std::io::Error::other)?;
        writer.finish().map_err(std::io::Error::other)
    }
}

/// Maps world coordinates `(u, v)` (meters, `v` up) to pixels with a uniform
/// scale and a margin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// World `u` at the left edge of the plot area.
    pub min_u: f64,
    /// World `v` at the top edge of the plot area.
    pub max_v: f64,
    /// Pixels per meter.
    pub scale: f64,
    /// Margin (pixels) around the plot area.
    pub margin: f64,
    /// Plot area width (pixels).
    pub plot_width: f64,
    /// Plot area height (pixels).
    pub plot_height: f64,
}

impl View {
    /// A view of the box `[min_u, max_u] × [min_v, max_v]` at `scale`
    /// pixels per meter.
    pub fn new(min_u: f64, max_u: f64, min_v: f64, max_v: f64, scale: f64, margin: f64) -> Self {
        View {
            min_u,
            max_v,
            scale,
            margin,
            plot_width: ((max_u - min_u) * scale).ceil(),
            plot_height: ((max_v - min_v) * scale).ceil(),
        }
    }

    /// A blank canvas of the right size.
    pub fn canvas(&self, background: Color) -> Canvas {
        Canvas::new(
            (self.plot_width + 2.0 * self.margin) as usize,
            (self.plot_height + 2.0 * self.margin) as usize,
            background,
        )
    }

    /// Pixel position of world point `(u, v)`.
    pub fn px(&self, u: f64, v: f64) -> (f64, f64) {
        (
            self.margin + (u - self.min_u) * self.scale,
            self.margin + (self.max_v - v) * self.scale,
        )
    }

    /// Draws light grid lines every `step` meters with labels on the left
    /// and bottom edges, and a frame around the plot area.
    pub fn grid(&self, canvas: &mut Canvas, step: f64) {
        let max_u = self.min_u + self.plot_width / self.scale;
        let min_v = self.max_v - self.plot_height / self.scale;
        let mut u = (self.min_u / step).ceil() * step;
        while u <= max_u {
            let (x, _) = self.px(u, 0.0);
            canvas.line(
                (x, self.margin),
                (x, self.margin + self.plot_height),
                colors::GRID,
            );
            let label = format!("{u:.0}");
            let w = Canvas::text_width(&label, 2) as f64;
            canvas.text(
                (x - w / 2.0) as i64,
                (self.margin + self.plot_height + 6.0) as i64,
                &label,
                2,
                colors::BLACK,
            );
            u += step;
        }
        let mut v = (min_v / step).ceil() * step;
        while v <= self.max_v {
            let (_, y) = self.px(0.0, v);
            canvas.line(
                (self.margin, y),
                (self.margin + self.plot_width, y),
                colors::GRID,
            );
            let label = format!("{v:.0}");
            let w = Canvas::text_width(&label, 2) as f64;
            canvas.text(
                (self.margin - w - 6.0) as i64,
                (y - 5.0) as i64,
                &label,
                2,
                colors::BLACK,
            );
            v += step;
        }
        self.frame(canvas);
    }

    /// A black frame around the plot area.
    pub fn frame(&self, canvas: &mut Canvas) {
        let (l, t) = (self.margin, self.margin);
        let (r, b) = (
            self.margin + self.plot_width,
            self.margin + self.plot_height,
        );
        canvas.polyline(&[(l, t), (r, t), (r, b), (l, b), (l, t)], 1, colors::BLACK);
    }
}

/// 3×5 glyphs: five rows, bit 2 is the leftmost column.
fn glyph(c: char) -> Option<[u8; 5]> {
    Some(match c {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b001, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        ',' => [0b000, 0b000, 0b000, 0b010, 0b100],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        '+' => [0b000, 0b010, 0b111, 0b010, 0b000],
        ':' => [0b000, 0b010, 0b000, 0b010, 0b000],
        '/' => [0b001, 0b001, 0b010, 0b100, 0b100],
        '(' => [0b010, 0b100, 0b100, 0b100, 0b010],
        ')' => [0b010, 0b001, 0b001, 0b001, 0b010],
        '=' => [0b000, 0b111, 0b000, 0b111, 0b000],
        '%' => [0b101, 0b001, 0b010, 0b100, 0b101],
        '_' => [0b000, 0b000, 0b000, 0b000, 0b111],
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b011, 0b100, 0b100, 0b100, 0b011],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'G' => [0b011, 0b100, 0b101, 0b101, 0b011],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'J' => [0b001, 0b001, 0b001, 0b101, 0b010],
        'K' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'N' => [0b110, 0b101, 0b101, 0b101, 0b101],
        'O' => [0b010, 0b101, 0b101, 0b101, 0b010],
        'P' => [0b110, 0b101, 0b110, 0b100, 0b100],
        'Q' => [0b010, 0b101, 0b101, 0b110, 0b011],
        'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b011, 0b100, 0b010, 0b001, 0b110],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        'Z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_endpoints_and_view_mapping() {
        let mut canvas = Canvas::new(10, 10, colors::WHITE);
        canvas.line((1.0, 1.0), (8.0, 5.0), colors::BLACK);
        assert_eq!(canvas.pixels[10 + 1], colors::BLACK);
        assert_eq!(canvas.pixels[5 * 10 + 8], colors::BLACK);
        let view = View::new(-1.0, 1.0, -1.0, 1.0, 10.0, 5.0);
        assert_eq!(view.px(-1.0, 1.0), (5.0, 5.0));
        assert_eq!(view.px(1.0, -1.0), (25.0, 25.0));
        assert_eq!(Canvas::text_width("AB", 1), 7);
    }
}
