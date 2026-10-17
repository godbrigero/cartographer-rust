//! Hand-rolled SVG charts: XY plots (trajectories, maps, time series),
//! horizontal log-scale bars with confidence intervals (benchmarks) and
//! stacked progress bars (porting status).
//!
//! Colors are CSS classes resolved through custom properties (`--series-1`,
//! `--ink-2`, ...), so the same markup follows the page's light/dark theme
//! when inlined and carries its own `<style>` when written as a standalone
//! asset.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Categorical slots and chart chrome (light).
pub const LIGHT: &str = "--surface:#fcfcfb;--page:#f6f6f3;--ink-1:#0b0b0b;--ink-2:#52514e;\
--ink-3:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);\
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#6250d6;\
--good:#0ca30c;--critical:#d03b3b;--good-ink:#006300;--critical-ink:#b02a2a;color-scheme:light;";
/// The same roles stepped for the dark surface.
pub const DARK: &str = "--surface:#1a1a19;--page:#0d0d0d;--ink-1:#f0efec;--ink-2:#c3c2b7;\
--ink-3:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);\
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#9085e9;\
--good:#0ca30c;--critical:#e66767;--good-ink:#0ca30c;--critical-ink:#e66767;color-scheme:dark;";

/// Page theme variables: dark is selected by `prefers-color-scheme` unless
/// the viewer forces `data-theme` on `<html>`.
pub fn theme_vars() -> String {
    format!(
        ":root{{{LIGHT}}}\n@media (prefers-color-scheme: dark){{:root:not([data-theme=\"light\"]){{{DARK}}}}}\n:root[data-theme=\"dark\"]{{{DARK}}}\n"
    )
}

/// [`CHART_CSS`] with every `var(--name)` replaced by its value in `vars`
/// (standalone SVGs: renderers without custom-property support).
fn resolved_css(vars: &str) -> String {
    let mut css = CHART_CSS.to_string();
    for decl in vars.split(';') {
        if let Some((name, value)) = decl.split_once(':')
            && name.starts_with("--")
        {
            css = css.replace(&format!("var({name})"), value);
        }
    }
    css
}

/// Classes used by the chart markup.
pub const CHART_CSS: &str = "\
.bg{fill:var(--surface)}\
.grid{stroke:var(--grid);stroke-width:1}\
.axis{stroke:var(--axis);stroke-width:1}\
.tick{fill:var(--ink-3);font:11px system-ui,-apple-system,'Segoe UI',sans-serif;font-variant-numeric:tabular-nums}\
.lbl{fill:var(--ink-2);font:12px system-ui,-apple-system,'Segoe UI',sans-serif}\
.lbl-b{fill:var(--ink-1);font:600 12px system-ui,-apple-system,'Segoe UI',sans-serif}\
.val{fill:var(--ink-1);font:12px system-ui,-apple-system,'Segoe UI',sans-serif;font-variant-numeric:tabular-nums}\
.s1{stroke:var(--series-1)}.s2{stroke:var(--series-2)}.s3{stroke:var(--series-3)}.s4{stroke:var(--series-4)}\
.sm{stroke:var(--ink-3)}.si{stroke:var(--ink-1)}.s-ink2{stroke:var(--ink-2)}\
.f1{fill:var(--series-1)}.f2{fill:var(--series-2)}.f3{fill:var(--series-3)}.f4{fill:var(--series-4)}\
.fm{fill:var(--ink-3)}.fi{fill:var(--ink-1)}.f-ink2{fill:var(--ink-2)}.ftrack{fill:var(--grid)}\
.fsurf{fill:var(--surface)}.ring{stroke:var(--surface)}\
.ln{fill:none;stroke-linejoin:round;stroke-linecap:round}\
.xhair{stroke:var(--ink-2);stroke-width:1;pointer-events:none}\
";

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn unique_id(prefix: &str) -> String {
    format!("{prefix}{}", NEXT_ID.fetch_add(1, Ordering::Relaxed))
}

pub fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// A rendered chart.
pub struct Svg {
    pub width: f64,
    pub height: f64,
    pub title: String,
    pub body: String,
    /// JSON for the crosshair script (`data-hover`).
    pub hover: Option<String>,
}

impl Svg {
    fn open(&self, extra: &str) -> String {
        format!(
            "<svg class=\"chart\" viewBox=\"0 0 {w:.0} {h:.0}\" width=\"{w:.0}\" height=\"{h:.0}\" \
             role=\"img\" aria-label=\"{t}\"{hover}{extra}><title>{t}</title>",
            w = self.width,
            h = self.height,
            t = esc(&self.title),
            hover = self
                .hover
                .as_ref()
                .map(|h| format!(" data-hover=\"{}\"", esc(h)))
                .unwrap_or_default(),
        )
    }

    /// Markup for inlining into the report page (styled by the page).
    pub fn inline(&self) -> String {
        format!("{}{}</svg>", self.open(""), self.body)
    }

    /// Self-contained SVG file (own theme variables and classes).
    pub fn standalone(&self) -> String {
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n{}<style>{}@media (prefers-color-scheme: dark){{{}}}</style>\
             <rect class=\"bg\" width=\"100%\" height=\"100%\"/>{}</svg>\n",
            self.open(
                " xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\""
            ),
            resolved_css(LIGHT),
            resolved_css(DARK),
            self.body
        )
    }
}

// ------------------------------------------------------------------ scales

#[derive(Clone, Copy, Debug)]
pub struct Scale {
    pub d0: f64,
    pub d1: f64,
    pub r0: f64,
    pub r1: f64,
    pub log: bool,
}

impl Scale {
    pub fn at(&self, v: f64) -> f64 {
        let f = |x: f64| if self.log { x.log10() } else { x };
        let (a, b) = (f(self.d0), f(self.d1));
        if b == a {
            return (self.r0 + self.r1) / 2.0;
        }
        self.r0 + (f(v) - a) / (b - a) * (self.r1 - self.r0)
    }

    fn json(&self) -> String {
        format!(
            "[{},{},{:.2},{:.2}]",
            num(self.d0),
            num(self.d1),
            self.r0,
            self.r1
        )
    }
}

fn num(v: f64) -> String {
    if v.is_finite() {
        let s = format!("{v:.6}");
        let s = s.trim_end_matches('0').trim_end_matches('.');
        if s.is_empty() || s == "-" {
            "0".into()
        } else {
            s.to_string()
        }
    } else {
        "null".into()
    }
}

/// 1-2-5 step giving about `n` intervals over `range`.
pub fn nice_step(range: f64, n: f64) -> f64 {
    let raw = (range / n).max(1e-12);
    let mag = 10_f64.powf(raw.log10().floor());
    let r = raw / mag;
    mag * if r < 1.5 {
        1.0
    } else if r < 3.5 {
        2.0
    } else if r < 7.5 {
        5.0
    } else {
        10.0
    }
}

pub fn ticks(lo: f64, hi: f64, step: f64) -> Vec<f64> {
    let mut v = Vec::new();
    let mut t = (lo / step - 1e-9).ceil() * step;
    while t <= hi + step * 1e-9 {
        v.push(if t.abs() < step * 1e-9 { 0.0 } else { t });
        t += step;
    }
    v
}

pub fn tick_label(v: f64, step: f64) -> String {
    let decimals = (-step.log10().floor()).max(0.0) as usize;
    format!("{v:.decimals$}")
}

fn text_width(s: &str, px: f64) -> f64 {
    s.chars().count() as f64 * px * 0.56
}

// ---------------------------------------------------------------- XY chart

#[derive(Clone, Debug)]
pub enum Data {
    /// Polyline; non-finite y values break it.
    Line(Vec<(f64, f64)>),
    /// Independent segments `[x0, y0, x1, y1]`.
    Segments(Vec<[f64; 4]>),
    /// Square markers.
    Markers(Vec<(f64, f64)>),
}

#[derive(Clone, Debug)]
pub struct Layer {
    pub data: Data,
    /// Color class suffix: `1`..`4`, `m` (muted), `i` (ink), `-ink2`.
    pub color: &'static str,
    pub label: String,
    pub width: f64,
    pub opacity: f64,
    pub dash: bool,
}

impl Layer {
    pub fn line(pts: Vec<(f64, f64)>, color: &'static str, label: &str) -> Self {
        Layer {
            data: Data::Line(pts),
            color,
            label: label.into(),
            width: 2.0,
            opacity: 1.0,
            dash: false,
        }
    }

    pub fn segments(segs: Vec<[f64; 4]>, color: &'static str, label: &str) -> Self {
        Layer {
            data: Data::Segments(segs),
            color,
            label: label.into(),
            width: 1.0,
            opacity: 0.35,
            dash: false,
        }
    }

    pub fn markers(pts: Vec<(f64, f64)>, color: &'static str, label: &str) -> Self {
        Layer {
            data: Data::Markers(pts),
            color,
            label: label.into(),
            width: 8.0,
            opacity: 1.0,
            dash: false,
        }
    }

    pub fn width(mut self, w: f64) -> Self {
        self.width = w;
        self
    }

    pub fn opacity(mut self, o: f64) -> Self {
        self.opacity = o;
        self
    }

    fn points(&self) -> Vec<(f64, f64)> {
        match &self.data {
            Data::Line(p) | Data::Markers(p) => p.clone(),
            Data::Segments(s) => s
                .iter()
                .flat_map(|s| [(s[0], s[1]), (s[2], s[3])])
                .collect(),
        }
    }
}

/// A raster drawn under the data, covering `extent = [x0, y0, x1, y1]`.
#[derive(Clone, Debug)]
pub struct Background {
    pub href: String,
    pub extent: [f64; 4],
}

#[derive(Clone, Debug)]
pub struct XyChart {
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub width: f64,
    /// Plot height when not `equal_aspect`.
    pub height: f64,
    pub equal_aspect: bool,
    pub layers: Vec<Layer>,
    pub background: Option<Background>,
    /// Reference lines `(y, label)`.
    pub hlines: Vec<(f64, String)>,
    /// Crosshair tooltip over the `Line` layers (time series).
    pub hover: bool,
    pub y_from_zero: bool,
    pub x_unit: String,
    pub y_unit: String,
}

impl XyChart {
    pub fn new(title: &str, x_label: &str, y_label: &str) -> Self {
        XyChart {
            title: title.into(),
            x_label: x_label.into(),
            y_label: y_label.into(),
            width: 560.0,
            height: 220.0,
            equal_aspect: false,
            layers: Vec::new(),
            background: None,
            hlines: Vec::new(),
            hover: false,
            y_from_zero: false,
            x_unit: String::new(),
            y_unit: String::new(),
        }
    }

    /// Renders with background hrefs prefixed by `asset_prefix`.
    pub fn render(&self, asset_prefix: &str) -> Svg {
        let (ml, mr, mb) = (56.0, 18.0, 42.0);
        // Legend layout.
        let mut legend = Vec::new();
        let (mut lx, mut ly) = (ml, 0usize);
        let entries: Vec<&Layer> = self.layers.iter().filter(|l| !l.label.is_empty()).collect();
        let mut extra: Vec<(&str, &'static str)> = Vec::new();
        for (_, label) in &self.hlines {
            extra.push((label.as_str(), "ref"));
        }
        for (i, label) in entries
            .iter()
            .map(|l| l.label.as_str())
            .chain(extra.iter().map(|e| e.0))
            .enumerate()
        {
            let w = 22.0 + text_width(label, 12.0) + 16.0;
            if lx + w > self.width - mr && lx > ml {
                lx = ml;
                ly += 1;
            }
            legend.push((i, lx, ly));
            lx += w;
        }
        let legend_rows = if legend.is_empty() { 0 } else { ly + 1 };
        let mt = 14.0 + 18.0 * legend_rows as f64;
        let pw = self.width - ml - mr;

        // Domains.
        let mut pts: Vec<(f64, f64)> = self
            .layers
            .iter()
            .flat_map(Layer::points)
            .filter(|p| p.0.is_finite() && p.1.is_finite())
            .collect();
        if let Some(bg) = &self.background {
            pts.push((bg.extent[0], bg.extent[1]));
            pts.push((bg.extent[2], bg.extent[3]));
        }
        for (y, _) in &self.hlines {
            if let Some(&(x, _)) = pts.first() {
                pts.push((x, *y));
            }
        }
        if pts.is_empty() {
            pts.push((0.0, 0.0));
            pts.push((1.0, 1.0));
        }
        let fold = |f: fn(&(f64, f64)) -> f64| {
            pts.iter()
                .map(f)
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| {
                    (a.min(v), b.max(v))
                })
        };
        let (mut x0, mut x1) = fold(|p| p.0);
        let (mut y0, mut y1) = fold(|p| p.1);
        if self.y_from_zero {
            y0 = y0.min(0.0);
        }
        if x1 - x0 < 1e-9 {
            x0 -= 0.5;
            x1 += 0.5;
        }
        if y1 - y0 < 1e-9 {
            y0 -= 0.5;
            y1 += 0.5;
        }
        let ph;
        if self.equal_aspect {
            let pad = 0.03 * (x1 - x0).max(y1 - y0);
            x0 -= pad;
            x1 += pad;
            y0 -= pad;
            y1 += pad;
            let mut h = pw * (y1 - y0) / (x1 - x0);
            let (hmin, hmax) = (130.0, 560.0);
            if h > hmax {
                h = hmax;
                let dx = (y1 - y0) * pw / h;
                let c = (x0 + x1) / 2.0;
                (x0, x1) = (c - dx / 2.0, c + dx / 2.0);
            } else if h < hmin {
                h = hmin;
                let dy = (x1 - x0) * h / pw;
                let c = (y0 + y1) / 2.0;
                (y0, y1) = (c - dy / 2.0, c + dy / 2.0);
            }
            ph = h;
        } else {
            let step = nice_step(y1 - y0, 4.0);
            y1 = (y1 / step).ceil() * step;
            if !self.y_from_zero {
                y0 = (y0 / step).floor() * step;
            }
            ph = self.height;
        }
        let xs = Scale {
            d0: x0,
            d1: x1,
            r0: ml,
            r1: ml + pw,
            log: false,
        };
        let ys = Scale {
            d0: y0,
            d1: y1,
            r0: mt + ph,
            r1: mt,
            log: false,
        };
        let height = mt + ph + mb;
        let mut b = String::new();

        // Grid and ticks.
        let x_step = nice_step(x1 - x0, (pw / 90.0).max(2.0));
        let y_step = nice_step(y1 - y0, (ph / 50.0).clamp(2.0, 6.0));
        for t in ticks(x0, x1, x_step) {
            let x = xs.at(t);
            let _ = write!(
                b,
                "<line class=\"grid\" x1=\"{x:.1}\" y1=\"{:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\"/>\
                 <text class=\"tick\" x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>",
                mt,
                mt + ph,
                mt + ph + 15.0,
                tick_label(t, x_step)
            );
        }
        for t in ticks(y0, y1, y_step) {
            let y = ys.at(t);
            let _ = write!(
                b,
                "<line class=\"grid\" x1=\"{ml:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\"/>\
                 <text class=\"tick\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\">{}</text>",
                ml + pw,
                ml - 6.0,
                y + 4.0,
                tick_label(t, y_step)
            );
        }
        let _ = write!(
            b,
            "<line class=\"axis\" x1=\"{ml}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\"/>\
             <text class=\"lbl\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>\
             <text class=\"lbl\" transform=\"translate(14 {:.1}) rotate(-90)\" text-anchor=\"middle\">{}</text>",
            mt + ph,
            ml + pw,
            mt + ph,
            ml + pw / 2.0,
            mt + ph + 34.0,
            esc(&self.x_label),
            mt + ph / 2.0,
            esc(&self.y_label)
        );

        // Data, clipped to the plot area.
        let clip = unique_id("clip");
        let _ = write!(
            b,
            "<clipPath id=\"{clip}\"><rect x=\"{ml}\" y=\"{mt}\" width=\"{pw:.1}\" height=\"{ph:.1}\"/></clipPath><g clip-path=\"url(#{clip})\">"
        );
        if let Some(bg) = &self.background {
            let (ax, ay) = (xs.at(bg.extent[0]), ys.at(bg.extent[3]));
            let (bx, by) = (xs.at(bg.extent[2]), ys.at(bg.extent[1]));
            let _ = write!(
                b,
                "<image href=\"{}{}\" x=\"{ax:.1}\" y=\"{ay:.1}\" width=\"{:.1}\" height=\"{:.1}\" preserveAspectRatio=\"none\" style=\"image-rendering:pixelated\"/>",
                esc(asset_prefix),
                esc(&bg.href),
                bx - ax,
                by - ay
            );
        }
        for layer in &self.layers {
            let dash = if layer.dash {
                " stroke-dasharray=\"5 4\""
            } else {
                ""
            };
            match &layer.data {
                Data::Line(p) => {
                    let mut d = String::new();
                    let mut pen = false;
                    for &(x, y) in p {
                        if !(x.is_finite() && y.is_finite()) {
                            pen = false;
                            continue;
                        }
                        let _ = write!(
                            d,
                            "{}{:.1} {:.1}",
                            if pen { "L" } else { "M" },
                            xs.at(x),
                            ys.at(y)
                        );
                        pen = true;
                    }
                    let _ = write!(
                        b,
                        "<path class=\"ln s{}\" stroke-width=\"{}\" opacity=\"{}\"{dash} d=\"{d}\"/>",
                        layer.color, layer.width, layer.opacity
                    );
                }
                Data::Segments(segs) => {
                    let mut d = String::new();
                    for s in segs {
                        let _ = write!(
                            d,
                            "M{:.1} {:.1}L{:.1} {:.1}",
                            xs.at(s[0]),
                            ys.at(s[1]),
                            xs.at(s[2]),
                            ys.at(s[3])
                        );
                    }
                    let _ = write!(
                        b,
                        "<path class=\"ln s{}\" stroke-width=\"{}\" opacity=\"{}\"{dash} d=\"{d}\"/>",
                        layer.color, layer.width, layer.opacity
                    );
                }
                Data::Markers(p) => {
                    let s = layer.width;
                    let _ = write!(
                        b,
                        "<g class=\"f{}\" opacity=\"{}\">",
                        layer.color, layer.opacity
                    );
                    for &(x, y) in p {
                        let _ = write!(
                            b,
                            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{s}\" height=\"{s}\" rx=\"1.5\" class=\"ring\" stroke-width=\"1.5\"/>",
                            xs.at(x) - s / 2.0,
                            ys.at(y) - s / 2.0
                        );
                    }
                    b.push_str("</g>");
                }
            }
        }
        for (y, _) in &self.hlines {
            let py = ys.at(*y);
            let _ = write!(
                b,
                "<line class=\"s-ink2\" stroke-width=\"1.5\" stroke-dasharray=\"2 3\" x1=\"{ml}\" y1=\"{py:.1}\" x2=\"{:.1}\" y2=\"{py:.1}\"/>",
                ml + pw
            );
        }
        b.push_str("</g>");

        // Legend.
        let labels: Vec<(&str, Option<&Layer>)> = entries
            .iter()
            .map(|l| (l.label.as_str(), Some(*l)))
            .chain(extra.iter().map(|e| (e.0, None)))
            .collect();
        for &(i, x, row) in &legend {
            let y = 12.0 + 18.0 * row as f64;
            let (label, layer) = labels[i];
            match layer {
                Some(l) => match l.data {
                    Data::Markers(_) => {
                        let _ = write!(
                            b,
                            "<rect class=\"f{}\" x=\"{:.1}\" y=\"{:.1}\" width=\"8\" height=\"8\" rx=\"1.5\"/>",
                            l.color,
                            x + 4.0,
                            y - 4.0
                        );
                    }
                    _ => {
                        let _ = write!(
                            b,
                            "<line class=\"ln s{}\" stroke-width=\"{}\" opacity=\"{}\"{} x1=\"{x:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\"/>",
                            l.color,
                            l.width.clamp(2.0, 5.0),
                            l.opacity.max(0.6),
                            if l.dash {
                                " stroke-dasharray=\"5 4\""
                            } else {
                                ""
                            },
                            x + 16.0
                        );
                    }
                },
                None => {
                    let _ = write!(
                        b,
                        "<line class=\"s-ink2\" stroke-width=\"1.5\" stroke-dasharray=\"2 3\" x1=\"{x:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\"/>",
                        x + 16.0
                    );
                }
            }
            let _ = write!(
                b,
                "<text class=\"lbl\" x=\"{:.1}\" y=\"{:.1}\">{}</text>",
                x + 22.0,
                y + 4.0,
                esc(label)
            );
        }

        let hover = self.hover.then(|| {
            let series: Vec<String> = self
                .layers
                .iter()
                .filter_map(|l| match &l.data {
                    Data::Line(p) if !l.label.is_empty() => Some(format!(
                        "{{\"l\":\"{}\",\"c\":\"{}\",\"p\":[{}]}}",
                        l.label.replace('"', "'"),
                        l.color,
                        p.iter()
                            .map(|(x, y)| format!("[{},{}]", num(*x), num(*y)))
                            .collect::<Vec<_>>()
                            .join(",")
                    )),
                    _ => None,
                })
                .collect();
            format!(
                "{{\"sx\":{},\"sy\":{},\"xu\":\"{}\",\"yu\":\"{}\",\"top\":{mt},\"bottom\":{:.1},\"s\":[{}]}}",
                xs.json(),
                ys.json(),
                self.x_unit,
                self.y_unit,
                mt + ph,
                series.join(",")
            )
        });
        Svg {
            width: self.width,
            height,
            title: self.title.clone(),
            body: b,
            hover,
        }
    }
}

// ------------------------------------------------------------ bar charts

/// Path of a bar from `x0` to `x1` with a rounded data end.
fn bar_path(x0: f64, x1: f64, y: f64, h: f64) -> String {
    let r = 4.0_f64.min((x1 - x0).max(0.0)).min(h / 2.0);
    format!(
        "M{x0:.1} {y:.1}H{:.1}A{r} {r} 0 0 1 {x1:.1} {:.1}V{:.1}A{r} {r} 0 0 1 {:.1} {:.1}H{x0:.1}Z",
        x1 - r,
        y + r,
        y + h - r,
        x1 - r,
        y + h
    )
}

pub struct BarRow {
    pub group: String,
    pub label: String,
    pub value: f64,
    pub lo: f64,
    pub hi: f64,
    pub reference: Option<f64>,
    pub value_text: String,
    pub tooltip: String,
}

/// Grouped horizontal bars on a log axis with CI whiskers and optional
/// reference markers; `axis_label` formats the decade ticks.
pub fn hbar_log_chart(
    title: &str,
    rows: &[BarRow],
    width: f64,
    axis_label: impl Fn(f64) -> String,
    reference_label: &str,
) -> Svg {
    let label_w = rows
        .iter()
        .map(|r| text_width(&r.label, 12.0))
        .fold(120.0, f64::max)
        .min(360.0)
        + 16.0;
    let value_w = rows
        .iter()
        .map(|r| text_width(&r.value_text, 12.0))
        .fold(60.0, f64::max)
        + 14.0;
    let (ml, mr, mt) = (label_w, value_w, 40.0);
    let pw = width - ml - mr;
    let lo = rows
        .iter()
        .flat_map(|r| [r.lo, r.value, r.reference.unwrap_or(r.value)])
        .fold(f64::INFINITY, f64::min);
    let hi = rows
        .iter()
        .flat_map(|r| [r.hi, r.value, r.reference.unwrap_or(r.value)])
        .fold(f64::NEG_INFINITY, f64::max);
    let d0 = 10_f64.powf(lo.max(1e-9).log10().floor());
    let d1 = 10_f64.powf(hi.max(d0 * 10.0).log10().ceil());
    let xs = Scale {
        d0,
        d1,
        r0: ml,
        r1: ml + pw,
        log: true,
    };
    let (row_h, group_h, bar_h) = (22.0, 28.0, 12.0);
    let mut groups: Vec<&str> = Vec::new();
    for r in rows {
        if groups.last() != Some(&r.group.as_str()) {
            groups.push(&r.group);
        }
    }
    let ph = rows.len() as f64 * row_h + groups.len() as f64 * group_h;
    let height = mt + ph + 30.0;
    let mut b = String::new();
    // Legend.
    let _ = write!(
        b,
        "<path class=\"f1\" d=\"{}\"/><text class=\"lbl\" x=\"{:.1}\" y=\"16\">median</text>\
         <line class=\"si\" stroke-width=\"1.5\" x1=\"{:.1}\" y1=\"12\" x2=\"{:.1}\" y2=\"12\"/>\
         <line class=\"si\" stroke-width=\"1.5\" x1=\"{:.1}\" y1=\"8\" x2=\"{:.1}\" y2=\"16\"/>\
         <line class=\"si\" stroke-width=\"1.5\" x1=\"{:.1}\" y1=\"8\" x2=\"{:.1}\" y2=\"16\"/>\
         <text class=\"lbl\" x=\"{:.1}\" y=\"16\">95 % CI of the median</text>",
        bar_path(ml, ml + 16.0, 6.0, 12.0),
        ml + 22.0,
        ml + 90.0,
        ml + 106.0,
        ml + 90.0,
        ml + 90.0,
        ml + 106.0,
        ml + 106.0,
        ml + 112.0
    );
    if !reference_label.is_empty() {
        let x = ml + 270.0;
        let _ = write!(
            b,
            "<path class=\"s-ink2 fsurf\" stroke-width=\"1.5\" d=\"M{x} 6L{} 12L{x} 18L{} 12Z\"/>\
             <text class=\"lbl\" x=\"{:.1}\" y=\"16\">{}</text>",
            x + 5.0,
            x - 5.0,
            x + 11.0,
            esc(reference_label)
        );
    }
    // Decade grid.
    let mut t = d0;
    while t <= d1 * 1.0001 {
        let x = xs.at(t);
        let _ = write!(
            b,
            "<line class=\"grid\" x1=\"{x:.1}\" y1=\"{mt}\" x2=\"{x:.1}\" y2=\"{:.1}\"/>\
             <text class=\"tick\" x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>\
             <text class=\"tick\" x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>",
            mt + ph,
            mt - 6.0,
            axis_label(t),
            mt + ph + 16.0,
            axis_label(t)
        );
        t *= 10.0;
    }
    let mut y = mt;
    let mut current = "";
    for r in rows {
        if r.group != current {
            current = &r.group;
            let _ = write!(
                b,
                "<text class=\"lbl-b\" x=\"8\" y=\"{:.1}\">{}</text>",
                y + 19.0,
                esc(current)
            );
            y += group_h;
        }
        let cy = y + row_h / 2.0;
        let x1 = xs.at(r.value);
        let _ = write!(
            b,
            "<g><title>{}</title><rect x=\"0\" y=\"{y:.1}\" width=\"{width}\" height=\"{row_h}\" fill=\"transparent\"/>\
             <text class=\"lbl\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\">{}</text>\
             <path class=\"f1\" d=\"{}\"/>",
            esc(&r.tooltip),
            ml - 8.0,
            cy + 4.0,
            esc(&r.label),
            bar_path(ml, x1, cy - bar_h / 2.0, bar_h)
        );
        if r.lo.is_finite() && r.hi.is_finite() && r.hi > r.lo {
            let (a, c) = (xs.at(r.lo), xs.at(r.hi));
            let _ = write!(
                b,
                "<path class=\"si ln\" stroke-width=\"1.5\" d=\"M{a:.1} {cy:.1}H{c:.1}M{a:.1} {:.1}V{:.1}M{c:.1} {:.1}V{:.1}\"/>",
                cy - 4.0,
                cy + 4.0,
                cy - 4.0,
                cy + 4.0
            );
        }
        if let Some(reference) = r.reference {
            let x = xs.at(reference);
            let _ = write!(
                b,
                "<path class=\"s-ink2 fsurf\" stroke-width=\"1.5\" d=\"M{x:.1} {:.1}L{:.1} {cy:.1}L{x:.1} {:.1}L{:.1} {cy:.1}Z\"/>",
                cy - 6.0,
                x + 5.0,
                cy + 6.0,
                x - 5.0
            );
        }
        let _ = write!(
            b,
            "<text class=\"val\" x=\"{:.1}\" y=\"{:.1}\">{}</text></g>",
            ml + pw + 8.0,
            cy + 4.0,
            esc(&r.value_text)
        );
        y += row_h;
    }
    Svg {
        width,
        height,
        title: title.into(),
        body: b,
        hover: None,
    }
}

pub struct ProgressRow {
    pub label: String,
    pub done: f64,
    pub na: f64,
    pub total: f64,
}

/// Stacked horizontal bars: done (series 1), not applicable (muted) and the
/// remaining track, absolute lengths (bar length = total).
pub fn progress_chart(title: &str, unit: &str, rows: &[ProgressRow], width: f64) -> Svg {
    let (ml, mr, mt) = (110.0, 200.0, 30.0);
    let pw = width - ml - mr;
    let max = rows.iter().map(|r| r.total).fold(1.0, f64::max);
    let xs = Scale {
        d0: 0.0,
        d1: max,
        r0: ml,
        r1: ml + pw,
        log: false,
    };
    let (row_h, bar_h) = (28.0, 14.0);
    let height = mt + rows.len() as f64 * row_h + 10.0;
    let mut b = String::new();
    let _ = write!(
        b,
        "<rect class=\"f1\" x=\"{ml}\" y=\"6\" width=\"12\" height=\"12\" rx=\"2\"/><text class=\"lbl\" x=\"{:.1}\" y=\"16\">ported</text>\
         <rect class=\"fm\" opacity=\"0.55\" x=\"{:.1}\" y=\"6\" width=\"12\" height=\"12\" rx=\"2\"/><text class=\"lbl\" x=\"{:.1}\" y=\"16\">not applicable</text>\
         <rect class=\"ftrack\" x=\"{:.1}\" y=\"6\" width=\"12\" height=\"12\" rx=\"2\"/><text class=\"lbl\" x=\"{:.1}\" y=\"16\">remaining</text>",
        ml + 18.0,
        ml + 80.0,
        ml + 98.0,
        ml + 200.0,
        ml + 218.0
    );
    for (i, r) in rows.iter().enumerate() {
        let y = mt + i as f64 * row_h;
        let cy = y + row_h / 2.0;
        let by = cy - bar_h / 2.0;
        let applicable = r.total - r.na;
        let pct = if applicable > 0.0 {
            100.0 * r.done / applicable
        } else {
            100.0
        };
        if r.total == 0.0 {
            let _ = write!(
                b,
                "<text class=\"lbl\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\">{}</text><text class=\"tick\" x=\"{ml}\" y=\"{:.1}\">no C++ {unit}</text>",
                ml - 10.0,
                cy + 4.0,
                esc(&r.label),
                cy + 4.0
            );
            continue;
        }
        let tooltip = format!(
            "{}: {} of {} {unit} ported ({pct:.0} % of applicable), {} not applicable",
            r.label, r.done, r.total, r.na
        );
        let _ = write!(
            b,
            "<g><title>{}</title><rect x=\"0\" y=\"{y:.1}\" width=\"{width}\" height=\"{row_h}\" fill=\"transparent\"/>\
             <text class=\"lbl\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\">{}</text>\
             <path class=\"ftrack\" d=\"{}\"/>",
            esc(&tooltip),
            ml - 10.0,
            cy + 4.0,
            esc(&r.label),
            bar_path(ml, xs.at(r.total), by, bar_h)
        );
        let x_done = xs.at(r.done);
        if r.done > 0.0 {
            let _ = write!(
                b,
                "<path class=\"f1\" d=\"{}\"/>",
                bar_path(ml, x_done, by, bar_h)
            );
        }
        if r.na > 0.0 {
            let a = x_done + if r.done > 0.0 { 2.0 } else { 0.0 };
            let c = xs.at(r.done + r.na);
            if c > a {
                let _ = write!(
                    b,
                    "<rect class=\"fm\" opacity=\"0.55\" x=\"{a:.1}\" y=\"{by:.1}\" width=\"{:.1}\" height=\"{bar_h}\"/>",
                    c - a
                );
            }
        }
        let _ = write!(
            b,
            "<text class=\"val\" x=\"{:.1}\" y=\"{:.1}\">{} / {} <tspan class=\"tick\">({pct:.0} %{})</tspan></text></g>",
            xs.at(r.total) + 8.0,
            cy + 4.0,
            r.done,
            applicable,
            if r.na > 0.0 {
                format!(", {} n/a", r.na)
            } else {
                String::new()
            }
        );
    }
    Svg {
        width,
        height,
        title: title.into(),
        body: b,
        hover: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nice_steps_and_ticks() {
        assert_eq!(nice_step(10.0, 5.0), 2.0);
        assert_eq!(nice_step(0.9, 4.0), 0.2);
        assert_eq!(ticks(-1.0, 1.0, 0.5), vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
        assert_eq!(tick_label(0.25, 0.05), "0.25");
    }

    #[test]
    fn xy_chart_renders_lines_with_gaps() {
        let mut c = XyChart::new("t", "x", "y");
        c.layers.push(Layer::line(
            vec![(0.0, 1.0), (1.0, f64::NAN), (2.0, 3.0)],
            "1",
            "a",
        ));
        c.hover = true;
        let svg = c.render("");
        let path = svg.body.split(" d=\"").nth(1).unwrap();
        assert_eq!(path.matches('M').count(), 2);
        assert!(svg.inline().contains("data-hover"));
        assert!(svg.standalone().starts_with("<?xml"));
    }
}
