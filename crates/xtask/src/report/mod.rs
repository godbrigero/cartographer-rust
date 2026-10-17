//! `cargo run --release -p xtask -- report`: builds the self-contained
//! visual report `report/index.html` (+ `report/assets/`) from the e2e test
//! artifacts (`target/e2e`), criterion results (`target/criterion`), the
//! porting status and the Unitree L1 demo outputs (`target/unitree_l1`).
//! See `report/README.md`.

mod diagrams;
mod json;
mod svg;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use json::Value;
use svg::{Background, BarRow, Layer, ProgressRow, Svg, XyChart, esc};

const USAGE: &str = "usage: cargo run --release -p xtask -- report [--no-tests] [--bench | --bench-quick] \
[--unitree-bag <bag>] [--unitree-dir <dir>] [--cpp <dir>] [--out <dir>]";

struct Args {
    run_tests: bool,
    bench: Option<bool>,
    unitree_bag: Option<PathBuf>,
    unitree_dir: Option<PathBuf>,
    cpp: Option<PathBuf>,
    out: Option<PathBuf>,
}

fn parse_args(args: &[String]) -> Args {
    let mut a = Args {
        run_tests: true,
        bench: None,
        unitree_bag: None,
        unitree_dir: None,
        cpp: None,
        out: None,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut path = |what: &str| PathBuf::from(it.next().unwrap_or_else(|| panic!("{what}")));
        match arg.as_str() {
            "--no-tests" => a.run_tests = false,
            "--bench" => a.bench = Some(false),
            "--bench-quick" => a.bench = Some(true),
            "--unitree-bag" => a.unitree_bag = Some(path("--unitree-bag <bag>")),
            "--unitree-dir" => a.unitree_dir = Some(path("--unitree-dir <dir>")),
            "--cpp" => a.cpp = Some(path("--cpp <dir>")),
            "--out" => a.out = Some(path("--out <dir>")),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument {other}\n{USAGE}");
                std::process::exit(2);
            }
        }
    }
    a
}

fn cargo(root: &Path, args: &[&str]) -> bool {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    println!("report: running cargo {}", args.join(" "));
    let status = Command::new(cargo)
        .args(args)
        .current_dir(root)
        .status()
        .expect("running cargo");
    if !status.success() {
        eprintln!(
            "report: warning: cargo {} failed ({status})",
            args.join(" ")
        );
    }
    status.success()
}

pub fn run(root: &Path, args: &[String]) {
    let args = parse_args(args);
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target"));
    let mut notes = Vec::new();
    if args.run_tests
        && !cargo(
            root,
            &[
                "test",
                "-p",
                "cartographer-slam",
                "--test",
                "e2e_2d",
                "--test",
                "e2e_3d",
            ],
        )
    {
        notes.push("Some e2e tests failed while generating this report (see the FAIL rows).");
    }
    if let Some(quick) = args.bench {
        let mut a = vec!["bench", "-p", "cartographer-slam"];
        for b in [
            "scan_matching",
            "grid_insertion",
            "optimization",
            "local_slam",
        ] {
            a.extend(["--bench", b]);
        }
        if quick {
            a.extend(["--", "--quick"]);
        }
        cargo(root, &a);
    }
    let unitree_dir = args
        .unitree_dir
        .clone()
        .unwrap_or_else(|| target.join("unitree_l1"));
    if let Some(bag) = &args.unitree_bag {
        let bag = bag.to_string_lossy().to_string();
        let out = unitree_dir.to_string_lossy().to_string();
        cargo(
            root,
            &[
                "run",
                "--release",
                "-p",
                "cartographer-demos",
                "--example",
                "unitree_l1_bag",
                "--",
                &bag,
                "--out",
                &out,
            ],
        );
    }

    let out = args.out.clone().unwrap_or_else(|| root.join("report"));
    let assets = out.join("assets");
    let _ = fs::remove_dir_all(&assets);
    fs::create_dir_all(&assets).expect("creating report/assets");
    let mut r = Report {
        assets,
        nav: Vec::new(),
        html: String::new(),
    };

    let porting = porting(root, args.cpp.as_deref());
    let e2e = load_e2e(&target.join("e2e"));
    let benches = load_benches(root, &target.join("criterion"));
    let env = environment(root);

    overview(&mut r, root, &porting, &e2e, &benches, &notes);
    porting_section(&mut r, &porting);
    results_2d(&mut r, &e2e, &target.join("e2e"));
    results_3d(&mut r, &e2e);
    unitree(&mut r, &unitree_dir, root);
    benchmarks(&mut r, &benches, args.bench.is_some());
    environment_section(&mut r, &env);

    let html = page(&r, &env);
    fs::write(out.join("index.html"), &html).expect("writing report/index.html");
    let size: u64 = walk(&out)
        .iter()
        .filter_map(|p| fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();
    println!(
        "report: wrote {} ({:.1} MB total in {})",
        out.join("index.html").display(),
        size as f64 / 1e6,
        out.display()
    );
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = fs::read_dir(dir) else {
        return out;
    };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            out.extend(walk(&p));
        } else {
            out.push(p);
        }
    }
    out
}

// ------------------------------------------------------------------ output

struct Report {
    assets: PathBuf,
    nav: Vec<(String, String)>,
    html: String,
}

impl Report {
    fn section(&mut self, id: &str, title: &str, lead: &str) {
        if !self.html.is_empty() {
            self.html.push_str("</section>\n");
        }
        self.nav.push((id.into(), title.into()));
        let _ = write!(
            self.html,
            "<section id=\"{id}\"><h2>{}</h2>{}",
            esc(title),
            if lead.is_empty() {
                String::new()
            } else {
                format!("<p class=\"lead\">{lead}</p>")
            }
        );
    }

    /// Writes `assets/<name>.svg` and returns the inline figure.
    fn svg(&self, name: &str, svg: &Svg, caption: &str) -> String {
        fs::write(self.assets.join(format!("{name}.svg")), svg.standalone()).expect("asset");
        format!(
            "<figure class=\"card\">{}<figcaption>{caption} <a class=\"dl\" href=\"assets/{name}.svg\">SVG</a></figcaption></figure>",
            svg.inline()
        )
    }

    fn chart(&self, name: &str, chart: &XyChart, caption: &str) -> String {
        let standalone = chart.render("");
        fs::write(
            self.assets.join(format!("{name}.svg")),
            standalone.standalone(),
        )
        .expect("asset");
        format!(
            "<figure class=\"card\">{}<figcaption>{caption} <a class=\"dl\" href=\"assets/{name}.svg\">SVG</a></figcaption></figure>",
            chart.render("assets/").inline()
        )
    }

    /// Copies `src` to `assets/<name>`; returns the relative path.
    fn copy(&self, src: &Path, name: &str) -> Option<String> {
        fs::copy(src, self.assets.join(name)).ok()?;
        Some(format!("assets/{name}"))
    }
}

// ------------------------------------------------------------ formatting

fn sig(v: f64, digits: i32) -> String {
    if !v.is_finite() {
        return "–".into();
    }
    if v == 0.0 {
        return "0".into();
    }
    let mag = v.abs().log10().floor() as i32;
    let decimals = (digits - 1 - mag).max(0) as usize;
    format!("{v:.decimals$}")
}

fn duration(ns: f64) -> String {
    let (v, unit) = if ns >= 1e9 {
        (ns / 1e9, "s")
    } else if ns >= 1e6 {
        (ns / 1e6, "ms")
    } else if ns >= 1e3 {
        (ns / 1e3, "µs")
    } else {
        (ns, "ns")
    };
    format!("{} {unit}", sig(v, 3))
}

/// Value with a display unit derived from the metric name suffix.
fn metric(name: &str, v: f64) -> String {
    if name.ends_with("_rad_per_m") {
        format!("{} mrad/m", sig(v * 1e3, 3))
    } else if name.ends_with("_m_per_m") {
        format!("{} cm/m", sig(v * 100.0, 3))
    } else if name.ends_with("_rad") {
        format!("{} mrad", sig(v * 1e3, 3))
    } else if name.ends_with("_m") && v.abs() < 2.0 {
        format!("{} cm", sig(v * 100.0, 3))
    } else if name.ends_with("_m") {
        format!("{} m", sig(v, 4))
    } else if name.ends_with("_s") || name.ends_with("seconds") {
        format!("{} s", sig(v, 3))
    } else if v.fract() == 0.0 && v.abs() < 1e9 {
        format!("{v:.0}")
    } else {
        let s = sig(v, 4);
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    }
}

/// Rate with an SI prefix (`3.19 M`).
fn si(v: f64) -> String {
    let (v, p) = if v >= 1e9 {
        (v / 1e9, " G")
    } else if v >= 1e6 {
        (v / 1e6, " M")
    } else if v >= 1e3 {
        (v / 1e3, " k")
    } else {
        (v, " ")
    };
    format!("{}{p}", sig(v, 3))
}

fn pretty(name: &str) -> String {
    let s = name
        .replace("rpe_1m", "RPE (1 m)")
        .replace("ate", "ATE")
        .replace("rmse", "RMSE")
        .replace("_m_per_m", "")
        .replace("_rad_per_m", "")
        .replace('_', " ");
    let s = s
        .strip_suffix(" m")
        .or_else(|| s.strip_suffix(" rad"))
        .or_else(|| s.strip_suffix(" s"))
        .unwrap_or(&s)
        .to_string();
    let mut c = s.chars();
    c.next()
        .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
        .unwrap_or_default()
}

fn badge(passed: bool) -> &'static str {
    if passed {
        "<span class=\"badge good\"><span aria-hidden=\"true\">✓</span> pass</span>"
    } else {
        "<span class=\"badge bad\"><span aria-hidden=\"true\">✗</span> fail</span>"
    }
}

fn checks_table(report: &Value) -> String {
    let mut t = String::from(
        "<table class=\"metrics\"><thead><tr><th>Check</th><th class=\"num\">Value</th><th class=\"num\">Threshold</th><th>Headroom</th><th>Status</th></tr></thead><tbody>",
    );
    for c in report.get("checks").map(Value::arr).unwrap_or_default() {
        let name = c.get("name").and_then(Value::str).unwrap_or("?");
        let (v, limit) = (
            c.get("value").map_or(f64::NAN, Value::f),
            c.get("limit").map_or(f64::NAN, Value::f),
        );
        let cmp = c.get("comparison").and_then(Value::str).unwrap_or("<");
        let passed = c.get("passed").and_then(Value::bool).unwrap_or(false);
        // Fraction of the allowed range used (upper bounds) or the margin
        // above a lower bound.
        let used = if cmp == "<" {
            (v / limit).clamp(0.0, 1.0)
        } else if v.is_finite() && v > 0.0 {
            (limit / v).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let tip = if cmp == "<" {
            format!("{:.0} % of the limit", 100.0 * v / limit)
        } else {
            format!("{:.2}x the minimum", v / limit)
        };
        let _ = write!(
            t,
            "<tr><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{} {}</td>\
             <td><span class=\"meter\" title=\"{tip}\"><span style=\"width:{:.1}%\"></span></span></td><td>{}</td></tr>",
            esc(&pretty(name)),
            metric(name, v),
            if cmp == "<" { "&lt;" } else { "&gt;" },
            metric(name, limit),
            100.0 * used,
            badge(passed)
        );
    }
    t.push_str("</tbody></table>");
    t
}

fn info_list(report: &Value) -> String {
    let mut t = String::from("<dl class=\"info\">");
    for (k, v) in report.get("info").map(Value::entries).unwrap_or_default() {
        let _ = write!(
            t,
            "<div><dt>{}</dt><dd>{}</dd></div>",
            esc(&pretty(k)),
            metric(k, v.f())
        );
    }
    t.push_str("</dl>");
    t
}

fn tile(value: &str, label: &str, sub: &str) -> String {
    format!(
        "<div class=\"tile\"><div class=\"tile-v\">{value}</div><div class=\"tile-l\">{label}</div><div class=\"tile-s\">{sub}</div></div>"
    )
}

// --------------------------------------------------------------- porting

struct Porting {
    modules: Vec<crate::porting_status::ModuleStats>,
    totals: crate::porting_status::Totals,
    source: String,
}

fn porting(root: &Path, cpp: Option<&Path>) -> Porting {
    let cpp = cpp
        .map(Path::to_path_buf)
        .or_else(|| crate::default_cpp_root(root));
    match cpp {
        Some(cpp) => {
            let status = crate::porting_status::compute(&cpp, root);
            Porting {
                modules: crate::porting_status::modules(&status),
                totals: status.totals(),
                // Last two path components (no absolute local paths).
                source: format!(
                    "computed from the C++ tree <code>…/{}</code>",
                    esc(&cpp
                        .components()
                        .rev()
                        .take(2)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<PathBuf>()
                        .display()
                        .to_string())
                ),
            }
        }
        None => {
            let md = fs::read_to_string(root.join("PORTING_STATUS.md")).unwrap_or_default();
            let (modules, totals) = crate::porting_status::from_markdown(&md);
            Porting {
                modules,
                totals,
                source: "parsed from <code>PORTING_STATUS.md</code> (C++ tree not found; pass <code>--cpp</code>)".into(),
            }
        }
    }
}

fn porting_section(r: &mut Report, p: &Porting) {
    r.section(
        "porting",
        "Porting progress",
        &format!(
            "Per top-level C++ module: compilation units (<code>.h</code>/<code>.cc</code> pairs) with a non-empty Rust counterpart, \
             and C++ <code>TEST*</code> cases matched by name in the Rust file. Units marked N/A have no Rust equivalent by design \
             (e.g. replaced by a crate). Full table: <a href=\"../PORTING_STATUS.md\">PORTING_STATUS.md</a>; {}.",
            p.source
        ),
    );
    let t = &p.totals;
    let applicable_units = t.units - t.na_units;
    let applicable_tests = t.cpp_tests - t.na_tests;
    let _ = write!(
        r.html,
        "<div class=\"tiles\">{}{}{}{}</div>",
        tile(
            &format!("{} / {}", t.ported_units, applicable_units),
            "C++ units ported",
            &format!(
                "{:.1} % · {} N/A of {} total",
                100.0 * t.ported_units as f64 / applicable_units.max(1) as f64,
                t.na_units,
                t.units
            )
        ),
        tile(
            &format!("{} / {}", t.ported_tests, applicable_tests),
            "C++ tests matched",
            &format!(
                "{:.1} % · {} in N/A units",
                100.0 * t.ported_tests as f64 / applicable_tests.max(1) as f64,
                t.na_tests
            )
        ),
        tile(
            &t.rust_tests.to_string(),
            "Rust #[test] functions",
            "all crates, incl. integration tests"
        ),
        tile(
            &p.modules.len().to_string(),
            "top-level modules",
            "common · transform · sensor · mapping · io · …"
        )
    );
    let units: Vec<ProgressRow> = p
        .modules
        .iter()
        .map(|m| ProgressRow {
            label: m.name.clone(),
            done: m.ported_units as f64,
            na: m.na_units as f64,
            total: m.units as f64,
        })
        .collect();
    let tests: Vec<ProgressRow> = p
        .modules
        .iter()
        .map(|m| ProgressRow {
            label: m.name.clone(),
            done: m.ported_tests as f64,
            na: m.na_tests as f64,
            total: m.cpp_tests as f64,
        })
        .collect();
    let a = r.svg(
        "porting_units",
        &svg::progress_chart("C++ units ported per module", "units", &units, 640.0),
        "Compilation units per module (bar length = number of units).",
    );
    let b = r.svg(
        "porting_tests",
        &svg::progress_chart("C++ tests ported per module", "tests", &tests, 640.0),
        "C++ test cases per module matched by a Rust test of the same name.",
    );
    let _ = write!(r.html, "<div class=\"grid2\">{a}{b}</div>");
    let mut table = String::from(
        "<table class=\"metrics\"><thead><tr><th>Module</th><th class=\"num\">Units</th><th class=\"num\">Ported</th><th class=\"num\">N/A</th>\
         <th class=\"num\">C++ tests</th><th class=\"num\">Matched</th><th class=\"num\">Rust tests</th></tr></thead><tbody>",
    );
    for m in &p.modules {
        let _ = write!(
            table,
            "<tr><td><code>{}</code></td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>",
            esc(&m.name),
            m.units,
            m.ported_units,
            m.na_units,
            m.cpp_tests,
            m.ported_tests,
            m.rust_tests
        );
    }
    let _ = write!(
        table,
        "</tbody><tfoot><tr><td>Total</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr></tfoot></table>",
        t.units, t.ported_units, t.na_units, t.cpp_tests, t.ported_tests, t.rust_tests
    );
    let _ = write!(
        r.html,
        "<details><summary>Per-module table</summary>{table}<p class=\"note\">Rust tests: tests in the Rust files of the module's units; \
         the total also counts integration tests and files without a C++ counterpart.</p></details>"
    );
}

// ------------------------------------------------------------------- e2e

fn load_e2e(dir: &Path) -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    for p in walk(dir) {
        if p.extension().is_some_and(|e| e == "json") {
            match fs::read_to_string(&p)
                .map_err(|e| e.to_string())
                .and_then(|s| Value::parse(&s))
            {
                Ok(v) => {
                    let name = p.file_stem().unwrap().to_string_lossy().to_string();
                    m.insert(name, v);
                }
                Err(e) => eprintln!("report: skipping {}: {e}", p.display()),
            }
        }
    }
    m
}

fn description(scenario: &str) -> &'static str {
    match scenario.trim_end_matches("_long") {
        "corridor_2d" => {
            "Two laps (2 × 108 m) of a loop corridor with LiDAR, an IMU with gyro/accelerometer bias and miscalibrated wheel odometry, started from rest; shipped 2D defaults."
        }
        "office_2d" => "Two laps of an office floor with LiDAR and wheel odometry only.",
        "multi_trajectory_2d" => {
            "Two trajectories in one map builder: the second starts unconnected elsewhere in the office and must be connected by the global (full-submap) constraint search; both must then fit the ground truth with a single rigid transform."
        }
        "pure_localization_2d" => {
            "An office map is built, serialized to a pbstream and loaded frozen; a new trajectory with the pure-localization trimmer localizes in it (no initial pose). Poses are the online <code>local_to_global × local</code> poses, with no re-alignment."
        }
        "corridor_3d" => {
            "Two handheld laps (2 × 108 m) of a 3D loop corridor (ceiling beams, pipes, floor clutter) with a VLP-16-like LiDAR and an IMU."
        }
        "room_3d" => {
            "Two handheld tours of a furnished 12 × 8 × 3 m room (pillars, table, shelf, hanging lamp)."
        }
        _ => "",
    }
}

/// Pose rows `[t, x, y, z, yaw]`, one list per trajectory.
fn pose_lists(v: Option<&Value>) -> Vec<Vec<[f64; 5]>> {
    let Some(v) = v else { return Vec::new() };
    let rows = |a: &Value| -> Vec<[f64; 5]> {
        a.arr()
            .iter()
            .map(|r| {
                let r = r.arr();
                std::array::from_fn(|i| r.get(i).map_or(f64::NAN, Value::f))
            })
            .collect()
    };
    let first_is_list = v
        .arr()
        .first()
        .and_then(|r| r.arr().first())
        .is_some_and(|e| matches!(e, Value::Arr(_)));
    if first_is_list {
        v.arr().iter().map(rows).collect()
    } else {
        vec![rows(v)]
    }
}

/// Sections holding trajectory metrics, with display names and a color.
fn metric_sections(report: &Value) -> Vec<(String, &'static str, &Value)> {
    let mut out = Vec::new();
    for (k, v) in report.entries() {
        if v.get("errors_over_time").is_none() {
            continue;
        }
        let (name, color) = match k.as_str() {
            "trajectory" => ("global SLAM".to_string(), "1"),
            "local_slam_trajectory" => ("local SLAM only".to_string(), "3"),
            "trajectory_0" => ("trajectory 0".to_string(), "1"),
            "trajectory_1" => ("trajectory 1".to_string(), "4"),
            "online_trajectory" => ("online pose".to_string(), "1"),
            "kept_nodes_trajectory" => ("kept nodes".to_string(), "3"),
            other => (other.replace('_', " "), "m"),
        };
        out.push((name, color, v));
    }
    out
}

fn check_limit(report: &Value, names: &[&str]) -> Option<f64> {
    report
        .get("checks")?
        .arr()
        .iter()
        .find(|c| {
            c.get("name")
                .and_then(Value::str)
                .is_some_and(|n| names.contains(&n))
        })
        .and_then(|c| c.get("limit"))
        .and_then(Value::num)
}

/// `(title, unit, errors_over_time column, scale, limit label, check names)`.
type ErrorSpec = (
    &'static str,
    &'static str,
    usize,
    f64,
    &'static str,
    &'static [&'static str],
);

fn error_charts(r: &Report, name: &str, report: &Value) -> String {
    let sections = metric_sections(report);
    if sections.is_empty() {
        return String::new();
    }
    let t0 = sections
        .iter()
        .filter_map(|(_, _, v)| {
            v.get("errors_over_time")?
                .arr()
                .first()?
                .arr()
                .first()?
                .num()
        })
        .fold(f64::INFINITY, f64::min);
    let specs: [ErrorSpec; 3] = [
        (
            "ATE translation",
            "cm",
            1,
            100.0,
            "ATE RMSE limit",
            &["ate_rmse_m", "online_ate_rmse_m", "trajectory_0_ate_rmse_m"],
        ),
        (
            "ATE rotation",
            "mrad",
            2,
            1e3,
            "ATE rotation RMSE limit",
            &["online_ate_rotation_rmse_rad"],
        ),
        (
            "RPE over 1 m, translation",
            "cm",
            3,
            100.0,
            "RPE RMSE limit",
            &[
                "rpe_1m_translation_rmse_m_per_m",
                "online_rpe_1m_translation_rmse_m_per_m",
                "trajectory_0_rpe_1m_translation_rmse_m_per_m",
            ],
        ),
    ];
    let mut html = String::from("<div class=\"grid3\">");
    for (i, (title, unit, col, scale, limit_label, limit_names)) in specs.iter().enumerate() {
        let mut c = XyChart::new(
            &format!("{name}: {title} over time"),
            "time since start (s)",
            &format!("{title} ({unit})"),
        );
        c.width = 400.0;
        c.height = 170.0;
        c.hover = true;
        c.y_from_zero = true;
        c.x_unit = "s".into();
        c.y_unit = (*unit).into();
        for (label, color, v) in &sections {
            let pts: Vec<(f64, f64)> = v
                .get("errors_over_time")
                .map(Value::arr)
                .unwrap_or_default()
                .iter()
                .map(|row| {
                    let row = row.arr();
                    (
                        row.first().map_or(f64::NAN, Value::f) - t0,
                        row.get(*col).map_or(f64::NAN, Value::f) * scale,
                    )
                })
                .collect();
            c.layers.push(Layer::line(pts, color, label).width(1.5));
        }
        if let Some(limit) = check_limit(report, limit_names) {
            c.hlines.push((limit * scale, (*limit_label).into()));
        }
        let _ = write!(
            html,
            "{}",
            r.chart(
                &format!("{name}_error_{i}"),
                &c,
                &format!(
                    "{title}{}.",
                    if i == 2 {
                        " between poses 1 m of travel apart"
                    } else {
                        " after the alignment, per pose"
                    }
                )
            )
        );
    }
    html.push_str("</div>");
    html
}

fn trajectory_layers(report: &Value, project: fn(&[f64; 5]) -> (f64, f64)) -> Vec<Layer> {
    let gt = pose_lists(report.get("ground_truth_poses"));
    let est = pose_lists(report.get("estimated_poses_gt_frame"));
    let mut layers = Vec::new();
    for (i, g) in gt.iter().enumerate() {
        layers.push(
            Layer::line(
                g.iter().map(project).collect(),
                "2",
                if i == 0 { "ground truth" } else { "" },
            )
            .width(5.0)
            .opacity(0.4),
        );
    }
    let multi = est.len() > 1;
    for (i, e) in est.iter().enumerate() {
        let (color, label) = match (multi, i) {
            (false, _) => ("1", "estimate".to_string()),
            (true, 0) => ("1", "estimate, trajectory 0".to_string()),
            (true, _) => ("4", format!("estimate, trajectory {i}")),
        };
        layers.push(Layer::line(e.iter().map(project).collect(), color, &label).width(1.5));
    }
    layers
}

fn pose_graph_chart(name: &str, report: &Value, three_d: bool) -> Option<XyChart> {
    let pg = report.get("pose_graph")?;
    let pos = |key: &str| -> BTreeMap<(i64, i64), (f64, f64)> {
        pg.get(key)
            .map(Value::arr)
            .unwrap_or_default()
            .iter()
            .map(|r| {
                let r = r.arr();
                let g = |i: usize| r.get(i).map_or(f64::NAN, Value::f);
                ((g(0) as i64, g(1) as i64), (g(2), g(3)))
            })
            .collect()
    };
    let nodes = pos("nodes");
    let submaps = pos("submaps");
    let mut kinds: [Vec<[f64; 4]>; 3] = Default::default();
    for c in report
        .path("constraints.inter_submap_constraints")
        .map(Value::arr)
        .unwrap_or_default()
    {
        let c: Vec<i64> = c.arr().iter().map(|v| v.f() as i64).collect();
        if c.len() < 4 {
            continue;
        }
        let (Some(s), Some(n)) = (submaps.get(&(c[0], c[1])), nodes.get(&(c[2], c[3]))) else {
            continue;
        };
        let kind = c.get(4).copied().unwrap_or(0).clamp(0, 2) as usize;
        kinds[kind].push([s.0, s.1, n.0, n.1]);
    }
    let mut chart = XyChart::new(&format!("{name}: pose graph (top-down)"), "x (m)", "y (m)");
    chart.equal_aspect = true;
    chart.width = if three_d { 560.0 } else { 420.0 };
    let [local, loops, inter] = kinds;
    let (nl, nloop, ninter) = (local.len(), loops.len(), inter.len());
    if nl > 0 {
        chart.layers.push(
            Layer::segments(local, "m", &format!("inter-submap, recent submaps ({nl})"))
                .opacity(0.25)
                .width(0.8),
        );
    }
    if nloop > 0 {
        chart.layers.push(
            Layer::segments(loops, "2", &format!("loop closures ({nloop})"))
                .opacity(0.3)
                .width(0.8),
        );
    }
    if ninter > 0 {
        chart.layers.push(
            Layer::segments(inter, "3", &format!("inter-trajectory ({ninter})"))
                .opacity(0.3)
                .width(0.8),
        );
    }
    let mut trajectories: BTreeMap<i64, Vec<(f64, f64)>> = BTreeMap::new();
    for ((t, _), p) in &nodes {
        trajectories.entry(*t).or_default().push(*p);
    }
    for (t, pts) in trajectories {
        let label = if nodes.keys().any(|k| k.0 != t) {
            format!("nodes, trajectory {t}")
        } else {
            "trajectory nodes".to_string()
        };
        chart
            .layers
            .push(Layer::line(pts, if t == 0 { "1" } else { "4" }, &label).width(1.5));
    }
    chart.layers.push(Layer::markers(
        submaps.values().copied().collect(),
        "i",
        &format!("submap origins ({})", submaps.len()),
    ));
    Some(chart)
}

fn scenario_header(name: &str, report: &Value) -> String {
    let passed = report.get("passed").and_then(Value::bool).unwrap_or(false);
    let checks = report.get("checks").map(Value::arr).unwrap_or_default();
    let ok = checks
        .iter()
        .filter(|c| c.get("passed").and_then(Value::bool) == Some(true))
        .count();
    format!(
        "<h3 id=\"{name}\"><code>{name}</code> {} <span class=\"muted\">{ok}/{} checks · {} s wall time</span></h3><p>{}</p>",
        badge(passed),
        checks.len(),
        sig(report.get("seconds").map_or(f64::NAN, Value::f), 3),
        description(name)
    )
}

fn results_2d(r: &mut Report, e2e: &BTreeMap<String, Value>, dir: &Path) {
    r.section(
        "results-2d",
        "2D SLAM results",
        "End-to-end acceptance tests (<code>crates/cartographer/tests/e2e_2d.rs</code>): <code>MapBuilder</code> configured from the shipped \
         Lua files and fed with simulated sensor data from <code>cartographer-sim</code>; estimates are rigidly aligned to the ground truth. \
         Map images: occupied black, free white, unknown grey, ground-truth walls missing from the map red, ground-truth path green, estimate blue.",
    );
    let scenarios: Vec<&String> = e2e.keys().filter(|k| k.contains("_2d")).collect();
    if scenarios.is_empty() {
        r.html.push_str("<p class=\"empty\">No 2D results in <code>target/e2e</code>; run without <code>--no-tests</code>.</p>");
        return;
    }
    let mut index = String::from("<ul class=\"toc\">");
    for name in &scenarios {
        let passed = e2e[*name]
            .get("passed")
            .and_then(Value::bool)
            .unwrap_or(false);
        let _ = write!(
            index,
            "<li><a href=\"#{name}\">{name}</a> {}</li>",
            badge(passed)
        );
    }
    index.push_str("</ul>");
    r.html.push_str(&index);
    for name in scenarios {
        let report = &e2e[name];
        let mut html = scenario_header(name, report);
        html.push_str("<div class=\"grid3\">");
        if let Some(src) = Some(dir.join(format!("{name}_map.png"))).filter(|p| p.exists())
            && let Some(href) = r.copy(&src, &format!("{name}_map.png"))
        {
            let m = report.get("map");
            let _ = write!(
                html,
                "<figure class=\"card\"><a href=\"{href}\"><img class=\"map\" src=\"{href}\" alt=\"{name} occupancy map\" loading=\"lazy\"></a>\
                 <figcaption>Global occupancy map composed from all submaps ({} cm cells). Precision {}, recall {}.</figcaption></figure>",
                sig(
                    m.and_then(|m| m.get("resolution_m"))
                        .map_or(f64::NAN, Value::f)
                        * 100.0,
                    2
                ),
                sig(
                    m.and_then(|m| m.get("precision"))
                        .map_or(f64::NAN, Value::f),
                    4
                ),
                sig(
                    m.and_then(|m| m.get("recall")).map_or(f64::NAN, Value::f),
                    4
                ),
            );
        }
        let mut c = XyChart::new(
            &format!("{name}: estimated vs ground-truth trajectory"),
            "x (m)",
            "y (m)",
        );
        c.equal_aspect = true;
        c.width = 420.0;
        c.layers = trajectory_layers(report, |p| (p[1], p[2]));
        html.push_str(&r.chart(
            &format!("{name}_trajectory"),
            &c,
            "Estimate (aligned to the ground-truth frame) over the ground-truth path.",
        ));
        if let Some(pg) = pose_graph_chart(name, report, false) {
            html.push_str(&r.chart(
                &format!("{name}_pose_graph"),
                &pg,
                "Optimized node path, submap origins and inter-submap constraints (submap origin to node).",
            ));
        }
        html.push_str("</div>");
        html.push_str(&error_charts(r, name, report));
        let _ = write!(
            html,
            "<div class=\"grid2 tables\"><div>{}</div><div><h4>Additional metrics</h4>{}</div></div>",
            checks_table(report),
            info_list(report)
        );
        r.html.push_str(&html);
    }
}

/// Density image of points projected with `project`, transparent where
/// empty; returns the extent `[x0, y0, x1, y1]`.
fn density_png(path: &Path, points: &[(f64, f64)], resolution: f64) -> Option<[f64; 4]> {
    if points.is_empty() {
        return None;
    }
    let (mut x0, mut y0, mut x1, mut y1) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for &(x, y) in points {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let (x0, y0) = (x0 - resolution, y0 - resolution);
    let w = ((x1 - x0) / resolution).ceil() as usize + 2;
    let h = ((y1 - y0) / resolution).ceil() as usize + 2;
    let mut counts = vec![0_u32; w * h];
    for &(x, y) in points {
        let ix = ((x - x0) / resolution) as usize;
        let iy = ((y - y0) / resolution) as usize;
        if ix < w && iy < h {
            counts[(h - 1 - iy) * w + ix] += 1;
        }
    }
    let max = counts.iter().copied().max().unwrap_or(1).max(1) as f64;
    let mut rgba = Vec::with_capacity(w * h * 4);
    for &c in &counts {
        let a = if c == 0 {
            0.0
        } else {
            0.3 + 0.7 * (c as f64).ln_1p() / max.ln_1p()
        };
        rgba.extend_from_slice(&[137, 135, 129, (a * 255.0).round() as u8]);
    }
    let file = fs::File::create(path).ok()?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header().ok()?.write_image_data(&rgba).ok()?;
    Some([
        x0,
        y0,
        x0 + w as f64 * resolution,
        y0 + h as f64 * resolution,
    ])
}

fn results_3d(r: &mut Report, e2e: &BTreeMap<String, Value>) {
    r.section(
        "results-3d",
        "3D SLAM results",
        "End-to-end acceptance tests (<code>crates/cartographer/tests/e2e_3d.rs</code>) with a simulated VLP-16-like LiDAR and an IMU, \
         handheld walking motion; rigid 3D alignment. Map projections show the occupied voxels (probability ≥ 0.6) of the high-resolution \
         submap grids placed with the optimized submap poses (subsampled; darker = more voxels along the projection).",
    );
    let scenarios: Vec<&String> = e2e.keys().filter(|k| k.contains("_3d")).collect();
    if scenarios.is_empty() {
        r.html.push_str("<p class=\"empty\">No 3D results in <code>target/e2e</code>; run without <code>--no-tests</code>.</p>");
        return;
    }
    for name in scenarios {
        let report = &e2e[name];
        let mut html = scenario_header(name, report);
        let points: Vec<[f64; 3]> = report
            .get("map_points")
            .map(Value::arr)
            .unwrap_or_default()
            .iter()
            .map(|p| {
                let p = p.arr();
                std::array::from_fn(|i| p.get(i).map_or(f64::NAN, Value::f))
            })
            .collect();
        let mut views = Vec::new();
        for (view, axes, (xl, yl), project) in [
            (
                "top",
                "top-down (x-y)",
                ("x (m)", "y (m)"),
                (|p: &[f64; 5]| (p[1], p[2])) as fn(&[f64; 5]) -> (f64, f64),
            ),
            ("side", "side (x-z)", ("x (m)", "z (m)"), |p: &[f64; 5]| {
                (p[1], p[3])
            }),
        ] {
            let mut c = XyChart::new(&format!("{name}: map and trajectory, {axes}"), xl, yl);
            c.equal_aspect = true;
            c.width = if view == "top" { 560.0 } else { 900.0 };
            // Top-down: without floor and ceiling (within 25 cm of the 2nd /
            // 98th height percentile) so walls and furniture stand out.
            let (floor, ceiling) = {
                let mut z: Vec<f64> = points
                    .iter()
                    .map(|p| p[2])
                    .filter(|z| z.is_finite())
                    .collect();
                z.sort_by(f64::total_cmp);
                let q = |f: f64| {
                    z.get(((z.len() as f64 - 1.0) * f).round() as usize)
                        .copied()
                        .unwrap_or(0.0)
                };
                (q(0.02) + 0.25, q(0.98) - 0.25)
            };
            let projected: Vec<(f64, f64)> = points
                .iter()
                .filter(|p| view != "top" || (p[2] > floor && p[2] < ceiling))
                .map(|p| {
                    if view == "top" {
                        (p[0], p[1])
                    } else {
                        (p[0], p[2])
                    }
                })
                .collect();
            let png = format!("{name}_map_{view}.png");
            let resolution = if view == "top" { 0.1 } else { 0.15 };
            if let Some(extent) = density_png(&r.assets.join(&png), &projected, resolution) {
                c.background = Some(Background { href: png, extent });
            }
            c.layers = trajectory_layers(report, project);
            views.push((view, c));
        }
        let mut charts = Vec::new();
        for (view, c) in &views {
            charts.push(r.chart(
                &format!("{name}_{view}"),
                c,
                if *view == "top" {
                    "Occupied voxels between floor and ceiling (grey) with the ground-truth path and the aligned estimate."
                } else {
                    "Side projection of all occupied voxels (equal axis scales)."
                },
            ));
        }
        let pose_graph = pose_graph_chart(name, report, true).map(|pg| {
            r.chart(
                &format!("{name}_pose_graph"),
                &pg,
                "Optimized node path, submap origins and inter-submap constraints, top-down.",
            )
        });
        let _ = write!(
            html,
            "<div class=\"grid2\">{}{}</div>{}",
            charts[0],
            pose_graph.unwrap_or_default(),
            charts[1]
        );
        // Height over time.
        let gt = pose_lists(report.get("ground_truth_poses"));
        let est = pose_lists(report.get("estimated_poses_gt_frame"));
        if let (Some(g), Some(e)) = (gt.first(), est.first()) {
            let t0 = g.first().map_or(0.0, |p| p[0]);
            let mut c = XyChart::new(
                &format!("{name}: height over time"),
                "time since start (s)",
                "z (m)",
            );
            c.width = 900.0;
            c.height = 160.0;
            c.hover = true;
            c.x_unit = "s".into();
            c.y_unit = "m".into();
            c.layers.push(
                Layer::line(
                    g.iter().map(|p| (p[0] - t0, p[3])).collect(),
                    "2",
                    "ground truth",
                )
                .width(4.0)
                .opacity(0.45),
            );
            c.layers.push(
                Layer::line(
                    e.iter().map(|p| (p[0] - t0, p[3])).collect(),
                    "1",
                    "estimate",
                )
                .width(1.5),
            );
            html.push_str(&r.chart(
                &format!("{name}_height"),
                &c,
                "Height of the tracking frame (handheld bobbing) — vertical is the weakly observed direction.",
            ));
        }
        html.push_str(&error_charts(r, name, report));
        let _ = write!(
            html,
            "<div class=\"grid2 tables\"><div>{}</div><div><h4>Additional metrics</h4>{}</div></div>",
            checks_table(report),
            info_list(report)
        );
        r.html.push_str(&html);
    }
}

// --------------------------------------------------------------- unitree

fn unitree(r: &mut Report, dir: &Path, root: &Path) {
    r.section(
        "unitree",
        "Unitree L1 MuJoCo demo",
        "3D SLAM (<code>cartographer::Slam</code>) on a ROS1 bag of a simulated Unitree 4D LiDAR L1 + IMU (MuJoCo, 90 s, ~3 laps in a \
         20 × 12 × 3 m room), read with the pure-Rust <code>cartographer-rosbag</code> reader \
         (<code>crates/demos/examples/unitree_l1_bag.rs</code>). Regenerate with <code>--unitree-bag &lt;bag&gt;</code>.",
    );
    let runs: Vec<PathBuf> = walk(dir)
        .into_iter()
        .filter(|p| p.file_name().is_some_and(|n| n == "metrics.json"))
        .filter_map(|p| p.parent().map(Path::to_path_buf))
        .collect();
    if runs.is_empty() {
        let _ = write!(
            r.html,
            "<p class=\"empty\">No demo outputs found in <code>{}</code>. Run the report with <code>--unitree-bag &lt;bag&gt;</code> \
             (or point <code>--unitree-dir</code> at an existing output directory).</p>",
            esc(&dir.strip_prefix(root).unwrap_or(dir).display().to_string())
        );
        return;
    }
    for run in runs {
        let label = run
            .strip_prefix(dir)
            .ok()
            .map(|p| p.display().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "default".into());
        let key = label.replace(['/', ' '], "_");
        let metrics = fs::read_to_string(run.join("metrics.json"))
            .ok()
            .and_then(|s| Value::parse(&s).ok())
            .unwrap_or(Value::Null);
        let _ = write!(r.html, "<h3>Run <code>{}</code></h3>", esc(&label));
        let ev = metrics.get("evaluation");
        let opt = ev.and_then(|e| e.get("optimized"));
        let local = ev.and_then(|e| e.get("local"));
        let g = |v: Option<&Value>, k: &str| v.and_then(|v| v.get(k)).map_or(f64::NAN, Value::f);
        let _ = write!(
            r.html,
            "<div class=\"tiles\">{}{}{}{}</div>",
            tile(
                &format!("{} cm", sig(g(opt, "ate_rmse_m") * 100.0, 3)),
                "ATE RMSE (optimized)",
                &format!(
                    "local SLAM only: {} cm",
                    sig(g(local, "ate_rmse_m") * 100.0, 3)
                )
            ),
            tile(
                &format!("{} %", sig(g(opt, "final_drift_percent"), 2)),
                "final drift",
                &format!("of {} m path", sig(g(ev, "path_length_m"), 3))
            ),
            tile(
                &format!(
                    "{}×",
                    sig(
                        metrics
                            .path("timing.realtime_factor")
                            .map_or(f64::NAN, Value::f),
                        3
                    )
                ),
                "real-time factor",
                &format!(
                    "{} s of data in {} s",
                    sig(g(Some(&metrics), "bag_duration_s"), 3),
                    sig(metrics.path("timing.total_s").map_or(f64::NAN, Value::f), 3)
                )
            ),
            tile(
                &metric(
                    "n",
                    metrics
                        .path("stats.loop_closures")
                        .map_or(f64::NAN, Value::f)
                ),
                "loop closures",
                &format!(
                    "{} nodes · {} submaps",
                    metric("n", metrics.path("stats.nodes").map_or(f64::NAN, Value::f)),
                    metric(
                        "n",
                        metrics.path("stats.submaps").map_or(f64::NAN, Value::f)
                    )
                )
            )
        );
        let mut figures = String::new();
        let mut pngs: Vec<PathBuf> = fs::read_dir(&run)
            .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default();
        pngs.retain(|p| p.extension().is_some_and(|e| e == "png"));
        pngs.sort_by_key(|p| {
            let n = p.file_name().unwrap().to_string_lossy().to_string();
            (
                [
                    "top_down.png",
                    "loop_closures.png",
                    "error_vs_time.png",
                    "side_xz.png",
                    "side_yz.png",
                ]
                .iter()
                .position(|o| *o == n)
                .unwrap_or(99),
                n,
            )
        });
        for p in pngs {
            let n = p.file_name().unwrap().to_string_lossy().to_string();
            if let Some(href) = r.copy(&p, &format!("unitree_{key}_{n}")) {
                let _ = write!(
                    figures,
                    "<figure class=\"card\"><a href=\"{href}\"><img src=\"{href}\" alt=\"{n}\" loading=\"lazy\"></a><figcaption><code>{n}</code></figcaption></figure>"
                );
            }
        }
        let _ = write!(r.html, "<div class=\"grid2\">{figures}</div>");
        if let (Some(opt), Some(local)) = (opt, local) {
            let mut t = String::from(
                "<table class=\"metrics\"><thead><tr><th>Metric</th><th class=\"num\">Optimized (global SLAM)</th><th class=\"num\">Local SLAM only</th></tr></thead><tbody>",
            );
            for (k, v) in opt.entries() {
                let fmt = |x: f64| {
                    if k.ends_with("_deg") {
                        format!("{} °", sig(x, 3))
                    } else if k.ends_with("_percent") {
                        format!("{} %", sig(x, 3))
                    } else {
                        metric(k, x)
                    }
                };
                let _ = write!(
                    t,
                    "<tr><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>",
                    esc(&pretty(
                        k.trim_end_matches("_deg").trim_end_matches("_percent")
                    )),
                    fmt(v.f()),
                    fmt(local.get(k).map_or(f64::NAN, Value::f))
                );
            }
            t.push_str("</tbody></table>");
            let _ = write!(
                r.html,
                "<div class=\"grid2 tables\"><div>{t}<p class=\"note\">{}; {}.</p></div><div><h4>Pose graph</h4>{}</div></div>",
                esc(ev
                    .and_then(|e| e.get("association"))
                    .and_then(Value::str)
                    .unwrap_or("")),
                esc(ev
                    .and_then(|e| e.get("alignment"))
                    .and_then(Value::str)
                    .unwrap_or("")),
                {
                    let mut d = String::from("<dl class=\"info\">");
                    for (k, v) in metrics.get("stats").map(Value::entries).unwrap_or_default() {
                        let _ = write!(
                            d,
                            "<div><dt>{}</dt><dd>{}</dd></div>",
                            esc(&pretty(k)),
                            metric(k, v.f())
                        );
                    }
                    d.push_str("</dl>");
                    d
                }
            );
        }
    }
    // Video.
    let videos: Vec<PathBuf> = walk(dir)
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "mp4"))
        .collect();
    for v in videos {
        let name = v.file_name().unwrap().to_string_lossy().to_string();
        let size = fs::metadata(&v).map(|m| m.len()).unwrap_or(0);
        if size < 50_000_000 {
            if let Some(href) = r.copy(&v, &format!("unitree_{name}")) {
                let _ = write!(
                    r.html,
                    "<figure class=\"card\"><video controls preload=\"metadata\" src=\"{href}\"></video><figcaption><code>{name}</code> ({:.1} MB). Video files are git-ignored in <code>report/</code> to keep the repository small; \
                     they are present after regenerating locally.</figcaption></figure>",
                    size as f64 / 1e6
                );
            }
        } else {
            let _ = write!(
                r.html,
                "<p class=\"note\">Video <code>{}</code> ({:.0} MB) is too large to copy into the report; open it from the workspace.</p>",
                esc(&v.strip_prefix(root).unwrap_or(&v).display().to_string()),
                size as f64 / 1e6
            );
        }
    }
}

// ------------------------------------------------------------ benchmarks

struct Bench {
    id: String,
    median: f64,
    lo: f64,
    hi: f64,
    elements: Option<f64>,
    reference: Option<f64>,
}

/// Medians listed in `docs/BENCHMARKS.md` (`| `id` | 309 µs ... |`), ns.
fn documented_medians(root: &Path) -> BTreeMap<String, f64> {
    let md = fs::read_to_string(root.join("docs/BENCHMARKS.md")).unwrap_or_default();
    let mut m = BTreeMap::new();
    for line in md.lines() {
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        if cells.len() < 4 || !cells[1].starts_with('`') {
            continue;
        }
        let id = cells[1].trim_matches('`');
        let mut words = cells[2].split_whitespace();
        let (Some(v), Some(unit)) = (
            words.next().and_then(|v| v.parse::<f64>().ok()),
            words.next(),
        ) else {
            continue;
        };
        let scale = match unit {
            "ns" => 1.0,
            "µs" | "us" => 1e3,
            "ms" => 1e6,
            "s" => 1e9,
            _ => continue,
        };
        m.insert(id.to_string(), v * scale);
    }
    m
}

fn load_benches(root: &Path, dir: &Path) -> Vec<Bench> {
    let documented = documented_medians(root);
    let mut out = Vec::new();
    for p in walk(dir) {
        if !(p.ends_with("new/estimates.json")) {
            continue;
        }
        let new = p.parent().unwrap();
        let Some(est) = fs::read_to_string(&p)
            .ok()
            .and_then(|s| Value::parse(&s).ok())
        else {
            continue;
        };
        let meta = fs::read_to_string(new.join("benchmark.json"))
            .ok()
            .and_then(|s| Value::parse(&s).ok());
        let id = meta
            .as_ref()
            .and_then(|m| m.get("full_id"))
            .and_then(Value::str)
            .map(String::from)
            .unwrap_or_else(|| {
                new.parent()
                    .unwrap()
                    .strip_prefix(dir)
                    .unwrap()
                    .display()
                    .to_string()
            });
        let elements = meta
            .as_ref()
            .and_then(|m| m.path("throughput.Elements"))
            .and_then(Value::num);
        let g = |k: &str| est.path(k).map_or(f64::NAN, Value::f);
        out.push(Bench {
            reference: documented.get(&id).copied(),
            id,
            median: g("median.point_estimate"),
            lo: g("median.confidence_interval.lower_bound"),
            hi: g("median.confidence_interval.upper_bound"),
            elements,
        });
    }
    out
}

/// Sort key with digit runs zero-padded (`1000_poses` < `4000` < `16000`).
fn natural_key(s: &str) -> String {
    let mut out = String::new();
    let mut digits = String::new();
    for c in s.chars().chain(std::iter::once('\0')) {
        if c.is_ascii_digit() {
            digits.push(c);
            continue;
        }
        if !digits.is_empty() {
            out.push_str(&format!("{digits:0>12}"));
            digits.clear();
        }
        if c != '\0' {
            out.push(c);
        }
    }
    out
}

fn benchmarks(r: &mut Report, benches: &[Bench], ran: bool) {
    r.section(
        "benchmarks",
        "Benchmarks",
        "Criterion benchmarks of the hot paths (<code>crates/cartographer/benches/</code>, ids and fixtures in \
         <a href=\"../docs/BENCHMARKS.md\">docs/BENCHMARKS.md</a>): median time per iteration with the 95 % confidence interval of the \
         median, log scale. Diamonds mark the medians recorded in docs/BENCHMARKS.md (Apple M3) for comparison.",
    );
    if benches.is_empty() {
        r.html.push_str("<p class=\"empty\">No criterion results in <code>target/criterion</code>. Run the report with <code>--bench</code> (full, ~1.5 min) or <code>--bench-quick</code>.</p>");
        return;
    }
    let mut groups: BTreeMap<String, Vec<&Bench>> = BTreeMap::new();
    for b in benches {
        let group = b.id.split('/').next().unwrap_or("").to_string();
        groups.entry(group).or_default().push(b);
    }
    let group_order = [
        "scan_matching_2d",
        "scan_matching_3d",
        "grid_insertion_2d",
        "grid_insertion_3d",
        "voxel_filter_3d",
        "local_slam_2d",
        "optimization_problem_2d",
        "optimization_problem_3d",
        "ceres_spa_2d_sparse_normal_cholesky",
    ];
    let mut names: Vec<&String> = groups.keys().collect();
    names.sort_by_key(|g| {
        (
            group_order.iter().position(|o| o == g).unwrap_or(99),
            (*g).clone(),
        )
    });
    let mut rows = Vec::new();
    for g in names {
        let mut items = groups[g].clone();
        items.sort_by_key(|b| natural_key(&b.id));
        for b in items {
            let label =
                b.id.split_once('/')
                    .map_or(b.id.as_str(), |x| x.1)
                    .to_string();
            let throughput = b
                .elements
                .map(|n| format!(" · {}elem/s", si(n / (b.median * 1e-9))))
                .unwrap_or_default();
            rows.push(BarRow {
                group: g.clone(),
                label,
                value: b.median,
                lo: b.lo,
                hi: b.hi,
                reference: b.reference,
                value_text: format!("{}{throughput}", duration(b.median)),
                tooltip: format!(
                    "{}: median {} (95 % CI {} – {}){}",
                    b.id,
                    duration(b.median),
                    duration(b.lo),
                    duration(b.hi),
                    b.reference
                        .map(|v| format!("; docs/BENCHMARKS.md: {}", duration(v)))
                        .unwrap_or_default()
                ),
            });
        }
    }
    let chart = svg::hbar_log_chart(
        "Criterion medians",
        &rows,
        1100.0,
        duration,
        "docs/BENCHMARKS.md (Apple M3)",
    );
    let mut html = r.svg(
        "benchmarks",
        &chart,
        &format!(
            "{} benchmarks{}. Throughput (elements per second) where the bench declares it.",
            benches.len(),
            if ran {
                ""
            } else {
                " (reused existing results)"
            }
        ),
    );
    html.push_str("<details><summary>Table</summary><table class=\"metrics\"><thead><tr><th>id</th><th class=\"num\">median</th><th class=\"num\">95 % CI</th><th class=\"num\">docs/BENCHMARKS.md</th></tr></thead><tbody>");
    for row in &rows {
        let _ = write!(
            html,
            "<tr><td><code>{}/{}</code></td><td class=\"num\">{}</td><td class=\"num\">{} – {}</td><td class=\"num\">{}</td></tr>",
            esc(&row.group),
            esc(&row.label),
            duration(row.value),
            duration(row.lo),
            duration(row.hi),
            row.reference.map(duration).unwrap_or_else(|| "–".into())
        );
    }
    html.push_str("</tbody></table></details>");
    r.html.push_str(&html);
}

// ------------------------------------------------------------- overview

fn mermaid_blocks(md: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current: Option<String> = None;
    for line in md.lines() {
        match &mut current {
            None if line.trim() == "```mermaid" => current = Some(String::new()),
            Some(block) if line.trim() == "```" => {
                out.push(std::mem::take(block));
                current = None;
            }
            Some(block) => {
                block.push_str(line);
                block.push('\n');
            }
            None => {}
        }
    }
    out
}

fn overview(
    r: &mut Report,
    root: &Path,
    porting: &Porting,
    e2e: &BTreeMap<String, Value>,
    benches: &[Bench],
    notes: &[&str],
) {
    r.section(
        "overview",
        "Overview",
        "<strong>cartographer-rust</strong> is a pure-Rust port of <a href=\"https://github.com/cartographer-project/cartographer\">Google Cartographer</a> \
         (real-time 2D/3D LiDAR SLAM): local SLAM (scan matching into submaps), the global pose graph (constraint search with \
         branch-and-bound scan matching, sparse pose adjustment with a Rust Ceres replacement), serialization (pbstream), \
         the Lua configuration files and the cloud API. This report collects what the port achieves on simulated data, \
         how much of the C++ code and its tests are ported, and how fast the hot paths are.",
    );
    for n in notes {
        let _ = write!(r.html, "<p class=\"warn\">{}</p>", esc(n));
    }
    let passed = e2e
        .values()
        .filter(|v| v.get("passed").and_then(Value::bool) == Some(true))
        .count();
    let t = &porting.totals;
    let _ = write!(
        r.html,
        "<div class=\"tiles\">{}{}{}{}</div>",
        tile(
            &format!("{passed} / {}", e2e.len()),
            "e2e scenarios passing",
            "simulated 2D/3D datasets vs ground truth"
        ),
        tile(
            &format!(
                "{:.0} %",
                100.0 * t.ported_units as f64 / (t.units - t.na_units).max(1) as f64
            ),
            "C++ units ported",
            &format!("{} of {} applicable", t.ported_units, t.units - t.na_units)
        ),
        tile(
            &t.rust_tests.to_string(),
            "Rust tests",
            &format!(
                "{} of {} applicable C++ tests matched",
                t.ported_tests,
                t.cpp_tests - t.na_tests
            )
        ),
        tile(
            &benches.len().to_string(),
            "benchmarks",
            "criterion, see below"
        ),
    );
    let md = fs::read_to_string(root.join("docs/architecture/OVERVIEW.md")).unwrap_or_default();
    let blocks = mermaid_blocks(&md);
    let titles = [
        (
            "Module dependency graph",
            "Arrows point from a module to the modules it depends on.",
        ),
        (
            "End-to-end data flow",
            "From sensor data to optimized poses and the serialized state.",
        ),
    ];
    for (i, src) in blocks.iter().enumerate() {
        let (title, caption) = titles.get(i).copied().unwrap_or(("Diagram", ""));
        let fallback = if src.trim_start().starts_with("sequenceDiagram") {
            diagrams::sequence(src, title)
        } else {
            diagrams::flowchart(src, title)
        };
        let name = format!("architecture_{i}");
        fs::write(r.assets.join(format!("{name}.svg")), fallback.standalone()).expect("asset");
        let _ = write!(
            r.html,
            "<h3>{title}</h3><figure class=\"card diagram\"><div class=\"mermaid-out\"></div><div class=\"fallback\">{}</div>\
             <pre class=\"mermaid-src\" hidden>{}</pre><figcaption>{caption} From <a href=\"../docs/architecture/OVERVIEW.md\">docs/architecture/OVERVIEW.md</a>; \
             rendered with Mermaid when online, otherwise with the built-in offline layout \
             (<a class=\"dl\" href=\"assets/{name}.svg\">SVG</a>).</figcaption>\
             <details><summary>Mermaid source</summary><pre>{}</pre></details></figure>",
            fallback.inline(),
            esc(src),
            esc(src)
        );
    }
}

// ----------------------------------------------------------- environment

struct Env {
    rows: Vec<(String, String)>,
    crates: Vec<(String, String, String)>,
    date: String,
    commit: String,
}

fn command(cmd: &str, args: &[&str], dir: &Path) -> Option<String> {
    let out = Command::new(cmd)
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// UTC date-time from the system clock (civil-from-days).
fn now_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02} UTC",
        rem / 3600,
        rem % 3600 / 60
    )
}

fn toml_value(toml: &str, key: &str) -> Option<String> {
    toml.lines().find_map(|l| {
        let (k, v) = l.split_once('=')?;
        (k.trim() == key).then(|| v.trim().trim_matches('"').to_string())
    })
}

fn environment(root: &Path) -> Env {
    let rustc = command("rustc", &["-vV"], root).unwrap_or_default();
    let field = |k: &str| {
        rustc
            .lines()
            .find_map(|l| l.strip_prefix(k))
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    };
    let commit = command("git", &["rev-parse", "--short=12", "HEAD"], root).unwrap_or_default();
    let dirty = command(
        "git",
        &["status", "--porcelain", "--untracked-files=no"],
        root,
    )
    .is_some_and(|s| !s.is_empty());
    let branch = command("git", &["rev-parse", "--abbrev-ref", "HEAD"], root).unwrap_or_default();
    let cargo = command("cargo", &["-V"], root).unwrap_or_default();
    let date = now_utc();
    let workspace = fs::read_to_string(root.join("Cargo.toml")).unwrap_or_default();
    let ws_version = toml_value(&workspace, "version").unwrap_or_default();
    let mut crates = Vec::new();
    let mut dirs: Vec<PathBuf> = fs::read_dir(root.join("crates"))
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    dirs.sort();
    for d in dirs {
        let Ok(toml) = fs::read_to_string(d.join("Cargo.toml")) else {
            continue;
        };
        let name = toml_value(&toml, "name").unwrap_or_default();
        let version = match toml_value(&toml, "version") {
            Some(v) if !v.starts_with('{') => v,
            _ => ws_version.clone(),
        };
        let version = if toml.contains("version.workspace = true") {
            ws_version.clone()
        } else {
            version
        };
        let publish = if toml_value(&toml, "publish").as_deref() == Some("false") {
            "not published"
        } else {
            "publishable"
        };
        crates.push((name, version, publish.to_string()));
    }
    Env {
        rows: vec![
            ("Generated".into(), date.clone()),
            (
                "Git commit".into(),
                format!(
                    "{commit}{}",
                    if dirty {
                        " (with uncommitted changes)"
                    } else {
                        ""
                    }
                ),
            ),
            ("Branch".into(), branch),
            ("rustc".into(), field("release:")),
            ("Host target".into(), field("host:")),
            ("LLVM".into(), field("LLVM version:")),
            ("cargo".into(), cargo),
            (
                "OS / arch".into(),
                format!("{} / {}", std::env::consts::OS, std::env::consts::ARCH),
            ),
            (
                "Logical CPUs".into(),
                std::thread::available_parallelism()
                    .map(|n| n.to_string())
                    .unwrap_or_default(),
            ),
        ],
        crates,
        date,
        commit,
    }
}

fn environment_section(r: &mut Report, env: &Env) {
    r.section("environment", "Environment", "");
    let mut html =
        String::from("<div class=\"grid2 tables\"><div><table class=\"metrics\"><tbody>");
    for (k, v) in &env.rows {
        let _ = write!(
            html,
            "<tr><th>{}</th><td><code>{}</code></td></tr>",
            esc(k),
            esc(v)
        );
    }
    html.push_str("</tbody></table></div><div><table class=\"metrics\"><thead><tr><th>Crate</th><th>Version</th><th></th></tr></thead><tbody>");
    for (n, v, p) in &env.crates {
        let _ = write!(
            html,
            "<tr><td><code>{}</code></td><td>{}</td><td class=\"muted\">{}</td></tr>",
            esc(n),
            esc(v),
            esc(p)
        );
    }
    html.push_str("</tbody></table></div></div>");
    html.push_str(
        "<h3>Regenerate</h3><pre>cargo run --release -p xtask -- report                 # e2e tests + reuse target/criterion\n\
cargo run --release -p xtask -- report --bench         # also run the criterion benches (~1.5 min)\n\
cargo run --release -p xtask -- report --unitree-bag &lt;bag&gt;   # also run the Unitree L1 demo\n\
cargo run --release -p xtask -- report --no-tests      # reuse target/e2e as is</pre>\
<p>See <a href=\"README.md\">report/README.md</a>. Simulated datasets use fixed seeds, so trajectory metrics are reproducible on the same \
platform; wall-clock times and benchmark numbers depend on the machine and its load.</p>",
    );
    r.html.push_str(&html);
}

// ------------------------------------------------------------------ page

const PAGE_CSS: &str = r#"
*{box-sizing:border-box}
html{background:var(--page);color:var(--ink-1);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
body{margin:0}
a{color:var(--series-1)}
code,pre{font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header.top{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--page) 88%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--border)}
header.top .in{max-width:1240px;margin:0 auto;padding:10px 24px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
header.top b{font-size:16px}
header.top nav{display:flex;gap:14px;flex-wrap:wrap;flex:1}
header.top nav a{color:var(--ink-2);text-decoration:none;font-size:14px}
header.top nav a:hover{color:var(--ink-1)}
button.theme{border:1px solid var(--border);background:var(--surface);color:var(--ink-2);border-radius:6px;padding:4px 10px;font:inherit;font-size:13px;cursor:pointer}
main{max-width:1240px;margin:0 auto;padding:8px 24px 64px}
h1{font-size:28px;margin:28px 0 4px}
.sub{color:var(--ink-2);margin:0 0 8px}
section{padding-top:18px;scroll-margin-top:56px}
h2{font-size:22px;margin:28px 0 8px;padding-top:14px;border-top:1px solid var(--border)}
h3{font-size:17px;margin:30px 0 6px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;scroll-margin-top:60px}
h4{font-size:14px;margin:4px 0 8px;color:var(--ink-2)}
.lead{color:var(--ink-2);max-width:900px}
.muted{color:var(--ink-3);font-weight:400;font-size:13px}
.note{color:var(--ink-2);font-size:13px}
.empty,.warn{padding:12px 16px;border:1px dashed var(--axis);border-radius:8px;color:var(--ink-2)}
.warn{border-style:solid;border-color:var(--critical)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px;margin:0}
figure.card{display:flex;flex-direction:column;gap:6px;min-width:0}
figcaption{color:var(--ink-2);font-size:13px}
a.dl{font-size:12px;color:var(--ink-3)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:14px;margin:12px 0}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px;margin:12px 0}
@media (max-width:560px){.grid2,.grid3{grid-template-columns:1fr}}
svg.chart{width:100%;height:auto;display:block}
img{max-width:100%;height:auto;display:block;border-radius:6px}
img.map{image-rendering:pixelated;width:100%}
video{width:100%;border-radius:6px}
.diagram .fallback svg{max-height:760px}
.diagram .mermaid-out{display:none;overflow:auto}
.diagram.has-mermaid .mermaid-out{display:block}
.diagram.has-mermaid .fallback{display:none}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:14px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile-v{font-size:28px;font-weight:600;line-height:1.2}
.tile-l{color:var(--ink-1);font-size:14px}
.tile-s{color:var(--ink-3);font-size:12.5px}
table.metrics{width:100%;border-collapse:collapse;font-size:13.5px;background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden}
table.metrics th,table.metrics td{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:middle}
table.metrics thead th{color:var(--ink-2);font-weight:600;font-size:12.5px}
table.metrics tbody th{color:var(--ink-2);font-weight:500;width:34%}
table.metrics tfoot td{font-weight:600}
.num{text-align:right!important;font-variant-numeric:tabular-nums;white-space:nowrap}
.meter{display:inline-block;width:90px;height:8px;border-radius:4px;background:var(--grid);overflow:hidden;vertical-align:middle}
.meter span{display:block;height:100%;background:var(--series-1);border-radius:4px}
.badge{display:inline-flex;gap:4px;align-items:center;font-size:12px;font-weight:600;padding:1px 8px;border-radius:999px;border:1px solid}
.badge.good{color:var(--good-ink);border-color:var(--good)}
.badge.bad{color:var(--critical-ink);border-color:var(--critical)}
dl.info{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:6px 14px;margin:0;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px 14px}
dl.info dt{color:var(--ink-3);font-size:12px}
dl.info dd{margin:0;font-variant-numeric:tabular-nums}
.tables{align-items:start}
ul.toc{display:flex;flex-wrap:wrap;gap:8px 18px;list-style:none;padding:0}
details{margin:10px 0}
summary{cursor:pointer;color:var(--ink-2)}
pre{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 12px;overflow:auto}
#tip{position:fixed;pointer-events:none;z-index:10;background:var(--surface);color:var(--ink-1);border:1px solid var(--border);border-radius:8px;padding:6px 9px;font-size:12.5px;box-shadow:0 4px 14px rgba(0,0,0,.15);display:none;font-variant-numeric:tabular-nums}
#tip i{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
footer{color:var(--ink-3);font-size:12.5px;text-align:center;padding:24px}
"#;

const PAGE_JS: &str = r#"
(function(){
  const root=document.documentElement, btn=document.getElementById('theme');
  const saved=localStorage.getItem('report-theme'); if(saved) root.dataset.theme=saved;
  const dark=()=>root.dataset.theme? root.dataset.theme==='dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  const label=()=>{btn.textContent=dark()?'Light theme':'Dark theme'}; label();
  btn.onclick=()=>{root.dataset.theme=dark()?'light':'dark'; localStorage.setItem('report-theme',root.dataset.theme); label();};
  const tip=document.getElementById('tip'), ns='http://www.w3.org/2000/svg';
  const fmt=v=>v==null?'–':Math.abs(v)>=100?v.toFixed(0):Math.abs(v)>=10?v.toFixed(1):v.toFixed(2);
  for(const svg of document.querySelectorAll('svg[data-hover]')){
    const h=JSON.parse(svg.dataset.hover);
    const inv=(s,v)=>s[0]+(v-s[2])/(s[3]-s[2])*(s[1]-s[0]), fwd=(s,v)=>s[2]+(v-s[0])/(s[1]-s[0])*(s[3]-s[2]);
    const line=document.createElementNS(ns,'line'); line.setAttribute('class','xhair'); line.setAttribute('y1',h.top); line.setAttribute('y2',h.bottom); line.style.display='none'; svg.appendChild(line);
    const dots=h.s.map(s=>{const c=document.createElementNS(ns,'circle'); c.setAttribute('r',4); c.setAttribute('class','f'+s.c); c.setAttribute('stroke','var(--surface)'); c.setAttribute('stroke-width',2); c.style.display='none'; c.style.pointerEvents='none'; svg.appendChild(c); return c;});
    const hide=()=>{line.style.display='none'; dots.forEach(d=>d.style.display='none'); tip.style.display='none';};
    svg.addEventListener('pointermove',e=>{
      const pt=svg.createSVGPoint(); pt.x=e.clientX; pt.y=e.clientY; const p=pt.matrixTransform(svg.getScreenCTM().inverse());
      if(p.x<h.sx[2]||p.x>h.sx[3]){hide();return;}
      const x=inv(h.sx,p.x); let rows='';
      h.s.forEach((s,i)=>{
        const a=s.p; if(!a.length) return; let lo=0,hi=a.length-1;
        while(hi-lo>1){const m=(lo+hi)>>1; if(a[m][0]<x) lo=m; else hi=m;}
        const q=Math.abs(a[lo][0]-x)<Math.abs(a[hi][0]-x)?a[lo]:a[hi];
        if(q[1]==null){dots[i].style.display='none';return;}
        dots[i].setAttribute('cx',fwd(h.sx,q[0])); dots[i].setAttribute('cy',fwd(h.sy,q[1])); dots[i].style.display='';
        rows+=`<div><i style="background:var(--series-${s.c==='m'?'1':s.c})"></i>${s.l}: <b>${fmt(q[1])}</b> ${h.yu}</div>`;
      });
      line.setAttribute('x1',p.x); line.setAttribute('x2',p.x); line.style.display='';
      tip.innerHTML=`<div>${fmt(x)} ${h.xu}</div>${rows}`; tip.style.display='block';
      const w=tip.offsetWidth; tip.style.left=Math.min(e.clientX+14,innerWidth-w-8)+'px'; tip.style.top=(e.clientY+14)+'px';
    });
    svg.addEventListener('pointerleave',hide);
  }
})();
"#;

const MERMAID_JS: &str = r#"
const blocks=[...document.querySelectorAll('figure.diagram')];
if(blocks.length){
  try{
    const mermaid=(await import('https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs')).default;
    const root=document.documentElement;
    const dark=root.dataset.theme? root.dataset.theme==='dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    mermaid.initialize({startOnLoad:false,theme:dark?'dark':'neutral',securityLevel:'strict'});
    let i=0;
    for(const fig of blocks){
      try{
        const {svg}=await mermaid.render('mermaid-'+(i++),fig.querySelector('pre.mermaid-src').textContent);
        fig.querySelector('.mermaid-out').innerHTML=svg; fig.classList.add('has-mermaid');
      }catch(e){console.warn('mermaid',e);}
    }
  }catch(e){/* offline: keep the built-in rendering */}
}
"#;

fn page(r: &Report, env: &Env) -> String {
    let nav: String = r
        .nav
        .iter()
        .map(|(id, t)| format!("<a href=\"#{id}\">{}</a>", esc(t)))
        .collect();
    format!(
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\
<meta name=\"color-scheme\" content=\"light dark\"><title>cartographer-rust report</title>\
<style>{}{}{}</style></head><body>\
<header class=\"top\"><div class=\"in\"><b>cartographer-rust</b><nav>{nav}</nav><button class=\"theme\" id=\"theme\" type=\"button\">Theme</button></div></header>\
<main><h1>cartographer-rust: visual report</h1><p class=\"sub\">Generated {} from commit <code>{}</code> by <code>cargo run --release -p xtask -- report</code>.</p>\
{}</section></main><footer>Static report: no network needed except to upgrade the architecture diagrams with Mermaid.</footer><div id=\"tip\"></div>\
<script>{PAGE_JS}</script><script type=\"module\">{MERMAID_JS}</script></body></html>\n",
        svg::theme_vars(),
        svg::CHART_CSS,
        PAGE_CSS,
        esc(&env.date),
        esc(&env.commit),
        r.html
    )
}
