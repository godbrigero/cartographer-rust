//! Developer tasks for the Cartographer Rust port.
//!
//! `cargo run -p xtask -- porting-status [--cpp <dir>] [--json <file>]` regenerates
//! `PORTING_STATUS.md` from the C++ tree and the Rust sources.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("porting-status") => {
            let root = workspace_root();
            let mut cpp = root.join("../cartographer/cartographer");
            let mut json = None;
            let mut it = args[1..].iter();
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--cpp" => cpp = PathBuf::from(it.next().expect("--cpp <dir>")),
                    "--json" => json = Some(PathBuf::from(it.next().expect("--json <file>"))),
                    other => panic!("unknown argument {other}"),
                }
            }
            let status = porting_status::compute(&cpp, &root);
            fs::write(
                root.join("PORTING_STATUS.md"),
                porting_status::to_markdown(&status),
            )
            .expect("writing PORTING_STATUS.md");
            if let Some(json) = json {
                fs::write(json, porting_status::to_json(&status)).expect("writing json");
            }
            let t = status.totals();
            println!(
                "units ported {}/{} (n/a {}), C++ tests {} ported {} (n/a {}), Rust tests {}",
                t.ported_units,
                t.units,
                t.na_units,
                t.cpp_tests,
                t.ported_tests,
                t.na_tests,
                t.rust_tests
            );
        }
        _ => {
            eprintln!("usage: cargo run -p xtask -- porting-status [--cpp <dir>] [--json <file>]");
            std::process::exit(2);
        }
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

mod porting_status {
    use super::*;

    /// One C++ compilation unit (`foo.h` + `foo.cc` + `foo_test.cc`).
    pub struct Unit {
        pub cpp: String,
        pub rust: String,
        pub rust_exists: bool,
        pub ported: bool,
        pub cpp_tests: Vec<String>,
        pub ported_tests: usize,
        pub rust_tests: usize,
        pub note: Option<String>,
        pub na: bool,
    }

    pub struct Status {
        pub units: Vec<Unit>,
        pub extra_rust_tests: BTreeMap<String, usize>,
    }

    #[derive(Default)]
    pub struct Totals {
        pub units: usize,
        pub ported_units: usize,
        pub na_units: usize,
        pub cpp_tests: usize,
        pub ported_tests: usize,
        pub na_tests: usize,
        pub rust_tests: usize,
    }

    impl Status {
        pub fn totals(&self) -> Totals {
            let mut t = Totals::default();
            for u in &self.units {
                t.units += 1;
                if u.na {
                    t.na_units += 1;
                    t.na_tests += u.cpp_tests.len();
                } else if u.ported {
                    t.ported_units += 1;
                }
                t.cpp_tests += u.cpp_tests.len();
                t.ported_tests += u.ported_tests;
                t.rust_tests += u.rust_tests;
            }
            t.rust_tests += self.extra_rust_tests.values().sum::<usize>();
            t
        }
    }

    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = fs::read_dir(dir) else { return };
        let mut v: Vec<_> = rd.map(|e| e.unwrap().path()).collect();
        v.sort();
        for p in v {
            if p.is_dir() {
                walk(&p, out);
            } else {
                out.push(p);
            }
        }
    }

    fn rust_dir(d: &str) -> String {
        d.split('/')
            .map(|p| match p {
                "2d" => "two_d",
                "3d" => "three_d",
                p => p,
            })
            .collect::<Vec<_>>()
            .join("/")
    }

    fn cpp_tests(src: &str) -> Vec<String> {
        let mut v = Vec::new();
        let mut offset = 0;
        for line in src.split_inclusive('\n') {
            for m in ["TEST(", "TEST_F(", "TYPED_TEST(", "TEST_P("] {
                if line.starts_with(m) {
                    let inner: String = src[offset + m.len()..]
                        .chars()
                        .take_while(|&c| c != ')')
                        .collect();
                    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
                    if parts.len() == 2 {
                        v.push(format!("{}.{}", parts[0], parts[1]));
                    }
                }
            }
            offset += line.len();
        }
        v
    }

    fn normalize(s: &str) -> String {
        s.chars()
            .filter(char::is_ascii_alphanumeric)
            .map(|c| c.to_ascii_lowercase())
            .collect()
    }

    fn rust_fn_names(src: &str) -> Vec<String> {
        src.split("fn ")
            .skip(1)
            .map(|r| {
                normalize(
                    &r.chars()
                        .take_while(|&c| c == '_' || c.is_ascii_alphanumeric())
                        .collect::<String>(),
                )
            })
            .collect()
    }

    fn count_rust_tests(src: &str) -> usize {
        src.matches("#[test]").count()
    }

    fn load_overrides(root: &Path) -> BTreeMap<String, (bool, String)> {
        // Format: `<cpp unit path> | n/a | reason` or `<cpp unit path> | note | text`.
        let mut m = BTreeMap::new();
        if let Ok(s) = fs::read_to_string(root.join("docs/porting_status_overrides.txt")) {
            for line in s.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let parts: Vec<&str> = line.splitn(3, '|').map(str::trim).collect();
                if parts.len() == 3 {
                    m.insert(
                        parts[0].to_string(),
                        (parts[1] == "n/a", parts[2].to_string()),
                    );
                }
            }
        }
        m
    }

    pub fn compute(cpp_root: &Path, root: &Path) -> Status {
        let mut files = Vec::new();
        walk(cpp_root, &mut files);
        let overrides = load_overrides(root);
        let mut units: BTreeMap<String, ()> = BTreeMap::new();
        for f in &files {
            let rel = f
                .strip_prefix(cpp_root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            for ext in [".h", ".cc"] {
                if let Some(stem) = rel.strip_suffix(ext) {
                    let stem = stem.strip_suffix("_test").unwrap_or(stem);
                    units.insert(stem.to_string(), ());
                }
            }
        }
        let mut out = Vec::new();
        let mut seen_rust = std::collections::BTreeSet::new();
        for stem in units.keys() {
            let (d, f) = stem.rsplit_once('/').unwrap_or(("", stem));
            let cloud = stem.starts_with("cloud/");
            let rust = if f.ends_with("_main") {
                format!(
                    "crates/cartographer/src/bin/cartographer_{}.rs",
                    f.trim_end_matches("_main")
                )
            } else if cloud {
                format!(
                    "crates/cartographer-cloud/src/{}/{f}.rs",
                    rust_dir(d.trim_start_matches("cloud/").trim_start_matches("cloud"))
                )
                .replace("src//", "src/")
            } else {
                format!("crates/cartographer/src/{}/{f}.rs", rust_dir(d))
            };
            let rust_path = root.join(&rust);
            let rust_src = fs::read_to_string(&rust_path).unwrap_or_default();
            seen_rust.insert(rust_path.clone());
            let ported = rust_src
                .lines()
                .filter(|l| !l.trim().is_empty() && !l.starts_with("//!"))
                .count()
                > 0;
            let test_src =
                fs::read_to_string(cpp_root.join(format!("{stem}_test.cc"))).unwrap_or_default();
            let tests = cpp_tests(&test_src);
            let fns = rust_fn_names(&rust_src);
            let ported_tests = tests
                .iter()
                .filter(|t| {
                    let (suite, name) = t.split_once('.').unwrap();
                    let n = normalize(name);
                    let sn = normalize(&format!("{suite}{name}"));
                    fns.iter().any(|f| f.starts_with(&n) || f.starts_with(&sn))
                })
                .count();
            let (na, note) = match overrides.get(stem.as_str()) {
                Some((na, n)) => (*na, Some(n.clone())),
                None => (false, None),
            };
            out.push(Unit {
                cpp: stem.clone(),
                rust,
                rust_exists: rust_path.exists(),
                ported,
                rust_tests: count_rust_tests(&rust_src),
                cpp_tests: tests,
                ported_tests,
                note,
                na,
            });
        }
        // Rust tests in files without a C++ counterpart (integration tests, new modules).
        let mut extra = BTreeMap::new();
        let mut all = Vec::new();
        for c in [
            "crates/cartographer/src",
            "crates/cartographer/tests",
            "crates/ceres",
            "crates/sim",
        ] {
            walk(&root.join(c), &mut all);
        }
        for p in all {
            if p.extension().is_some_and(|e| e == "rs") && !seen_rust.contains(&p) {
                let n = count_rust_tests(&fs::read_to_string(&p).unwrap_or_default());
                if n > 0 {
                    extra.insert(
                        p.strip_prefix(root)
                            .unwrap()
                            .to_string_lossy()
                            .replace('\\', "/"),
                        n,
                    );
                }
            }
        }
        Status {
            units: out,
            extra_rust_tests: extra,
        }
    }

    pub fn to_markdown(s: &Status) -> String {
        let t = s.totals();
        let mut m = String::new();
        m.push_str("# Porting status\n\n");
        m.push_str("Generated by `cargo run -p xtask -- porting-status`; do not edit by hand.\n");
        m.push_str("Notes and N/A justifications live in `docs/porting_status_overrides.txt`.\n\n");
        m.push_str(&format!(
            "* C++ units (`.h`/`.cc` pairs incl. `*_main.cc`): **{}**, ported: **{}**, N/A: **{}**\n",
            t.units, t.ported_units, t.na_units
        ));
        m.push_str(&format!(
            "* C++ test cases (`TEST*` in `*_test.cc`): **{}**, ported (matched by name in the Rust file): **{}**, in N/A units: **{}**\n",
            t.cpp_tests, t.ported_tests, t.na_tests
        ));
        m.push_str(&format!(
            "* Rust `#[test]` functions (all crates): **{}**\n\n",
            t.rust_tests
        ));
        m.push_str(
            "Test matching is by snake-cased name (`TEST(Suite, FooBar)` -> `fn foo_bar`); a\n",
        );
        m.push_str(
            "ported test with a different name shows as unmatched, see the notes column.\n\n",
        );
        m.push_str(
            "| C++ unit | Rust file | Status | C++ tests | Matched | Rust tests | Notes |\n",
        );
        m.push_str("|---|---|---|---|---|---|---|\n");
        for u in &s.units {
            let status = if u.na {
                "N/A"
            } else if u.ported {
                if u.ported_tests == u.cpp_tests.len() {
                    "done"
                } else {
                    "ported (tests partial)"
                }
            } else if u.rust_exists {
                "pending"
            } else {
                "missing"
            };
            m.push_str(&format!(
                "| `{}` | `{}` | {} | {} | {} | {} | {} |\n",
                u.cpp,
                u.rust.trim_start_matches("crates/"),
                status,
                u.cpp_tests.len(),
                u.ported_tests,
                u.rust_tests,
                u.note.clone().unwrap_or_default()
            ));
        }
        m.push_str("\n## Additional Rust test files\n\n| File | Tests |\n|---|---|\n");
        for (f, n) in &s.extra_rust_tests {
            m.push_str(&format!("| `{f}` | {n} |\n"));
        }
        m
    }

    pub fn to_json(s: &Status) -> String {
        let t = s.totals();
        let mut j = format!(
            "{{\"units\":{},\"ported_units\":{},\"na_units\":{},\"cpp_tests\":{},\"ported_tests\":{},\"na_tests\":{},\"rust_tests\":{},\"modules\":{{",
            t.units,
            t.ported_units,
            t.na_units,
            t.cpp_tests,
            t.ported_tests,
            t.na_tests,
            t.rust_tests
        );
        let mut per: BTreeMap<String, [usize; 4]> = BTreeMap::new();
        for u in &s.units {
            let module = u.cpp.split('/').next().unwrap().to_string();
            let e = per.entry(module).or_default();
            e[0] += 1;
            e[1] += usize::from(u.ported || u.na);
            e[2] += u.cpp_tests.len();
            e[3] += if u.na {
                u.cpp_tests.len()
            } else {
                u.ported_tests
            };
        }
        let parts: Vec<String> = per
            .iter()
            .map(|(k, v)| format!("\"{k}\":[{},{},{},{}]", v[0], v[1], v[2], v[3]))
            .collect();
        j.push_str(&parts.join(","));
        j.push_str("}}\n");
        j
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn parses_multiline_and_typed_tests() {
            let src =
                "TEST(A, B) {}\nTYPED_TEST(C, D) {}\nTEST(E,\n     F) {}\n  TEST(no, indent)\n";
            assert_eq!(super::cpp_tests(src), vec!["A.B", "C.D", "E.F"]);
        }
    }
}
