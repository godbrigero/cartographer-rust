//! Prints a `rosbag info`-style summary of a ROS1 bag.
//!
//! ```text
//! cargo run -p cartographer-rosbag --example bag_info -- path/to/file.bag
//! ```

use std::process::ExitCode;

use cartographer_rosbag::{Bag, Time};

fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u + 1 < UNITS.len() {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {} ({bytes} bytes)", UNITS[u])
    }
}

fn duration(start: Time, end: Time) -> String {
    let secs = end.as_secs_f64() - start.as_secs_f64();
    let whole = secs as u64;
    if whole >= 60 {
        format!("{}:{:02}s ({secs:.3}s)", whole / 60, whole % 60)
    } else {
        format!("{secs:.3}s")
    }
}

fn run(path: &str) -> cartographer_rosbag::Result<()> {
    let mut bag = Bag::open(path)?;
    println!("path:        {path}");
    println!("version:     2.0");
    match (bag.start_time(), bag.end_time()) {
        (Some(s), Some(e)) => {
            println!("duration:    {}", duration(s, e));
            println!("start:       {s}");
            println!("end:         {e}");
        }
        _ => println!("duration:    0s (empty bag)"),
    }
    println!("size:        {}", human_size(bag.file_len()));
    println!("messages:    {}", bag.message_count());
    let compressions = bag.chunk_compressions()?;
    let total: usize = compressions.iter().map(|(_, n)| n).sum();
    let comp: Vec<String> = compressions
        .iter()
        .map(|(c, n)| format!("{c} [{n}/{total} chunks]"))
        .collect();
    println!("compression: {}", comp.join(", "));

    let topics = bag.topics();
    let mut types: Vec<(&str, &str)> = topics
        .iter()
        .map(|t| (t.msg_type.as_str(), t.md5sum.as_str()))
        .collect();
    types.sort_unstable();
    types.dedup();
    let tw = types.iter().map(|t| t.0.len()).max().unwrap_or(0);
    for (i, (ty, md5)) in types.iter().enumerate() {
        let label = if i == 0 { "types:" } else { "" };
        println!("{label:<13}{ty:<tw$} [{md5}]");
    }
    let nw = topics.iter().map(|t| t.topic.len()).max().unwrap_or(0);
    let cw = topics
        .iter()
        .map(|t| t.message_count.to_string().len())
        .max()
        .unwrap_or(0);
    for (i, t) in topics.iter().enumerate() {
        let label = if i == 0 { "topics:" } else { "" };
        let conns = if t.connections > 1 {
            format!(" ({} connections)", t.connections)
        } else {
            String::new()
        };
        println!(
            "{label:<13}{:<nw$}  {:>cw$} msgs : {}{conns}",
            t.topic, t.message_count, t.msg_type
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bag_info <file.bag> [more.bag ...]");
        return ExitCode::from(2);
    }
    let mut code = ExitCode::SUCCESS;
    for (i, path) in args.iter().enumerate() {
        if i > 0 {
            println!();
        }
        if let Err(e) = run(path) {
            eprintln!("{path}: {e}");
            code = ExitCode::FAILURE;
        }
    }
    code
}
