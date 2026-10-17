//! Offline fallback renderings of the two Mermaid diagram kinds used in
//! `docs/architecture/OVERVIEW.md`: a layered layout for `graph TD` and a
//! column layout for `sequenceDiagram` (only the syntax those use).

use std::collections::BTreeMap;
use std::fmt::Write as _;

use super::svg::{Svg, esc};

fn lines_of(label: &str) -> Vec<String> {
    label
        .split("<br/>")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// `id[label]` or `id` -> `(id, Some(label))`.
fn node_ref(s: &str) -> (String, Option<String>) {
    let s = s.trim();
    match s.split_once('[') {
        Some((id, rest)) => (
            id.trim().to_string(),
            Some(rest.trim_end_matches(']').to_string()),
        ),
        None => (s.to_string(), None),
    }
}

/// Layered rendering of a `graph TD` diagram.
pub fn flowchart(src: &str, title: &str) -> Svg {
    let mut labels: BTreeMap<String, String> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut edges: Vec<(String, String)> = Vec::new();
    let mut grouped: BTreeMap<String, String> = BTreeMap::new();
    let mut group: Option<String> = None;
    let mut add = |id: &str, label: Option<String>, order: &mut Vec<String>| {
        if !order.iter().any(|o| o == id) {
            order.push(id.to_string());
        }
        if let Some(l) = label {
            labels.insert(id.to_string(), l);
        }
    };
    for line in src.lines().map(str::trim) {
        if line.starts_with("graph") || line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("subgraph ") {
            let (_, l) = node_ref(rest);
            group = Some(l.unwrap_or_else(|| rest.to_string()));
            continue;
        }
        if line == "end" {
            group = None;
            continue;
        }
        if let Some((a, b)) = line.split_once("-->") {
            let (ia, la) = node_ref(a);
            let (ib, lb) = node_ref(b);
            add(&ia, la, &mut order);
            add(&ib, lb, &mut order);
            edges.push((ia, ib));
        } else {
            let (id, l) = node_ref(line);
            add(&id, l, &mut order);
            if let Some(g) = &group {
                grouped.insert(id, g.clone());
            }
        }
    }
    // Longest-path layering (edges point down, to dependencies).
    let mut layer: BTreeMap<&str, usize> = order.iter().map(|n| (n.as_str(), 0)).collect();
    for _ in 0..order.len() {
        let mut changed = false;
        for (a, b) in &edges {
            let la = layer[a.as_str()];
            if layer[b.as_str()] < la + 1 {
                layer.insert(b, la + 1);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let depth = layer.values().copied().max().unwrap_or(0) + 1;
    let size = |id: &str| {
        let l = lines_of(labels.get(id).map_or(id, String::as_str));
        let w = l.iter().map(|s| s.chars().count()).max().unwrap_or(1) as f64 * 6.4 + 20.0;
        (w, l.len() as f64 * 14.0 + 12.0, l)
    };
    let mut rows: Vec<Vec<&str>> = vec![Vec::new(); depth];
    for id in &order {
        rows[layer[id.as_str()]].push(id);
    }
    // Order each row by the mean position of its parents (one sweep).
    let mut pos: BTreeMap<&str, f64> = BTreeMap::new();
    for row in rows.iter_mut() {
        let key = |id: &str| {
            let parents: Vec<f64> = edges
                .iter()
                .filter(|(_, b)| b == id)
                .filter_map(|(a, _)| pos.get(a.as_str()).copied())
                .collect();
            if parents.is_empty() {
                f64::MAX
            } else {
                parents.iter().sum::<f64>() / parents.len() as f64
            }
        };
        let mut keyed: Vec<(f64, usize, &str)> = row
            .iter()
            .enumerate()
            .map(|(i, id)| (key(id), i, *id))
            .collect();
        keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        *row = keyed.into_iter().map(|k| k.2).collect();
        for (i, id) in row.iter().enumerate() {
            pos.insert(id, i as f64 / row.len().max(1) as f64);
        }
    }
    let gap = 18.0;
    let row_widths: Vec<f64> = rows
        .iter()
        .map(|r| r.iter().map(|id| size(id).0 + gap).sum::<f64>() - gap)
        .collect();
    let width = row_widths.iter().copied().fold(400.0, f64::max) + 40.0;
    let row_h: Vec<f64> = rows
        .iter()
        .map(|r| r.iter().map(|id| size(id).1).fold(0.0, f64::max))
        .collect();
    let mut boxes: BTreeMap<&str, (f64, f64, f64, f64)> = BTreeMap::new();
    let mut y = 20.0;
    for (ri, r) in rows.iter().enumerate() {
        let mut x = (width - row_widths[ri]) / 2.0;
        for id in r {
            let (w, h, _) = size(id);
            boxes.insert(id, (x, y + (row_h[ri] - h) / 2.0, w, h));
            x += w + gap;
        }
        y += row_h[ri] + 46.0;
    }
    let height = y - 26.0;
    let mut b = String::from(
        "<defs><marker id=\"fc-arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\"><path class=\"fm\" d=\"M0 0L10 5L0 10z\"/></marker></defs>",
    );
    for (a, c) in &edges {
        let (ax, ay, aw, ah) = boxes[a.as_str()];
        let (cx, cy, cw, _) = boxes[c.as_str()];
        let (x0, y0, x1, y1) = (ax + aw / 2.0, ay + ah, cx + cw / 2.0, cy);
        let m = (y0 + y1) / 2.0;
        let _ = write!(
            b,
            "<path class=\"ln sm\" stroke-width=\"1\" opacity=\"0.8\" marker-end=\"url(#fc-arrow)\" d=\"M{x0:.1} {y0:.1}C{x0:.1} {m:.1} {x1:.1} {m:.1} {x1:.1} {y1:.1}\"/>"
        );
    }
    for id in &order {
        let (x, y, w, h) = boxes[id.as_str()];
        let (_, _, l) = size(id);
        let external = grouped.contains_key(id);
        let _ = write!(
            b,
            "<g><title>{}</title><rect class=\"fsurf {}\" x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" rx=\"6\" stroke-width=\"1.2\"{}/>",
            esc(&l.join(" ")),
            if external { "s-ink2" } else { "s1" },
            if external {
                " stroke-dasharray=\"4 3\""
            } else {
                ""
            }
        );
        for (i, line) in l.iter().enumerate() {
            let _ = write!(
                b,
                "<text class=\"{}\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>",
                if i == 0 { "lbl-b" } else { "tick" },
                x + w / 2.0,
                y + 18.0 + 14.0 * i as f64,
                esc(line)
            );
        }
        b.push_str("</g>");
    }
    Svg {
        width,
        height,
        title: title.into(),
        body: b,
        hover: None,
    }
}

enum Step {
    Message {
        from: usize,
        to: usize,
        text: String,
        dashed: bool,
    },
    Note {
        at: usize,
        lines: Vec<String>,
    },
}

/// Column rendering of a `sequenceDiagram`.
pub fn sequence(src: &str, title: &str) -> Svg {
    let mut participants: Vec<(String, String)> = Vec::new();
    let mut steps = Vec::new();
    let index = |participants: &mut Vec<(String, String)>, id: &str| {
        let id = id.trim();
        participants
            .iter()
            .position(|p| p.0 == id)
            .unwrap_or_else(|| {
                participants.push((id.to_string(), id.to_string()));
                participants.len() - 1
            })
    };
    for line in src.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix("participant ") {
            let (id, name) = rest.split_once(" as ").unwrap_or((rest, rest));
            participants.push((id.trim().to_string(), name.trim().to_string()));
        } else if let Some(rest) = line.strip_prefix("Note over ") {
            let (who, text) = rest.split_once(':').unwrap_or((rest, ""));
            let at = index(&mut participants, who.split(',').next().unwrap_or(who));
            steps.push(Step::Note {
                at,
                lines: lines_of(text),
            });
        } else if let Some((lhs, text)) = line.split_once(':') {
            let (dashed, sep) = if lhs.contains("-->>") {
                (true, "-->>")
            } else if lhs.contains("->>") {
                (false, "->>")
            } else {
                continue;
            };
            let (a, c) = lhs.split_once(sep).unwrap();
            let from = index(&mut participants, a);
            let to = index(&mut participants, c);
            steps.push(Step::Message {
                from,
                to,
                text: text.trim().to_string(),
                dashed,
            });
        }
    }
    let col = 150.0;
    let width = col * participants.len() as f64 + 20.0;
    let cx = |i: usize| 10.0 + col * (i as f64 + 0.5);
    let head = 52.0;
    let mut y = head + 26.0;
    let mut body = String::new();
    for s in &steps {
        match s {
            Step::Message {
                from,
                to,
                text,
                dashed,
            } => {
                let dash = if *dashed {
                    " stroke-dasharray=\"5 4\""
                } else {
                    ""
                };
                if from == to {
                    let x = cx(*from);
                    let _ = write!(
                        body,
                        "<path class=\"ln si\" stroke-width=\"1.2\"{dash} marker-end=\"url(#sq-arrow)\" d=\"M{x:.1} {y:.1}h34v14h-34\"/>\
                         <text class=\"tick\" x=\"{:.1}\" y=\"{:.1}\">{}</text>",
                        x + 40.0,
                        y + 11.0,
                        esc(text)
                    );
                    y += 34.0;
                } else {
                    let (x0, x1) = (cx(*from), cx(*to));
                    let _ = write!(
                        body,
                        "<text class=\"tick\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>\
                         <line class=\"si\" stroke-width=\"1.2\"{dash} marker-end=\"url(#sq-arrow)\" x1=\"{x0:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\"/>",
                        (x0 + x1) / 2.0,
                        y - 5.0,
                        esc(text),
                        x1 + if x1 > x0 { -2.0 } else { 2.0 }
                    );
                    y += 30.0;
                }
            }
            Step::Note { at, lines } => {
                let w =
                    lines.iter().map(|l| l.chars().count()).max().unwrap_or(1) as f64 * 5.9 + 20.0;
                let h = lines.len() as f64 * 14.0 + 10.0;
                let x = cx(*at) - w / 2.0;
                let _ = write!(
                    body,
                    "<rect class=\"ftrack s-ink2\" stroke-width=\"0.8\" x=\"{x:.1}\" y=\"{:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" rx=\"4\"/>",
                    y - 12.0
                );
                for (i, l) in lines.iter().enumerate() {
                    let _ = write!(
                        body,
                        "<text class=\"tick\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>",
                        cx(*at),
                        y + 3.0 + 14.0 * i as f64,
                        esc(l)
                    );
                }
                y += h + 16.0;
            }
        }
    }
    let height = y + 4.0;
    let mut b = String::from(
        "<defs><marker id=\"sq-arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\"><path class=\"fi\" d=\"M0 0L10 5L0 10z\"/></marker></defs>",
    );
    for (i, (_, name)) in participants.iter().enumerate() {
        let x = cx(i);
        let words = name.replace(" (", "\n(");
        let lines: Vec<&str> = words.lines().collect();
        let _ = write!(
            b,
            "<line class=\"sm\" stroke-dasharray=\"3 4\" x1=\"{x:.1}\" y1=\"{head}\" x2=\"{x:.1}\" y2=\"{height:.1}\"/>\
             <rect class=\"fsurf s1\" stroke-width=\"1.2\" x=\"{:.1}\" y=\"8\" width=\"{:.1}\" height=\"{:.1}\" rx=\"6\"/>",
            x - col / 2.0 + 8.0,
            col - 16.0,
            head - 8.0
        );
        for (k, l) in lines.iter().enumerate() {
            let _ = write!(
                b,
                "<text class=\"lbl-b\" x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>",
                26.0 + 14.0 * k as f64 + if lines.len() == 1 { 8.0 } else { 0.0 },
                esc(l)
            );
        }
    }
    b.push_str(&body);
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
    #[test]
    fn layouts_parse_overview_syntax() {
        let g = super::flowchart(
            "graph TD\n  subgraph ext[E]\n    n[N = x]\n  end\n  a --> b[B<br/>c]\n  b --> n\n",
            "g",
        );
        assert_eq!(g.body.matches("<rect").count(), 3);
        let s = super::sequence(
            "sequenceDiagram\n  participant A as Alpha\n  A->>B: hi\n  B-->>A: ok\n  Note over A: x<br/>y\n  B->>B: self\n",
            "s",
        );
        assert_eq!(s.body.matches("marker-end").count(), 3);
    }
}
