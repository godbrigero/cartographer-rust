//! Tiny hand-rolled ROS1 bag v2.0 writer and message serializer for tests.

#![allow(dead_code)]

use std::io::Write;

/// Serializes ROS1 messages field by field.
#[derive(Default)]
pub struct Ser(pub Vec<u8>);

impl Ser {
    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.0.push(v);
        self
    }
    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn f64(&mut self, v: f64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn f64s(&mut self, v: &[f64]) -> &mut Self {
        for &x in v {
            self.f64(x);
        }
        self
    }
    pub fn string(&mut self, s: &str) -> &mut Self {
        self.u32(s.len() as u32);
        self.0.extend_from_slice(s.as_bytes());
        self
    }
    pub fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.0.extend_from_slice(b);
        self
    }
    pub fn header(&mut self, seq: u32, sec: u32, nsec: u32, frame: &str) -> &mut Self {
        self.u32(seq).u32(sec).u32(nsec).string(frame)
    }
    pub fn take(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.0)
    }
}

fn field(name: &str, value: &[u8]) -> Vec<u8> {
    let mut f = Vec::new();
    f.extend_from_slice(&((name.len() + 1 + value.len()) as u32).to_le_bytes());
    f.extend_from_slice(name.as_bytes());
    f.push(b'=');
    f.extend_from_slice(value);
    f
}

fn record(out: &mut Vec<u8>, header: &[(&str, Vec<u8>)], data: &[u8]) {
    let h: Vec<u8> = header.iter().flat_map(|(n, v)| field(n, v)).collect();
    out.extend_from_slice(&(h.len() as u32).to_le_bytes());
    out.extend_from_slice(&h);
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(data);
}

fn time_bytes(nanos: u64) -> Vec<u8> {
    let sec = (nanos / 1_000_000_000) as u32;
    let nsec = (nanos % 1_000_000_000) as u32;
    let mut b = sec.to_le_bytes().to_vec();
    b.extend_from_slice(&nsec.to_le_bytes());
    b
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Compression {
    None,
    Lz4,
}

pub struct TestConnection {
    pub topic: String,
    pub msg_type: String,
}

/// Messages are grouped into explicit chunks so tests control chunk layout.
pub struct BagWriter {
    pub compression: Compression,
    pub connections: Vec<TestConnection>,
    /// Each chunk: list of `(conn id, time ns, payload)` in file order.
    pub chunks: Vec<Vec<(u32, u64, Vec<u8>)>>,
}

impl BagWriter {
    pub fn new(compression: Compression) -> Self {
        Self {
            compression,
            connections: Vec::new(),
            chunks: vec![Vec::new()],
        }
    }

    pub fn add_connection(&mut self, topic: &str, msg_type: &str) -> u32 {
        self.connections.push(TestConnection {
            topic: topic.into(),
            msg_type: msg_type.into(),
        });
        (self.connections.len() - 1) as u32
    }

    pub fn write(&mut self, conn: u32, time_ns: u64, data: Vec<u8>) {
        self.chunks.last_mut().unwrap().push((conn, time_ns, data));
    }

    pub fn new_chunk(&mut self) {
        self.chunks.push(Vec::new());
    }

    fn connection_record(&self, out: &mut Vec<u8>, id: u32) {
        let c = &self.connections[id as usize];
        let data: Vec<u8> = [
            field("topic", c.topic.as_bytes()),
            field("type", c.msg_type.as_bytes()),
            field("md5sum", b"0123456789abcdef0123456789abcdef"),
            field("message_definition", b"# test\n"),
            field("callerid", b"/test_node"),
            field("latching", b"0"),
        ]
        .concat();
        record(
            out,
            &[
                ("op", vec![0x07]),
                ("conn", id.to_le_bytes().to_vec()),
                ("topic", c.topic.as_bytes().to_vec()),
            ],
            &data,
        );
    }

    pub fn finish(&self) -> Vec<u8> {
        let mut out = b"#ROSBAG V2.0\n".to_vec();
        let header_pos = out.len();
        // Placeholder bag header; rewritten at the end.
        let bag_header = |index_pos: u64, conns: u32, chunks: u32| {
            let mut r = Vec::new();
            let h: Vec<(&str, Vec<u8>)> = vec![
                ("op", vec![0x03]),
                ("index_pos", index_pos.to_le_bytes().to_vec()),
                ("conn_count", conns.to_le_bytes().to_vec()),
                ("chunk_count", chunks.to_le_bytes().to_vec()),
            ];
            let hb: Vec<u8> = h.iter().flat_map(|(n, v)| field(n, v)).collect();
            r.extend_from_slice(&(hb.len() as u32).to_le_bytes());
            r.extend_from_slice(&hb);
            let pad = 4096 - 4 - r.len();
            r.extend_from_slice(&(pad as u32).to_le_bytes());
            r.extend(std::iter::repeat_n(b' ', pad));
            r
        };
        out.extend(bag_header(0, 0, 0));

        let mut chunk_infos = Vec::new();
        for (ci, chunk) in self
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_empty())
        {
            let mut raw = Vec::new();
            if ci == 0 {
                // Like rosbag, connection records also appear inside chunks.
                for id in 0..self.connections.len() as u32 {
                    self.connection_record(&mut raw, id);
                }
            }
            let mut index: Vec<(u32, Vec<(u64, u32)>)> = Vec::new();
            for (conn, t, data) in chunk {
                let offset = raw.len() as u32;
                record(
                    &mut raw,
                    &[
                        ("op", vec![0x02]),
                        ("conn", conn.to_le_bytes().to_vec()),
                        ("time", time_bytes(*t)),
                    ],
                    data,
                );
                match index.iter_mut().find(|(c, _)| c == conn) {
                    Some((_, v)) => v.push((*t, offset)),
                    None => index.push((*conn, vec![(*t, offset)])),
                }
            }
            let (name, compressed) = match self.compression {
                Compression::None => ("none", raw.clone()),
                Compression::Lz4 => {
                    let mut enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
                    enc.write_all(&raw).unwrap();
                    ("lz4", enc.finish().unwrap())
                }
            };
            let pos = out.len() as u64;
            record(
                &mut out,
                &[
                    ("op", vec![0x05]),
                    ("compression", name.as_bytes().to_vec()),
                    ("size", (raw.len() as u32).to_le_bytes().to_vec()),
                ],
                &compressed,
            );
            for (conn, items) in &index {
                let data: Vec<u8> = items
                    .iter()
                    .flat_map(|(t, o)| [time_bytes(*t), o.to_le_bytes().to_vec()].concat())
                    .collect();
                record(
                    &mut out,
                    &[
                        ("op", vec![0x04]),
                        ("ver", 1u32.to_le_bytes().to_vec()),
                        ("conn", conn.to_le_bytes().to_vec()),
                        ("count", (items.len() as u32).to_le_bytes().to_vec()),
                    ],
                    &data,
                );
            }
            let start = chunk.iter().map(|m| m.1).min().unwrap();
            let end = chunk.iter().map(|m| m.1).max().unwrap();
            chunk_infos.push((pos, start, end, index));
        }

        let index_pos = out.len() as u64;
        for id in 0..self.connections.len() as u32 {
            self.connection_record(&mut out, id);
        }
        for (pos, start, end, index) in &chunk_infos {
            let data: Vec<u8> = index
                .iter()
                .flat_map(|(c, v)| [c.to_le_bytes(), (v.len() as u32).to_le_bytes()].concat())
                .collect();
            record(
                &mut out,
                &[
                    ("op", vec![0x06]),
                    ("ver", 1u32.to_le_bytes().to_vec()),
                    ("chunk_pos", pos.to_le_bytes().to_vec()),
                    ("start_time", time_bytes(*start)),
                    ("end_time", time_bytes(*end)),
                    ("count", (index.len() as u32).to_le_bytes().to_vec()),
                ],
                &data,
            );
        }
        let hdr = bag_header(
            index_pos,
            self.connections.len() as u32,
            chunk_infos.len() as u32,
        );
        out[header_pos..header_pos + hdr.len()].copy_from_slice(&hdr);
        out
    }
}
