//! Bag container: header, index (connections + chunk infos) and time-ordered
//! message iteration across chunks.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::decode::RosMsg;
use crate::record::{self, Fields};
use crate::{Error, Result, Time, normalize_type};

const MAGIC: &[u8] = b"#ROSBAG V2.0\n";

/// A connection (topic + message type) recorded in the bag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Connection {
    /// Connection id referenced by message records.
    pub id: u32,
    /// Topic name.
    pub topic: String,
    /// Message type, normalized to ROS1 form (`pkg/Name`).
    pub msg_type: String,
    /// MD5 sum of the message definition.
    pub md5sum: String,
    /// Full text of the message definition.
    pub message_definition: String,
    /// Publishing node, if recorded.
    pub callerid: Option<String>,
    /// Whether the publisher was latched.
    pub latching: bool,
}

/// Index entry describing one chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkInfo {
    /// File offset of the chunk record.
    pub pos: u64,
    /// Earliest message time in the chunk.
    pub start_time: Time,
    /// Latest message time in the chunk.
    pub end_time: Time,
    /// `(connection id, message count)` pairs.
    pub connection_counts: Vec<(u32, u32)>,
}

/// Per-topic statistics, as shown by `rosbag info`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopicSummary {
    /// Topic name.
    pub topic: String,
    /// Message type.
    pub msg_type: String,
    /// MD5 sum of the message type.
    pub md5sum: String,
    /// Number of messages (summed over connections).
    pub message_count: u64,
    /// Number of connections publishing on this topic.
    pub connections: usize,
}

/// A raw (still serialized) message read from the bag.
#[derive(Debug, Clone)]
pub struct RawMessage<'a> {
    /// The connection the message was recorded on.
    pub connection: &'a Connection,
    /// Record time (time the message was received by the recorder).
    pub time: Time,
    /// ROS1 serialized message bytes.
    pub data: Vec<u8>,
}

impl RawMessage<'_> {
    /// Topic name.
    pub fn topic(&self) -> &str {
        &self.connection.topic
    }

    /// Decodes the payload as `T`, checking the connection's type first.
    pub fn decode<T: RosMsg>(&self) -> Result<T> {
        if self.connection.msg_type != T::MSG_TYPE {
            return Err(Error::TypeMismatch {
                expected: T::MSG_TYPE,
                actual: self.connection.msg_type.clone(),
            });
        }
        T::decode(&self.data)
    }
}

/// An open ROS1 bag (format 2.0).
#[derive(Debug)]
pub struct Bag<R = BufReader<File>> {
    reader: R,
    file_len: u64,
    connections: Vec<Connection>,
    chunks: Vec<ChunkInfo>,
}

impl Bag<BufReader<File>> {
    /// Opens a bag file and reads its index.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_reader(BufReader::new(File::open(path)?))
    }
}

impl<R: Read + Seek> Bag<R> {
    /// Reads the bag header and index from any seekable source.
    pub fn from_reader(mut reader: R) -> Result<Self> {
        let file_len = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(0))?;
        let mut magic = [0u8; MAGIC.len()];
        reader.read_exact(&mut magic).map_err(|_| Error::BadMagic)?;
        if magic != MAGIC {
            return Err(Error::BadMagic);
        }

        let header = record::read_header(&mut reader, file_len)?
            .ok_or_else(|| Error::Malformed("missing bag header record".into()))?;
        if header.op != record::OP_BAG_HEADER {
            return Err(Error::Malformed(format!(
                "first record has op 0x{:02x}, expected bag header",
                header.op
            )));
        }
        let index_pos = header.u64("index_pos")?;
        let conn_count = header.u32("conn_count")?;
        let chunk_count = header.u32("chunk_count")?;
        if index_pos == 0 {
            return Err(Error::Unindexed);
        }
        if index_pos >= file_len {
            return Err(Error::Malformed(format!(
                "index_pos {index_pos} beyond file length {file_len}"
            )));
        }

        reader.seek(SeekFrom::Start(index_pos))?;
        let mut connections = Vec::with_capacity(conn_count.min(1 << 16) as usize);
        let mut chunks = Vec::with_capacity(chunk_count.min(1 << 20) as usize);
        loop {
            let remaining = file_len.saturating_sub(reader.stream_position()?);
            let Some(fields) = record::read_header(&mut reader, remaining)? else {
                break;
            };
            let remaining = file_len.saturating_sub(reader.stream_position()?);
            let data = record::read_data(&mut reader, remaining)?;
            match fields.op {
                record::OP_CONNECTION => connections.push(parse_connection(&fields, &data)?),
                record::OP_CHUNK_INFO => chunks.push(parse_chunk_info(&fields, &data)?),
                _ => {} // Unknown records in the index section are skipped.
            }
        }
        if connections.len() != conn_count as usize {
            return Err(Error::Malformed(format!(
                "bag header announces {conn_count} connections, index has {}",
                connections.len()
            )));
        }
        if chunks.len() != chunk_count as usize {
            return Err(Error::Malformed(format!(
                "bag header announces {chunk_count} chunks, index has {}",
                chunks.len()
            )));
        }
        connections.sort_by_key(|c| c.id);
        Ok(Self {
            reader,
            file_len,
            connections,
            chunks,
        })
    }

    /// All connections, sorted by id.
    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// Chunk index entries, in file order.
    pub fn chunks(&self) -> &[ChunkInfo] {
        &self.chunks
    }

    /// Size of the underlying file in bytes.
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// Looks up a connection by id.
    pub fn connection(&self, id: u32) -> Option<&Connection> {
        self.connections
            .binary_search_by_key(&id, |c| c.id)
            .ok()
            .map(|i| &self.connections[i])
    }

    /// Earliest message time (from the chunk index), if any messages exist.
    pub fn start_time(&self) -> Option<Time> {
        self.non_empty_chunks().map(|c| c.start_time).min()
    }

    /// Latest message time (from the chunk index), if any messages exist.
    pub fn end_time(&self) -> Option<Time> {
        self.non_empty_chunks().map(|c| c.end_time).max()
    }

    fn non_empty_chunks(&self) -> impl Iterator<Item = &ChunkInfo> {
        self.chunks
            .iter()
            .filter(|c| c.connection_counts.iter().any(|&(_, n)| n > 0))
    }

    /// Total number of messages (from the chunk index).
    pub fn message_count(&self) -> u64 {
        self.chunks
            .iter()
            .flat_map(|c| &c.connection_counts)
            .map(|&(_, n)| u64::from(n))
            .sum()
    }

    /// Per-connection message counts (from the chunk index).
    pub fn connection_message_count(&self, id: u32) -> u64 {
        self.chunks
            .iter()
            .flat_map(|c| &c.connection_counts)
            .filter(|&&(c, _)| c == id)
            .map(|&(_, n)| u64::from(n))
            .sum()
    }

    /// Per-topic summaries sorted by topic name.
    pub fn topics(&self) -> Vec<TopicSummary> {
        let mut out: Vec<TopicSummary> = Vec::new();
        for c in &self.connections {
            let n = self.connection_message_count(c.id);
            match out.iter_mut().find(|t| t.topic == c.topic) {
                Some(t) => {
                    t.message_count += n;
                    t.connections += 1;
                }
                None => out.push(TopicSummary {
                    topic: c.topic.clone(),
                    msg_type: c.msg_type.clone(),
                    md5sum: c.md5sum.clone(),
                    message_count: n,
                    connections: 1,
                }),
            }
        }
        out.sort_by(|a, b| a.topic.cmp(&b.topic));
        out
    }

    /// Counts chunks per compression scheme, e.g. `[("lz4", 12)]`. Reads
    /// each chunk's record header (but not its data).
    pub fn chunk_compressions(&mut self) -> Result<Vec<(String, usize)>> {
        let mut out: Vec<(String, usize)> = Vec::new();
        for i in 0..self.chunks.len() {
            let (fields, _) =
                read_chunk_header(&mut self.reader, self.file_len, self.chunks[i].pos)?;
            let c = fields.string("compression")?;
            match out.iter_mut().find(|(n, _)| *n == c) {
                Some((_, k)) => *k += 1,
                None => out.push((c, 1)),
            }
        }
        Ok(out)
    }

    /// Iterates over all messages in time order.
    pub fn messages(&mut self) -> Messages<'_, R> {
        Messages::new(self, None)
    }

    /// Iterates over messages on the given topics, in time order. Errors if
    /// none of the topics exist in the bag.
    pub fn messages_for_topics(&mut self, topics: &[&str]) -> Result<Messages<'_, R>> {
        let ids: HashSet<u32> = self
            .connections
            .iter()
            .filter(|c| topics.contains(&c.topic.as_str()))
            .map(|c| c.id)
            .collect();
        if ids.is_empty() && !topics.is_empty() {
            return Err(Error::Malformed(format!(
                "none of the topics {topics:?} are in the bag"
            )));
        }
        Ok(Messages::new(self, Some(ids)))
    }
}

/// Seeks to `pos` and reads a chunk record header. Returns the header and
/// the number of bytes left in the file after it.
fn read_chunk_header<R: Read + Seek>(
    reader: &mut R,
    file_len: u64,
    pos: u64,
) -> Result<(Fields, u64)> {
    if pos >= file_len {
        return Err(Error::Malformed(format!("chunk position {pos} beyond EOF")));
    }
    reader.seek(SeekFrom::Start(pos))?;
    let fields = record::read_header(reader, file_len - pos)?
        .ok_or_else(|| Error::Malformed(format!("no chunk record at {pos}")))?;
    if fields.op != record::OP_CHUNK {
        return Err(Error::Malformed(format!(
            "record at {pos} has op 0x{:02x}, expected chunk",
            fields.op
        )));
    }
    let remaining = file_len.saturating_sub(reader.stream_position()?);
    Ok((fields, remaining))
}

fn parse_connection(fields: &Fields, data: &[u8]) -> Result<Connection> {
    let id = fields.u32("conn")?;
    let topic = fields.string("topic")?;
    let mut d = Fields::parse(data, false)?;
    d.op = record::OP_CONNECTION; // For error messages.
    Ok(Connection {
        id,
        topic,
        msg_type: normalize_type(&d.string("type")?).into_owned(),
        md5sum: d.string("md5sum")?,
        message_definition: d.opt_string("message_definition")?.unwrap_or_default(),
        callerid: d.opt_string("callerid")?,
        latching: d.get("latching").is_some_and(|v| v == b"1"),
    })
}

fn parse_chunk_info(fields: &Fields, data: &[u8]) -> Result<ChunkInfo> {
    let ver = fields.u32("ver")?;
    if ver != 1 {
        return Err(Error::Malformed(format!(
            "unsupported chunk info version {ver}"
        )));
    }
    let count = fields.u32("count")? as usize;
    if data.len() != count * 8 {
        return Err(Error::Malformed(format!(
            "chunk info data is {} bytes, expected {}",
            data.len(),
            count * 8
        )));
    }
    let connection_counts = data
        .chunks_exact(8)
        .map(|b| {
            (
                u32::from_le_bytes(b[..4].try_into().unwrap()),
                u32::from_le_bytes(b[4..].try_into().unwrap()),
            )
        })
        .collect();
    Ok(ChunkInfo {
        pos: fields.u64("chunk_pos")?,
        start_time: fields.time("start_time")?,
        end_time: fields.time("end_time")?,
        connection_counts,
    })
}

struct Pending {
    time: Time,
    seq: u64,
    conn_index: usize,
    data: Vec<u8>,
}

impl PartialEq for Pending {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Pending {}
impl PartialOrd for Pending {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Pending {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.time, self.seq).cmp(&(other.time, other.seq))
    }
}

/// Time-ordered message iterator returned by [`Bag::messages`].
///
/// Chunks are visited in order of their start time and decompressed lazily;
/// a chunk is only loaded once the next pending message is not earlier than
/// its start time, so overlapping chunks are merged correctly while memory
/// stays bounded by the set of overlapping chunks. Messages with equal times
/// keep file order.
pub struct Messages<'a, R> {
    reader: &'a mut R,
    file_len: u64,
    connections: &'a [Connection],
    chunks: Vec<ChunkInfo>,
    next_chunk: usize,
    filter: Option<HashSet<u32>>,
    heap: BinaryHeap<Reverse<Pending>>,
    seq: u64,
    failed: bool,
}

impl<'a, R: Read + Seek> Messages<'a, R> {
    fn new(bag: &'a mut Bag<R>, filter: Option<HashSet<u32>>) -> Self {
        let mut chunks: Vec<ChunkInfo> = bag
            .chunks
            .iter()
            .filter(|c| {
                c.connection_counts
                    .iter()
                    .any(|(id, n)| *n > 0 && filter.as_ref().is_none_or(|f| f.contains(id)))
            })
            .cloned()
            .collect();
        chunks.sort_by_key(|c| (c.start_time, c.pos));
        Self {
            reader: &mut bag.reader,
            file_len: bag.file_len,
            connections: &bag.connections,
            chunks,
            next_chunk: 0,
            filter,
            heap: BinaryHeap::new(),
            seq: 0,
            failed: false,
        }
    }

    fn load_chunk(&mut self, info: &ChunkInfo) -> Result<()> {
        let (fields, remaining) = read_chunk_header(self.reader, self.file_len, info.pos)?;
        let compression = fields.string("compression")?;
        let size = fields.u32("size")? as usize;
        let raw = record::read_data(self.reader, remaining)?;
        let data = record::decompress(&compression, raw, size)?;

        let mut buf = data.as_slice();
        let mut msgs = Vec::new();
        while let Some((f, payload)) = record::next_in_buf(&mut buf)? {
            if f.op != record::OP_MSG_DATA {
                continue; // Connection records are duplicated in the index.
            }
            let conn = f.u32("conn")?;
            if self.filter.as_ref().is_some_and(|s| !s.contains(&conn)) {
                continue;
            }
            let conn_index = self
                .connections
                .binary_search_by_key(&conn, |c| c.id)
                .map_err(|_| Error::Malformed(format!("message on unknown connection {conn}")))?;
            msgs.push(Pending {
                time: f.time("time")?,
                seq: 0,
                conn_index,
                data: payload.to_vec(),
            });
        }
        // Order within the chunk by time (stable: ties keep file order).
        msgs.sort_by_key(|m| m.time);
        for mut m in msgs {
            m.seq = self.seq;
            self.seq += 1;
            self.heap.push(Reverse(m));
        }
        Ok(())
    }

    fn advance(&mut self) -> Result<Option<RawMessage<'a>>> {
        loop {
            let next_start = self.chunks.get(self.next_chunk).map(|c| c.start_time);
            let should_load = match (next_start, self.heap.peek()) {
                (None, _) => false,
                (Some(_), None) => true,
                (Some(start), Some(Reverse(top))) => start <= top.time,
            };
            if !should_load {
                break;
            }
            let info = self.chunks[self.next_chunk].clone();
            self.next_chunk += 1;
            self.load_chunk(&info)?;
        }
        Ok(self.heap.pop().map(|Reverse(m)| RawMessage {
            connection: &self.connections[m.conn_index],
            time: m.time,
            data: m.data,
        }))
    }
}

impl<'a, R: Read + Seek> Iterator for Messages<'a, R> {
    type Item = Result<RawMessage<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        match self.advance() {
            Ok(m) => m.map(Ok),
            Err(e) => {
                self.failed = true;
                Some(Err(e))
            }
        }
    }
}
