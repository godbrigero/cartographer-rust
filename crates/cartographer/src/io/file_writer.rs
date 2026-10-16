//! Port of `cartographer/io/file_writer` (.h, .cc).

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::sync::Arc;

/// Simple abstraction for a file.
pub trait FileWriter: Send {
    /// Write `data` to the beginning of the file. This is required to
    /// overwrite fixed size headers which contain the number of points once we
    /// actually know how many points there are.
    fn write_header(&mut self, data: &[u8]) -> bool;

    /// Appends `data` at the current position.
    fn write(&mut self, data: &[u8]) -> bool;

    /// Closes the file. Returns false on failure.
    fn close(&mut self) -> bool;

    /// The name of the file.
    fn get_filename(&self) -> String;
}

/// Creates a [`FileWriter`] for a filename.
///
/// RUST PORT NOTE: C++ `std::function<std::unique_ptr<FileWriter>(const
/// std::string&)>`; shared (`Arc`) because processors keep a copy.
pub type FileWriterFactory = Arc<dyn Fn(&str) -> Box<dyn FileWriter> + Send + Sync>;

/// An implementation of [`FileWriter`] using a [`File`] (C++ `std::ofstream`
/// opened with `out | binary`).
///
/// RUST PORT NOTE: like the C++ stream, failing to open the file is not
/// reported by the constructor; all subsequent operations return false.
#[derive(Debug)]
pub struct StreamFileWriter {
    filename: String,
    // `None` once the file failed to open or was closed.
    out: Option<BufWriter<File>>,
    // C++ `badbit`.
    bad: bool,
}

impl StreamFileWriter {
    /// Creates (truncates) `filename`.
    pub fn new(filename: &str) -> Self {
        let out = match File::create(filename) {
            Ok(file) => Some(BufWriter::new(file)),
            Err(e) => {
                log::error!("Failed to open '{filename}': {e}");
                None
            }
        };
        let bad = out.is_none();
        Self {
            filename: filename.to_string(),
            out,
            bad,
        }
    }

    /// Returns a [`FileWriterFactory`] creating `StreamFileWriter`s.
    pub fn factory() -> FileWriterFactory {
        Arc::new(|filename: &str| Box::new(StreamFileWriter::new(filename)) as Box<dyn FileWriter>)
    }
}

impl FileWriter for StreamFileWriter {
    fn write(&mut self, data: &[u8]) -> bool {
        if self.bad {
            return false;
        }
        let Some(out) = self.out.as_mut() else {
            self.bad = true;
            return false;
        };
        if out.write_all(data).is_err() {
            self.bad = true;
        }
        !self.bad
    }

    fn write_header(&mut self, data: &[u8]) -> bool {
        if self.bad {
            return false;
        }
        let Some(out) = self.out.as_mut() else {
            self.bad = true;
            return false;
        };
        if out.flush().is_err() || out.seek(SeekFrom::Start(0)).is_err() {
            self.bad = true;
            return false;
        }
        self.write(data)
    }

    fn close(&mut self) -> bool {
        if self.bad {
            return false;
        }
        if let Some(mut out) = self.out.take()
            && out.flush().is_err()
        {
            self.bad = true;
        }
        !self.bad
    }

    fn get_filename(&self) -> String {
        self.filename.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_file_writer_overwrites_header() {
        let path = std::env::temp_dir().join(format!(
            "cartographer_stream_file_writer_{}.txt",
            std::process::id()
        ));
        let filename = path.to_str().unwrap();
        let mut writer = StreamFileWriter::new(filename);
        assert_eq!(writer.get_filename(), filename);
        assert!(writer.write_header(b"HEADER 000\n"));
        assert!(writer.write(b"payload"));
        assert!(writer.write_header(b"HEADER 123\n"));
        assert!(writer.close());
        // Writing after closing fails.
        assert!(!writer.write(b"x"));
        assert_eq!(std::fs::read(&path).unwrap(), b"HEADER 123\npayload");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn stream_file_writer_reports_open_failure() {
        let mut writer = StreamFileWriter::new("/nonexistent_directory_for_test/x/y.txt");
        assert!(!writer.write(b"data"));
        assert!(!writer.close());
    }
}
