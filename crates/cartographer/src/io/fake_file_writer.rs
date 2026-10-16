//! Port of `cartographer/io/fake_file_writer` (.h, .cc).

use std::sync::{Arc, Mutex};

use crate::io::file_writer::{FileWriter, FileWriterFactory};

/// Fakes a [`FileWriter`] by just writing the data to a shared byte vector.
///
/// RUST PORT NOTE: C++ `std::shared_ptr<std::vector<char>>` ->
/// `Arc<Mutex<Vec<u8>>>`. The gtest `EXPECT_FALSE(is_closed_)` checks become
/// assertions.
#[derive(Debug)]
pub struct FakeFileWriter {
    is_closed: bool,
    content: Arc<Mutex<Vec<u8>>>,
    filename: String,
}

impl FakeFileWriter {
    /// Creates a writer appending to `content`.
    pub fn new(filename: &str, content: Arc<Mutex<Vec<u8>>>) -> Self {
        Self {
            is_closed: false,
            content,
            filename: filename.to_string(),
        }
    }

    /// Returns a factory creating `FakeFileWriter`s that all write into
    /// `content`.
    pub fn factory(content: Arc<Mutex<Vec<u8>>>) -> FileWriterFactory {
        Arc::new(move |filename: &str| {
            Box::new(FakeFileWriter::new(filename, Arc::clone(&content))) as Box<dyn FileWriter>
        })
    }
}

impl FileWriter for FakeFileWriter {
    fn write(&mut self, data: &[u8]) -> bool {
        assert!(!self.is_closed, "Write on a closed FakeFileWriter");
        self.content.lock().unwrap().extend_from_slice(data);
        true
    }

    fn close(&mut self) -> bool {
        assert!(!self.is_closed, "Close on a closed FakeFileWriter");
        self.is_closed = true;
        true
    }

    fn write_header(&mut self, data: &[u8]) -> bool {
        assert!(!self.is_closed, "WriteHeader on a closed FakeFileWriter");
        let mut content = self.content.lock().unwrap();
        if content.len() < data.len() {
            content.resize(data.len(), 0);
        }
        content[..data.len()].copy_from_slice(data);
        true
    }

    fn get_filename(&self) -> String {
        self.filename.clone()
    }
}

#[cfg(test)]
mod tests {
    mod fake_file_writer {
        use std::sync::{Arc, Mutex};

        use crate::io::fake_file_writer::FakeFileWriter;
        use crate::io::file_writer::FileWriter;

        fn new_content() -> Arc<Mutex<Vec<u8>>> {
            Arc::new(Mutex::new(Vec::new()))
        }

        fn to_string(content: &Arc<Mutex<Vec<u8>>>) -> String {
            String::from_utf8(content.lock().unwrap().clone()).unwrap()
        }

        #[test]
        fn filename() {
            let content = new_content();
            let writer = FakeFileWriter::new("file", content);
            assert_eq!("file", writer.get_filename());
        }

        #[test]
        fn close_stream() {
            let content = new_content();
            let mut writer = FakeFileWriter::new("file", Arc::clone(&content));
            assert!(writer.close());
            assert_eq!("", to_string(&content));
        }

        #[test]
        fn write_header() {
            let content = new_content();
            let header = "dummy header";
            let header_2 = "dummy header 2";
            let mut writer = FakeFileWriter::new("file", Arc::clone(&content));
            assert!(writer.write_header(header.as_bytes()));
            assert_eq!("dummy header", to_string(&content));
            assert!(writer.write_header(header_2.as_bytes()));
            assert_eq!("dummy header 2", to_string(&content));
            assert!(writer.close());
            assert_eq!("dummy header 2", to_string(&content));
        }

        #[test]
        fn write() {
            let content = new_content();
            let data_stream = ["data 1", "data 2"];
            let mut writer = FakeFileWriter::new("file", Arc::clone(&content));
            for data in data_stream {
                assert!(writer.write(data.as_bytes()));
            }
            assert_eq!("data 1data 2", to_string(&content));
            assert!(writer.close());
            assert_eq!("data 1data 2", to_string(&content));
        }

        #[test]
        fn header_and_write() {
            let content = new_content();
            let header = "dummy header";
            let data_stream = ["data 1", "data 2"];
            let mut writer = FakeFileWriter::new("file", Arc::clone(&content));
            assert!(writer.write_header(header.as_bytes()));
            assert_eq!("dummy header", to_string(&content));
            for data in data_stream {
                assert!(writer.write(data.as_bytes()));
            }
            assert!(writer.close());
            assert_eq!("dummy headerdata 1data 2", to_string(&content));
        }

        #[test]
        fn write_terminated_string() {
            let content = new_content();
            let data_stream = b"data\0 1".to_vec();
            let mut writer = FakeFileWriter::new("file", Arc::clone(&content));
            assert!(writer.write(&data_stream));
            assert_eq!(data_stream, *content.lock().unwrap());
        }

        #[test]
        fn write_terminated_header_string() {
            let content = new_content();
            let header = b"head\0 er".to_vec();
            let mut writer = FakeFileWriter::new("file", Arc::clone(&content));
            assert!(writer.write_header(&header));
            assert_eq!(header, *content.lock().unwrap());
        }

        #[test]
        fn header_and_write_terminated_string() {
            let content = new_content();
            let header = b"data\0 1".to_vec();
            let data = b"head\0 er".to_vec();
            let mut writer = FakeFileWriter::new("file", Arc::clone(&content));
            assert!(writer.write_header(&header));
            assert_eq!(header, *content.lock().unwrap());
            assert!(writer.write(&data));
            let mut expected_output = header.clone();
            expected_output.extend_from_slice(&data);
            assert_eq!(expected_output, *content.lock().unwrap());
            assert!(writer.close());
            assert_eq!(expected_output, *content.lock().unwrap());
        }

        #[test]
        fn write_header_never_shrinks() {
            let content = new_content();
            let mut writer = FakeFileWriter::new("file", Arc::clone(&content));
            assert!(writer.write(b"0123456789"));
            assert!(writer.write_header(b"abc"));
            assert_eq!("abc3456789", to_string(&content));
        }

        #[test]
        #[should_panic(expected = "closed")]
        fn write_after_close_panics() {
            let mut writer = FakeFileWriter::new("file", new_content());
            assert!(writer.close());
            writer.write(b"x");
        }
    }
}
