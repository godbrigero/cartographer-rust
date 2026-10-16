//! Port of `cartographer/common/configuration_file_resolver` (.h, .cc).

use super::config::CONFIGURATION_FILES_DIRECTORY;
use super::lua_parameter_dictionary::FileResolver;

/// A `FileResolver` for the `LuaParameterDictionary` that reads files from
/// disk. It searches the `configuration_files_directories` in order to find
/// the requested filename. The last place searched is always the
/// `configuration_files/` directory shipped with Cartographer
/// ([`CONFIGURATION_FILES_DIRECTORY`]). It contains reasonable configuration
/// for the various Cartographer components which provide a good starting
/// ground for new platforms.
#[derive(Clone, Debug)]
pub struct ConfigurationFileResolver {
    configuration_files_directories: Vec<String>,
}

impl ConfigurationFileResolver {
    /// Creates a resolver searching `configuration_files_directories` (in
    /// order), then [`CONFIGURATION_FILES_DIRECTORY`].
    pub fn new<I, S>(configuration_files_directories: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut configuration_files_directories: Vec<String> = configuration_files_directories
            .into_iter()
            .map(Into::into)
            .collect();
        configuration_files_directories.push(CONFIGURATION_FILES_DIRECTORY.to_string());
        ConfigurationFileResolver {
            configuration_files_directories,
        }
    }

    /// The directories searched, in order.
    pub fn configuration_files_directories(&self) -> &[String] {
        &self.configuration_files_directories
    }
}

impl FileResolver for ConfigurationFileResolver {
    fn get_full_path_or_die(&self, basename: &str) -> String {
        for path in &self.configuration_files_directories {
            // String concatenation like the C++ code (not Path::join).
            let filename = format!("{path}/{basename}");
            if std::fs::File::open(&filename).is_ok() {
                log::info!("Found '{filename}' for '{basename}'.");
                return filename;
            }
        }
        panic!("File '{basename}' was not found.");
    }

    fn get_file_content_or_die(&self, basename: &str) -> String {
        assert!(
            !basename.is_empty(),
            "Check failed: !basename.empty() File basename cannot be empty.{basename}"
        );
        let filename = self.get_full_path_or_die(basename);
        let bytes =
            std::fs::read(&filename).unwrap_or_else(|e| panic!("Failed to read '{filename}': {e}"));
        // RUST PORT NOTE: invalid UTF-8 is replaced (lossy conversion).
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::config::SOURCE_DIRECTORY;
    use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
    use std::io::Write;

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "cartographer_resolver_test_{}_{}",
                name,
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }

        fn write(&self, name: &str, content: &str) {
            let mut f = std::fs::File::create(self.0.join(name)).unwrap();
            f.write_all(content.as_bytes()).unwrap();
        }

        fn path(&self) -> String {
            self.0.to_str().unwrap().to_string()
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn default_directory_is_searched_last() {
        let dir = TempDir::new("order");
        dir.write("map_builder.lua", "MAP_BUILDER = { overridden = true }");
        let resolver = ConfigurationFileResolver::new([dir.path()]);
        assert_eq!(
            resolver.get_full_path_or_die("map_builder.lua"),
            format!("{}/map_builder.lua", dir.path())
        );
        assert_eq!(
            resolver.get_full_path_or_die("pose_graph.lua"),
            format!("{SOURCE_DIRECTORY}/configuration_files/pose_graph.lua")
        );
        assert_eq!(
            resolver.configuration_files_directories().last().unwrap(),
            CONFIGURATION_FILES_DIRECTORY
        );
    }

    #[test]
    #[should_panic(expected = "File 'does_not_exist.lua' was not found.")]
    fn missing_file_panics() {
        ConfigurationFileResolver::new(Vec::<String>::new())
            .get_full_path_or_die("does_not_exist.lua");
    }

    #[test]
    #[should_panic(expected = "File basename cannot be empty.")]
    fn empty_basename_panics() {
        ConfigurationFileResolver::new(Vec::<String>::new()).get_file_content_or_die("");
    }

    #[test]
    fn include_and_read() {
        let dir = TempDir::new("include");
        dir.write("a.lua", "A = 1 include 'b.lua'");
        dir.write("b.lua", "B = A + 1 return 'unused'");
        dir.write("data.txt", "some data");
        dir.write("ret.lua", "return { x = 5 }");
        let dict = LuaParameterDictionary::new(
            "include 'a.lua' local r = include 'ret.lua' return { a = A, b = B, data = read('data.txt'), x = r.x }",
            Box::new(ConfigurationFileResolver::new([dir.path()])),
        );
        assert_eq!(dict.get_int("a"), 1);
        assert_eq!(dict.get_int("b"), 2);
        assert_eq!(dict.get_string("data"), "some data");
        assert_eq!(dict.get_int("x"), 5);
    }

    #[test]
    #[should_panic(expected = "twice. Already included files in order of inclusion: ")]
    fn double_include_panics() {
        let dir = TempDir::new("twice");
        dir.write("a.lua", "A = 1");
        LuaParameterDictionary::new(
            "include 'a.lua' include 'a.lua' return {}",
            Box::new(ConfigurationFileResolver::new([dir.path()])),
        );
    }

    #[test]
    #[should_panic(expected = "twice.")]
    fn recursive_include_panics() {
        let dir = TempDir::new("recursive");
        dir.write("a.lua", "include 'b.lua'");
        dir.write("b.lua", "include 'a.lua'");
        LuaParameterDictionary::new(
            "include 'a.lua' return {}",
            Box::new(ConfigurationFileResolver::new([dir.path()])),
        );
    }

    #[test]
    #[should_panic(expected = "include takes a filename.")]
    fn include_requires_a_string() {
        LuaParameterDictionary::new(
            "include({}) return {}",
            Box::new(ConfigurationFileResolver::new(Vec::<String>::new())),
        );
    }

    #[test]
    fn errors_in_included_files_name_the_file() {
        let dir = TempDir::new("errors");
        dir.write("bad.lua", "\nx = nil + 1");
        let path = format!("{}/bad.lua", dir.path());
        let result = std::panic::catch_unwind(|| {
            LuaParameterDictionary::new(
                "include 'bad.lua' return {}",
                Box::new(ConfigurationFileResolver::new([dir.path()])),
            );
        });
        let payload = result.unwrap_err();
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default();
        let chunk = crate::common::lua::lexer::chunk_id(&path);
        assert!(
            message.contains(&format!(
                "{chunk}:2: attempt to perform arithmetic on a nil value"
            )),
            "{message}"
        );
    }
}
