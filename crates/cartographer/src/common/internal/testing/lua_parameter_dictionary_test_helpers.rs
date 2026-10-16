//! Port of `cartographer/common/internal/testing/lua_parameter_dictionary_test_helpers` (.h).

use crate::common::lua_parameter_dictionary::{FileResolver, LuaParameterDictionary};

/// A file resolver for tests: `include` and `read` are not available.
#[derive(Debug, Default, Clone, Copy)]
pub struct DummyFileResolver;

impl FileResolver for DummyFileResolver {
    fn get_full_path_or_die(&self, _basename: &str) -> String {
        panic!("Not implemented");
    }

    fn get_file_content_or_die(&self, _basename: &str) -> String {
        panic!("Not implemented");
    }
}

/// Creates a reference counted dictionary from `code` (C++ `MakeDictionary`).
pub fn make_dictionary(code: &str) -> LuaParameterDictionary {
    LuaParameterDictionary::new(code, Box::new(DummyFileResolver))
}
