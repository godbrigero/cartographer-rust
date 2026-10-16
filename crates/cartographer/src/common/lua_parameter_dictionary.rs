//! Port of `cartographer/common/lua_parameter_dictionary` (.h, .cc).
//!
//! When a `LuaParameterDictionary` is constructed, a new Lua state (i.e. an
//! independent Lua interpreter) is fired up to evaluate the Lua code. The
//! code is expected to return a Lua table that contains key/value pairs that
//! are the key/value pairs of our parameter dictionary.
//!
//! We keep the Lua table around and reference it in the `get_*` methods
//! instead of moving its contents into a Rust map since we can only know in
//! the `get_*` methods what data type to expect in the table.
//!
//! RUST PORT NOTE: the C++ class drives the C Lua API (a table on the stack
//! of a `lua_State`, sub-dictionaries are coroutines). Here the Lua code is
//! run by the pure-Rust interpreter in [`crate::common::lua`] and a
//! dictionary holds a reference to its (shared, reference-counted) table.
//! Sub-dictionaries therefore no longer need to be destroyed before their
//! parent. All observable behavior (type coercions, error messages, the
//! exactly-once key usage check, `to_string` output) is kept.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;

use super::lua::number::str_to_number;
use super::lua::stdlib::quote_string;
use super::lua::{Interpreter, LuaError, TableRef, Value};
use super::printf::format_f;

/// Resolves file paths and file content for the Lua `read` and `include`
/// functions. Use this to configure where those functions load other files
/// from.
pub trait FileResolver {
    /// Returns the full path of `basename`, or panics if it does not exist.
    fn get_full_path_or_die(&self, basename: &str) -> String;
    /// Returns the content of `basename`, or panics if it does not exist.
    fn get_file_content_or_die(&self, basename: &str) -> String;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReferenceCount {
    Yes,
    No,
}

/// The Lua state shared by a dictionary and all its sub-dictionaries.
struct LuaState {
    // Kept alive as long as any dictionary refers to the state, like the
    // C++ `lua_State` (globals stay reachable).
    _interpreter: Interpreter,
}

/// A parameter dictionary that gets loaded from Lua code.
///
/// Getters take `&self` (the C++ ones are non-const because they update the
/// key reference counts, which here use interior mutability).
pub struct LuaParameterDictionary {
    table: TableRef,
    _state: Rc<LuaState>,
    // This is shared with all the sub dictionaries.
    file_resolver: Rc<dyn FileResolver>,
    // If `Yes`, checks that all keys were used on destruction.
    reference_count: ReferenceCount,
    // This is modified with every call to get_* in order to verify that all
    // parameters are read exactly once.
    reference_counts: RefCell<BTreeMap<String, i32>>,
}

fn check_for_lua_errors(status: i32, error: &LuaError) -> ! {
    panic!(
        "Check failed: status == 0 ({status} vs. 0) {}",
        error.message()
    );
}

/// Lua status code of a syntax error (`LUA_ERRSYNTAX`).
const LUA_ERRSYNTAX: i32 = 3;
/// Lua status code of a runtime error (`LUA_ERRRUN`).
const LUA_ERRRUN: i32 = 2;

/// Converts a Lua string to a Rust string like
/// `std::string(lua_tostring(...))`: stops at the first NUL byte.
/// RUST PORT NOTE: invalid UTF-8 is replaced (lossy conversion).
fn c_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Returns `a` if `condition` is true, else `b`.
fn lua_choose(_: &mut Interpreter, args: Vec<Value>) -> Result<Vec<Value>, LuaError> {
    assert_eq!(
        args.len(),
        3,
        "Check failed: lua_gettop(L) == 3 ({} vs. 3) choose() takes (condition, a, b).",
        args.len()
    );
    let Value::Boolean(condition) = args[0] else {
        panic!("Check failed: lua_isboolean(L, 1) condition is not a boolean value.");
    };
    Ok(vec![if condition {
        args[1].clone()
    } else {
        args[2].clone()
    }])
}

fn filename_argument(args: &[Value], function: &str) -> String {
    assert_eq!(
        args.len(),
        1,
        "Check failed: lua_gettop(L) == 1 ({} vs. 1)",
        args.len()
    );
    match args[0].to_lua_string() {
        Some(s) => c_string(s.as_bytes()),
        None => panic!("Check failed: lua_isstring(L, -1) {function} takes a filename."),
    }
}

/// Lua function to run a script in the current Lua context. Takes the
/// filename as its only argument. Handles inclusion of other Lua files and
/// prevents double inclusion.
fn lua_include(
    interpreter: &mut Interpreter,
    args: Vec<Value>,
    file_resolver: &dyn FileResolver,
    included_files: &RefCell<Vec<String>>,
) -> Result<Vec<Value>, LuaError> {
    let basename = filename_argument(&args, "include");
    let filename = file_resolver.get_full_path_or_die(&basename);
    {
        let mut included_files = included_files.borrow_mut();
        if included_files.contains(&filename) {
            let mut error_msg = format!(
                "Tried to include {filename} twice. Already included files in order of inclusion: "
            );
            for f in included_files.iter() {
                error_msg.push_str(f);
                error_msg.push('\n');
            }
            panic!("{error_msg}");
        }
        included_files.push(filename.clone());
    }
    let content = file_resolver.get_file_content_or_die(&basename);
    let chunk = interpreter
        .load(content.as_bytes(), &filename)
        .unwrap_or_else(|e| check_for_lua_errors(LUA_ERRSYNTAX, &e));
    let results = interpreter
        .call(&chunk, Vec::new())
        .unwrap_or_else(|e| check_for_lua_errors(LUA_ERRRUN, &e));
    Ok(results)
}

/// Lua function to read a file into a string.
fn lua_read(args: Vec<Value>, file_resolver: &dyn FileResolver) -> Result<Vec<Value>, LuaError> {
    let basename = filename_argument(&args, "read");
    let content = file_resolver.get_file_content_or_die(&basename);
    // lua_pushstring stops at the first NUL byte.
    Ok(vec![Value::str(&c_string(content.as_bytes()))])
}

impl LuaParameterDictionary {
    /// Constructs the dictionary from a Lua Table specification.
    pub fn new(code: &str, file_resolver: Box<dyn FileResolver>) -> Self {
        Self::with_reference_count(code, ReferenceCount::Yes, file_resolver)
    }

    /// Constructs a `LuaParameterDictionary` without reference counting.
    pub fn non_reference_counted(code: &str, file_resolver: Box<dyn FileResolver>) -> Self {
        Self::with_reference_count(code, ReferenceCount::No, file_resolver)
    }

    fn with_reference_count(
        code: &str,
        reference_count: ReferenceCount,
        file_resolver: Box<dyn FileResolver>,
    ) -> Self {
        let file_resolver: Rc<dyn FileResolver> = Rc::from(file_resolver);
        let included_files: Rc<RefCell<Vec<String>>> = Rc::default();
        let mut interpreter = Interpreter::new();

        interpreter.register("choose", lua_choose);
        {
            let file_resolver = Rc::clone(&file_resolver);
            let included_files = Rc::clone(&included_files);
            interpreter.register("include", move |interpreter, args| {
                lua_include(interpreter, args, &*file_resolver, &included_files)
            });
        }
        {
            let file_resolver = Rc::clone(&file_resolver);
            interpreter.register("read", move |_, args| lua_read(args, &*file_resolver));
        }

        // luaL_loadstring uses the code itself as the chunk name.
        let chunk_name = c_string(code.as_bytes());
        let chunk = interpreter
            .load(chunk_name.as_bytes(), &chunk_name)
            .unwrap_or_else(|e| check_for_lua_errors(LUA_ERRSYNTAX, &e));
        let results = interpreter
            .call(&chunk, Vec::new())
            .unwrap_or_else(|e| check_for_lua_errors(LUA_ERRRUN, &e));
        let Some(Value::Table(table)) = results.into_iter().next() else {
            panic!("Check failed: lua_istable(L, -1) Topmost item on Lua stack is not a table!");
        };
        LuaParameterDictionary {
            table,
            _state: Rc::new(LuaState {
                _interpreter: interpreter,
            }),
            file_resolver,
            reference_count,
            reference_counts: RefCell::default(),
        }
    }

    /// For `get_dictionary()`: a dictionary for a table of this dictionary's
    /// Lua state.
    fn sub_dictionary(&self, table: TableRef, reference_count: ReferenceCount) -> Self {
        LuaParameterDictionary {
            table,
            _state: Rc::clone(&self._state),
            file_resolver: Rc::clone(&self.file_resolver),
            reference_count,
            reference_counts: RefCell::default(),
        }
    }

    /// Returns all available keys (string keys that do not look like
    /// numbers, in table traversal order).
    pub fn get_keys(&self) -> Vec<String> {
        get_keys(&self.table)
    }

    /// Returns true if the key is in this dictionary.
    pub fn has_key(&self, key: &str) -> bool {
        !self.table.get_str(key).is_nil()
    }

    /// Returns the string value of `key`; numbers are converted (`10` ->
    /// `"10"`). Panics if the key is missing or of a different type.
    pub fn get_string(&self, key: &str) -> String {
        self.check_has_key_and_reference(key);
        pop_string(&self.table.get_str(key))
    }

    /// Returns the number value of `key`; numeric strings are converted.
    pub fn get_double(&self, key: &str) -> f64 {
        self.check_has_key_and_reference(key);
        pop_double(&self.table.get_str(key))
    }

    /// Returns the value of `key` as an int (truncated toward zero, like
    /// Lua 5.2's `lua_tointeger`).
    pub fn get_int(&self, key: &str) -> i32 {
        self.check_has_key_and_reference(key);
        pop_int(&self.table.get_str(key))
    }

    /// Returns the boolean value of `key` (no truthiness conversion).
    pub fn get_bool(&self, key: &str) -> bool {
        self.check_has_key_and_reference(key);
        pop_bool(&self.table.get_str(key))
    }

    /// Returns the sub-dictionary at `key`. It is reference counted iff this
    /// dictionary is.
    pub fn get_dictionary(&self, key: &str) -> LuaParameterDictionary {
        self.check_has_key_and_reference(key);
        self.pop_dictionary(self.table.get_str(key), self.reference_count)
    }

    /// Gets an int from the dictionary and checks that it is non-negative.
    pub fn get_non_negative_int(&self, key: &str) -> i32 {
        let temp = self.get_int(key); // Will increase reference count.
        assert!(
            temp >= 0,
            "Check failed: temp >= 0 ({temp} vs. 0) {temp} is negative."
        );
        temp
    }

    /// Returns the values of the keys `1`, `2`, `3`, ... (up to the first
    /// missing one) as doubles. Array values are not reference counted.
    pub fn get_array_values_as_doubles(&self) -> Vec<f64> {
        array_values(&self.table).iter().map(pop_double).collect()
    }

    /// Like [`Self::get_array_values_as_doubles`] for strings.
    pub fn get_array_values_as_strings(&self) -> Vec<String> {
        array_values(&self.table).iter().map(pop_string).collect()
    }

    /// Like [`Self::get_array_values_as_doubles`] for dictionaries.
    pub fn get_array_values_as_dictionaries(&self) -> Vec<LuaParameterDictionary> {
        array_values(&self.table)
            .into_iter()
            .map(|v| self.pop_dictionary(v, self.reference_count))
            .collect()
    }

    fn pop_dictionary(
        &self,
        value: Value,
        reference_count: ReferenceCount,
    ) -> LuaParameterDictionary {
        let Value::Table(table) = value else {
            panic!("Check failed: lua_istable(L, -1) Topmost item on Lua stack is not a table!");
        };
        self.sub_dictionary(table, reference_count)
    }

    // CHECK() that 'key' is in the dictionary.
    fn check_has_key(&self, key: &str) {
        assert!(
            self.has_key(key),
            "Check failed: HasKey(key) Key '{key}' not in dictionary:\n{self}"
        );
    }

    // CHECK() that 'key' is in this dictionary and reference it as being
    // used.
    fn check_has_key_and_reference(&self, key: &str) {
        self.check_has_key(key);
        *self
            .reference_counts
            .borrow_mut()
            .entry(key.to_string())
            .or_insert(0) += 1;
    }

    // CHECK()s that all keys defined in the configuration have been used
    // exactly once and resets the reference counter.
    fn check_all_keys_were_used_exactly_once_and_reset(&self) {
        let mut reference_counts = self.reference_counts.borrow_mut();
        for key in self.get_keys() {
            match reference_counts.get(&key) {
                None => panic!(
                    "Check failed: 1 == reference_counts_.count(key) (1 vs. 0) Key '{key}' was used the wrong number of times."
                ),
                Some(&count) if count != 1 => panic!(
                    "Check failed: 1 == reference_counts_.at(key) (1 vs. {count}) Key '{key}' was used the wrong number of times."
                ),
                Some(_) => {}
            }
        }
        reference_counts.clear();
    }
}

/// Returns a string representation (valid Lua when prefixed by `return `)
/// for this `LuaParameterDictionary` (C++ `ToString()`).
impl fmt::Display for LuaParameterDictionary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&do_to_string(&self.table, ""))
    }
}

impl fmt::Debug for LuaParameterDictionary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LuaParameterDictionary({self})")
    }
}

impl Drop for LuaParameterDictionary {
    fn drop(&mut self) {
        // RUST PORT NOTE: skipped while unwinding from another panic (a
        // second panic would abort the process).
        if self.reference_count == ReferenceCount::Yes && !std::thread::panicking() {
            self.check_all_keys_were_used_exactly_once_and_reset();
        }
    }
}

fn get_keys(table: &TableRef) -> Vec<String> {
    table
        .entries()
        .into_iter()
        .filter_map(|(key, _)| match key {
            // `lua_isnumber` is also true for numeric strings.
            Value::String(s) if str_to_number(s.as_bytes()).is_none() => {
                Some(c_string(s.as_bytes()))
            }
            // RUST PORT NOTE: numbers are skipped like in C++; other key
            // types (booleans, tables) are UB in C++ and skipped here.
            _ => None,
        })
        .collect()
}

fn array_values(table: &TableRef) -> Vec<Value> {
    let mut values = Vec::new();
    for idx in 1.. {
        let value = table.get_int(idx);
        if value.is_nil() {
            break;
        }
        values.push(value);
    }
    values
}

fn pop_string(value: &Value) -> String {
    match value.to_lua_string() {
        Some(s) => c_string(s.as_bytes()),
        None => panic!("Check failed: lua_isstring(L_, -1) Top of stack is not a string value."),
    }
}

fn pop_double(value: &Value) -> f64 {
    match value.to_number() {
        Some(n) => n,
        None => panic!("Check failed: lua_isnumber(L_, -1) Top of stack is not a number value."),
    }
}

fn pop_int(value: &Value) -> i32 {
    // lua_tointeger (C cast to a 64-bit lua_Integer), then cast to int.
    (pop_double(value) as i64) as i32
}

fn pop_bool(value: &Value) -> bool {
    match value {
        Value::Boolean(b) => *b,
        _ => panic!("Check failed: lua_isboolean(L_, -1) Top of stack is not a boolean value."),
    }
}

fn value_to_string(value: &Value, indent: &str) -> String {
    match value {
        Value::Boolean(b) => if *b { "true" } else { "false" }.to_string(),
        Value::String(s) => String::from_utf8_lossy(&quote_string(s.as_bytes())).into_owned(),
        Value::Number(n) => {
            if n.is_infinite() {
                if *n < 0. { "-math.huge" } else { "math.huge" }.to_string()
            } else {
                // std::to_string(double)
                format_f(*n)
            }
        }
        Value::Table(t) => do_to_string(t, &format!("{indent}  ")),
        other => panic!("Unhandled type {}", other.type_name()),
    }
}

// Function that recurses to keep track of indent for ToString().
fn do_to_string(table: &TableRef, indent: &str) -> String {
    let mut result = String::from("{");
    let mut dictionary_is_empty = true;

    // Integer (array) keys.
    for value in array_values(table) {
        dictionary_is_empty = false;
        result.push('\n');
        result.push_str(indent);
        result.push_str("  ");
        result.push_str(&value_to_string(&value, indent));
        result.push(',');
    }

    // String keys.
    let mut keys = get_keys(table);
    keys.sort();
    for key in keys {
        dictionary_is_empty = false;
        let value = table.get_str(&key);
        result.push('\n');
        result.push_str(indent);
        result.push_str("  ");
        result.push_str(&key);
        result.push_str(" = ");
        result.push_str(&value_to_string(&value, indent));
        result.push(',');
    }
    result.push('\n');
    result.push_str(indent);
    result.push('}');

    if dictionary_is_empty {
        return "{}".to_string();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::{
        DummyFileResolver, make_dictionary,
    };

    fn make_non_reference_counted(code: &str) -> LuaParameterDictionary {
        LuaParameterDictionary::non_reference_counted(code, Box::new(DummyFileResolver))
    }

    fn reference_all_keys_as_integers(dict: &LuaParameterDictionary) {
        for key in dict.get_keys() {
            dict.get_int(&key);
        }
    }

    mod lua_parameter_dictionary_test {
        use super::*;

        #[test]
        fn get_int() {
            let dict = make_dictionary("return { blah = 100 }");
            assert_eq!(dict.get_int("blah"), 100);
        }

        #[test]
        fn get_string() {
            let dict = make_dictionary("return { blah = 'is_a_string' }\n");
            assert_eq!(dict.get_string("blah"), "is_a_string");
        }

        #[test]
        #[allow(clippy::approx_constant)] // The C++ test input.
        fn get_double() {
            let dict = make_dictionary("return { blah = 3.1415 }");
            assert_eq!(dict.get_double("blah"), 3.1415);
        }

        #[test]
        fn get_bool_true() {
            let dict = make_dictionary("return { blah = true }");
            assert!(dict.get_bool("blah"));
        }

        #[test]
        fn get_bool_false() {
            let dict = make_dictionary("return { blah = false }");
            assert!(!dict.get_bool("blah"));
        }

        #[test]
        fn get_dictionary() {
            let dict = make_dictionary("return { blah = { blue = 100, red = 200 }, fasel = 10 }");

            let sub_dict = dict.get_dictionary("blah");
            let mut keys = sub_dict.get_keys();
            assert_eq!(keys.len(), 2);
            keys.sort();
            assert_eq!(keys[0], "blue");
            assert_eq!(keys[1], "red");
            assert!(sub_dict.has_key("blue"));
            assert!(sub_dict.has_key("red"));
            assert_eq!(sub_dict.get_int("blue"), 100);
            assert_eq!(sub_dict.get_int("red"), 200);

            assert_eq!(dict.get_string("fasel"), "10");
        }

        #[test]
        fn get_keys() {
            let dict = make_dictionary("return { blah = 100, blah1 = 200}");

            let mut keys = dict.get_keys();
            assert_eq!(keys.len(), 2);
            keys.sort();
            assert_eq!(keys[0], "blah");
            assert_eq!(keys[1], "blah1");

            reference_all_keys_as_integers(&dict);
        }

        #[test]
        #[should_panic(expected = "Key 'blah_fasel' not in dictionary:")]
        fn non_existing_key() {
            let dict = make_dictionary("return { blah = 100 }");
            reference_all_keys_as_integers(&dict);
            dict.get_int("blah_fasel");
        }

        #[test]
        #[should_panic(expected = "-100 is negative.")]
        fn uint_negative() {
            let dict = make_dictionary("return { blah = -100}");
            dict.get_non_negative_int("blah");
        }

        #[test]
        fn to_string() {
            let dict = make_dictionary(
                r#"return {
  ceta = { yolo = "hurray" },
  fasel = 1234.456786,
  fasel1 = -math.huge,
  fasel2 = math.huge,
  blubber = 123,
  blub = 'hello',
  alpha = true,
  alpha1 = false,
  }"#,
            );

            let golden = r#"{
  alpha = true,
  alpha1 = false,
  blub = "hello",
  blubber = 123.000000,
  ceta = {
    yolo = "hurray",
  },
  fasel = 1234.456786,
  fasel1 = -math.huge,
  fasel2 = math.huge,
}"#;
            assert_eq!(golden, dict.to_string());

            let dict1 = make_dictionary(&format!("return {dict}"));

            assert!(dict1.get_bool("alpha"));
            assert!(!dict1.get_bool("alpha1"));
            assert_eq!(dict1.get_int("blubber"), 123);
            assert_eq!(dict1.get_string("blub"), "hello");
            assert_eq!(dict1.get_dictionary("ceta").get_string("yolo"), "hurray");
            assert!((dict1.get_double("fasel") - 1234.456786).abs() <= 1e-3);
            assert!((-dict1.get_double("fasel1")).is_infinite());
            assert!(dict1.get_double("fasel2").is_infinite());

            assert!(dict.get_bool("alpha"));
            assert!(!dict.get_bool("alpha1"));
            assert_eq!(dict.get_int("blubber"), 123);
            assert_eq!(dict.get_string("blub"), "hello");
            assert_eq!(dict.get_dictionary("ceta").get_string("yolo"), "hurray");
            assert!((dict.get_double("fasel") - 1234.456786).abs() <= 1e-3);
            assert!((-dict.get_double("fasel1")).is_infinite());
            assert!(dict.get_double("fasel2").is_infinite());
        }

        #[test]
        fn to_string_for_arrays() {
            let dict = make_non_reference_counted(
                r#"return {
      "blub", 3, 3.1,
      foo = "ups",
  }"#,
            );

            let golden = r#"{
  "blub",
  3.000000,
  3.100000,
  foo = "ups",
}"#;
            assert_eq!(golden, dict.to_string());
        }

        #[test]
        fn get_array_values_as_strings() {
            let dict = make_dictionary("return { 'a', 'b', 'c' }");
            assert_eq!(0, dict.get_keys().len());
            let values = dict.get_array_values_as_strings();
            assert_eq!(3, values.len());
            assert_eq!("a", values[0]);
            assert_eq!("b", values[1]);
            assert_eq!("c", values[2]);
        }

        #[test]
        fn get_array_values_as_doubles() {
            let dict = make_dictionary("return { 1., 2., 3. }");
            assert_eq!(0, dict.get_keys().len());
            let values = dict.get_array_values_as_doubles();
            assert_eq!(3, values.len());
            assert!((1. - values[0]).abs() <= 1e-7);
            assert!((2. - values[1]).abs() <= 1e-7);
            assert!((3. - values[2]).abs() <= 1e-7);
        }

        #[test]
        fn get_array_values_as_dictionaries() {
            let dict = make_dictionary("return { { a = 1 }, { b = 3 } }");
            assert_eq!(0, dict.get_keys().len());
            let values = dict.get_array_values_as_dictionaries();
            assert_eq!(2, values.len());
            assert_eq!(1, values[0].get_int("a"));
            assert_eq!(3, values[1].get_int("b"));
        }

        #[test]
        fn test_choose_true() {
            let dict = make_dictionary("return { a = choose(true, 1, 0) }");
            assert_eq!(1, dict.get_int("a"));
        }

        #[test]
        fn test_chose_false() {
            let dict = make_dictionary("return { a = choose(false, 1, 0) }");
            assert_eq!(0, dict.get_int("a"));
        }

        #[test]
        #[should_panic(expected = "condition is not a boolean value.")]
        fn test_choose_invalid_argument() {
            make_dictionary("return { a = choose('truish', 1, 0) }");
        }
    }

    // Additional tests of the C++ semantics.

    #[test]
    #[should_panic(expected = "Key 'blah' was used the wrong number of times.")]
    fn unused_key_panics_on_drop() {
        let _dict = make_dictionary("return { blah = 100 }");
    }

    #[test]
    #[should_panic(expected = "(1 vs. 2) Key 'blah' was used the wrong number of times.")]
    fn key_used_twice_panics_on_drop() {
        let dict = make_dictionary("return { blah = 100 }");
        dict.get_int("blah");
        dict.get_int("blah");
    }

    #[test]
    #[should_panic(expected = "Key 'b' was used the wrong number of times.")]
    fn sub_dictionaries_inherit_reference_counting() {
        let dict = make_dictionary("return { a = { b = 1 } }");
        let _sub = dict.get_dictionary("a");
    }

    #[test]
    fn non_reference_counted_does_not_check() {
        let dict = make_non_reference_counted("return { a = { b = 1 }, c = 2 }");
        let _sub = dict.get_dictionary("a");
        dict.get_int("c");
        dict.get_int("c");
    }

    #[test]
    fn coercions() {
        let dict = make_non_reference_counted(
            "return { s = '12.5', n = 1e15, f = -2.7, x = 0.1, big = 2^31 + 5, h = ' 0x10 ' }",
        );
        assert_eq!(dict.get_double("s"), 12.5);
        assert_eq!(dict.get_int("s"), 12);
        assert_eq!(dict.get_string("n"), "1e+15");
        assert_eq!(dict.get_string("x"), "0.1");
        assert_eq!(dict.get_int("f"), -2);
        // Truncation to 64 bits, then to int.
        assert_eq!(dict.get_int("big"), i32::MIN + 5);
        assert_eq!(dict.get_int("h"), 16);
    }

    #[test]
    #[should_panic(expected = "Top of stack is not a number value.")]
    fn get_double_of_string_panics() {
        let dict = make_non_reference_counted("return { s = 'abc' }");
        dict.get_double("s");
    }

    #[test]
    #[should_panic(expected = "Top of stack is not a boolean value.")]
    fn get_bool_is_strict() {
        let dict = make_non_reference_counted("return { b = 1 }");
        dict.get_bool("b");
    }

    #[test]
    #[should_panic(expected = "Top of stack is not a string value.")]
    fn get_string_of_table_panics() {
        let dict = make_non_reference_counted("return { t = {} }");
        dict.get_string("t");
    }

    #[test]
    #[should_panic(expected = "Topmost item on Lua stack is not a table!")]
    fn code_must_return_a_table() {
        make_dictionary("return 1");
    }

    #[test]
    #[should_panic(expected = "Topmost item on Lua stack is not a table!")]
    fn get_dictionary_of_number_panics() {
        let dict = make_non_reference_counted("return { a = 1 }");
        dict.get_dictionary("a");
    }

    #[test]
    #[should_panic(
        expected = "Check failed: status == 0 (3 vs. 0) [string \"return {\"]:1: unexpected symbol near <eof>"
    )]
    fn syntax_errors_panic() {
        make_dictionary("return {");
    }

    #[test]
    #[should_panic(
        expected = "Check failed: status == 0 (2 vs. 0) [string \"return { a = b.c }\"]:1: attempt to index a nil value (global 'b')"
    )]
    fn runtime_errors_panic() {
        make_dictionary("return { a = b.c }");
    }

    #[test]
    #[should_panic(expected = "Not implemented")]
    fn dummy_resolver_include_panics() {
        make_dictionary("include 'x.lua' return {}");
    }

    #[test]
    fn get_keys_skips_numeric_keys() {
        let dict =
            make_non_reference_counted("return { 1, 2, ['3'] = 3, [4.5] = 4, a = 5, ['0x1'] = 6 }");
        assert_eq!(dict.get_keys(), vec!["a".to_string()]);
        assert_eq!(dict.get_array_values_as_doubles(), vec![1., 2.]);
        assert!(dict.has_key("3"));
        assert!(!dict.has_key("b"));
    }

    #[test]
    fn to_string_details() {
        let dict = make_non_reference_counted(
            "return { e = {}, s = 'a\"b\\\\c\\nd', n = -0.5, t = { 1, { x = 2 } } }",
        );
        let golden = "{\n  e = {},\n  n = -0.500000,\n  s = \"a\\\"b\\\\c\\\nd\",\n  t = {\n    1.000000,\n    {\n      x = 2.000000,\n    },\n  },\n}";
        assert_eq!(dict.to_string(), golden);
        // Round trip.
        let dict1 = make_non_reference_counted(&format!("return {dict}"));
        assert_eq!(dict1.get_string("s"), "a\"b\\c\nd");
        assert_eq!(dict1.to_string(), golden);
        assert_eq!(make_non_reference_counted("return {}").to_string(), "{}");
    }

    #[test]
    #[should_panic(expected = "Unhandled type function")]
    fn to_string_of_function_panics() {
        make_non_reference_counted("return { f = print }").to_string();
    }

    #[test]
    fn tables_are_shared_by_reference() {
        let dict = make_dictionary(
            "A = { x = 1 } B = { a = A } A.x = 2 B.a.y = 3 return { b = B, a = A }",
        );
        let b = dict.get_dictionary("b");
        let ba = b.get_dictionary("a");
        assert_eq!(ba.get_int("x"), 2);
        assert_eq!(ba.get_int("y"), 3);
        // Reference counts are per dictionary object.
        let a = dict.get_dictionary("a");
        assert_eq!(a.get_int("x"), 2);
        assert_eq!(a.get_int("y"), 3);
    }
}
