//! Lua values: nil, booleans, numbers (always `f64`, like Lua 5.2), byte
//! strings, tables (shared by reference) and functions.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use super::ast::FuncBody;
use super::interpreter::{Env, Interpreter, LuaError};
use super::number::{number_to_string, str_to_number};

/// An immutable Lua string (arbitrary bytes).
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LuaString(Rc<[u8]>);

impl LuaString {
    /// Creates a string from bytes.
    pub fn new(bytes: &[u8]) -> Self {
        LuaString(Rc::from(bytes))
    }

    /// The raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True for the empty string.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Lossy UTF-8 conversion.
    pub fn to_str_lossy(&self) -> String {
        String::from_utf8_lossy(&self.0).into_owned()
    }
}

impl From<&str> for LuaString {
    fn from(s: &str) -> Self {
        LuaString::new(s.as_bytes())
    }
}

impl From<String> for LuaString {
    fn from(s: String) -> Self {
        LuaString::new(s.as_bytes())
    }
}

impl From<Vec<u8>> for LuaString {
    fn from(s: Vec<u8>) -> Self {
        LuaString(Rc::from(s))
    }
}

impl fmt::Debug for LuaString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.to_str_lossy())
    }
}

impl fmt::Display for LuaString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_str_lossy())
    }
}

/// Signature of functions implemented in Rust.
pub type NativeFn = dyn Fn(&mut Interpreter, Vec<Value>) -> Result<Vec<Value>, LuaError>;

/// A Lua closure: a function body plus its captured lexical environment.
pub struct LuaClosure {
    pub(crate) body: Rc<FuncBody>,
    pub(crate) env: Env,
    pub(crate) chunk: Rc<str>,
}

/// A callable value.
pub enum Function {
    /// Defined in Lua.
    Lua(LuaClosure),
    /// Implemented in Rust.
    Native {
        /// Name used in error messages (`bad argument #1 to 'name'`).
        name: String,
        /// The implementation.
        f: Box<NativeFn>,
    },
}

/// Reference to a function value.
#[derive(Clone)]
pub struct FunctionRef(pub(crate) Rc<Function>);

impl FunctionRef {
    /// Wraps a Rust function.
    pub fn native(
        name: &str,
        f: impl Fn(&mut Interpreter, Vec<Value>) -> Result<Vec<Value>, LuaError> + 'static,
    ) -> Self {
        FunctionRef(Rc::new(Function::Native {
            name: name.to_string(),
            f: Box::new(f),
        }))
    }

    fn addr(&self) -> usize {
        Rc::as_ptr(&self.0) as *const u8 as usize
    }
}

/// A Lua value.
#[derive(Clone, Default)]
pub enum Value {
    /// `nil`.
    #[default]
    Nil,
    /// `true`/`false`.
    Boolean(bool),
    /// A number (Lua 5.2 numbers are doubles).
    Number(f64),
    /// A string.
    String(LuaString),
    /// A table (reference semantics).
    Table(TableRef),
    /// A function (reference semantics).
    Function(FunctionRef),
}

impl Value {
    /// Lua `type()` name.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Nil => "nil",
            Value::Boolean(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Table(_) => "table",
            Value::Function(_) => "function",
        }
    }

    /// True for `nil`.
    pub fn is_nil(&self) -> bool {
        matches!(self, Value::Nil)
    }

    /// Lua truthiness: everything but `nil` and `false` is true.
    pub fn truthy(&self) -> bool {
        !matches!(self, Value::Nil | Value::Boolean(false))
    }

    /// Creates a string value.
    pub fn str(s: &str) -> Value {
        Value::String(LuaString::from(s))
    }

    /// `lua_isnumber` + `lua_tonumber`: numbers and strings convertible to
    /// numbers.
    pub fn to_number(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            Value::String(s) => str_to_number(s.as_bytes()),
            _ => None,
        }
    }

    /// `lua_isstring` + `lua_tostring`: strings and numbers (formatted with
    /// `%.14g`).
    pub fn to_lua_string(&self) -> Option<LuaString> {
        match self {
            Value::String(s) => Some(s.clone()),
            Value::Number(n) => Some(LuaString::from(number_to_string(*n))),
            _ => None,
        }
    }

    /// Raw equality (`rawequal`, `==` without metamethods).
    pub fn raw_equals(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Nil, Value::Nil) => true,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Number(a), Value::Number(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Table(a), Value::Table(b)) => a.ptr_eq(b),
            (Value::Function(a), Value::Function(b)) => Rc::ptr_eq(&a.0, &b.0),
            _ => false,
        }
    }

    /// `tostring()` without metamethods.
    pub fn to_display_string(&self) -> LuaString {
        match self {
            Value::Nil => LuaString::from("nil"),
            Value::Boolean(b) => LuaString::from(if *b { "true" } else { "false" }),
            Value::Number(n) => LuaString::from(number_to_string(*n)),
            Value::String(s) => s.clone(),
            Value::Table(t) => LuaString::from(format!("table: 0x{:08x}", t.addr())),
            Value::Function(f) => LuaString::from(format!("function: 0x{:08x}", f.addr())),
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::String(s) => write!(f, "{s:?}"),
            other => f.write_str(&other.to_display_string().to_str_lossy()),
        }
    }
}

impl From<f64> for Value {
    fn from(n: f64) -> Self {
        Value::Number(n)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Boolean(b)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::str(s)
    }
}

impl From<LuaString> for Value {
    fn from(s: LuaString) -> Self {
        Value::String(s)
    }
}

impl From<TableRef> for Value {
    fn from(t: TableRef) -> Self {
        Value::Table(t)
    }
}

/// Hashable identity of a table key.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Key {
    Boolean(bool),
    Number(u64),
    String(LuaString),
    Reference(usize),
}

/// Error of [`TableRef::next`]: the key is not in the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidKey;

/// Errors when using a value as a table key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyError {
    /// `t[nil] = v`.
    Nil,
    /// `t[0/0] = v`.
    NaN,
}

impl fmt::Display for KeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeyError::Nil => f.write_str("table index is nil"),
            KeyError::NaN => f.write_str("table index is NaN"),
        }
    }
}

fn key_of(value: &Value) -> Result<Key, KeyError> {
    Ok(match value {
        Value::Nil => return Err(KeyError::Nil),
        Value::Boolean(b) => Key::Boolean(*b),
        Value::Number(n) => {
            if n.is_nan() {
                return Err(KeyError::NaN);
            }
            // 0.0 and -0.0 are the same key.
            Key::Number(if *n == 0. {
                0f64.to_bits()
            } else {
                n.to_bits()
            })
        }
        Value::String(s) => Key::String(s.clone()),
        Value::Table(t) => Key::Reference(t.addr()),
        Value::Function(f) => Key::Reference(f.addr()),
    })
}

/// Returns `Some(i)` (0-based) if `n` is a positive integer key `i + 1`.
fn array_index(n: f64) -> Option<usize> {
    if n >= 1. && n.fract() == 0. && n <= (usize::MAX >> 1) as f64 {
        Some(n as usize - 1)
    } else {
        None
    }
}

/// A Lua table with an array part (keys `1..=n`) and a hash part that
/// iterates in insertion order.
#[derive(Default)]
pub struct Table {
    array: Vec<Value>,
    // Entries of the hash part; removed entries keep their slot with a nil
    // value so that `next` keeps working while fields are cleared during a
    // traversal (as Lua allows).
    entries: Vec<(Value, Value)>,
    index: HashMap<Key, usize>,
    num_removed: usize,
}

impl Table {
    fn get(&self, key: &Value) -> Value {
        if let Value::Number(n) = key
            && let Some(i) = array_index(*n)
            && i < self.array.len()
        {
            return self.array[i].clone();
        }
        let Ok(k) = key_of(key) else {
            return Value::Nil;
        };
        match self.index.get(&k) {
            Some(&i) => self.entries[i].1.clone(),
            None => Value::Nil,
        }
    }

    fn set(&mut self, key: Value, value: Value) -> Result<(), KeyError> {
        let k = key_of(&key)?;
        if let Value::Number(n) = key
            && let Some(i) = array_index(n)
        {
            if i < self.array.len() {
                self.array[i] = value;
                if i + 1 == self.array.len() {
                    while matches!(self.array.last(), Some(Value::Nil)) {
                        self.array.pop();
                    }
                }
                return Ok(());
            }
            if i == self.array.len() && !value.is_nil() {
                self.remove_from_hash(&k);
                self.array.push(value);
                self.migrate_to_array();
                return Ok(());
            }
        }
        if value.is_nil() {
            self.remove_from_hash(&k);
            return Ok(());
        }
        if let Some(&i) = self.index.get(&k) {
            if self.entries[i].1.is_nil() {
                self.num_removed -= 1;
            }
            self.entries[i].1 = value;
        } else {
            self.compact_if_needed();
            self.index.insert(k, self.entries.len());
            self.entries.push((key, value));
        }
        Ok(())
    }

    fn remove_from_hash(&mut self, k: &Key) {
        if let Some(&i) = self.index.get(k)
            && !self.entries[i].1.is_nil()
        {
            self.entries[i].1 = Value::Nil;
            self.num_removed += 1;
        }
    }

    // Moves keys n+1, n+2, ... from the hash part to the array part.
    fn migrate_to_array(&mut self) {
        loop {
            let next = Value::Number((self.array.len() + 1) as f64);
            let k = key_of(&next).expect("valid key");
            match self.index.get(&k) {
                Some(&i) if !self.entries[i].1.is_nil() => {
                    let v = std::mem::take(&mut self.entries[i].1);
                    self.num_removed += 1;
                    self.array.push(v);
                }
                _ => break,
            }
        }
    }

    // Removes cleared entries. Only called when inserting a new key, during
    // which traversals are undefined in Lua anyway.
    fn compact_if_needed(&mut self) {
        if self.num_removed < 8 || self.num_removed * 2 < self.entries.len() {
            return;
        }
        self.entries.retain(|(_, v)| !v.is_nil());
        self.index.clear();
        for (i, (k, _)) in self.entries.iter().enumerate() {
            self.index.insert(key_of(k).expect("valid key"), i);
        }
        self.num_removed = 0;
    }

    fn first_hash_entry_from(&self, start: usize) -> Option<(Value, Value)> {
        self.entries[start..]
            .iter()
            .find(|(_, v)| !v.is_nil())
            .cloned()
    }

    fn next(&self, key: &Value) -> Result<Option<(Value, Value)>, InvalidKey> {
        let array_start = match key {
            Value::Nil => Some(0),
            Value::Number(n) => match array_index(*n) {
                Some(i) if i < self.array.len() => Some(i + 1),
                _ => None,
            },
            _ => None,
        };
        let hash_start = match array_start {
            Some(start) => {
                for i in start..self.array.len() {
                    if !self.array[i].is_nil() {
                        return Ok(Some((Value::Number((i + 1) as f64), self.array[i].clone())));
                    }
                }
                0
            }
            None => {
                let k = key_of(key).map_err(|_| InvalidKey)?;
                match self.index.get(&k) {
                    Some(&i) => i + 1,
                    None => {
                        // An array index that was trimmed during traversal.
                        if matches!(key, Value::Number(n) if array_index(*n).is_some()) {
                            0
                        } else {
                            return Err(InvalidKey);
                        }
                    }
                }
            }
        };
        Ok(self.first_hash_entry_from(hash_start))
    }
}

/// Shared reference to a table.
#[derive(Clone, Default)]
pub struct TableRef(Rc<RefCell<Table>>);

impl TableRef {
    /// Creates an empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// True if both refer to the same table.
    pub fn ptr_eq(&self, other: &TableRef) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    fn addr(&self) -> usize {
        Rc::as_ptr(&self.0) as *const u8 as usize
    }

    /// `rawget`.
    pub fn get(&self, key: &Value) -> Value {
        self.0.borrow().get(key)
    }

    /// `rawget` with a string key.
    pub fn get_str(&self, key: &str) -> Value {
        self.get(&Value::str(key))
    }

    /// `rawget` with an integer key.
    pub fn get_int(&self, key: i64) -> Value {
        self.get(&Value::Number(key as f64))
    }

    /// `rawset`. Assigning nil removes the key.
    pub fn set(&self, key: Value, value: Value) -> Result<(), KeyError> {
        self.0.borrow_mut().set(key, value)
    }

    /// `rawset` with a string key.
    pub fn set_str(&self, key: &str, value: Value) {
        self.set(Value::str(key), value)
            .expect("string keys are valid");
    }

    /// Removes all entries (used to break reference cycles such as
    /// `_G._G` when a Lua state is dropped).
    pub fn clear(&self) {
        let table = std::mem::take(&mut *self.0.borrow_mut());
        drop(table);
    }

    /// The border used by the `#` operator.
    pub fn len(&self) -> usize {
        self.0.borrow().array.len()
    }

    /// True if the table has no entries at all.
    pub fn is_empty(&self) -> bool {
        self.next(&Value::Nil).ok().flatten().is_none()
    }

    /// Lua `next`: the entry after `key` (or the first entry for nil).
    /// Returns `Err(InvalidKey)` for a key that is not in the table.
    pub fn next(&self, key: &Value) -> Result<Option<(Value, Value)>, InvalidKey> {
        self.0.borrow().next(key)
    }

    /// All entries in traversal order.
    pub fn entries(&self) -> Vec<(Value, Value)> {
        let mut result = Vec::new();
        let mut key = Value::Nil;
        while let Ok(Some((k, v))) = self.next(&key) {
            result.push((k.clone(), v));
            key = k;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(x: f64) -> Value {
        Value::Number(x)
    }

    #[test]
    fn array_and_hash_parts() {
        let t = TableRef::new();
        assert!(t.is_empty());
        t.set(n(2.), Value::str("b")).unwrap();
        assert_eq!(t.len(), 0);
        t.set(n(1.), Value::str("a")).unwrap();
        assert_eq!(t.len(), 2);
        t.set(Value::str("x"), n(1.)).unwrap();
        assert_eq!(t.get(&n(2.)).to_display_string().to_str_lossy(), "b");
        assert_eq!(t.get(&n(-0.)).type_name(), "nil");
        t.set(n(0.), n(5.)).unwrap();
        assert_eq!(t.get(&n(-0.)).to_number(), Some(5.));
        t.set(n(2.), Value::Nil).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t.set(Value::Nil, n(1.)), Err(KeyError::Nil));
        assert_eq!(t.set(n(f64::NAN), n(1.)), Err(KeyError::NaN));
        let keys: Vec<String> = t
            .entries()
            .iter()
            .map(|(k, _)| k.to_display_string().to_str_lossy())
            .collect();
        assert_eq!(keys, vec!["1", "x", "0"]);
    }

    #[test]
    fn next_survives_clearing_during_traversal() {
        let t = TableRef::new();
        for i in 0..20 {
            t.set(Value::str(&format!("k{i}")), n(f64::from(i)))
                .unwrap();
        }
        let mut key = Value::Nil;
        let mut count = 0;
        while let Ok(Some((k, _))) = t.next(&key) {
            t.set(k.clone(), Value::Nil).unwrap();
            key = k;
            count += 1;
        }
        assert_eq!(count, 20);
        assert!(t.is_empty());
        // Compaction on insert.
        t.set(Value::str("new"), n(1.)).unwrap();
        assert_eq!(t.entries().len(), 1);
        assert!(t.next(&Value::str("missing")).is_err());
    }
}
