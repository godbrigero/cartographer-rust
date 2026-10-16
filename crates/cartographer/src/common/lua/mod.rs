//! A pure-Rust interpreter for the subset of Lua 5.2 used by Cartographer
//! configuration files (replacement for `cartographer/common/lua.h`, which
//! just includes the C Lua library).
//!
//! RUST PORT NOTE: the C++ code embeds the reference Lua 5.2 implementation.
//! This port implements the language in Rust instead (no C code, no
//! `unsafe`). Supported: the whole Lua 5.2 grammar except `goto`/labels;
//! numbers are `f64`; strings are byte strings; tables are shared by
//! reference; closures, varargs, multiple results and assignment, numeric and
//! generic `for`, `while`, `repeat`, `if`; error messages mirror Lua 5.2.
//! Library: base functions (`assert error ipairs next pairs pcall print
//! rawequal rawget rawlen rawset select tonumber tostring type unpack`),
//! `math` (all functions except `random`/`randomseed`), `string` (`byte char
//! find(plain) format len lower rep reverse sub upper`, method syntax on
//! strings), `table` (`concat insert remove sort unpack`) and `os` (`clock
//! getenv time`). Not supported: metatables, coroutines, `load`/`require`,
//! `io`, Lua patterns, `goto`.

pub mod ast;
pub mod interpreter;
pub mod lexer;
pub mod number;
pub mod parser;
pub mod stdlib;
pub mod value;

#[cfg(test)]
mod tests;

pub use interpreter::{Interpreter, LuaError};
pub use value::{FunctionRef, LuaString, TableRef, Value};
