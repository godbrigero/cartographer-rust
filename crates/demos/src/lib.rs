//! Support code for the end-to-end demos of the Cartographer Rust port
//! (see `examples/` and `README.md` of this crate).
//!
//! * [`canvas`]: a tiny pure-Rust RGB raster (lines, discs, a 3×5 bitmap
//!   font) written as PNG, used to plot maps, trajectories and error curves
//!   without any plotting dependency.

#![forbid(unsafe_code)]

pub mod canvas;
