//! Edge Impulse FFI Rust Bindings
//!
//! This crate provides safe Rust bindings for the Edge Impulse C++ SDK,
//! allowing you to run inference on trained models from Rust applications.

pub mod bindings;
pub mod model_metadata;
pub mod thresholds;

// Re-export the bindings for convenience
pub use bindings::*;
