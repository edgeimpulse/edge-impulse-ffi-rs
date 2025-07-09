#![allow(warnings, non_camel_case_types, non_snake_case, non_upper_case_globals, unused_variables, unpredictable_function_pointer_comparisons)]

// Dummy bindings for development when no model is present
// These are pure Rust implementations that allow the crate to compile

use std::os::raw::{c_int, c_void, c_char};

// Define EI_IMPULSE_ERROR as a type alias (like a C enum)
pub type EI_IMPULSE_ERROR = i32;

// Define the error constants
pub const EI_IMPULSE_OK: EI_IMPULSE_ERROR = 0;
pub const EI_IMPULSE_ERROR_SHAPES_DONT_MATCH: EI_IMPULSE_ERROR = 1;
pub const EI_IMPULSE_CANCELED: EI_IMPULSE_ERROR = 2;
pub const EI_IMPULSE_ALLOC_FAILED: EI_IMPULSE_ERROR = 3;
pub const EI_IMPULSE_OUT_OF_MEMORY: EI_IMPULSE_ERROR = 4;
pub const EI_IMPULSE_INPUT_TENSOR_WAS_NULL: EI_IMPULSE_ERROR = 5;
pub const EI_IMPULSE_OUTPUT_TENSOR_WAS_NULL: EI_IMPULSE_ERROR = 6;
pub const EI_IMPULSE_TFLITE_ERROR: EI_IMPULSE_ERROR = 7;
pub const EI_IMPULSE_TFLITE_ARENA_ALLOC_FAILED: EI_IMPULSE_ERROR = 8;
pub const EI_IMPULSE_DSP_ERROR: EI_IMPULSE_ERROR = 9;
pub const EI_IMPULSE_INVALID_SIZE: EI_IMPULSE_ERROR = 10;
pub const EI_IMPULSE_ONLY_SUPPORTED_FOR_IMAGES: EI_IMPULSE_ERROR = 11;
pub const EI_IMPULSE_UNSUPPORTED_INFERENCING_ENGINE: EI_IMPULSE_ERROR = 12;
pub const EI_IMPULSE_INFERENCE_ERROR: EI_IMPULSE_ERROR = 13;

// Define dummy struct types that the code expects
#[repr(C)]
pub struct ei_signal_t {
    pub get_data: [u64; 4], // std::function equivalent
    pub total_length: usize,
}

#[repr(C)]
pub struct ei_impulse_result_classification_t {
    pub label: *const c_char,
    pub value: f32,
}

#[repr(C)]
pub struct ei_impulse_result_bounding_box_t {
    pub label: *const c_char,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub value: f32,
}

#[repr(C)]
pub struct ei_impulse_result_timing_t {
    pub dsp: u64,
    pub classification: u64,
    pub anomaly: u64,
}

#[repr(C)]
pub struct ei_impulse_result_t {
    pub classification: *mut ei_impulse_result_classification_t,
    pub bounding_boxes: *mut ei_impulse_result_bounding_box_t,
    pub bounding_boxes_count: c_int,
    pub timing: ei_impulse_result_timing_t,
}

pub type ei_impulse_handle_t = *mut *mut c_void;
pub type ei_feature_t = *mut *mut c_void;

// Pure Rust dummy function implementations that return error codes
pub fn ei_ffi_run_classifier_init() -> EI_IMPULSE_ERROR {
    -1 // Return error
}

pub fn ei_ffi_run_classifier_deinit() -> EI_IMPULSE_ERROR {
    -1 // Return error
}

pub fn ei_ffi_init_impulse(_handle: *mut c_void) -> EI_IMPULSE_ERROR {
    -1 // Return error
}

pub fn ei_ffi_run_classifier(_signal: *mut ei_signal_t, _result: *mut *mut ei_impulse_result_t, _debug: c_int) -> EI_IMPULSE_ERROR {
    -1 // Return error
}

pub fn ei_ffi_run_classifier_continuous(_signal: *mut ei_signal_t, _result: *mut *mut ei_impulse_result_t, _debug: c_int, _enable_maf: c_int) -> EI_IMPULSE_ERROR {
    -1 // Return error
}

pub fn ei_ffi_run_inference(_handle: *mut c_void, _fmatrix: *mut ei_feature_t, _result: *mut *mut ei_impulse_result_t, _debug: c_int) -> EI_IMPULSE_ERROR {
    -1 // Return error
}

pub fn ei_ffi_signal_from_buffer(_data: *const f32, _data_len: usize, _signal: *mut *mut ei_signal_t) -> EI_IMPULSE_ERROR {
    -1 // Return error
}