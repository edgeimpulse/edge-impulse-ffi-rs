//! Edge Impulse FFI Rust Bindings
//!
//! This crate provides safe Rust bindings for the Edge Impulse C++ SDK,
//! allowing you to run inference on trained models from Rust applications.

pub mod bindings;

use bindings::*;

use std::error::Error;
use std::fmt;


impl fmt::Display for ClassificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:.4}", self.label, self.value)
    }
}

impl fmt::Display for BoundingBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.4} (x={}, y={}, w={}, h={})",
            self.label, self.value, self.x, self.y, self.width, self.height
        )
    }
}

impl fmt::Display for TimingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Timing: dsp={} ms, classification={} ms, anomaly={} ms",
            self.dsp, self.classification, self.anomaly
        )
    }
}

/// Error type for Edge Impulse operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeImpulseError {
    Ok,
    ShapesDontMatch,
    Canceled,
    MemoryAllocationFailed,
    OutOfMemory,
    InputTensorWasNull,
    OutputTensorWasNull,
    AllocatedTensorWasNull,
    TfliteError,
    TfliteArenaAllocFailed,
    ReadSensor,
    MinSizeRatio,
    MaxSizeRatio,
    OnlySupportImages,
    ModelInputTensorWasNull,
    ModelOutputTensorWasNull,
    UnsupportedInferencingEngine,
    AllocWhileCacheLocked,
    NoValidImpulse,
    Other,
}

impl From<bindings::EI_IMPULSE_ERROR> for EdgeImpulseError {
    fn from(error: bindings::EI_IMPULSE_ERROR) -> Self {
        match error {
            bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OK => EdgeImpulseError::Ok,
            bindings::EI_IMPULSE_ERROR::EI_IMPULSE_ERROR_SHAPES_DONT_MATCH => {
                EdgeImpulseError::ShapesDontMatch
            }
            bindings::EI_IMPULSE_ERROR::EI_IMPULSE_CANCELED => EdgeImpulseError::Canceled,
            bindings::EI_IMPULSE_ERROR::EI_IMPULSE_ALLOC_FAILED => EdgeImpulseError::MemoryAllocationFailed,
            bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OUT_OF_MEMORY => EdgeImpulseError::OutOfMemory,
            bindings::EI_IMPULSE_ERROR::EI_IMPULSE_INPUT_TENSOR_WAS_NULL => {
                EdgeImpulseError::InputTensorWasNull
            }
            bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OUTPUT_TENSOR_WAS_NULL => {
                EdgeImpulseError::OutputTensorWasNull
            }
            bindings::EI_IMPULSE_ERROR::EI_IMPULSE_TFLITE_ERROR => EdgeImpulseError::TfliteError,
            bindings::EI_IMPULSE_ERROR::EI_IMPULSE_TFLITE_ARENA_ALLOC_FAILED => {
                EdgeImpulseError::TfliteArenaAllocFailed
            }
            bindings::EI_IMPULSE_ERROR::EI_IMPULSE_DSP_ERROR => EdgeImpulseError::ReadSensor,
            bindings::EI_IMPULSE_ERROR::EI_IMPULSE_INVALID_SIZE => EdgeImpulseError::MinSizeRatio,
            bindings::EI_IMPULSE_ERROR::EI_IMPULSE_ONLY_SUPPORTED_FOR_IMAGES => {
                EdgeImpulseError::OnlySupportImages
            }
            bindings::EI_IMPULSE_ERROR::EI_IMPULSE_UNSUPPORTED_INFERENCING_ENGINE => {
                EdgeImpulseError::UnsupportedInferencingEngine
            }
            bindings::EI_IMPULSE_ERROR::EI_IMPULSE_INFERENCE_ERROR => EdgeImpulseError::NoValidImpulse,
            _ => EdgeImpulseError::Other,
        }
    }
}

impl fmt::Display for EdgeImpulseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdgeImpulseError::Ok => write!(f, "Operation completed successfully"),
            EdgeImpulseError::ShapesDontMatch => {
                write!(f, "Input shapes don't match expected dimensions")
            }
            EdgeImpulseError::Canceled => write!(f, "Operation was canceled"),
            EdgeImpulseError::MemoryAllocationFailed => write!(f, "Memory allocation failed"),
            EdgeImpulseError::OutOfMemory => write!(f, "Out of memory"),
            EdgeImpulseError::InputTensorWasNull => write!(f, "Input tensor was null"),
            EdgeImpulseError::OutputTensorWasNull => write!(f, "Output tensor was null"),
            EdgeImpulseError::AllocatedTensorWasNull => write!(f, "Allocated tensor was null"),
            EdgeImpulseError::TfliteError => write!(f, "TensorFlow Lite error"),
            EdgeImpulseError::TfliteArenaAllocFailed => {
                write!(f, "TensorFlow Lite arena allocation failed")
            }
            EdgeImpulseError::ReadSensor => write!(f, "Error reading sensor data"),
            EdgeImpulseError::MinSizeRatio => write!(f, "Minimum size ratio not met"),
            EdgeImpulseError::MaxSizeRatio => write!(f, "Maximum size ratio exceeded"),
            EdgeImpulseError::OnlySupportImages => write!(f, "Only image input is supported"),
            EdgeImpulseError::ModelInputTensorWasNull => write!(f, "Model input tensor was null"),
            EdgeImpulseError::ModelOutputTensorWasNull => write!(f, "Model output tensor was null"),
            EdgeImpulseError::UnsupportedInferencingEngine => {
                write!(f, "Unsupported inferencing engine")
            }
            EdgeImpulseError::AllocWhileCacheLocked => {
                write!(f, "Allocation attempted while cache is locked")
            }
            EdgeImpulseError::NoValidImpulse => write!(f, "No valid impulse found"),
            EdgeImpulseError::Other => write!(f, "Unknown error occurred"),
        }
    }
}

impl Error for EdgeImpulseError {}

/// Result type for Edge Impulse operations
pub type EdgeImpulseResult<T> = Result<T, EdgeImpulseError>;

/// Opaque handle for Edge Impulse operations
pub struct EdgeImpulseHandle {
    handle: *mut ei_impulse_handle_t,
}

impl EdgeImpulseHandle {
    /// Create a new Edge Impulse handle
    pub fn new() -> EdgeImpulseResult<Self> {
        let handle = std::ptr::null_mut::<bindings::ei_impulse_handle_t>();
        let result = unsafe { ei_ffi_init_impulse(handle) };
        let error = EdgeImpulseError::from(result);
        if error != EdgeImpulseError::Ok {
            return Err(error);
        }

        Ok(Self { handle })
    }
}

impl Drop for EdgeImpulseHandle {
    fn drop(&mut self) {
        // Cleanup if needed
    }
}

/// Signal structure for holding audio or sensor data
pub struct Signal {
    c_signal: Box<bindings::ei_signal_t>,
}

impl Signal {
    /// Create a new signal from raw data (f32 slice) using the SDK's signal_from_buffer
    pub fn from_raw_data(data: &[f32]) -> EdgeImpulseResult<Self> {
        let mut c_signal = Box::new(bindings::ei_signal_t {
            get_data: [0u64; 4], // Initialize with zeros for std::function
            total_length: 0,
        });
        let c_signal_ptr: *mut bindings::ei_signal_t = &mut *c_signal;
        let result = unsafe { ei_ffi_signal_from_buffer(data.as_ptr(), data.len(), c_signal_ptr) };

        if result == bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OK {
            Ok(Self { c_signal })
        } else {
            Err(EdgeImpulseError::from(result))
        }
    }

    pub fn as_ptr(&self) -> *mut bindings::ei_signal_t {
        Box::as_ref(&self.c_signal) as *const bindings::ei_signal_t as *mut bindings::ei_signal_t
    }
}

/// Result structure for Edge Impulse inference
pub struct InferenceResult {
    result: *mut ei_impulse_result_t,
}

impl InferenceResult {
    /// Create a new inference result
    pub fn new() -> Self {
        let result = unsafe {
            let ptr = std::alloc::alloc_zeroed(std::alloc::Layout::new::<ei_impulse_result_t>())
                as *mut ei_impulse_result_t;
            ptr
        };
        Self { result }
    }

    /// Get a raw pointer to the underlying ei_impulse_result_t (for advanced result parsing)
    pub fn as_ptr(&self) -> *const ei_impulse_result_t {
        self.result as *const ei_impulse_result_t
    }
    /// Get a mutable raw pointer to the underlying ei_impulse_result_t (for advanced result parsing)
    pub fn as_mut_ptr(&mut self) -> *mut ei_impulse_result_t {
        self.result
    }

    /// Get all classification results as safe Rust structs
    pub fn classifications(&self, label_count: usize) -> Vec<ClassificationResult> {
        unsafe {
            let result = &*self.result;
            (0..label_count)
                .map(|i| {
                    let c = result.classification[i];
                    let label = if !c.label.is_null() {
                        std::ffi::CStr::from_ptr(c.label)
                            .to_string_lossy()
                            .into_owned()
                    } else {
                        String::new()
                    };
                    ClassificationResult {
                        label,
                        value: c.value,
                    }
                })
                .collect()
        }
    }

    /// Get all bounding boxes as safe Rust structs
    pub fn bounding_boxes(&self) -> Vec<BoundingBox> {
        unsafe {
            let result = &*self.result;
            if result.bounding_boxes_count == 0 || result.bounding_boxes.is_null() {
                return vec![];
            }
            let bbs = std::slice::from_raw_parts(
                result.bounding_boxes,
                result.bounding_boxes_count as usize,
            );
            bbs.iter()
                .filter_map(|bb| {
                    if bb.value == 0.0 {
                        return None;
                    }
                    let label = if !bb.label.is_null() {
                        std::ffi::CStr::from_ptr(bb.label)
                            .to_string_lossy()
                            .into_owned()
                    } else {
                        String::new()
                    };
                    Some(BoundingBox {
                        label,
                        value: bb.value,
                        x: bb.x,
                        y: bb.y,
                        width: bb.width,
                        height: bb.height,
                    })
                })
                .collect()
        }
    }

    /// Get timing information
    pub fn timing(&self) -> TimingResult {
        unsafe {
            let result = &*self.result;
            let t = &result.timing;
            TimingResult {
                dsp: t.dsp as i32,
                classification: t.classification as i32,
                anomaly: t.anomaly as i32,
            }
        }
    }
}

impl Drop for InferenceResult {
    fn drop(&mut self) {
        if !self.result.is_null() {
            unsafe {
                std::alloc::dealloc(
                    self.result as *mut u8,
                    std::alloc::Layout::new::<ei_impulse_result_t>(),
                );
            }
        }
    }
}

/// Main Edge Impulse classifier
pub struct EdgeImpulseClassifier {
    initialized: bool,
}

impl EdgeImpulseClassifier {
    /// Create a new Edge Impulse classifier
    pub fn new() -> Self {
        Self { initialized: false }
    }

    /// Initialize the classifier
    pub fn init(&mut self) -> EdgeImpulseResult<()> {
        unsafe { ei_ffi_run_classifier_init() };
        self.initialized = true;
        Ok(())
    }

    /// Deinitialize the classifier
    pub fn deinit(&mut self) -> EdgeImpulseResult<()> {
        if self.initialized {
            unsafe { ei_ffi_run_classifier_deinit() };
            self.initialized = false;
        }
        Ok(())
    }

    /// Run classification on signal data
    pub fn run_classifier(
        &self,
        signal: &Signal,
        debug: bool,
    ) -> EdgeImpulseResult<InferenceResult> {
        if !self.initialized {
            return Err(EdgeImpulseError::Other);
        }
        let result = InferenceResult::new();
        let result_code = unsafe { ei_ffi_run_classifier(signal.as_ptr(), result.result, if debug { 1 } else { 0 }) };
        if result_code == bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OK {
            Ok(result)
        } else {
            Err(EdgeImpulseError::from(result_code))
        }
    }

    /// Run continuous classification on signal data
    pub fn run_classifier_continuous(
        &self,
        signal: &Signal,
        debug: bool,
        enable_maf: bool,
    ) -> EdgeImpulseResult<InferenceResult> {
        if !self.initialized {
            return Err(EdgeImpulseError::Other);
        }
        let result = InferenceResult::new();
                    let result_code = unsafe { ei_ffi_run_classifier_continuous(
                signal.as_ptr(),
                result.result,
                if debug { 1 } else { 0 },
                if enable_maf { 1 } else { 0 },
            ) };
            if result_code == bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OK {
            Ok(result)
        } else {
            Err(EdgeImpulseError::from(result_code))
        }
    }

    /// Run inference on pre-processed features
    pub fn run_inference(
        &self,
        handle: &mut EdgeImpulseHandle,
        fmatrix: *mut ei_feature_t,
        debug: bool,
    ) -> EdgeImpulseResult<InferenceResult> {
        if !self.initialized {
            return Err(EdgeImpulseError::Other);
        }
        let result = InferenceResult::new();
        let result_code = unsafe { ei_ffi_run_inference(
            handle.handle,
            fmatrix,
            result.result,
            if debug { 1 } else { 0 },
        ) };
        let error = EdgeImpulseError::from(result_code);
        if error == EdgeImpulseError::Ok {
            Ok(result)
        } else {
            Err(error)
        }
    }
}

impl Drop for EdgeImpulseClassifier {
    fn drop(&mut self) {
        let _ = self.deinit();
    }
}

// Model metadata constants (generated by build.rs)
pub mod model_metadata;

/// Helper functions to access model metadata
pub struct ModelMetadata;

#[derive(Debug, Clone)]
pub struct ModelMetadataInfo {
    pub input_width: usize,
    pub input_height: usize,
    pub input_frames: usize,
    pub label_count: usize,
    pub project_name: &'static str,
    pub project_owner: &'static str,
    pub project_id: usize,
    pub deploy_version: usize,
    pub sensor: i32,
    pub inferencing_engine: usize,
    pub interval_ms: usize,
    pub frequency: usize,
    pub slice_size: usize,
    pub has_anomaly: bool,
    pub has_object_detection: bool,
    pub has_object_tracking: bool,
    pub raw_sample_count: usize,
    pub raw_samples_per_frame: usize,
    pub input_features_count: usize,
}

#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub label: String,
    pub value: f32,
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub label: String,
    pub value: f32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct TimingResult {
    pub dsp: i32,
    pub classification: i32,
    pub anomaly: i32,
}

impl fmt::Display for ModelMetadataInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model Metadata:")?;
        writeln!(
            f,
            "  Project: {} (ID: {})",
            self.project_name, self.project_id
        )?;
        writeln!(f, "  Owner: {}", self.project_owner)?;
        writeln!(f, "  Deploy version: {}", self.deploy_version)?;
        writeln!(
            f,
            "  Input: {}x{} frames: {}",
            self.input_width, self.input_height, self.input_frames
        )?;
        writeln!(f, "  Label count: {}", self.label_count)?;
        writeln!(f, "  Sensor: {}", self.sensor)?;
        writeln!(f, "  Inferencing engine: {}", self.inferencing_engine)?;
        writeln!(f, "  Interval (ms): {}", self.interval_ms)?;
        writeln!(f, "  Frequency: {}", self.frequency)?;
        writeln!(f, "  Slice size: {}", self.slice_size)?;
        writeln!(f, "  Has anomaly: {}", self.has_anomaly)?;
        writeln!(f, "  Has object detection: {}", self.has_object_detection)?;
        writeln!(f, "  Has object tracking: {}", self.has_object_tracking)?;
        writeln!(f, "  Raw sample count: {}", self.raw_sample_count)?;
        writeln!(f, "  Raw samples per frame: {}", self.raw_samples_per_frame)?;
        writeln!(f, "  Input features count: {}", self.input_features_count)?;
        Ok(())
    }
}

impl ModelMetadata {
    /// Get the model's required input width
    pub fn input_width() -> usize {
        model_metadata::EI_CLASSIFIER_INPUT_WIDTH
    }
    /// Get the model's required input height
    pub fn input_height() -> usize {
        model_metadata::EI_CLASSIFIER_INPUT_HEIGHT
    }
    /// Get the model's required input frame size (width * height)
    pub fn input_frame_size() -> usize {
        Self::input_width() * Self::input_height()
    }
    /// Get the number of input frames
    pub fn input_frames() -> usize {
        model_metadata::EI_CLASSIFIER_INPUT_FRAMES
    }
    /// Get the number of labels
    pub fn label_count() -> usize {
        model_metadata::EI_CLASSIFIER_LABEL_COUNT
    }
    /// Get the project name
    pub fn project_name() -> &'static str {
        model_metadata::EI_CLASSIFIER_PROJECT_NAME
    }
    /// Get the project owner
    pub fn project_owner() -> &'static str {
        model_metadata::EI_CLASSIFIER_PROJECT_OWNER
    }
    /// Get the project ID
    pub fn project_id() -> usize {
        model_metadata::EI_CLASSIFIER_PROJECT_ID
    }
    /// Get the deploy version
    pub fn deploy_version() -> usize {
        model_metadata::EI_CLASSIFIER_PROJECT_DEPLOY_VERSION
    }
    /// Get the sensor type
    pub fn sensor() -> i32 {
        model_metadata::EI_CLASSIFIER_SENSOR
    }
    /// Get the inferencing engine
    pub fn inferencing_engine() -> usize {
        model_metadata::EI_CLASSIFIER_INFERENCING_ENGINE
    }
    /// Get the model's interval in ms
    pub fn interval_ms() -> usize {
        model_metadata::EI_CLASSIFIER_INTERVAL_MS
    }
    /// Get the model's frequency
    pub fn frequency() -> usize {
        model_metadata::EI_CLASSIFIER_FREQUENCY
    }
    /// Get the model's slice size
    pub fn slice_size() -> usize {
        model_metadata::EI_CLASSIFIER_SLICE_SIZE
    }
    /// Whether the model has anomaly detection
    pub fn has_anomaly() -> bool {
        model_metadata::EI_CLASSIFIER_HAS_ANOMALY != 0
    }
    /// Whether the model has object detection
    pub fn has_object_detection() -> bool {
        model_metadata::EI_CLASSIFIER_OBJECT_DETECTION != 0
    }
    /// Whether the model has object tracking
    pub fn has_object_tracking() -> bool {
        model_metadata::EI_CLASSIFIER_OBJECT_TRACKING_ENABLED != 0
    }
    /// Get the model's raw sample count
    pub fn raw_sample_count() -> usize {
        model_metadata::EI_CLASSIFIER_RAW_SAMPLE_COUNT
    }
    /// Get the model's raw samples per frame
    pub fn raw_samples_per_frame() -> usize {
        model_metadata::EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME
    }
    /// Get the model's input feature count
    pub fn input_features_count() -> usize {
        model_metadata::EI_CLASSIFIER_NN_INPUT_FRAME_SIZE
    }

    pub fn get() -> ModelMetadataInfo {
        ModelMetadataInfo {
            input_width: model_metadata::EI_CLASSIFIER_INPUT_WIDTH,
            input_height: model_metadata::EI_CLASSIFIER_INPUT_HEIGHT,
            input_frames: model_metadata::EI_CLASSIFIER_INPUT_FRAMES,
            label_count: model_metadata::EI_CLASSIFIER_LABEL_COUNT,
            project_name: model_metadata::EI_CLASSIFIER_PROJECT_NAME,
            project_owner: model_metadata::EI_CLASSIFIER_PROJECT_OWNER,
            project_id: model_metadata::EI_CLASSIFIER_PROJECT_ID,
            deploy_version: model_metadata::EI_CLASSIFIER_PROJECT_DEPLOY_VERSION,
            sensor: model_metadata::EI_CLASSIFIER_SENSOR,
            inferencing_engine: model_metadata::EI_CLASSIFIER_INFERENCING_ENGINE,
            interval_ms: model_metadata::EI_CLASSIFIER_INTERVAL_MS,
            frequency: model_metadata::EI_CLASSIFIER_FREQUENCY,
            slice_size: model_metadata::EI_CLASSIFIER_SLICE_SIZE,
            has_anomaly: model_metadata::EI_CLASSIFIER_HAS_ANOMALY != 0,
            has_object_detection: model_metadata::EI_CLASSIFIER_OBJECT_DETECTION != 0,
            has_object_tracking: model_metadata::EI_CLASSIFIER_OBJECT_TRACKING_ENABLED != 0,
            raw_sample_count: model_metadata::EI_CLASSIFIER_RAW_SAMPLE_COUNT,
            raw_samples_per_frame: model_metadata::EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME,
            input_features_count: model_metadata::EI_CLASSIFIER_NN_INPUT_FRAME_SIZE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_initialization() {
        let mut classifier = EdgeImpulseClassifier::new();
        assert!(classifier.init().is_ok());
        assert!(classifier.deinit().is_ok());
    }
}
