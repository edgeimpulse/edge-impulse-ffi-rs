//! Edge Impulse Runner API compatibility layer
//!
//! This module provides an API that mirrors the `EimModel` from `edge-impulse-runner-rs`
//! but uses the FFI bindings instead of socket communication. This allows for
//! interchangeable usage between the two crates.

use crate::{EdgeImpulseClassifier, EdgeImpulseError, EdgeImpulseHandle, Signal};
use std::collections::HashMap;
use std::fmt;

// Import types from the local module
use self::types::RunnerHelloHasAnomaly;

/// Debug callback type for receiving debug messages
pub type DebugCallback = Box<dyn Fn(&str) + Send + Sync>;

/// Edge Impulse Model compatible with edge-impulse-runner-rs API
///
/// This struct provides the same interface as `EimModel` from `edge-impulse-runner-rs`
/// but uses the FFI bindings for direct model execution instead of socket communication.
///
/// # Example Usage
///
/// ```no_run
/// use edge_impulse_ffi::runner_api::{EimModel, InferenceResponse};
///
/// // Create a new model instance (no path needed for FFI)
/// let mut model = EimModel::new().unwrap();
///
/// // Prepare normalized features (e.g., image pixels, audio samples)
/// let features: Vec<f32> = vec![0.1, 0.2, 0.3];
///
/// // Run inference
/// let result = model.infer(features, None).unwrap();
///
/// // Process results
/// match result.result {
///     InferenceResult::Classification { classification } => {
///         println!("Classification: {:?}", classification);
///     }
///     InferenceResult::ObjectDetection { bounding_boxes, classification } => {
///         println!("Detected objects: {:?}", bounding_boxes);
///         if !classification.is_empty() {
///             println!("Classification: {:?}", classification);
///         }
///     }
///     InferenceResult::VisualAnomaly { visual_anomaly_grid, visual_anomaly_max, visual_anomaly_mean, anomaly } => {
///         let (normalized_anomaly, normalized_max, normalized_mean, normalized_regions) =
///             model.normalize_visual_anomaly(
///                 anomaly,
///                 visual_anomaly_max,
///                 visual_anomaly_mean,
///                 &visual_anomaly_grid.iter()
///                     .map(|bbox| (bbox.value, bbox.x as u32, bbox.y as u32, bbox.width as u32, bbox.height as u32))
///                     .collect::<Vec<_>>()
///             );
///         println!("Anomaly score: {:.2}%", normalized_anomaly * 100.0);
///     }
/// }
/// ```
pub struct EimModel {
    /// Internal classifier instance
    classifier: EdgeImpulseClassifier,
    /// Internal handle for model operations
    handle: Option<EdgeImpulseHandle>,
    /// Enable debug logging
    debug: bool,
    /// Optional debug callback for receiving debug messages
    debug_callback: Option<DebugCallback>,
    /// Cached model parameters
    model_parameters: Option<ModelParameters>,
    /// Continuous mode state
    continuous_state: Option<ContinuousState>,
}

#[derive(Debug)]
struct ContinuousState {
    feature_matrix: Vec<f32>,
    feature_buffer_full: bool,
    maf_buffers: HashMap<String, MovingAverageFilter>,
    slice_size: usize,
}

impl ContinuousState {
    fn new(labels: Vec<String>, slice_size: usize) -> Self {
        Self {
            feature_matrix: Vec::new(),
            feature_buffer_full: false,
            maf_buffers: labels
                .into_iter()
                .map(|label| (label, MovingAverageFilter::new(4)))
                .collect(),
            slice_size,
        }
    }

    fn update_features(&mut self, features: &[f32]) {
        // Add new features to the matrix
        self.feature_matrix.extend_from_slice(features);

        // Check if buffer is full
        if self.feature_matrix.len() >= self.slice_size {
            self.feature_buffer_full = true;
            // Keep only the most recent features if we've exceeded the buffer size
            if self.feature_matrix.len() > self.slice_size {
                self.feature_matrix
                    .drain(0..self.feature_matrix.len() - self.slice_size);
            }
        }
    }

    fn apply_maf(&mut self, classification: &mut HashMap<String, f32>) {
        for (label, value) in classification.iter_mut() {
            if let Some(maf) = self.maf_buffers.get_mut(label) {
                *value = maf.update(*value);
            }
        }
    }
}

#[derive(Debug)]
struct MovingAverageFilter {
    buffer: std::collections::VecDeque<f32>,
    window_size: usize,
    sum: f32,
}

impl MovingAverageFilter {
    fn new(window_size: usize) -> Self {
        Self {
            buffer: std::collections::VecDeque::with_capacity(window_size),
            window_size,
            sum: 0.0,
        }
    }

    fn update(&mut self, value: f32) -> f32 {
        if self.buffer.len() >= self.window_size {
            if let Some(old_value) = self.buffer.pop_front() {
                self.sum -= old_value;
            }
        }
        self.buffer.push_back(value);
        self.sum += value;
        self.sum / self.buffer.len() as f32
    }
}

impl fmt::Debug for EimModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EimModel")
            .field("debug", &self.debug)
            .field("model_parameters", &self.model_parameters)
            .field("continuous_state", &self.continuous_state.is_some())
            .finish()
    }
}

impl EimModel {
    /// Create a new Edge Impulse model instance
    ///
    /// Unlike the socket-based version, this doesn't require a path to an .eim file
    /// since the model is compiled into the binary via FFI.
    pub fn new() -> Result<Self, EimError> {
        let mut classifier = EdgeImpulseClassifier::new();
        match classifier.init() {
            Ok(_) => println!("✓ Classifier initialized successfully"),
            Err(e) => {
                eprintln!("✗ Failed to initialize classifier: {:?}", e);
                return Err(EimError::ExecutionError(format!("Classifier initialization failed: {:?}", e)));
            }
        }

        // Print model metadata for debugging
        println!("✓ Model metadata:");
        println!("  - Input size: {}x{}", crate::ModelMetadata::input_width(), crate::ModelMetadata::input_height());
        println!("  - Input features: {}", crate::ModelMetadata::input_features_count());
        println!("  - Labels: {}", crate::ModelMetadata::label_count());
        println!("  - Sensor: {}", crate::ModelMetadata::sensor());
        println!("  - Has object detection: {}", crate::ModelMetadata::has_object_detection());
        println!("  - Has anomaly: {}", crate::ModelMetadata::has_anomaly());

        let model_parameters = Some(ModelParameters {
            axis_count: crate::ModelMetadata::frequency() as u32,
            frequency: crate::ModelMetadata::frequency() as f32,
            has_anomaly: if crate::ModelMetadata::has_anomaly() {
                RunnerHelloHasAnomaly::KMeans
            } else {
                RunnerHelloHasAnomaly::None
            },
            has_object_tracking: crate::ModelMetadata::has_object_tracking(),
            image_channel_count: 3, // Default to RGB
            image_input_frames: crate::ModelMetadata::input_frames() as u32,
            image_input_height: crate::ModelMetadata::input_height() as u32,
            image_input_width: crate::ModelMetadata::input_width() as u32,
            image_resize_mode: "fit".to_string(),
            inferencing_engine: crate::ModelMetadata::inferencing_engine() as u32,
            input_features_count: crate::ModelMetadata::input_features_count() as u32,
            interval_ms: crate::ModelMetadata::interval_ms() as f32,
            label_count: crate::ModelMetadata::label_count() as u32,
            labels: vec![], // Will be populated from model metadata
            model_type: if crate::ModelMetadata::has_object_detection() {
                "object-detection".to_string()
            } else {
                "classification".to_string()
            },
            sensor: crate::ModelMetadata::sensor(),
            slice_size: crate::ModelMetadata::slice_size() as u32,
            thresholds: vec![],
            use_continuous_mode: false,
        });

        Ok(Self {
            classifier,
            handle: None, // We don't need the handle for basic inference
            debug: false,
            debug_callback: None,
            model_parameters,
            continuous_state: None,
        })
    }

    /// Create a new model instance with debug enabled
    pub fn new_with_debug(debug: bool) -> Result<Self, EimError> {
        let mut model = Self::new()?;
        model.debug = debug;
        Ok(model)
    }

    /// Set a debug callback for receiving debug messages
    pub fn set_debug_callback<F>(&mut self, callback: F)
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        self.debug_callback = Some(Box::new(callback));
    }

    fn debug_message(&self, message: &str) {
        if self.debug {
            if let Some(callback) = &self.debug_callback {
                callback(message);
            } else {
                eprintln!("[DEBUG] {}", message);
            }
        }
    }

    /// Get the model parameters
    pub fn parameters(&self) -> Result<&ModelParameters, EimError> {
        self.model_parameters
            .as_ref()
            .ok_or(EimError::ModelNotInitialized)
    }

    /// Get the sensor type
    pub fn sensor_type(&self) -> Result<SensorType, EimError> {
        let sensor_id = crate::ModelMetadata::sensor();
        Ok(SensorType::from(sensor_id))
    }

    /// Get the required input size for the model
    pub fn input_size(&self) -> Result<usize, EimError> {
        Ok(crate::ModelMetadata::input_features_count())
    }

    /// Run inference on the provided features
    pub fn infer(
        &mut self,
        features: Vec<f32>,
        debug: Option<bool>,
    ) -> Result<InferenceResponse, EimError> {
        let debug_enabled = debug.unwrap_or(self.debug);

        // Check if continuous mode is required
        if self.requires_continuous_mode() {
            self.infer_continuous_internal(features, debug_enabled)
        } else {
            self.infer_single(features, debug_enabled)
        }
    }

    fn infer_continuous_internal(
        &mut self,
        features: Vec<f32>,
        debug: bool,
    ) -> Result<InferenceResponse, EimError> {
        // Get labels before any mutable borrows
        let labels = self.get_labels()?;

        // Initialize continuous state if needed
        if self.continuous_state.is_none() {
            let slice_size = self.parameters()?.slice_size as usize;
            self.continuous_state = Some(ContinuousState::new(labels.clone(), slice_size));
        }

        let continuous_state = self.continuous_state.as_mut().unwrap();
        continuous_state.update_features(&features);

        if !continuous_state.feature_buffer_full {
            // Return empty classification until buffer is full
            let mut empty_classification = HashMap::new();
            for label in &labels {
                empty_classification.insert(label.clone(), 0.0);
            }
            continuous_state.apply_maf(&mut empty_classification);

            return Ok(InferenceResponse {
                success: true,
                id: 1,
                result: InferenceResult::Classification {
                    classification: empty_classification,
                },
            });
        }

        // Use the full feature matrix for inference
        let signal = Signal::from_raw_data(&continuous_state.feature_matrix)?;
        let result = self.classifier.run_classifier_continuous(&signal, debug, true)?;

        // Convert to runner API format
        self.convert_inference_result(result)
    }

    fn infer_single(
        &mut self,
        features: Vec<f32>,
        debug: bool,
    ) -> Result<InferenceResponse, EimError> {
        let signal = Signal::from_raw_data(&features)?;
        let result = self.classifier.run_classifier(&signal, debug)?;

        // Convert to runner API format
        self.convert_inference_result(result)
    }

    fn convert_inference_result(&self, result: crate::InferenceResult) -> Result<InferenceResponse, EimError> {
        let classifications = result.classifications(crate::ModelMetadata::label_count());
        let bounding_boxes = result.bounding_boxes();
        let _timing = result.timing();

        // Convert classifications to HashMap
        let mut classification_map = HashMap::new();
        for classification in classifications {
            classification_map.insert(classification.label, classification.value);
        }

        // Determine result type based on model capabilities
        let inference_result = if !bounding_boxes.is_empty() {
            InferenceResult::ObjectDetection {
                bounding_boxes: bounding_boxes
                    .into_iter()
                    .map(|bb| self::types::BoundingBox {
                        label: bb.label,
                        value: bb.value,
                        x: bb.x as i32,
                        y: bb.y as i32,
                        width: bb.width as i32,
                        height: bb.height as i32,
                    })
                    .collect(),
                classification: classification_map,
            }
        } else if crate::ModelMetadata::has_anomaly() {
            // For anomaly detection, we need to check if it's visual anomaly
            // This is a simplified implementation - you might need to adjust based on your model
            InferenceResult::VisualAnomaly {
                visual_anomaly_grid: vec![],
                visual_anomaly_max: 0.0,
                visual_anomaly_mean: 0.0,
                anomaly: classification_map.get("anomaly").copied().unwrap_or(0.0),
            }
        } else {
            InferenceResult::Classification {
                classification: classification_map,
            }
        };

        Ok(InferenceResponse {
            success: true,
            id: 1,
            result: inference_result,
        })
    }

    fn requires_continuous_mode(&self) -> bool {
        if let Ok(params) = self.parameters() {
            params.use_continuous_mode
        } else {
            false
        }
    }

    fn get_labels(&self) -> Result<Vec<String>, EimError> {
        // This would need to be implemented based on your model metadata
        // For now, return a default set
        Ok((0..crate::ModelMetadata::label_count())
            .map(|i| format!("label_{}", i))
            .collect())
    }

    /// Normalize visual anomaly results
    pub fn normalize_visual_anomaly(
        &self,
        anomaly: f32,
        max: f32,
        mean: f32,
        regions: &[(f32, u32, u32, u32, u32)],
    ) -> self::types::VisualAnomalyResult {
        // Simple normalization - clamp to [0, 1] range
        let normalized_anomaly = anomaly.max(0.0).min(1.0);
        let normalized_max = max.max(0.0).min(1.0);
        let normalized_mean = mean.max(0.0).min(1.0);

        let normalized_regions = regions
            .iter()
            .map(|(value, x, y, w, h)| {
                (value.max(0.0).min(1.0), *x, *y, *w, *h)
            })
            .collect();

        (normalized_anomaly, normalized_max, normalized_mean, normalized_regions)
    }
}

impl Drop for EimModel {
    fn drop(&mut self) {
        // Cleanup is handled by the classifier's Drop implementation
    }
}

// Re-export types for compatibility
pub use self::types::{BoundingBox, ModelParameters, ProjectInfo, SensorType, TimingInfo};
pub use self::inference::messages::{InferenceResponse, InferenceResult};

// Import the types from the runner crate for compatibility
pub mod types {

    /// Enum representing different types of anomaly detection
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum RunnerHelloHasAnomaly {
        None = 0,
        KMeans = 1,
        GMM = 2,
        VisualGMM = 3,
    }

    impl From<u32> for RunnerHelloHasAnomaly {
        fn from(value: u32) -> Self {
            match value {
                0 => Self::None,
                1 => Self::KMeans,
                2 => Self::GMM,
                3 => Self::VisualGMM,
                _ => Self::None,
            }
        }
    }

    /// Parameters that define a model's configuration and capabilities
    #[derive(Debug, Clone)]
    pub struct ModelParameters {
        pub axis_count: u32,
        pub frequency: f32,
        pub has_anomaly: RunnerHelloHasAnomaly,
        pub has_object_tracking: bool,
        pub image_channel_count: u32,
        pub image_input_frames: u32,
        pub image_input_height: u32,
        pub image_input_width: u32,
        pub image_resize_mode: String,
        pub inferencing_engine: u32,
        pub input_features_count: u32,
        pub interval_ms: f32,
        pub label_count: u32,
        pub labels: Vec<String>,
        pub model_type: String,
        pub sensor: i32,
        pub slice_size: u32,
        pub thresholds: Vec<ModelThreshold>,
        pub use_continuous_mode: bool,
    }

    #[derive(Debug, Clone)]
    pub enum ModelThreshold {
        ObjectDetection { id: u32, min_score: f32 },
        AnomalyGMM { id: u32, min_anomaly_score: f32 },
        ObjectTracking {
            id: u32,
            keep_grace: u32,
            max_observations: u32,
            threshold: f32,
        },
        Unknown { id: u32, unknown: f32 },
    }

    /// Information about the Edge Impulse project
    #[derive(Debug)]
    pub struct ProjectInfo {
        pub deploy_version: u32,
        pub id: u32,
        pub name: String,
        pub owner: String,
    }

    /// Performance timing information
    #[derive(Debug)]
    pub struct TimingInfo {
        pub dsp: u32,
        pub classification: u32,
        pub anomaly: u32,
        pub json: u32,
        pub stdin: u32,
    }

    /// Represents a detected object's location and classification
    #[derive(Debug, Clone)]
    pub struct BoundingBox {
        pub height: i32,
        pub label: String,
        pub value: f32,
        pub width: i32,
        pub x: i32,
        pub y: i32,
    }

    /// Represents the normalized results of visual anomaly detection
    pub type VisualAnomalyResult = (f32, f32, f32, Vec<(f32, u32, u32, u32, u32)>);

    /// Represents the type of sensor used for data collection
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SensorType {
        Unknown = -1,
        Microphone = 1,
        Accelerometer = 2,
        Camera = 3,
        Positional = 4,
    }

    impl From<i32> for SensorType {
        fn from(value: i32) -> Self {
            match value {
                1 => Self::Microphone,
                2 => Self::Accelerometer,
                3 => Self::Camera,
                4 => Self::Positional,
                _ => Self::Unknown,
            }
        }
    }
}

pub mod inference {
    pub mod messages {
        use crate::runner_api::types::*;
        use std::collections::HashMap;
        use std::fmt;

        /// Represents different types of inference results
        #[derive(Debug, Clone)]
        pub enum InferenceResult {
            /// Result from a classification model
            Classification {
                /// Map of class names to their probability scores
                classification: HashMap<String, f32>,
            },
            /// Result from an object detection model
            ObjectDetection {
                /// Vector of detected objects with their bounding boxes
                bounding_boxes: Vec<BoundingBox>,
                /// Optional classification results for the entire image
                classification: HashMap<String, f32>,
            },
            /// Result from a visual anomaly detection model
            VisualAnomaly {
                /// Grid of anomaly scores for different regions of the image
                visual_anomaly_grid: Vec<BoundingBox>,
                /// Maximum anomaly score across all regions
                visual_anomaly_max: f32,
                /// Mean anomaly score across all regions
                visual_anomaly_mean: f32,
                /// Overall anomaly score for the image
                anomaly: f32,
            },
        }

        /// Response containing inference results
        #[derive(Debug, Clone)]
        pub struct InferenceResponse {
            /// Indicates if the inference was successful
            pub success: bool,
            /// Message identifier matching the request
            pub id: u32,
            /// The actual inference results
            pub result: InferenceResult,
        }

        impl fmt::Display for InferenceResponse {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match &self.result {
                    InferenceResult::Classification { classification } => {
                        write!(f, "Classification results: ")?;
                        for (class, probability) in classification {
                            write!(f, "{}={:.2}% ", class, probability * 100.0)?;
                        }
                        Ok(())
                    }
                    InferenceResult::ObjectDetection {
                        bounding_boxes,
                        classification,
                    } => {
                        if !classification.is_empty() {
                            write!(f, "Image classification: ")?;
                            for (class, probability) in classification {
                                write!(f, "{}={:.2}% ", class, probability * 100.0)?;
                            }
                            writeln!(f)?;
                        }
                        write!(f, "Detected objects: ")?;
                        for bbox in bounding_boxes {
                            write!(
                                f,
                                "{}({:.2}%) at ({},{},{},{}) ",
                                bbox.label,
                                bbox.value * 100.0,
                                bbox.x,
                                bbox.y,
                                bbox.width,
                                bbox.height
                            )?;
                        }
                        Ok(())
                    }
                    InferenceResult::VisualAnomaly {
                        visual_anomaly_grid,
                        visual_anomaly_max,
                        visual_anomaly_mean,
                        anomaly,
                    } => {
                        write!(
                            f,
                            "Visual anomaly detection: max={:.2}%, mean={:.2}%, overall={:.2}%",
                            visual_anomaly_max * 100.0,
                            visual_anomaly_mean * 100.0,
                            anomaly * 100.0
                        )?;
                        if !visual_anomaly_grid.is_empty() {
                            writeln!(f)?;
                            write!(f, "Anomaly grid: ")?;
                            for bbox in visual_anomaly_grid {
                                write!(
                                    f,
                                    "{}({:.2}%) at ({},{},{},{}) ",
                                    bbox.label,
                                    bbox.value * 100.0,
                                    bbox.x,
                                    bbox.y,
                                    bbox.width,
                                    bbox.height
                                )?;
                            }
                        }
                        Ok(())
                    }
                }
            }
        }
    }
}

/// Error type for Edge Impulse operations (compatible with edge-impulse-runner-rs)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EimError {
    InvalidPath,
    SocketError(String),
    ExecutionError(String),
    ModelNotInitialized,
    InvalidInput(String),
    JsonError(String),
    TimeoutError(String),
    Other(String),
}

impl std::fmt::Display for EimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EimError::InvalidPath => write!(f, "Invalid file path"),
            EimError::SocketError(msg) => write!(f, "Socket error: {}", msg),
            EimError::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            EimError::ModelNotInitialized => write!(f, "Model not initialized"),
            EimError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            EimError::JsonError(msg) => write!(f, "JSON error: {}", msg),
            EimError::TimeoutError(msg) => write!(f, "Timeout error: {}", msg),
            EimError::Other(msg) => write!(f, "Other error: {}", msg),
        }
    }
}

impl std::error::Error for EimError {}

impl From<EdgeImpulseError> for EimError {
    fn from(error: EdgeImpulseError) -> Self {
        match error {
            EdgeImpulseError::Ok => EimError::Other("Unexpected OK status".to_string()),
            EdgeImpulseError::ShapesDontMatch => EimError::InvalidInput("Shapes don't match".to_string()),
            EdgeImpulseError::Canceled => EimError::ExecutionError("Operation canceled".to_string()),
            EdgeImpulseError::MemoryAllocationFailed => EimError::ExecutionError("Memory allocation failed".to_string()),
            EdgeImpulseError::OutOfMemory => EimError::ExecutionError("Out of memory".to_string()),
            EdgeImpulseError::InputTensorWasNull => EimError::InvalidInput("Input tensor was null".to_string()),
            EdgeImpulseError::OutputTensorWasNull => EimError::ExecutionError("Output tensor was null".to_string()),
            EdgeImpulseError::AllocatedTensorWasNull => EimError::ExecutionError("Allocated tensor was null".to_string()),
            EdgeImpulseError::TfliteError => EimError::ExecutionError("TensorFlow Lite error".to_string()),
            EdgeImpulseError::TfliteArenaAllocFailed => EimError::ExecutionError("TensorFlow Lite arena allocation failed".to_string()),
            EdgeImpulseError::ReadSensor => EimError::ExecutionError("Error reading sensor data".to_string()),
            EdgeImpulseError::MinSizeRatio => EimError::InvalidInput("Minimum size ratio not met".to_string()),
            EdgeImpulseError::MaxSizeRatio => EimError::InvalidInput("Maximum size ratio exceeded".to_string()),
            EdgeImpulseError::OnlySupportImages => EimError::InvalidInput("Only image input is supported".to_string()),
            EdgeImpulseError::ModelInputTensorWasNull => EimError::ExecutionError("Model input tensor was null".to_string()),
            EdgeImpulseError::ModelOutputTensorWasNull => EimError::ExecutionError("Model output tensor was null".to_string()),
            EdgeImpulseError::UnsupportedInferencingEngine => EimError::ExecutionError("Unsupported inferencing engine".to_string()),
            EdgeImpulseError::AllocWhileCacheLocked => EimError::ExecutionError("Allocation attempted while cache is locked".to_string()),
            EdgeImpulseError::NoValidImpulse => EimError::ModelNotInitialized,
            EdgeImpulseError::Other => EimError::Other("Unknown error occurred".to_string()),
        }
    }
}