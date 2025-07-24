//! Audio Classification Example using Edge Impulse FFI Raw Bindings
//!
//! Usage:
//!   cargo run --example ffi_audio_infer -- --audio <path_to_audio.wav> [--debug]
//!
//! This example demonstrates how to:
//! 1. Load audio data from a WAV file
//! 2. Convert it to the format expected by Edge Impulse models
//! 3. Run inference using the FFI bindings
//! 4. Display classification results

use clap::Parser;
use edge_impulse_ffi_rs::bindings::*;
use edge_impulse_ffi_rs::model_metadata::*;
use hound;
use std::error::Error;

/// Command line parameters for the audio classification example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the audio file to process (WAV format)
    #[arg(short, long)]
    audio: String,

    /// Enable debug output
    #[arg(short, long, default_value_t = false)]
    debug: bool,

    /// Optional threshold for classification
    #[arg(long)]
    threshold: Option<f32>,
}

/// Load audio data from a WAV file and convert to the format expected by Edge Impulse
fn load_audio_file(file_path: &str, target_sample_rate: u32) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut reader = hound::WavReader::open(file_path)?;
    let spec = reader.spec();

    println!("Audio file info:");
    println!("  Sample rate: {} Hz", spec.sample_rate);
    println!("  Channels: {}", spec.channels);
    println!("  Bits per sample: {}", spec.bits_per_sample);
    println!("  Sample format: {:?}", spec.sample_format);

    // Read all samples
    let samples: Vec<i16> = reader.samples().collect::<Result<Vec<i16>, _>>()?;
    println!("  Total samples: {}", samples.len());

    // Convert to mono if stereo
    let mono_samples = if spec.channels == 2 {
        println!("  Converting stereo to mono...");
        let mut mono = Vec::with_capacity(samples.len() / 2);
        for chunk in samples.chunks(2) {
            if chunk.len() == 2 {
                // Average the two channels
                let avg = (chunk[0] as i32 + chunk[1] as i32) / 2;
                mono.push(avg as i16);
            }
        }
        mono
    } else {
        samples
    };

    // Resample if needed
    let resampled_samples = if spec.sample_rate != target_sample_rate {
        println!(
            "  Resampling from {} Hz to {} Hz...",
            spec.sample_rate, target_sample_rate
        );
        resample_audio(&mono_samples, spec.sample_rate, target_sample_rate)
    } else {
        mono_samples
    };

    // Convert to f32 (no normalization - models expect raw i16 values as f32)
    let normalized_samples: Vec<f32> = resampled_samples
        .iter()
        .map(|&sample| sample as f32)
        .collect();

    println!(
        "  Final samples: {} ({} seconds)",
        normalized_samples.len(),
        normalized_samples.len() as f32 / target_sample_rate as f32
    );

    Ok(normalized_samples)
}

/// Simple resampling function (linear interpolation)
fn resample_audio(samples: &[i16], from_rate: u32, to_rate: u32) -> Vec<i16> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f32 / to_rate as f32;
    let new_length = (samples.len() as f32 / ratio).round() as usize;
    let mut resampled = Vec::with_capacity(new_length);

    for i in 0..new_length {
        let src_index = i as f32 * ratio;
        let src_index_floor = src_index.floor() as usize;
        let src_index_ceil = (src_index.ceil() as usize).min(samples.len() - 1);
        let fraction = src_index - src_index_floor as f32;

        let sample1 = samples[src_index_floor] as f32;
        let sample2 = samples[src_index_ceil] as f32;
        let interpolated = sample1 + (sample2 - sample1) * fraction;

        resampled.push(interpolated.round() as i16);
    }

    resampled
}

/// Print classification results from raw C struct
fn print_classification_results(result: &ei_impulse_result_t, label_count: u16) {
    if label_count > 0 {
        println!("\n🎵 Audio Classification Results:");
        println!("┌─────────────────────────────────────────────────────────────┐");
        println!("│ Label                    │ Confidence │ Value              │");
        println!("├─────────────────────────────────────────────────────────────┤");

        for i in 0..label_count {
            let label = unsafe {
                std::ffi::CStr::from_ptr(result.classification[i as usize].label)
                    .to_string_lossy()
                    .to_string()
            };
            let value = result.classification[i as usize].value;
            let confidence = if value > 0.5 {
                "HIGH"
            } else if value > 0.2 {
                "MED"
            } else {
                "LOW"
            };

            println!(
                "│ {:<24} │ {:<9} │ {:.6}              │",
                label, confidence, value
            );
        }
        println!("└─────────────────────────────────────────────────────────────┘");
    }
}

/// Print timing information
fn print_timing(timing: &ei_impulse_result_timing_t) {
    println!("\n⏱️  Timing Information:");
    println!("  DSP: {:.2} ms", timing.dsp);
    println!("  Classification: {:.2} ms", timing.classification);
    println!("  Anomaly: {:.2} ms", timing.anomaly);
    println!(
        "  Total: {:.2} ms",
        timing.dsp + timing.classification + timing.anomaly
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // Get model metadata from the generated constants
    let input_frames = EI_CLASSIFIER_INPUT_FRAMES as usize;
    let input_frequency = EI_CLASSIFIER_FREQUENCY as u32;
    let label_count = EI_CLASSIFIER_LABEL_COUNT as u16;

    println!("🎵 Audio Classification Example");
    println!("=================================");
    println!("Using input frames: {}", input_frames);
    println!("Target frequency: {} Hz", input_frequency);
    println!(
        "Model: {} (v{})",
        EI_CLASSIFIER_PROJECT_NAME, EI_CLASSIFIER_PROJECT_DEPLOY_VERSION
    );

    // Print model info using available constants
    println!("\n📋 Model Metadata:");
    println!(
        "  Project: {} (ID: {})",
        EI_CLASSIFIER_PROJECT_NAME, EI_CLASSIFIER_PROJECT_ID
    );
    println!("  Owner: {}", EI_CLASSIFIER_PROJECT_OWNER);
    println!("  Deploy version: {}", EI_CLASSIFIER_PROJECT_DEPLOY_VERSION);
    println!(
        "  Input: {} frames at {} Hz",
        EI_CLASSIFIER_INPUT_FRAMES, EI_CLASSIFIER_FREQUENCY
    );
    println!("  Label count: {}", EI_CLASSIFIER_LABEL_COUNT);
    println!("  Sensor: {}", EI_CLASSIFIER_SENSOR);
    println!("  Inferencing engine: {}", EI_CLASSIFIER_INFERENCING_ENGINE);
    println!("  Interval (ms): {}", EI_CLASSIFIER_INTERVAL_MS);
    println!("  Slice size: {}", EI_CLASSIFIER_SLICE_SIZE);
    println!("  Raw sample count: {}", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
    println!(
        "  Raw samples per frame: {}",
        EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME
    );
    println!(
        "  Input features count: {}",
        EI_CLASSIFIER_NN_INPUT_FRAME_SIZE
    );

    // Print extracted threshold information
    println!("\n🎯 Extracted Threshold Information:");
    let thresholds = edge_impulse_ffi_rs::thresholds::get_model_thresholds();
    for threshold in &thresholds.thresholds {
        println!(
            "  Block {}: {} (threshold: {})",
            threshold.id, threshold.threshold_type, threshold.min_score
        );
    }
    println!();

    // Load and process the audio file
    println!("📁 Loading audio file: {}", args.audio);
    let audio_samples = load_audio_file(&args.audio, input_frequency)?;

    // For audio models, use the raw sample count directly
    let required_samples = EI_CLASSIFIER_RAW_SAMPLE_COUNT as usize;
    println!(
        "📊 Model expects {} samples ({} seconds at {} Hz)",
        required_samples,
        required_samples as f32 / input_frequency as f32,
        input_frequency
    );

    if audio_samples.len() < required_samples {
        println!(
            "⚠️  Warning: Audio file has {} samples, but model expects {}",
            audio_samples.len(),
            required_samples
        );
        println!("   The model will be padded with zeros or truncated as needed.");
    }

    // Take the required number of samples (or pad with zeros)
    let mut features = Vec::with_capacity(required_samples);
    for i in 0..required_samples {
        if i < audio_samples.len() {
            features.push(audio_samples[i]);
        } else {
            features.push(0.0); // Pad with zeros
        }
    }

    if args.debug {
        let min = features.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = features.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f32 = features.iter().sum();
        let mean = sum / features.len() as f32;
        println!(
            "Feature statistics: min={:.3}, max={:.3}, mean={:.3}, count={}",
            min,
            max,
            mean,
            features.len()
        );

        // Show first and last few samples
        println!(
            "First 10 samples: {:?}",
            &features[..10.min(features.len())]
        );
        println!(
            "Last 10 samples: {:?}",
            &features[features.len().saturating_sub(10)..]
        );
    }

    // Initialize the classifier using raw C function
    println!("\n🚀 Initializing classifier...");
    unsafe {
        ei_ffi_run_classifier_init();
    }

    // Create signal from buffer using the FFI function
    let mut signal = ei_signal_t::default();
    let result_code =
        unsafe { ei_ffi_signal_from_buffer(features.as_ptr(), features.len(), &mut signal) };

    if result_code != EI_IMPULSE_ERROR::EI_IMPULSE_OK {
        eprintln!("❌ Failed to create signal from buffer: {:?}", result_code);
        return Err("Failed to create signal".into());
    }

    // Create result struct
    let mut result = ei_impulse_result_t::default();

    // Run inference using raw C function
    println!("🔍 Running inference...");
    let debug_int = if args.debug { 1 } else { 0 };
    let result_code = unsafe { ei_ffi_run_classifier(&mut signal, &mut result, debug_int) };

    match result_code {
        EI_IMPULSE_ERROR::EI_IMPULSE_OK => {
            println!("✅ Inference completed successfully!");

            // Print classification results
            print_classification_results(&result, label_count);

            // Print timing info
            print_timing(&result.timing);
        }
        error_code => {
            eprintln!(
                "❌ Error running inference: {:?} (code: {})",
                error_code, error_code as i32
            );
            println!("This might be expected if:");
            println!("1. No model is loaded/initialized");
            println!("2. The signal format doesn't match what the model expects");
            println!("3. The model requires additional setup");
            println!("4. The model is not an audio classification model");
        }
    }

    // Clean up using raw C function
    println!("\n🧹 Cleaning up...");
    unsafe {
        ei_ffi_run_classifier_deinit();
    }

    println!("✅ Done!");
    Ok(())
}
