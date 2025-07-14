//! Image Classification Example using Edge Impulse FFI Raw Bindings
//!
//! Usage:
//!   cargo run --example ffi_image_infer -- --image <path_to_image> [--debug]

use clap::Parser;
use edge_impulse_ffi_rs::bindings::*;
use edge_impulse_ffi_rs::model_metadata::*;
use image::{self, GenericImageView};
use image::{imageops::FilterType, DynamicImage, RgbImage};
use std::error::Error;

/// Command line parameters for the image classification example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the image file to process
    #[arg(short, long)]
    image: String,

    /// Enable debug output
    #[arg(short, long, default_value_t = false)]
    debug: bool,
}

/// Resize and crop the image according to Edge Impulse model metadata
fn resize_and_crop(
    img: &DynamicImage,
    input_width: u32,
    input_height: u32,
    resize_mode: usize,
) -> RgbImage {
    let (w, h) = img.dimensions();
    match resize_mode {
        x if x == 0 => img // EI_CLASSIFIER_RESIZE_SQUASH
            .resize_exact(input_width, input_height, FilterType::Triangle)
            .to_rgb8(),
        x if x == 1 => {
            // EI_CLASSIFIER_RESIZE_FIT_SHORTEST
            let factor = (input_width as f32 / w as f32).min(input_height as f32 / h as f32);
            let resize_w = (w as f32 * factor).round() as u32;
            let resize_h = (h as f32 * factor).round() as u32;
            let resized = img.resize_exact(resize_w, resize_h, FilterType::Triangle);
            let crop_x = if resize_w > input_width {
                (resize_w - input_width) / 2
            } else {
                0
            };
            let crop_y = if resize_h > input_height {
                (resize_h - input_height) / 2
            } else {
                0
            };
            image::DynamicImage::ImageRgba8(
                image::imageops::crop_imm(&resized, crop_x, crop_y, input_width, input_height)
                    .to_image(),
            )
            .to_rgb8()
        }
        x if x == 2 => {
            // EI_CLASSIFIER_RESIZE_FIT_LONGEST
            let factor = (input_width as f32 / w as f32).max(input_height as f32 / h as f32);
            let resize_w = (w as f32 * factor).round() as u32;
            let resize_h = (h as f32 * factor).round() as u32;
            let resized = img.resize_exact(resize_w, resize_h, FilterType::Triangle);
            // Pad to center
            let mut out = RgbImage::new(input_width, input_height);
            let x_offset = if input_width > resize_w {
                (input_width - resize_w) / 2
            } else {
                0
            };
            let y_offset = if input_height > resize_h {
                (input_height - resize_h) / 2
            } else {
                0
            };
            image::imageops::replace(
                &mut out,
                &resized.to_rgb8(),
                x_offset as i64,
                y_offset as i64,
            );
            out
        }
        _ => {
            // Default to squash if unknown
            img.resize_exact(input_width, input_height, FilterType::Triangle)
                .to_rgb8()
        }
    }
}

/// Print classification results from raw C struct
fn print_classification_results(result: &ei_impulse_result_t, label_count: u16) {
    if label_count > 0 {
        println!("Classification results:");
        // The classification array is fixed size with only 1 element
        // In a real implementation, you'd need to get the actual results differently
        let classification = &result.classification[0];
        if !classification.label.is_null() {
            let label = unsafe {
                std::ffi::CStr::from_ptr(classification.label)
                    .to_string_lossy()
                    .to_string()
            };
            println!("  {}: {:.3}", label, classification.value);
        } else {
            println!("  No classification result available");
        }

        // Note: For multiple labels, you'd typically need to access the results
        // through a different mechanism or pointer provided by the C library
        if label_count > 1 {
            println!(
                "  Note: Model has {} labels but only first result shown",
                label_count
            );
        }
    }
}

/// Print bounding box results from raw C struct
fn print_bounding_boxes(result: &ei_impulse_result_t) {
    if result.bounding_boxes_count > 0 && !result.bounding_boxes.is_null() {
        println!("Object detection results:");
        let boxes = unsafe {
            std::slice::from_raw_parts(result.bounding_boxes, result.bounding_boxes_count as usize)
        };
        for (i, bb) in boxes.iter().enumerate() {
            if !bb.label.is_null() {
                let label = unsafe {
                    std::ffi::CStr::from_ptr(bb.label)
                        .to_string_lossy()
                        .to_string()
                };
                println!(
                    "  Box {}: {} at ({}, {}) {}x{} with confidence {:.3}",
                    i, label, bb.x, bb.y, bb.width, bb.height, bb.value
                );
            }
        }
    }
}

/// Print timing information from raw C struct
fn print_timing(timing: &ei_impulse_result_timing_t) {
    println!("Timing:");
    println!("  DSP: {} ms ({} μs)", timing.dsp, timing.dsp_us);
    println!(
        "  Classification: {} ms ({} μs)",
        timing.classification, timing.classification_us
    );
    if timing.anomaly > 0 {
        println!(
            "  Anomaly: {} ms ({} μs)",
            timing.anomaly, timing.anomaly_us
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // Get model metadata from the generated constants
    let input_width = EI_CLASSIFIER_INPUT_WIDTH as u32;
    let input_height = EI_CLASSIFIER_INPUT_HEIGHT as u32;
    let resize_mode = EI_CLASSIFIER_RESIZE_MODE;
    let label_count = EI_CLASSIFIER_LABEL_COUNT as u16;

    println!("Using input dimensions: {}x{}", input_width, input_height);
    println!(
        "Model: {} (v{})",
        EI_CLASSIFIER_PROJECT_NAME, EI_CLASSIFIER_PROJECT_DEPLOY_VERSION
    );
    println!("Resize mode: {}", resize_mode);

    // Load and process the image
    let img = image::open(&args.image)?;
    let (width, height) = img.dimensions();
    println!("Loaded image: {} ({}x{})", args.image, width, height);

    // Resize and crop using hardcoded dimensions
    let rgb = resize_and_crop(&img, input_width, input_height, resize_mode);
    println!("Processed image to {}x{} RGB", input_width, input_height);

    // Pack each pixel as (r << 16) + (g << 8) + b, as f32
    let mut features = Vec::with_capacity((input_width * input_height) as usize);
    for pixel in rgb.pixels() {
        let [r, g, b] = pixel.0;
        let packed = ((r as u32) << 16) + ((g as u32) << 8) + (b as u32);
        features.push(packed as f32);
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
    }

    // Initialize the classifier using raw C function
    unsafe {
        ei_ffi_run_classifier_init();
    }

    // Create signal from buffer using the FFI function
    let mut signal = ei_signal_t::default();
    let result_code =
        unsafe { ei_ffi_signal_from_buffer(features.as_ptr(), features.len(), &mut signal) };

    if result_code != EI_IMPULSE_ERROR::EI_IMPULSE_OK {
        eprintln!("Failed to create signal from buffer: {:?}", result_code);
        return Err("Failed to create signal".into());
    }

    // Create result struct
    let mut result = ei_impulse_result_t::default();

    // Run inference using raw C function
    let debug_int = if args.debug { 1 } else { 0 };
    let result_code = unsafe { ei_ffi_run_classifier(&mut signal, &mut result, debug_int) };

    match result_code {
        EI_IMPULSE_ERROR::EI_IMPULSE_OK => {
            println!("Inference ran successfully!");

            // Print classification results
            print_classification_results(&result, label_count);

            // Print bounding boxes for object detection
            print_bounding_boxes(&result);

            // Print timing info
            print_timing(&result.timing);
        }
        error_code => {
            eprintln!(
                "Error running inference: {:?} (code: {})",
                error_code, error_code as i32
            );
            println!("This might be expected if:");
            println!("1. No model is loaded/initialized");
            println!("2. The signal format doesn't match what the model expects");
            println!("3. The model requires additional setup");
        }
    }

    // Clean up using raw C function
    unsafe {
        ei_ffi_run_classifier_deinit();
    }

    Ok(())
}
