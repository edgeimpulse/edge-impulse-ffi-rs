//! Image Classification Example using Edge Impulse FFI Runner API
//!
//! This example demonstrates how to use the Edge Impulse FFI bindings with
//! the runner API compatibility layer, providing the same interface as
//! edge-impulse-runner-rs but using direct FFI calls instead of socket communication.
//!
//! Usage:
//!   cargo run --example image_infer -- --image <path_to_image> [--debug]

use clap::Parser;
use edge_impulse_ffi_rs::model_metadata;
use edge_impulse_ffi_rs::runner_api::{EimModel, InferenceResult};
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
        x if x == model_metadata::EI_CLASSIFIER_RESIZE_SQUASH => img
            .resize_exact(input_width, input_height, FilterType::Triangle)
            .to_rgb8(),
        x if x == model_metadata::EI_CLASSIFIER_RESIZE_FIT_SHORTEST => {
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
        x if x == model_metadata::EI_CLASSIFIER_RESIZE_FIT_LONGEST => {
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

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    println!("Edge Impulse FFI Runner API - Image Classification Example");
    println!("=========================================================");

    // Create a new model instance using the runner API
    let mut model = EimModel::new()?;
    println!("✓ Model initialized successfully");

    // Get model parameters
    let params = model.parameters()?;
    println!("✓ Model parameters:");
    println!("  - Input size: {}x{}", params.image_input_width, params.image_input_height);
    println!("  - Input features: {}", params.input_features_count);
    println!("  - Labels: {}", params.label_count);
    println!("  - Sensor type: {:?}", model.sensor_type()?);
    println!("  - Model type: {}", params.model_type);
    println!("  - Has anomaly detection: {:?}", params.has_anomaly);
    println!("  - Has object detection: {}", !params.model_type.is_empty() && params.model_type.contains("object"));

    // Load and process the image
    let img = image::open(&args.image)?;
    let (width, height) = img.dimensions();
    println!("✓ Loaded image: {} ({}x{})", args.image, width, height);

    // Resize and crop using model metadata
    let (iw, ih) = (params.image_input_width, params.image_input_height);
    let rgb = resize_and_crop(&img, iw, ih, model_metadata::EI_CLASSIFIER_RESIZE_MODE);
    println!("✓ Processed image to {}x{} RGB", iw, ih);

    // Pack each pixel as (r << 16) + (g << 8) + b, as f32
    let mut features = Vec::with_capacity((iw * ih) as usize);
    for pixel in rgb.pixels() {
        let [r, g, b] = pixel.0;
        let packed = ((r as u32) << 16) + ((g as u32) << 8) + (b as u32);
        features.push(packed as f32);
    }

    // For Edge Impulse image models, we expect one packed RGB value per pixel
    // The model metadata input_features_count is calculated as width * height * channels,
    // but the SDK expects packed RGB values (one feature per pixel)
    let expected_pixels = (params.image_input_width * params.image_input_height * params.image_input_frames) as usize;
    if features.len() != expected_pixels {
        eprintln!(
            "Warning: feature count is {} but expected {} pixels ({}x{}x{})",
            features.len(),
            expected_pixels,
            params.image_input_width,
            params.image_input_height,
            params.image_input_frames
        );
    } else {
        println!("✓ Feature extraction: {} packed RGB pixels", features.len());
    }

    if args.debug {
        let min = features.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = features.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f32 = features.iter().sum();
        let mean = sum / features.len() as f32;
        println!(
            "Feature statistics: min={:.3}, max={:.3}, mean={:.3}, count={}",
            min, max, mean, features.len()
        );
    }

    // Run inference using the runner API
    println!("Running inference...");
    let result = model.infer(features, Some(args.debug));

    match result {
        Ok(inference_response) => {
            println!("✓ Inference completed successfully!");

            // Process results based on the inference type
            match inference_response.result {
                InferenceResult::Classification { classification } => {
                    println!("Classification results:");
                    for (class, probability) in classification {
                        println!("  - {}: {:.2}%", class, probability * 100.0);
                    }
                }
                InferenceResult::ObjectDetection {
                    bounding_boxes,
                    classification,
                } => {
                    if !classification.is_empty() {
                        println!("Image classification:");
                        for (class, probability) in classification {
                            println!("  - {}: {:.2}%", class, probability * 100.0);
                        }
                    }
                    println!("Object detection results:");
                    if bounding_boxes.is_empty() {
                        println!("  No objects detected.");
                    } else {
                        for bbox in bounding_boxes {
                            println!(
                                "  - {} ({:.2}%) at ({},{},{},{})",
                                bbox.label,
                                bbox.value * 100.0,
                                bbox.x,
                                bbox.y,
                                bbox.width,
                                bbox.height
                            );
                        }
                    }
                }
                InferenceResult::VisualAnomaly {
                    visual_anomaly_grid,
                    visual_anomaly_max,
                    visual_anomaly_mean,
                    anomaly,
                } => {
                    println!("Visual anomaly detection results:");
                    println!("  - Overall anomaly: {:.2}%", anomaly * 100.0);
                    println!("  - Maximum anomaly: {:.2}%", visual_anomaly_max * 100.0);
                    println!("  - Mean anomaly: {:.2}%", visual_anomaly_mean * 100.0);

                    if !visual_anomaly_grid.is_empty() {
                        println!("  Anomaly regions:");
                        for bbox in visual_anomaly_grid {
                            println!(
                                "    - {} ({:.2}%) at ({},{},{},{})",
                                bbox.label,
                                bbox.value * 100.0,
                                bbox.x,
                                bbox.y,
                                bbox.width,
                                bbox.height
                            );
                        }
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("✗ Error running inference: {}", e);
            println!("This might be expected if:");
            println!("1. No model is loaded/initialized");
            println!("2. The signal format doesn't match what the model expects");
            println!("3. The model requires additional setup");
        }
    }

    println!("✓ Example completed successfully!");
    Ok(())
}
