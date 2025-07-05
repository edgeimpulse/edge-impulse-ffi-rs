//! Image Classification Example using Edge Impulse FFI
//!
//! Usage:
//!   cargo run --example image_infer -- --image <path_to_image> [--debug]

use clap::Parser;
use image::{self, GenericImageView};
use std::error::Error;
use edge_impulse_ffi_rs::{EdgeImpulseClassifier, Signal, ModelMetadata};
use edge_impulse_ffi_rs::model_metadata;
use image::{DynamicImage, imageops::FilterType, RgbImage};

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
        x if x == model_metadata::EI_CLASSIFIER_RESIZE_SQUASH => {
            img.resize_exact(input_width, input_height, FilterType::Triangle).to_rgb8()
        }
        x if x == model_metadata::EI_CLASSIFIER_RESIZE_FIT_SHORTEST => {
            let factor = (input_width as f32 / w as f32).min(input_height as f32 / h as f32);
            let resize_w = (w as f32 * factor).round() as u32;
            let resize_h = (h as f32 * factor).round() as u32;
            let resized = img.resize_exact(resize_w, resize_h, FilterType::Triangle);
            let crop_x = if resize_w > input_width { (resize_w - input_width) / 2 } else { 0 };
            let crop_y = if resize_h > input_height { (resize_h - input_height) / 2 } else { 0 };
            image::DynamicImage::ImageRgba8(image::imageops::crop_imm(&resized, crop_x, crop_y, input_width, input_height).to_image()).to_rgb8()
        }
        x if x == model_metadata::EI_CLASSIFIER_RESIZE_FIT_LONGEST => {
            let factor = (input_width as f32 / w as f32).max(input_height as f32 / h as f32);
            let resize_w = (w as f32 * factor).round() as u32;
            let resize_h = (h as f32 * factor).round() as u32;
            let resized = img.resize_exact(resize_w, resize_h, FilterType::Triangle);
            // Pad to center
            let mut out = RgbImage::new(input_width, input_height);
            let x_offset = if input_width > resize_w { (input_width - resize_w) / 2 } else { 0 };
            let y_offset = if input_height > resize_h { (input_height - resize_h) / 2 } else { 0 };
            image::imageops::replace(&mut out, &resized.to_rgb8(), x_offset as i64, y_offset as i64);
            out
        }
        _ => {
            // Default to squash if unknown
            img.resize_exact(input_width, input_height, FilterType::Triangle).to_rgb8()
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // Get and print all model metadata at once
    let meta = ModelMetadata::get();
    println!("{}", meta);

    // Load and process the image
    let img = image::open(&args.image)?;
    let (width, height) = img.dimensions();
    println!("Loaded image: {} ({}x{})", args.image, width, height);

    // Resize and crop using model metadata
    let (iw, ih) = (meta.input_width as u32, meta.input_height as u32);
    let rgb = resize_and_crop(
        &img,
        iw,
        ih,
        model_metadata::EI_CLASSIFIER_RESIZE_MODE,
    );
    println!("Processed image to {}x{} RGB", iw, ih);

    // Pack each pixel as (r << 16) + (g << 8) + b, as f32
    let mut features = Vec::with_capacity((iw * ih) as usize);
    for pixel in rgb.pixels() {
        let [r, g, b] = pixel.0;
        let packed = ((r as u32) << 16) + ((g as u32) << 8) + (b as u32);
        features.push(packed as f32);
    }
    if features.len() != meta.input_width * meta.input_height {
        eprintln!(
            "Warning: feature count is {} but expected {} ({}x{})",
            features.len(),
            meta.input_width * meta.input_height,
            meta.input_width,
            meta.input_height
        );
    }
    if args.debug {
        let min = features.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = features.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f32 = features.iter().sum();
        let mean = sum / features.len() as f32;
        println!("Feature statistics: min={:.3}, max={:.3}, mean={:.3}, count={}", min, max, mean, features.len());
    }

    // Create a signal from the features
    let signal = Signal::from_raw_data(&features)?;

    // Initialize the classifier
    let mut classifier = EdgeImpulseClassifier::new();
    classifier.init()?;

    // Run inference with the real signal
    let result = classifier.run_classifier(&signal, args.debug);

    match result {
        Ok(inference) => {
            println!("Inference ran successfully!");
            // Print classification results if present
            if !meta.has_object_detection && meta.label_count > 0 {
                let results = inference.classifications(meta.label_count);
                if results.is_empty() {
                    println!("No classification results.");
                } else {
                    println!("Classification results:");
                    for c in results {
                        println!("  {}", c);
                    }
                }
            }
            // Print bounding boxes for object detection
            if meta.has_object_detection {
                let bbs = inference.bounding_boxes();
                if bbs.is_empty() {
                    println!("No bounding boxes found.");
                } else {
                    println!("Object detection results:");
                    for bb in bbs {
                        println!("  {}", bb);
                    }
                }
            }
            // Print timing info
            let timing = inference.timing();
            println!("{}", timing);
        }
        Err(e) => {
            // Print both the error and its integer value
            let code = e as i32;
            eprintln!("Error running inference: {} (error code: {:?}, value: {})", e, e, code);
            println!("Raw error code from C++: {}", code);
            println!("This might be expected if:");
            println!("1. No model is loaded/initialized");
            println!("2. The signal format doesn't match what the model expects");
            println!("3. The model requires additional setup");
        }
    }

    classifier.deinit()?;
    Ok(())
}