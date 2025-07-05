# Edge Impulse Rust FFI Example

This project demonstrates how to run Edge Impulse machine learning models from Rust using a safe FFI (Foreign Function Interface) layer over the Edge Impulse C++ SDK.

## Features
- **Model-agnostic:** Easily swap in any Edge Impulse exported C++ model by replacing the contents of the `model/` folder.
- **Automatic FFI glue:** C/C++ glue code in `ffi_glue/` is automatically copied and built alongside your model code.
- **Rust bindings:** Rust code calls into the C++ SDK using generated bindings, with ergonomic Rust APIs for model metadata, inference, and result parsing.
- **Examples:** See `examples/image_infer.rs` for a complete image classification and object detection example.

## How it Works
1. Place your Edge Impulse exported C++ model (including `edge-impulse-sdk/`, `model-parameters/`, etc.) in the `model/` directory.
2. The FFI glue code in `ffi_glue/` is copied into `model/` at build time.
3. The `build.rs` script builds the C++ static library, generates Rust bindings, and extracts model metadata for ergonomic use in Rust.
4. The Rust API provides safe wrappers for model initialization, signal creation, inference, and result parsing.

## Building and Running

```sh
cargo build
cargo run --example image_infer -- --image <path_to_image>
```

- The build process is fully automated via `build.rs`.
- You can swap in a new Edge Impulse model by replacing the contents of `model/` and rebuilding.

## FFI Glue Code
- The `ffi_glue/` folder contains C/C++ wrappers and CMake logic to expose the Edge Impulse C++ API to Rust.
- These files are copied into `model/` at build time so you never lose your FFI logic when updating the model.

## Model Metadata
- Model constants (input size, label count, etc.) are extracted from the model's generated headers and made available in Rust for ergonomic use.

## Example: Image Inference
- See `examples/image_infer.rs` for a complete example of loading an image, preprocessing, running inference, and printing results for both classification and object detection models.

---

## Build Process (build.rs)

The `build.rs` script automates:
- Copying FFI glue code from `ffi_glue/` to `model/`.
- Building the C++ static library (`libedge-impulse-sdk.a`) in `model/` using CMake.
- Generating Rust FFI bindings for the C++ API headers with `bindgen` (output to `src/bindings.rs`).
- Extracting model metadata from `model_metadata.h` and writing Rust constants to `src/model_metadata.rs`.
- Printing build progress and diagnostics to help debug integration issues.

This ensures the Rust code always has up-to-date bindings and metadata for the current model, and that the C++ code is rebuilt as needed.

---

For more details, see the comments in `build.rs` and the example code in `examples/`.
