# Edge Impulse Rust FFI Example

This project lets you run Edge Impulse machine learning models from Rust using a safe FFI (Foreign Function Interface) layer over the Edge Impulse C++ SDK.

## Features
- **Model-agnostic:** Easily swap in any Edge Impulse exported C++ model by replacing the contents of the `model/` folder.
- **Automatic FFI glue:** C/C++ glue code in `ffi_glue/` is automatically copied and built alongside your model code.
- **Rust bindings:** Rust code calls into the C++ SDK using generated bindings, with ergonomic Rust APIs for model metadata, inference, and result parsing.
- **Examples:** See `examples/ffi_image_infer.rs` for a complete image classification and object detection example.

## Quick Start

### Option 1: Manual Model Setup
1. Place your Edge Impulse exported C++ model (including `edge-impulse-sdk/`, `model-parameters/`, etc.) in the `model/` directory.
2. Build and run the example:
   ```sh
   cargo build
   cargo run --example ffi_image_infer -- --image <path_to_image>
   ```

### Option 2: Automated Model Download
You can automatically download and build your Edge Impulse model during the build process by adding metadata to your `Cargo.toml`:

```toml
[package.metadata.edge-impulse]
project_id = 12345
api_key = "ei_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

This will:
1. Download the latest model from your Edge Impulse project
2. Build the C++ library automatically
3. Generate the necessary Rust bindings

**Note**: The download process may take several minutes on the first build.

**Security**: Never commit your API key to version control.

## Building

This project supports both TensorFlow Lite Micro and full TensorFlow Lite builds.

**⚠️ Important**: You must explicitly specify which TensorFlow Lite version to use. There is no automatic detection.

### Build Modes

- **Default (TensorFlow Lite Micro - for microcontrollers):**
  ```sh
  cargo build
  ```
- **Full TensorFlow Lite (for desktop/server):**
  ```sh
  USE_FULL_TFLITE=1 cargo build
  ```

### Platform-Specific Builds

You can specify the target platform explicitly using these environment variables:

#### macOS
```sh
# Apple Silicon (M1/M2/M3)
TARGET_MAC_ARM64=1 USE_FULL_TFLITE=1 cargo build
# Intel Mac
TARGET_MAC_X86_64=1 USE_FULL_TFLITE=1 cargo build
```
#### Linux
```sh
# Linux x86_64
TARGET_LINUX_X86=1 USE_FULL_TFLITE=1 cargo build
# Linux ARM64
TARGET_LINUX_AARCH64=1 USE_FULL_TFLITE=1 cargo build
# Linux ARMv7
TARGET_LINUX_ARMV7=1 USE_FULL_TFLITE=1 cargo build
```
#### NVIDIA Jetson
```sh
# Jetson Nano
TARGET_JETSON_NANO=1 USE_FULL_TFLITE=1 cargo build
# Jetson Orin
TARGET_JETSON_ORIN=1 USE_FULL_TFLITE=1 cargo build
```

### Platform Support

Full TensorFlow Lite uses prebuilt binaries from the `tflite/` directory:

| Platform           | Directory                |
|--------------------|-------------------------|
| macOS ARM64        | tflite/mac-arm64/       |
| macOS x86_64       | tflite/mac-x86_64/      |
| Linux x86          | tflite/linux-x86/       |
| Linux ARM64        | tflite/linux-aarch64/   |
| Linux ARMv7        | tflite/linux-armv7/     |
| Jetson Nano        | tflite/linux-jetson-nano/|

**Note:** If no platform is specified, the build system will auto-detect based on your current system architecture.

---

## Advanced Build Flags

This project supports a wide range of advanced build flags for hardware accelerators, backends, and cross-compilation, mirroring the Makefile from Edge Impulse's [example-standalone-inferencing-linux](https://github.com/edgeimpulse/example-standalone-inferencing-linux). You can combine these flags as needed:

| Flag                          | Purpose / Effect                                                                                 |
|-------------------------------|-----------------------------------------------------------------------------------------------|
| `USE_TVM=1`                   | Enable Apache TVM backend (requires `TVM_HOME` env var)                                         |
| `USE_ONNX=1`                  | Enable ONNX Runtime backend                                                                    |
| `USE_QUALCOMM_QNN=1`          | Enable Qualcomm QNN delegate (requires `QNN_SDK_ROOT` env var)                                 |
| `USE_ETHOS=1`                 | Enable ARM Ethos-U delegate                                                                   |
| `USE_AKIDA=1`                 | Enable BrainChip Akida backend                                                                |
| `USE_MEMRYX=1`                | Enable MemryX backend                                                                         |
| `LINK_TFLITE_FLEX_LIBRARY=1`  | Link TensorFlow Lite Flex library                                                             |
| `EI_CLASSIFIER_USE_MEMRYX_SOFTWARE=1` | Use MemryX software mode (with Python bindings)                                    |
| `TENSORRT_VERSION=8.5.2`      | Set TensorRT version for Jetson platforms                                                     |
| `TVM_HOME=/path/to/tvm`       | Path to TVM installation (required for `USE_TVM=1`)                                           |
| `QNN_SDK_ROOT=/path/to/qnn`   | Path to Qualcomm QNN SDK (required for `USE_QUALCOMM_QNN=1`)                                  |
| `PYTHON_CROSS_PATH=...`       | Path prefix for cross-compiling Python bindings                                               |

### Example Advanced Builds

```sh
# Build with ONNX Runtime and full TensorFlow Lite for TI AM68A
USE_ONNX=1 TARGET_AM68A=1 USE_FULL_TFLITE=1 cargo build

# Build with TVM backend (requires TVM_HOME)
USE_TVM=1 TVM_HOME=/opt/tvm TARGET_RENESAS_RZV2L=1 USE_FULL_TFLITE=1 cargo build

# Build with Qualcomm QNN delegate (requires QNN_SDK_ROOT)
USE_QUALCOMM_QNN=1 QNN_SDK_ROOT=/opt/qnn TARGET_LINUX_AARCH64=1 USE_FULL_TFLITE=1 cargo build

# Build with ARM Ethos-U delegate
USE_ETHOS=1 TARGET_LINUX_AARCH64=1 USE_FULL_TFLITE=1 cargo build

# Build with MemryX backend in software mode
USE_MEMRYX=1 EI_CLASSIFIER_USE_MEMRYX_SOFTWARE=1 TARGET_LINUX_X86=1 USE_FULL_TFLITE=1 cargo build

# Build with TensorFlow Lite Flex library
LINK_TFLITE_FLEX_LIBRARY=1 USE_FULL_TFLITE=1 cargo build

# Build for Jetson Nano with specific TensorRT version
TARGET_JETSON_NANO=1 TENSORRT_VERSION=8.5.2 USE_FULL_TFLITE=1 cargo build
```

See the Makefile in Edge Impulse's [example-standalone-inferencing-linux](https://github.com/edgeimpulse/example-standalone-inferencing-linux) for more details on what each flag does. Not all combinations are valid for all models/platforms.

---

## Running Examples

```sh
# Build and run with TensorFlow Lite Micro
cargo build
cargo run --example ffi_image_infer -- --image <path_to_image>

# Build and run with full TensorFlow Lite
USE_FULL_TFLITE=1 cargo build
cargo run --example ffi_image_infer -- --image <path_to_image>

# Build and run with platform-specific flags (Apple Silicon example)
TARGET_MAC_ARM64=1 USE_FULL_TFLITE=1 cargo build
cargo run --example ffi_image_infer -- --image <path_to_image>
```

**Note**: Once built, you can run the binary directly without the environment variable:
```sh
./target/debug/examples/ffi_image_infer --image <path_to_image>
```

## Cleaning the Model Folder

To clean the `model/` folder and remove all generated files (keeping only `README.md` and `.gitignore`):

```sh
CLEAN_MODEL=1 cargo build && cargo clean
```

This is useful when you want to:
- Remove build artifacts and generated files
- Prepare the folder for a new model
- Clean up after development/testing

## How the Build System Works

The `build.rs` script automates the entire build process:

### Model Download (if configured)
If `[package.metadata.edge-impulse]` is configured in `Cargo.toml`:
1. Fetches project information from Edge Impulse REST API
2. Triggers a build job for the latest model
3. Polls job status until completion
4. Downloads the model ZIP file
5. Extracts to the `model/` directory
6. Preserves existing `model/.gitignore` and `model/README.md` files

### Model Processing
- The `ffi_glue/` folder contains C/C++ wrappers and CMake logic to expose the Edge Impulse C++ API to Rust. These files are copied into `model/` at build time so you never lose your FFI logic when updating the model.
- Model constants (input size, label count, etc.) are extracted from the model's generated headers and made available in Rust for ergonomic use.

### Build Automation
The build script handles:
- Copying FFI glue code from `ffi_glue/` to `model/`.
- Detecting and processing build flags (platform targets, TensorFlow Lite mode, hardware accelerators, etc.).
- Building the C++ static library (`libedge-impulse-sdk.a`) in `model/` using CMake with appropriate flags.
- Generating Rust FFI bindings for the C++ API headers with `bindgen` (output to `src/bindings.rs`).
- Extracting model metadata from `model_metadata.h` and writing Rust constants to `src/model_metadata.rs`.
- Printing build progress and diagnostics to help debug integration issues.

This ensures the Rust code always has up-to-date bindings and metadata for the current model, and that the C++ code is rebuilt as needed with the correct configuration for your target platform and hardware.

## Example: Image Inference

See `examples/ffi_image_infer.rs` for a complete example of loading an image, preprocessing, running inference, and printing results for both classification and object detection models.

## Troubleshooting Automated Downloads

### Common Issues

**Download fails with authentication error:**
- Verify your API key is correct and has access to the project
- Check that the project ID exists and is accessible

**Download times out:**
- The download process can take several minutes for large models
- Check your internet connection

**Build job fails:**
- Ensure your Edge Impulse project has a valid model deployed
- Check the Edge Impulse Studio for any build errors
- Verify the project has the correct deployment target (Linux)

### Manual Override

If automated download fails, you can:
1. Manually download the model from Edge Impulse Studio
2. Extract it to the `model/` directory
3. Remove the `[package.metadata.edge-impulse]` section from `Cargo.toml`
4. Build normally with `cargo build`

## Additional Notes
- You can swap in a new Edge Impulse model by replacing the contents of `model/` and rebuilding.
- For more details, see the comments in `build.rs` and the example code in `examples/`.
