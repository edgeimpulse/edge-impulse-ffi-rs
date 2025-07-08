# Edge Impulse Rust FFI Example

This project lets you run Edge Impulse machine learning models from Rust using a safe FFI (Foreign Function Interface) layer over the Edge Impulse C++ SDK.

## Features
- **Model-agnostic:** Easily swap in any Edge Impulse exported C++ model by replacing the contents of the `model/` folder.
- **Automatic FFI glue:** C/C++ glue code in `ffi_glue/` is automatically copied and built alongside your model code.
- **Rust bindings:** Rust code calls into the C++ SDK using generated bindings, with ergonomic Rust APIs for model metadata, inference, and result parsing.
- **Examples:** See `examples/image_infer.rs` for a complete image classification and object detection example.

## Quick Start
1. Place your Edge Impulse exported C++ model (including `edge-impulse-sdk/`, `model-parameters/`, etc.) in the `model/` directory.
2. Build and run the example:
   ```sh
   cargo build
   cargo run --example image_infer -- --image <path_to_image>
   ```

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

This project supports a wide range of advanced build flags for hardware accelerators, backends, and cross-compilation, mirroring the Makefile from Edge Impulse's example-standalone-inferencing-linux. You can combine these flags as needed:

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

See the Makefile in Edge Impulse's example-standalone-inferencing-linux for more details on what each flag does. Not all combinations are valid for all models/platforms.

---

## Running Examples

```sh
# Build and run with TensorFlow Lite Micro
cargo build
cargo run --example image_infer -- --image <path_to_image>

# Build and run with full TensorFlow Lite
USE_FULL_TFLITE=1 cargo build
cargo run --example image_infer -- --image <path_to_image>

# Build and run with platform-specific flags (Apple Silicon example)
TARGET_MAC_ARM64=1 USE_FULL_TFLITE=1 cargo build
cargo run --example image_infer -- --image <path_to_image>
```

**Note**: Once built, you can run the binary directly without the environment variable:
```sh
./target/debug/examples/image_infer --image <path_to_image>
```

## Cleaning the Model Folder

To clean the `model/` folder and remove all generated files (keeping only `README.md` and `.gitignore`):

```sh
CLEAN_MODEL=1 cargo build
```

This is useful when you want to:
- Remove build artifacts and generated files
- Prepare the folder for a new model
- Clean up after development/testing

## How the Build System Works

- The `ffi_glue/` folder contains C/C++ wrappers and CMake logic to expose the Edge Impulse C++ API to Rust. These files are copied into `model/` at build time so you never lose your FFI logic when updating the model.
- Model constants (input size, label count, etc.) are extracted from the model's generated headers and made available in Rust for ergonomic use.
- The `build.rs` script automates:
  - Copying FFI glue code from `ffi_glue/` to `model/`.
  - Building the C++ static library (`libedge-impulse-sdk.a`) in `model/` using CMake.
  - Generating Rust FFI bindings for the C++ API headers with `bindgen` (output to `src/bindings.rs`).
  - Extracting model metadata from `model_metadata.h` and writing Rust constants to `src/model_metadata.rs`.
  - Printing build progress and diagnostics to help debug integration issues.

This ensures the Rust code always has up-to-date bindings and metadata for the current model, and that the C++ code is rebuilt as needed.

## Example: Image Inference

See `examples/image_infer.rs` for a complete example of loading an image, preprocessing, running inference, and printing results for both classification and object detection models.

## Additional Notes
- You can swap in a new Edge Impulse model by replacing the contents of `model/` and rebuilding.
- For more details, see the comments in `build.rs` and the example code in `examples/`.
