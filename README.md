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

### Option 2: Copy Model from Custom Path
You can copy your Edge Impulse model from a custom directory path using the `EI_MODEL` environment variable:

```sh
export EI_MODEL=/path/to/your/edge-impulse-model
cargo build
```

This will:
1. Copy the model files from the specified path to the `model/` directory
2. Build the C++ library automatically
3. Generate the necessary Rust bindings

**Note**: The source directory should contain the standard Edge Impulse model structure (`edge-impulse-sdk/`, `model-parameters/`, `tflite-model/`, etc.).

### Option 3: Automated Model Download
You can automatically download and build your Edge Impulse model during the build process by setting environment variables:

```sh
export EI_PROJECT_ID=12345
export EI_API_KEY=ei_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
cargo build
```

This will:
1. Download the latest model from your Edge Impulse project
2. Build the C++ library automatically
3. Generate the necessary Rust bindings

**Note**: The download process may take several minutes on the first build.

**Security**: Never commit your API key to version control. Environment variables are the recommended approach for managing secrets.

#### Model Source Priority

The build system checks for models in the following order:

1. **Local model files** in the `model/` directory (if they already exist)
2. **Custom model path** specified by `EI_MODEL` environment variable
3. **Edge Impulse API download** using `EI_PROJECT_ID` and `EI_API_KEY`

This means you can:
- Use pre-existing model files (fastest)
- Copy from a custom path (useful for Docker builds, CI/CD)
- Download from Edge Impulse Studio (requires API credentials)

#### Engine Selection
By default, the model is built with the `tflite-eon` engine (optimized for microcontrollers). To use the standard `tflite` engine (compatible with full TensorFlow Lite), set the `EI_ENGINE` environment variable:

```sh
# Use standard TensorFlow Lite (compatible with full TFLite builds)
EI_ENGINE=tflite cargo build

# Use EON-optimized TensorFlow Lite (default, for microcontrollers)
EI_ENGINE=tflite-eon cargo build
# or simply
cargo build
```

### EI_MODEL Usage Examples

```sh
# Copy model from a mounted volume in Docker
EI_MODEL=/mnt/models/my-project cargo build

# Copy model from a relative path
EI_MODEL=../shared-models/project-123 cargo build

# Copy model and use full TensorFlow Lite
EI_MODEL=/opt/models/my-project USE_FULL_TFLITE=1 cargo build

# Copy model with platform-specific flags
EI_MODEL=/path/to/model TARGET_MAC_ARM64=1 USE_FULL_TFLITE=1 cargo build
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

## Cross-Compilation

This project supports cross-compilation to aarch64-unknown-linux-gnu using both local tools and Docker.

### Prerequisites

#### Local Cross-Compilation
To build locally, you need the aarch64 cross-compilation tools:

**Ubuntu/Debian:**
```sh
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

**macOS:**
```sh
brew install aarch64-linux-gnu-binutils
```

#### Docker Cross-Compilation
For Docker-based cross-compilation, you only need Docker and Docker Compose installed.

### Building for aarch64

#### Option 1: Using Make (Recommended)
```sh
# Build locally
make build-aarch64 EI_MODEL=~/Downloads/model-person-detection

# Build with Docker
make build-aarch64-docker EI_MODEL=~/Downloads/model-person-detection
```

#### Option 2: Using the Build Script Directly
```sh
# Build locally
./build-aarch64.sh

# Build with specific model
EI_MODEL=~/Downloads/model-person-detection ./build-aarch64.sh
```

#### Option 3: Using Docker Compose Directly
```sh
# Build with Docker Compose
EI_MODEL=~/Downloads/model-person-detection docker-compose up --build aarch64-build
```

#### Option 4: Manual Cargo Build
```sh
# Set environment variables
export TARGET_LINUX_AARCH64=1
export USE_FULL_TFLITE=1
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
export EI_MODEL=~/Downloads/model-person-detection

# Add target and build
rustup target add aarch64-unknown-linux-gnu
cargo build --target aarch64-unknown-linux-gnu --release
```

### Output Files

After successful cross-compilation, you'll find the built files in:
```
target/aarch64-unknown-linux-gnu/release/
```

The output includes:
- `libedge_impulse_ffi_rs.a` - Static library
- `libedge_impulse_ffi_rs.so` - Shared library
- `edge_impulse_ffi_rs.rlib` - Rust library

### Testing Cross-Compiled Binaries

You can test the cross-compiled binaries on an aarch64 system (like a Raspberry Pi 4, AWS Graviton, or ARM64 server):

```sh
# Copy the binary to your aarch64 system
scp target/aarch64-unknown-linux-gnu/release/examples/ffi_image_infer user@aarch64-host:/tmp/

# Run on the aarch64 system
ssh user@aarch64-host "cd /tmp && ./ffi_image_infer --image test.jpg"
```

### Troubleshooting Cross-Compilation

**Common Issues:**

1. **Missing cross-compiler:**
   ```
   Error: aarch64-linux-gnu-gcc not found
   ```
   Solution: Install the cross-compilation tools (see Prerequisites above)

2. **CMake configuration fails:**
   ```
   CMake configuration failed
   ```
   Solution: Ensure you have cmake installed and the cross-compilers are in your PATH

3. **Linking errors:**
   ```
   cannot find -lstdc++
   ```
   Solution: Install the ARM64 standard library: `sudo apt-get install libstdc++-11-dev:arm64`

4. **Model not found:**
   ```
   FFI crate requires a valid Edge Impulse model
   ```
   Solution: Ensure the EI_MODEL environment variable points to a valid model directory

**Docker-specific Issues:**

1. **Permission denied:**
   ```
   permission denied: unknown
   ```
   Solution: Ensure Docker has access to the model directory and current directory

2. **Volume mount issues:**
   ```
   cannot find model files
   ```
   Solution: Check that the EI_MODEL path is accessible from within the Docker container

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

### Local Development

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

### Testing in Docker (aarch64 Cross-Compilation)

For testing the aarch64 cross-compiled version in Docker, several scripts are provided:

#### Quick Test
```sh
# Run with default test image
./test-aarch64-example.sh

# Run with specific image and debug output
./test-aarch64-example.sh -i my_image.jpg -d

# Run with custom model path
./test-aarch64-example.sh -e EI_MODEL=/path/to/model -c
```

#### Interactive Testing
```sh
# Open interactive shell in Docker container
./docker-shell.sh

# With custom environment variables
./docker-shell.sh EI_MODEL=/path/to/model USE_FULL_TFLITE=1
```

#### Manual Testing
```sh
# Build only
./test-aarch64-example.sh -b

# Run only (if already built)
./test-aarch64-example.sh -r

# Clean build
./test-aarch64-example.sh -c
```

#### Example Commands in Docker Shell
```bash
# Build the example
cargo build --example ffi_image_infer --target aarch64-unknown-linux-gnu --release

# Run with test image
./target/aarch64-unknown-linux-gnu/release/examples/ffi_image_infer --image test_image.png --debug

# Check binary architecture
file ./target/aarch64-unknown-linux-gnu/release/examples/ffi_image_infer
```

## Using as a Dependency

When using this crate as a dependency in another Rust project, you must set the appropriate environment variables before building:

```toml
[dependencies]
edge-impulse-ffi-rs = { path = "../edge-impulse-ffi-rs" }
```

Then build with the required environment variables. You can use any of the model source options:

```sh
# Copy model from custom path
EI_MODEL=/path/to/model cargo build

# Download from Edge Impulse Studio
EI_PROJECT_ID=12345 EI_API_KEY=your-api-key cargo build

# For full TensorFlow Lite on Apple Silicon
TARGET_MAC_ARM64=1 USE_FULL_TFLITE=1 cargo build

# For full TensorFlow Lite on Linux x86_64
TARGET_LINUX_X86=1 USE_FULL_TFLITE=1 cargo build

# For TensorFlow Lite Micro (default)
cargo build
```

**Note**: The environment variables must be set in the shell session where you run `cargo build`. The build system will auto-detect your platform if no target variables are specified, but you must still set `USE_FULL_TFLITE=1` if you want full TensorFlow Lite instead of the micro version.

## Cleaning the Model Folder

To clean the `model/` folder and remove all generated files (keeping only `README.md` and `.gitignore`), use the provided script:

```sh
sh clean-model.sh
```

This is useful when you want to:
- Remove build artifacts and generated files
- Prepare the folder for a new model
- Clean up after development/testing

## How the Build System Works

The `build.rs` script automates the entire build process:

### Model Acquisition
The build system supports three ways to get model files:

#### Option 1: Local Files (if configured)
If model files already exist in the `model/` directory, they are used directly.

#### Option 2: Custom Path Copy (if configured)
If `EI_MODEL` environment variable is set:
1. Validates the source path exists
2. Copies `edge-impulse-sdk/`, `model-parameters/`, `tflite-model/` directories
3. Optionally copies `tensorflow-lite/` directory (for full TFLite builds)
4. Preserves existing `model/.gitignore` and `model/README.md` files

#### Option 3: API Download (if configured)
If `EI_PROJECT_ID` and `EI_API_KEY` environment variables are set:
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
- Ensure environment variables are set correctly: `EI_PROJECT_ID` and `EI_API_KEY`

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
3. Unset the environment variables: `unset EI_PROJECT_ID EI_API_KEY`
4. Build normally with `cargo build`

## Additional Notes
- You can swap in a new Edge Impulse model by replacing the contents of `model/` and rebuilding.
- For more details, see the comments in `build.rs` and the example code in `examples/`.
