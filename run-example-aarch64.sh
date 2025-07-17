#!/bin/bash

# Script to build and run the ffi_image_infer example in aarch64 Docker container

set -e

echo "=== Building and running ffi_image_infer example in aarch64 Docker container ==="

# Check if we have a test image
if [ ! -f "test_image.png" ]; then
    echo "Creating test image..."
    echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==" | base64 -d > test_image.png
fi

# Build the example in Docker
echo "Building example in Docker container..."
docker-compose run --rm aarch64-build bash -c "
    echo 'Building ffi_image_infer example...'
    cargo build --example ffi_image_infer --target aarch64-unknown-linux-gnu --release

    echo 'Checking if example was built successfully...'
    ls -la target/aarch64-unknown-linux-gnu/release/examples/ffi_image_infer

    echo 'Running example with test image...'
    ./target/aarch64-unknown-linux-gnu/release/examples/ffi_image_infer --image test_image.png --debug

    echo 'Example completed successfully!'
"

echo "=== Example run completed ==="