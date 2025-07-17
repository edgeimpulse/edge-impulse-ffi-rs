#!/bin/bash

# Comprehensive script to test the ffi_image_infer example in aarch64 Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -i, --image PATH        Use specific image file (default: creates test image)"
    echo "  -d, --debug             Enable debug output"
    echo "  -c, --clean             Clean build before testing"
    echo "  -e, --env VAR=VALUE     Set environment variable"
    echo "  -b, --build-only        Only build, don't run"
    echo "  -r, --run-only          Only run existing binary"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run with default test image"
    echo "  $0 -i my_image.jpg -d                 # Run with specific image and debug"
    echo "  $0 -e EI_MODEL=/path/to/model -c      # Use custom model and clean build"
    echo "  $0 -b                                  # Only build the example"
}

# Parse command line arguments
IMAGE_PATH=""
DEBUG_FLAG=""
CLEAN_BUILD=false
BUILD_ONLY=false
RUN_ONLY=false
ENV_VARS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -i|--image)
            IMAGE_PATH="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG_FLAG="--debug"
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -b|--build-only)
            BUILD_ONLY=true
            shift
            ;;
        -r|--run-only)
            RUN_ONLY=true
            shift
            ;;
        -e|--env)
            ENV_VARS+=("$2")
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed or not in PATH"
    exit 1
fi

print_status "=== Testing ffi_image_infer example in aarch64 Docker container ==="

# Create test image if not provided
if [ -z "$IMAGE_PATH" ]; then
    print_status "Creating test image..."
    echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==" | base64 -d > test_image.png
    IMAGE_PATH="test_image.png"
    print_success "Created test image: $IMAGE_PATH"
else
    if [ ! -f "$IMAGE_PATH" ]; then
        print_error "Image file not found: $IMAGE_PATH"
        exit 1
    fi
    print_success "Using image: $IMAGE_PATH"
fi

# Build environment variables string
ENV_STRING=""
for env_var in "${ENV_VARS[@]}"; do
    ENV_STRING="$ENV_STRING -e $env_var"
done

# Docker command prefix
DOCKER_CMD="docker-compose run --rm $ENV_STRING aarch64-build"

if [ "$CLEAN_BUILD" = true ]; then
    print_status "Cleaning previous build..."
    $DOCKER_CMD bash -c "cargo clean"
fi

if [ "$RUN_ONLY" = false ]; then
    print_status "Building ffi_image_infer example..."
    $DOCKER_CMD bash -c "
        echo 'Building example with target aarch64-unknown-linux-gnu...'
        cargo build --example ffi_image_infer --target aarch64-unknown-linux-gnu --release

        if [ \$? -eq 0 ]; then
            echo 'Build successful!'
            ls -la target/aarch64-unknown-linux-gnu/release/examples/ffi_image_infer
        else
            echo 'Build failed!'
            exit 1
        fi
    "

    if [ $? -ne 0 ]; then
        print_error "Build failed!"
        exit 1
    fi

    print_success "Example built successfully!"
fi

if [ "$BUILD_ONLY" = false ]; then
    print_status "Running ffi_image_infer example..."
    $DOCKER_CMD bash -c "
        echo 'Running example with image: $IMAGE_PATH'
        echo 'Debug flag: $DEBUG_FLAG'

        # Check if binary exists
        if [ ! -f './target/aarch64-unknown-linux-gnu/release/examples/ffi_image_infer' ]; then
            echo 'Binary not found!'
            exit 1
        fi

        # Run the example
        ./target/aarch64-unknown-linux-gnu/release/examples/ffi_image_infer --image $IMAGE_PATH $DEBUG_FLAG

        echo 'Example execution completed!'
    "

    if [ $? -eq 0 ]; then
        print_success "Example ran successfully!"
    else
        print_error "Example execution failed!"
        exit 1
    fi
fi

print_success "=== Test completed successfully ==="