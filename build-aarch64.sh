#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Edge Impulse FFI Rust aarch64 Cross-Compilation ===${NC}"

# Check if we're running in Docker
if [ -f /.dockerenv ]; then
    echo -e "${YELLOW}Running inside Docker container${NC}"
    IN_DOCKER=true
else
    echo -e "${YELLOW}Running on host system${NC}"
    IN_DOCKER=false
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! command_exists cargo; then
    echo -e "${RED}Error: cargo not found. Please install Rust.${NC}"
    exit 1
fi

if ! command_exists aarch64-linux-gnu-gcc; then
    echo -e "${RED}Error: aarch64-linux-gnu-gcc not found.${NC}"
    if [ "$IN_DOCKER" = false ]; then
        echo -e "${YELLOW}On Ubuntu/Debian, install with: sudo apt-get install gcc-aarch64-linux-gnu${NC}"
        echo -e "${YELLOW}On macOS, install with: brew install aarch64-linux-gnu-binutils${NC}"
    fi
    exit 1
fi

if ! command_exists aarch64-linux-gnu-g++; then
    echo -e "${RED}Error: aarch64-linux-gnu-g++ not found.${NC}"
    if [ "$IN_DOCKER" = false ]; then
        echo -e "${YELLOW}On Ubuntu/Debian, install with: sudo apt-get install g++-aarch64-linux-gnu${NC}"
    fi
    exit 1
fi

if ! command_exists cmake; then
    echo -e "${RED}Error: cmake not found.${NC}"
    exit 1
fi

echo -e "${GREEN}Prerequisites check passed!${NC}"

# Add aarch64 target if not already added
echo -e "${BLUE}Adding aarch64-unknown-linux-gnu target...${NC}"
rustup target add aarch64-unknown-linux-gnu

# Set environment variables for cross-compilation
export TARGET_LINUX_AARCH64=1
export USE_FULL_TFLITE=1
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++

# Check if EI_MODEL is provided
if [ -n "$EI_MODEL" ]; then
    echo -e "${BLUE}Using model from: $EI_MODEL${NC}"
    export EI_MODEL
fi

# Check if EI_ENGINE is provided
if [ -n "$EI_ENGINE" ]; then
    echo -e "${BLUE}Using engine: $EI_ENGINE${NC}"
    export EI_ENGINE
fi

# Build for aarch64
echo -e "${BLUE}Building for aarch64-unknown-linux-gnu...${NC}"
echo -e "${YELLOW}Environment variables:${NC}"
echo -e "  TARGET_LINUX_AARCH64=$TARGET_LINUX_AARCH64"
echo -e "  USE_FULL_TFLITE=$USE_FULL_TFLITE"
echo -e "  CC_aarch64_unknown_linux_gnu=$CC_aarch64_unknown_linux_gnu"
echo -e "  CXX_aarch64_unknown_linux_gnu=$CXX_aarch64_unknown_linux_gnu"
if [ -n "$EI_MODEL" ]; then
    echo -e "  EI_MODEL=$EI_MODEL"
fi
if [ -n "$EI_ENGINE" ]; then
    echo -e "  EI_ENGINE=$EI_ENGINE"
fi

# Clean previous build if requested
if [ "$1" = "--clean" ]; then
    echo -e "${YELLOW}Cleaning previous build...${NC}"
    cargo clean --target aarch64-unknown-linux-gnu
fi

# Build
echo -e "${BLUE}Starting build...${NC}"
cargo build --target aarch64-unknown-linux-gnu --release

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo -e "${BLUE}Output files:${NC}"
    ls -la target/aarch64-unknown-linux-gnu/release/

    # Show file types
    echo -e "${BLUE}File types:${NC}"
    file target/aarch64-unknown-linux-gnu/release/*

    echo -e "${GREEN}Cross-compilation completed!${NC}"
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi