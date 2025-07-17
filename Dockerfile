FROM rust:latest

# Install cross-compilation tools and dependencies
RUN dpkg --add-architecture arm64 && \
    apt-get update && \
    apt-get install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    build-essential \
    pkg-config \
    libclang-dev \
    clang \
    cmake \
    make \
    libstdc++-11-dev:arm64 \
    libc6-dev:arm64 \
    curl \
    wget \
    unzip \
    git

WORKDIR /app

# Install rustfmt for aarch64 target
RUN rustup target add aarch64-unknown-linux-gnu && \
    rustup component add rustfmt

# Copy the project files
COPY . .

# Set up environment for cross-compilation
ENV TARGET_LINUX_AARCH64=1
ENV USE_FULL_TFLITE=1
ENV CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
ENV CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++

# Create a build script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Building for aarch64-unknown-linux-gnu..."\n\
\n\
# Set cross-compilation environment variables\n\
export TARGET_LINUX_AARCH64=1\n\
export USE_FULL_TFLITE=1\n\
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc\n\
export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++\n\
\n\
# Build for aarch64\n\
cargo build --target aarch64-unknown-linux-gnu --release\n\
\n\
echo "Build completed successfully!"\n\
echo "Output files:"\n\
ls -la target/aarch64-unknown-linux-gnu/release/\n\
' > /usr/local/bin/build-aarch64.sh && chmod +x /usr/local/bin/build-aarch64.sh

# Default command
CMD ["/usr/local/bin/build-aarch64.sh"]