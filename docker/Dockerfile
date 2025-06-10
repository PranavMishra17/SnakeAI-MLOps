# Multi-stage build for SnakeAI-MLOps
FROM ubuntu:22.04 AS build

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    curl \
    zip \
    unzip \
    tar \
    ninja-build \
    # SFML dependencies
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libgl1-mesa-dev \
    libudev-dev \
    libfreetype6-dev \
    libopenal-dev \
    libflac-dev \
    libvorbis-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy vcpkg configuration
COPY vcpkg.json .

# Install vcpkg
RUN git clone https://github.com/Microsoft/vcpkg.git && \
    ./vcpkg/bootstrap-vcpkg.sh && \
    ./vcpkg/vcpkg install

# Copy source code
COPY src/ src/
COPY CMakeLists.txt .
COPY CMakePresets.json .

# Create assets directory (placeholder)
RUN mkdir -p assets/fonts && \
    echo "Font placeholder for Docker" > assets/fonts/arial.ttf

# Build the project
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/app/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -G Ninja

RUN cmake --build build --config Release

# Runtime stage
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libgl1-mesa-glx \
    libudev1 \
    libfreetype6 \
    libopenal1 \
    libflac8 \
    libvorbis0a \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -s /bin/bash snakeai

# Set working directory
WORKDIR /app

# Copy built executable and assets
COPY --from=build /app/build/SnakeAI-MLOps .
COPY --from=build /app/assets/ assets/

# Create necessary directories
RUN mkdir -p models data logs && \
    chown -R snakeai:snakeai /app

# Switch to app user
USER snakeai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD test -f /app/SnakeAI-MLOps || exit 1

# Default command
CMD ["echo", "SnakeAI-MLOps container ready. Note: This is a GUI application and requires display forwarding for interactive use."]

# Labels for metadata
LABEL org.opencontainers.image.title="SnakeAI-MLOps"
LABEL org.opencontainers.image.description="Reinforcement Learning Snake Game with MLOps Pipeline"
LABEL org.opencontainers.image.version="1.0"
LABEL org.opencontainers.image.vendor="SnakeAI-MLOps Project"