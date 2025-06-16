# Multi-stage build for SnakeAI-MLOps
FROM ubuntu:22.04 AS build

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

WORKDIR /app

# Copy configuration files
COPY vcpkg.json .
COPY CMakeLists.txt .
COPY CMakePresets.json .

# Install vcpkg and dependencies
RUN git clone https://github.com/Microsoft/vcpkg.git && \
    ./vcpkg/bootstrap-vcpkg.sh && \
    ./vcpkg/vcpkg install

# Copy source code
COPY src/ src/

# Create required directories and assets
RUN mkdir -p assets/fonts models data logs && \
    echo "Font placeholder for Docker" > assets/fonts/arial.ttf

# Build the project
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/app/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -G Ninja && \
    cmake --build build --config Release

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

WORKDIR /app

# Copy built executable and required files
COPY --from=build /app/build/SnakeAI-MLOps .
COPY --from=build /app/assets/ assets/

# Create necessary directories with proper permissions
RUN mkdir -p models data logs && \
    chown -R snakeai:snakeai /app

# Copy pre-trained models if they exist (optional)
RUN mkdir -p models
COPY models* models/

USER snakeai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD test -f /app/SnakeAI-MLOps || exit 1

# Default command
CMD ["echo", "SnakeAI-MLOps container ready. Use with X11 forwarding: docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix snakeai-mlops"]

# Metadata labels
LABEL org.opencontainers.image.title="SnakeAI-MLOps"
LABEL org.opencontainers.image.description="Reinforcement Learning Snake Game with MLOps Pipeline"
LABEL org.opencontainers.image.version="2.0"
LABEL org.opencontainers.image.vendor="SnakeAI-MLOps Project"