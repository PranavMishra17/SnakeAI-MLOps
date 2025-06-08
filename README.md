# SnakeAI-MLOps

Reinforcement Learning Snake Game with Production MLOps Pipeline

[![CI/CD Pipeline](https://github.com/PranavMishra17/SnakeAI-MLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/PranavMishra17/SnakeAI-MLOps/actions/workflows/ci.yml) ✅ **PASSING**

## Quick Start

### Prerequisites
- Visual Studio 2022 Community
- Git
- Docker Desktop
- vcpkg (installed separately)

### Setup
```bash
# 1. Clone and enter directory
git clone https://github.com/PranavMishra17/SnakeAI-MLOps.git
cd SnakeAI-MLOps

# 2. Install vcpkg (in parent directory)
cd ..
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# 3. Set environment variable
setx VCPKG_ROOT "%CD%"

# 4. Install dependencies
.\vcpkg install sfml:x64-windows nlohmann-json:x64-windows spdlog:x64-windows

# 5. Return to project and build
cd ..\SnakeAI-MLOps
cmake --preset windows-default
cmake --build out/build/windows-default
```

### Run Game
```bash
.\out\build\windows-default\SnakeAI-MLOps.exe
```

## Development

### Project Structure
```
src/
├── game/          # SFML game logic
├── rl/            # RL agent implementation  
├── mlops/         # Logging, metrics, model management
└── main.cpp       # Entry point

models/            # Saved ML models
data/              # Training logs and metrics
tests/             # Unit and integration tests
docker/            # Docker configuration
.github/workflows/ # CI/CD pipelines
```

### Build Commands
```bash
# Debug build
cmake --preset windows-default
cmake --build out/build/windows-default

# Release build  
cmake --preset windows-default -DCMAKE_BUILD_TYPE=Release
cmake --build out/build/windows-default --config Release

# Run tests
ctest --test-dir out/build/windows-default
```

### Docker

#### Build Container ✅ VERIFIED
```bash
docker build -f docker/Dockerfile -t snakeai-mlops .
```

#### Verify Container
```bash
# Check image
docker images snakeai-mlops

# Test run
docker run --rm snakeai-mlops

# Explore interactively
docker run -it --rm snakeai-mlops /bin/bash

# Verify project files
docker run --rm snakeai-mlops find /app -name "*.cpp"
```

#### Production
```bash
docker run -p 8080:8080 snakeai-mlops
```

#### Docker Compose (Full Stack)
```bash
# Start all services (game + monitoring)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## CI/CD Pipeline

### Automated Checks
- **Build Verification**: CMake + vcpkg build
- **Unit Tests**: Game logic and RL algorithms
- **Code Quality**: Static analysis
- **Docker Build**: Container image creation

### Manual Verification
```bash
# Check build status
git push origin main
# Monitor: https://github.com/PranavMishra17/SnakeAI-MLOps/actions

# Local CI simulation
.\scripts\run-local-ci.bat
```

## MLOps Features

### Experiment Tracking
- Training metrics logged to `data/experiments/`
- Model checkpoints saved to `models/`
- Performance dashboards via Docker containers

### Model Management
```bash
# Train new model
.\SnakeAI-MLOps.exe --train --episodes 1000

# Evaluate model
.\SnakeAI-MLOps.exe --evaluate --model models/best_model.bin

# Compare models
.\SnakeAI-MLOps.exe --compare --models models/v1.bin,models/v2.bin
```

### Monitoring
- Real-time training visualization
- Performance regression testing
- Automated model deployment

## Verification Checklist

### ✅ Local Development
- [ ] Game window opens and renders
- [ ] Mouse clicks place apples
- [ ] Snake AI moves and learns
- [ ] Models save/load correctly

### ✅ Build System
- [ ] CMake configures without errors
- [ ] All dependencies found via vcpkg
- [ ] Release build optimization works
- [ ] Tests pass locally

### ✅ Docker
- [ ] Container builds successfully
- [ ] Game runs inside container
- [ ] Port forwarding works
- [ ] Volume mounts preserve data

### ✅ CI/CD
- [ ] GitHub Actions build passes
- [ ] Tests run automatically
- [ ] Docker image publishes
- [ ] No credential leaks

## Troubleshooting

### vcpkg Issues
```bash
# Reinstall integration
.\vcpkg integrate install

# Check installed packages
.\vcpkg list

# Update packages
.\vcpkg update
```

### CMake Issues
```bash
# Clear cache
rm -rf out/
cmake --preset windows-default

# Verbose build
cmake --build out/build/windows-default --verbose
```

### Docker Issues
```bash
# Check Docker status
docker --version
docker run hello-world

# Clean build cache
docker system prune -a
```

## Contributing

### Next Development Phase
```cpp
// Immediate tasks (Week 1-2):
1. Grid system implementation
2. Snake class with movement
3. Mouse-click apple placement  
4. Basic collision detection
5. Q-Learning agent foundation
```

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.