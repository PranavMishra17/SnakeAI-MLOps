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

## Project File Structure

```
SnakeAI-MLOps/
├── src/
│   ├── main.cpp              # Entry point with logging setup
│   ├── Game.hpp              # Main game class header
│   ├── Game.cpp              # Game loop, state management, rendering
│   ├── GameState.hpp         # Enums and state structures
│   ├── Grid.hpp              # Grid system header
│   ├── Grid.cpp              # Grid rendering and coordinate conversion
│   ├── Snake.hpp             # Snake entity header
│   ├── Snake.cpp             # Snake movement, collision, rendering
│   ├── Apple.hpp             # Apple/food header
│   ├── Apple.cpp             # Apple spawning and rendering
│   ├── Menu.hpp              # Main menu header
│   ├── Menu.cpp              # Menu navigation and mode selection
│   ├── QLearningAgent.hpp    # Q-Learning AI header
│   ├── QLearningAgent.cpp    # Q-table management, action selection
│   ├── DataCollector.hpp     # Data collection header
│   ├── DataCollector.cpp     # Episode tracking, metrics logging
│   ├── InputManager.hpp      # Input handling header
│   └── InputManager.cpp      # Keyboard/mouse input processing
├── assets/
│   ├── head.jpg              # Snake head texture (required)
│   ├── skin.jpg              # Snake body texture (required)
│   └── fonts/
│       └── arial.ttf         # UI font (required)
├── models/                   # Saved ML models
│   └── qtable.json          # Q-Learning table (auto-generated)
├── data/                     # Training logs and metrics
│   ├── training_data.json   # Episode history (auto-generated)
│   └── training_summary.json # Performance statistics (auto-generated)
├── logs/                     # Application logs
│   └── game.log             # Runtime logs (auto-generated)
├── tests/                    # Unit and integration tests
├── docker/                   # Docker configuration
│   └── Dockerfile           # Container setup
├── .github/workflows/        # CI/CD pipelines
│   └── ci.yml              # GitHub Actions workflow
├── CMakeLists.txt           # Build configuration
├── CMakePresets.json        # Build presets
├── vcpkg.json              # Dependencies
├── KT.md                   # Knowledge transfer document
└── README.md               # This file
```

## Index

### Quick Navigation
- [Setup & Installation](#quick-start)
- [Project File Structure](#project-file-structure)
- [Game Modes](#game-modes)
  - [Single Player](#1-single-player)
  - [Agent vs Player](#2-agent-vs-player)
  - [Agent vs System](#3-agent-vs-system)
- [Controls](#controls)
- [Q-Learning Implementation](#q-learning-implementation)
  - [State Representation](#state-representation)
  - [Reward System](#reward-system)
  - [Training Process](#training-process)
- [Development](#development)
  - [Build Commands](#build-commands)
  - [Required Assets](#required-assets)
- [Docker](#docker)
- [CI/CD Pipeline](#cicd-pipeline)
- [MLOps Features](#mlops-features)
- [Data Collection](#data-collection)
  - [Episode Data](#episode-data)
  - [Training Files](#training-files)
  - [Metrics Tracked](#metrics-tracked)
- [Troubleshooting](#troubleshooting)
  - [vcpkg Issues](#vcpkg-issues)
  - [CMake Issues](#cmake-issues)
  - [Docker Issues](#docker-issues)
  - [Game Issues](#game-issues)
- [Contributing](#contributing)

## Game Modes

### 1. Single Player
- Human controls snake with arrow keys (↑↓←→) or WASD
- System spawns apples randomly
- Classic snake gameplay

### 2. Agent vs Player
- ML agent controls snake automatically
- Human places apples with mouse clicks
- Click empty cell to place apple
- Red preview shows next apple placement

### 3. Agent vs System
- ML agent controls snake
- System spawns apples randomly
- Training mode for Q-Learning

## Controls

### Menu Navigation
- **↑/↓**: Select game mode
- **Enter**: Start game
- **ESC**: Return to menu

### In-Game Controls
- **Arrow Keys/WASD**: Move snake (Single Player only)
- **Mouse Click**: Place apple (Agent vs Player only)
- **+/-**: Increase/decrease speed (0.5-3.0 blocks/sec)
- **ESC**: Pause/unpause game

## Q-Learning Implementation

### State Representation
The agent perceives 8 features:
1. Danger straight ahead
2. Danger to the left
3. Danger to the right
4. Current direction
5. Food is left
6. Food is right
7. Food is up
8. Food is down

### Reward System
- Eat food: +10 points
- Hit wall/self: -10 points
- Move toward food: +1 point
- Move away from food: -1 point

### Training Process
1. Agent starts with empty Q-table
2. Uses epsilon-greedy strategy (explores vs exploits)
3. Updates Q-values after each action
4. Epsilon decays each episode (more exploitation over time)
5. Q-table auto-saves to `models/qtable.json`

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

### Required Assets
Before running, ensure these files exist:
- `assets/head.jpg` - Snake head texture
- `assets/skin.jpg` - Snake body texture
- `assets/fonts/arial.ttf` - UI font

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

## Data Collection

### Episode Data
Each game episode records:
- Episode number
- Steps taken
- Final score
- Total reward accumulated
- Death status
- Duration in milliseconds
- Epsilon value

### Training Files
- `data/training_data.json`: Complete episode history
- `data/training_summary.json`: Aggregated statistics
- `logs/game.log`: Runtime events and errors

### Metrics Tracked
- Average score per episode
- Average steps per episode
- Maximum score achieved
- Success rate (non-death episodes)
- Q-table convergence

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

### Game Issues
- **Snake won't move**: Check if game is paused (ESC key)
- **No textures**: Ensure `assets/head.jpg` and `assets/skin.jpg` exist
- **Font error**: Add `arial.ttf` to `assets/fonts/`
- **Q-table not saving**: Check write permissions for `models/` directory

## Contributing

### Next Development Phase
```cpp
// Immediate tasks (Week 1-2):
1. Grid system implementation ✅
2. Snake class with movement ✅
3. Mouse-click apple placement ✅
4. Basic collision detection ✅
5. Q-Learning agent foundation ✅

// Next phase (Week 3-4):
1. Deep Q-Network (DQN) implementation
2. Performance visualization dashboard
3. Multi-agent scenarios
4. Advanced reward shaping
5. Curriculum learning
```

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.