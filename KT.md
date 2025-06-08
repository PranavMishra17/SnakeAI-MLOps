# SnakeAI-MLOps Project Knowledge Transfer Document

## Executive Summary

**Project:** SnakeAI-MLOps  
**Objective:** Reinforcement Learning Snake Game with Production MLOps Pipeline  
**Status:** Development Setup Complete - Ready for Game Logic Implementation  
**Technology Stack:** C++ (SFML), vcpkg, CMake, Docker, GitHub Actions  

## Project Architecture

### High-Level Design
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Game Engine   │    │   RL Agent      │    │   MLOps         │
│   (SFML + C++)  │◄──►│   (Q-Learning)  │◄──►│   (Docker/CI)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Choices & Rationale

#### **Game Framework: SFML**
- **Why:** Lightweight, C++ native, excellent for 2D games
- **Alternatives considered:** SDL2, Raylib
- **Version:** 3.0.0 (latest with modern C++ API)

#### **Build System: CMake + vcpkg**
- **Why:** Industry standard, excellent dependency management
- **vcpkg advantages:** Precompiled binaries, consistent versioning
- **Version:** CMake 4.0.2, vcpkg latest

#### **Dependencies Management**
```json
{
  "sfml": "Graphics, Window, System components",
  "nlohmann-json": "Configuration and data serialization", 
  "spdlog": "Structured logging",
  "fmt": "String formatting (dependency of spdlog)"
}
```

## Development Environment Setup

### Prerequisites
- **Visual Studio 2022 Community** (Free, excellent C++ tooling)
- **vcpkg** (Package manager, installed at `E:\vcpkg`)
- **Git** (Version control)
- **Docker Desktop** (Containerization)

### Directory Structure
```
C:\Users\prana\Desktop\SnakeAI-MLOps\
├── src/
│   ├── main.cpp              # Entry point
│   ├── game/                 # Game logic (future)
│   ├── rl/                   # RL algorithms (future)
│   └── mlops/                # Metrics & logging (future)
├── assets/                   # Game assets (images, etc.)
├── models/                   # Saved ML models
├── data/                     # Training logs
├── tests/                    # Unit tests
├── docker/                   # Docker configuration
├── .github/workflows/        # CI/CD pipelines
├── out/build/windows-default/ # Build output
├── CMakeLists.txt            # Build configuration
├── CMakePresets.json         # Build presets
├── vcpkg.json               # Dependencies
└── README.md                # Project documentation
```

### Build Process & Commands

#### **Initial Setup (One-time)**
```bash
# 1. Clone repository
git clone https://github.com/PranavMishra17/SnakeAI-MLOps.git
cd SnakeAI-MLOps

# 2. Install vcpkg (separate location)
cd E:\
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
setx VCPKG_ROOT "E:\vcpkg"

# 3. Install dependencies
.\vcpkg install sfml:x64-windows nlohmann-json:x64-windows spdlog:x64-windows
```

#### **Development Workflow**
```bash
# Configure (creates build files)
cmake --preset windows-default

# Build (compiles code)
cmake --build out/build/windows-default

# Run executable
.\out\build\windows-default\Debug\SnakeAI-MLOps.exe
```

#### **Critical Answer: Do Changes Require Rebuilding?**

**YES** - Any source code changes require rebuilding:

| Change Type | Rebuild Required | Command |
|-------------|------------------|---------|
| **C++ source files** (.cpp) | **YES** | `cmake --build out/build/windows-default` |
| **Header files** (.hpp) | **YES** | `cmake --build out/build/windows-default` |
| **CMakeLists.txt** | **YES** | `cmake --preset windows-default` then build |
| **vcpkg.json** | **YES** | Full reconfigure + build |
| **Assets** (images, etc.) | **NO** | Just copy to build directory |

**Build Process Explanation:**
1. **CMake Configure:** Generates Visual Studio project files
2. **MSBuild Compile:** Compiles C++ to executable
3. **Link:** Combines with SFML/other libraries
4. **Output:** Creates .exe in `out/build/windows-default/Debug/`

## Current Implementation Status

### ✅ Completed Components

#### **1. Development Environment**
- Visual Studio 2022 integration
- vcpkg dependency management
- CMake build system
- SFML 3.0 integration with modern C++ API

#### **2. Basic Application Framework**
```cpp
// Current main.cpp capabilities:
- SFML window creation (800x600)
- Event handling (close window)
- Image loading and display
- Structured logging with spdlog
- JSON configuration support (ready)
```

#### **3. CI/CD Pipeline**
- GitHub Actions workflow
- Automated building on Windows
- vcpkg dependency caching
- Error reporting and logs

#### **4. Project Infrastructure**
- Git repository with proper .gitignore
- README with setup instructions
- CMake presets for consistent builds
- Environment variable configuration

### 🔄 In Progress

#### **Image Display Implementation**
```cpp
// Current main.cpp features:
sf::Texture texture;
texture.loadFromFile("assets/test.png");  // Load PNG
sf::Sprite sprite(texture);               // Create sprite
window.draw(sprite);                      // Display
```

### 🎯 Next Development Phases

#### **Phase 1: Game Foundation (Week 1-2)**
```cpp
// Planned implementations:
class Grid {
    static const int WIDTH = 15;
    static const int HEIGHT = 15;
    // Grid rendering and coordinate systems
};

class Snake {
    std::vector<Vec2> body;
    Direction direction;
    // Snake movement and collision
};

class Apple {
    Vec2 position;
    // Apple placement and rendering
};
```

#### **Phase 2: Human Interaction (Week 2-3)**
```cpp
// Mouse click handling for apple placement
if (event->is<sf::Event::MouseButtonPressed>()) {
    Vec2 gridPos = screenToGrid(mousePosition);
    apple.setPosition(gridPos);
}
```

#### **Phase 3: RL Agent (Week 3-4)**
```cpp
class QLearningAgent {
    std::map<State, std::array<double, 4>> qTable;
    double epsilon, alpha, gamma;
    // Q-learning algorithm implementation
};
```

## MLOps Integration Strategy

### Experiment Tracking
```cpp
// Planned logging structure:
{
  "episode": 1,
  "steps": 45,
  "reward": 10.5,
  "epsilon": 0.8,
  "timestamp": "2025-01-07T10:30:00Z"
}
```

### Model Management
```
models/
├── checkpoints/
│   ├── episode_100.bin
│   ├── episode_500.bin
│   └── best_model.bin
└── metadata/
    ├── training_config.json
    └── performance_metrics.json
```

### CI/CD Pipeline Features
- **Automated Testing:** Unit tests for game logic and RL algorithms
- **Performance Regression:** Check agent performance doesn't degrade
- **Model Validation:** Automated model quality checks
- **Docker Deployment:** Containerized training and inference

## Common Issues & Troubleshooting

### Build Issues

#### **"cannot open source file SFML/Graphics.hpp"**
**Solution:**
```bash
# Reinstall vcpkg integration
cd E:\vcpkg
.\vcpkg integrate install

# Verify packages installed
.\vcpkg list | grep sfml
```

#### **"vcpkg install failed"**
**Solution:**
```bash
# Update vcpkg baseline in vcpkg.json
"builtin-baseline": "984f9232b2fe0eb94f5e9f161d6c632c581fff0c"

# Update vcpkg itself
cd E:\vcpkg
git pull
.\bootstrap-vcpkg.bat
```

#### **CMake preset not found**
**Solution:**
```bash
# Ensure VCPKG_ROOT environment variable set
setx VCPKG_ROOT "E:\vcpkg"

# Restart command prompt and try again
cmake --preset windows-default
```

### Runtime Issues

#### **"Failed to load assets/test.png"**
**Solutions:**
1. Create `assets/` directory in project root
2. Add any PNG file as `test.png`
3. Ensure working directory is correct when running

#### **Window doesn't respond**
**Solution:** Event loop is working correctly, this is expected behavior

### GitHub Actions Issues

#### **vcpkg baseline errors**
**Solution:** Use latest commit SHA in `vcpkg.json`, not date strings

## Development Guidelines

### Code Standards
```cpp
// Naming conventions:
class SnakeGame {};        // PascalCase for classes
void updateGame() {};      // camelCase for functions
int gridWidth = 15;        // camelCase for variables
const int GRID_SIZE = 30;  // UPPER_CASE for constants
```

### Git Workflow
```bash
# Feature development
git checkout -b feature/game-logic
git add .
git commit -m "Add basic snake movement"
git push origin feature/game-logic

# Pull request → main branch
```

### Testing Strategy
```cpp
// Unit tests planned:
TEST(SnakeTest, InitialPosition) {
    Snake snake;
    EXPECT_EQ(snake.getPosition(), Vec2(7, 7));
}

TEST(QLearningTest, ActionSelection) {
    QLearningAgent agent;
    Action action = agent.selectAction(state);
    EXPECT_NE(action, Action::INVALID);
}
```

## Performance Considerations

### Current Performance
- **Build time:** ~30 seconds (clean build)
- **Runtime:** 60 FPS target with SFML
- **Memory usage:** <50MB for basic application

### Scalability Plans
- **Training parallelization:** Multiple agent instances
- **GPU acceleration:** CUDA for neural networks (future)
- **Distributed training:** Docker Swarm/Kubernetes (future)

## Security & Best Practices

### Secrets Management
```yaml
# GitHub Actions - no hardcoded secrets
# Use GitHub secrets for:
- Docker registry credentials
- Model deployment keys
- Cloud storage access
```

### Code Quality
- **Static analysis:** Planned integration with SonarQube
- **Memory safety:** Modern C++ practices, smart pointers
- **Error handling:** Comprehensive logging with spdlog

## Knowledge Gaps & Learning Resources

### Immediate Learning Needs
1. **SFML 3.0 API changes** - Official documentation
2. **Q-Learning implementation** - Sutton & Barto textbook
3. **CMake advanced features** - Professional CMake guide

### Recommended Resources
- **SFML:** https://www.sfml-dev.org/tutorials/
- **Reinforcement Learning:** OpenAI Spinning Up
- **MLOps:** "Building Machine Learning Powered Applications"

## Project Roadmap

### Short-term (2-4 weeks)
- [ ] Complete basic Snake game mechanics
- [ ] Implement mouse-based apple placement
- [ ] Add simple Q-learning agent
- [ ] Basic experiment logging

### Medium-term (1-2 months)
- [ ] Advanced RL algorithms (DQN, PPO)
- [ ] Comprehensive MLOps pipeline
- [ ] Performance optimization
- [ ] Model comparison framework

### Long-term (3+ months)
- [ ] Multi-agent scenarios
- [ ] Cloud deployment
- [ ] Real-time training dashboard
- [ ] Academic paper/publication

## Contact & Support

### Key Personnel
- **Developer:** Pranav Mishra
- **Repository:** https://github.com/PranavMishra17/SnakeAI-MLOps

### Getting Help
1. **Build issues:** Check GitHub Actions logs
2. **Runtime errors:** Check application logs in console
3. **vcpkg problems:** Consult vcpkg documentation
4. **RL questions:** Refer to OpenAI Gym documentation

---

**Document Version:** 1.0  
**Last Updated:** January 7, 2025  
**Next Review:** January 14, 2025