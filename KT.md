# SnakeAI-MLOps Knowledge Transfer Document

## Project Overview

**Status**: ✅ **FUNCTIONAL** - All core systems operational  
**Language**: C++17 with CMake build system  
**Framework**: SFML 3.x for graphics and input  
**Dependencies**: vcpkg for package management  

This is a reinforcement learning Snake game with 3 gameplay modes, Q-Learning AI agent, and production MLOps data collection pipeline.

## Current Working Features

### ✅ Completed & Functional
- **Basic Snake Game Logic**: Movement, growth, collision detection
- **Visual System**: Colored rectangles (no textures needed)
- **Three Game Modes**: All operational
- **Q-Learning Agent**: Functional AI with state representation
- **Data Collection**: Episode tracking, metrics logging
- **Build System**: CMake + vcpkg + SFML 3.x integration
- **Menu System**: Basic navigation between modes

### 🎮 Game Modes (All Working)
1. **Single Player**: Human controls snake, system spawns apples
2. **Agent vs Player**: AI controls snake, human places apples with mouse
3. **Agent vs System**: AI controls snake, system spawns apples (training mode)

## File Structure & Architecture

### 🔴 **CRITICAL - DO NOT MODIFY** (Build Configuration)
```
.vscode/
├── c_cpp_properties.json          # IntelliSense configuration - VCPKG paths
├── settings.json                  # VS Code C++ associations
CMakePresets.json                  # Build presets with VCPKG_ROOT path
CMakeLists.txt                     # Main build configuration
vcpkg.json                         # Package dependencies
```

**⚠️ Warning**: These files contain hardcoded paths to `E:/vcpkg/`. Only modify if vcpkg installation path changes.

### 🟢 **SAFE TO MODIFY** (Source Code)
```
src/
├── main.cpp                       # Entry point with debug logging
├── Game.{hpp,cpp}                 # Main game controller & state management
├── GameState.hpp                  # Enums, structs, constants
├── Grid.{hpp,cpp}                 # Grid rendering (light brown background)
├── Snake.{hpp,cpp}                # Snake entity (light green rectangles)
├── Apple.{hpp,cpp}                # Apple spawning (red circles)
├── Menu.{hpp,cpp}                 # Basic menu navigation
├── QLearningAgent.{hpp,cpp}       # Q-Learning implementation
├── DataCollector.{hpp,cpp}        # MLOps data collection
└── InputManager.{hpp,cpp}         # Keyboard/mouse input handling
```

### 📁 Runtime Directories (Auto-created)
```
models/          # Q-table saves (qtable.json)
data/           # Training episodes & metrics  
logs/           # Debug logs (debug.log)
out/            # Build output - can be deleted for clean builds
```

## Core Architecture

### Game Flow
```
main.cpp → Game() constructor → Game::run() loop
         ↓
    Menu System → Mode Selection → Game Logic → Data Collection
```

### Key Classes
- **Game**: Central controller, manages all subsystems
- **Snake**: Entity with collision detection, uses colored rectangles
- **Grid**: Coordinate system, visual grid rendering
- **QLearningAgent**: AI brain with state/action/reward logic
- **DataCollector**: MLOps metrics tracking

### State Management
```cpp
enum class GameState { MENU, PLAYING, PAUSED, GAME_OVER };
enum class GameMode { SINGLE_PLAYER, AGENT_VS_PLAYER, AGENT_VS_SYSTEM };
```

## Visual Design (No Assets Required)

### Color Scheme
- **Snake Head**: Dark green `sf::Color(34, 139, 34)`
- **Snake Body**: Light green `sf::Color(144, 238, 144)`
- **Grid Background**: Light brown `sf::Color(222, 184, 135)`
- **Grid Lines**: Dark grey `sf::Color(105, 105, 105)`
- **Apples**: Red circles `sf::Color::Red`

### Rendering System
- Uses `sf::RectangleShape` for snake segments
- Uses `sf::CircleShape` for apples
- No texture loading - eliminates asset dependencies
- 20x20 grid, auto-scaled to screen size

## Q-Learning Implementation

### State Representation (8 features)
```cpp
struct AgentState {
    bool dangerStraight, dangerLeft, dangerRight;  // Collision detection
    Direction currentDirection;                     // Current heading
    bool foodLeft, foodRight, foodUp, foodDown;    // Food direction
};
```

### Reward System
```cpp
EAT_FOOD = +10.0f
DEATH = -10.0f  
MOVE_TOWARDS_FOOD = +1.0f
MOVE_AWAY_FROM_FOOD = -1.0f
```

### Training Process
- Epsilon-greedy exploration (starts high, decays each episode)
- Q-table auto-saves to `models/qtable.json`
- Episode data logged to `data/training_data.json`

## Controls & Input

### Menu Navigation
- **↑/↓**: Select mode
- **Enter**: Start game
- **ESC**: Return to menu

### In-Game Controls
- **WASD/Arrow Keys**: Snake movement (Single Player only)
- **Mouse Click**: Place apple (Agent vs Player mode)
- **+/-**: Speed adjustment (0.5-3.0 blocks/sec)
- **ESC**: Return to menu (⚠️ No pause yet - see Next Steps)

## Build System

### Prerequisites
- Visual Studio 2022 Community
- vcpkg installed at `E:/vcpkg/` (or update paths in config files)
- VCPKG_ROOT environment variable set

### Build Commands
```bash
# Clean build
rm -rf out/
cmake --preset windows-default
cmake --build out/build/windows-default

# Incremental build
cmake --build out/build/windows-default

# Run (from project root)
.\out\build\windows-default\Debug\SnakeAI-MLOps.exe
```

### SFML 3.x Compatibility Notes
- Uses `sf::Text` and `sf::Sprite` with separate property setting
- No default constructors - uses `std::unique_ptr` where needed
- `loadFromFile()` for textures, `openFromFile()` for fonts

## Debugging & Logging

### Debug Output
- **Console**: Real-time logging during execution
- **File**: `logs/debug.log` with full history
- **Level**: Set to `debug` in `main.cpp`

### Key Debug Points
```cpp
spdlog::info("Game: Creating snake...");      // Component initialization
spdlog::info("Snake: Growing! New length: {}", size);  // Game events
spdlog::warn("Snake: Wall collision detected");        // Error conditions
```

### Crash Debugging
- Run from project root directory
- Check `logs/debug.log` for last logged operation
- Console shows real-time progress during startup

## Data & MLOps

### Files Generated
```
data/training_data.json      # Complete episode history
data/training_summary.json   # Aggregated statistics  
models/qtable.json          # Q-Learning state-action values
logs/debug.log              # Application logs
```

### Metrics Tracked
- Episodes played, steps taken, scores achieved
- Q-table convergence, epsilon decay
- Success/failure rates, average performance

## Next Development Steps

### 🎯 **Priority 1: Enhanced Menu System**
**Goal**: Descriptive menu with mode explanations

**Current State**: Basic text menu with mode names  
**Needed**: 
- Mode descriptions explaining what each does
- Visual indicators for selected mode
- Instructions display (controls, objectives)
- Maybe add mode icons/graphics

**Files to Modify**: `Menu.{hpp,cpp}`

### 🎯 **Priority 2: Pause Functionality** 
**Goal**: ESC pauses game instead of returning to menu

**Current State**: ESC exits to menu immediately  
**Needed**:
- Pause state management
- Pause overlay/UI
- Resume functionality
- Pause during gameplay only (not menu)

**Files to Modify**: `Game.cpp` (event handling), `GameState.hpp`

### 🎯 **Priority 3: Settings System**
**Goal**: Configurable game parameters

**Needed**:
- Settings menu/overlay
- Speed adjustment UI (currently +/- keys only)
- Grid size options
- AI parameters (epsilon, learning rate)
- Save/load settings

**Files to Modify**: New `Settings.{hpp,cpp}`, `Game.cpp`

### 🎯 **Priority 4: Enhanced Visuals**
**Goal**: Better visual feedback and polish

**Ideas**:
- Snake movement animations
- Apple spawn effects  
- Score display improvements
- Training progress visualization
- Performance graphs

### 🎯 **Priority 5: Advanced AI Features**
**Goal**: More sophisticated learning

**Ideas**:
- Deep Q-Network (DQN) implementation
- Multiple AI agents comparison
- Curriculum learning
- Advanced reward shaping

## Common Issues & Solutions

### Build Problems
- **vcpkg not found**: Check `VCPKG_ROOT` environment variable
- **SFML errors**: Ensure using SFML 3.x compatible code
- **Linker errors**: Verify all source files in `CMakeLists.txt`

### Runtime Issues  
- **Immediate crash**: Run from project root where `logs/` directory exists
- **No window**: Check debug logs for SFML initialization errors
- **Missing assets**: Not needed anymore - uses colored shapes

### Development Tips
- Always run from project root directory
- Check `logs/debug.log` for detailed operation history
- Use incremental builds for faster iteration
- Q-table persists between runs for AI training continuity

## Code Quality Guidelines

### Logging Standards
```cpp
spdlog::info("Component: Action description");      // Normal operations
spdlog::warn("Component: Warning condition");       // Recoverable issues  
spdlog::error("Component: Error details");          // Serious problems
```

### SFML 3.x Patterns
```cpp
// Text creation
auto text = std::make_unique<sf::Text>(font);
text->setString("content");
text->setCharacterSize(size);

// Shape positioning  
shape.setPosition(sf::Vector2f(x, y));  // Always use Vector2f
```

### Memory Management
- Use `std::unique_ptr` for SFML objects without default constructors
- RAII principles throughout
- No manual memory management needed

---

**Last Updated**: December 2024  
**Project Status**: Core functionality complete, ready for enhancements  
**Build Status**: ✅ Functional on Windows with VS2022 + vcpkg