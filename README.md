# SnakeAI-MLOps

Advanced Reinforcement Learning Snake Game with Production MLOps Pipeline & Multiple AI Agents

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
│   ├── Game.cpp              # Enhanced game loop with all features
│   ├── GameState.hpp         # Extended enums and state structures
│   ├── Grid.hpp/.cpp         # Grid rendering and coordinate conversion
│   ├── Snake.hpp/.cpp        # Snake movement, collision, rendering
│   ├── Apple.hpp/.cpp        # Apple spawning and rendering
│   ├── Menu.hpp/.cpp         # Main menu navigation and mode selection
│   ├── PauseMenu.hpp/.cpp    # NEW: Enhanced pause functionality
│   ├── AgentSelection.hpp/.cpp # NEW: AI agent selection system
│   ├── Leaderboard.hpp/.cpp  # NEW: Score tracking and username entry
│   ├── QLearningAgent.hpp/.cpp # Original Q-Learning implementation
│   ├── MLAgents.hpp/.cpp     # NEW: Multiple AI agent framework
│   ├── DataCollector.hpp/.cpp # Episode tracking, metrics logging
│   └── InputManager.hpp/.cpp # Keyboard/mouse input processing
├── assets/
│   └── fonts/
│       └── arial.ttf         # UI font (required)
├── models/                   # Saved ML models
│   ├── qtable.json          # Q-Learning table (auto-generated)
│   ├── dqn_model.bin        # DQN weights (placeholder)
│   ├── policy_model.bin     # Policy Gradient model (placeholder)
│   └── actor_critic.bin     # Actor-Critic model (placeholder)
├── data/                     # Training logs and metrics
│   ├── training_data.json   # Episode history (auto-generated)
│   └── training_summary.json # Performance statistics (auto-generated)
├── logs/                     # Application logs
│   └── debug.log            # Runtime logs (auto-generated)
├── leaderboard.json         # Persistent high scores (auto-generated)
├── tests/                    # Unit and integration tests
├── docker/                   # Docker configuration
├── .github/workflows/        # CI/CD pipelines
├── CMakeLists.txt           # Enhanced build configuration
├── CMakePresets.json        # Build presets
├── vcpkg.json              # Dependencies
├── KT.md                   # Knowledge transfer document
├── ML.md                   # Machine learning implementation guide
└── README.md               # This file
```

## Index

### Quick Navigation
- [Setup & Installation](#quick-start)
- [New Features](#new-features-v20)
- [Game Modes](#game-modes)
- [AI Agents](#ai-agents)
  - [Q-Learning Agent](#q-learning-agent)
  - [Deep Q-Network](#deep-q-network-dqn)
  - [Policy Gradient](#policy-gradient)
  - [Actor-Critic](#actor-critic)
  - [Genetic Algorithm](#genetic-algorithm)
- [Controls](#controls)
  - [Menu Navigation](#menu-navigation)
  - [In-Game Controls](#in-game-controls)
  - [Pause Menu](#pause-menu)
  - [Agent Selection](#agent-selection)
- [Leaderboard System](#leaderboard-system)
- [State Representation](#state-representation)
- [Development](#development)
- [Docker](#docker)
- [MLOps Features](#mlops-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## New Features (v2.0)

### 🎮 Enhanced Gameplay
- **Pause Menu**: Full pause functionality with speed control and game stats
- **Agent Selection**: Choose from 5 different AI agents before playing
- **Leaderboard**: Persistent high score tracking with player names and agent types
- **Improved UI**: Enhanced menus, better visual feedback, and comprehensive game stats

### 🤖 Multiple AI Agents
- **Q-Learning**: Enhanced tabular RL with better state representation (✅ Implemented)
- **Deep Q-Network**: Neural network-based Q-learning (🔄 Framework ready)
- **Policy Gradient**: Direct policy optimization (🔄 Framework ready)
- **Actor-Critic**: Combined value and policy learning (🔄 Framework ready)
- **Genetic Algorithm**: Evolution-based approach (🔄 Framework ready)

### 📊 Advanced Analytics
- **Enhanced State Features**: 20-dimensional state vector for neural networks
- **Performance Tracking**: Efficiency metrics and comparative analysis
- **Real-time Agent Info**: Live display of learning parameters and performance

## Game Modes

### 1. Single Player
- **Control**: Human controls snake with arrow keys (↑↓←→) or WASD
- **Objective**: System spawns apples randomly, achieve highest score
- **Features**: Classic snake gameplay with modern enhancements

### 2. Agent vs Player
- **Control**: AI agent controls snake automatically
- **Interaction**: Human places apples with mouse clicks
- **Strategy**: Test AI behavior by controlling food placement
- **Visual**: Red preview shows next apple placement location

### 3. Agent vs System
- **Control**: AI agent controls snake automatically
- **Objective**: System spawns apples randomly
- **Purpose**: Pure AI training mode for reinforcement learning
- **Analytics**: Full data collection and performance tracking

## AI Agents

### Q-Learning Agent
**Status**: ✅ Fully Implemented
- **Method**: Tabular reinforcement learning with epsilon-greedy exploration
- **State Space**: 8-dimensional discrete features (danger detection + food direction)
- **Action Space**: 4 directions (Up, Down, Left, Right)
- **Features**: Experience replay, model persistence, adaptive exploration
- **Performance**: Achieves consistent scores of 10-20 after training

### Deep Q-Network (DQN)
**Status**: 🔄 Framework Ready
- **Method**: Neural network approximation of Q-values
- **Architecture**: 3 hidden layers, 128 neurons each
- **Input**: 20-dimensional enhanced state vector
- **Features**: Target network, experience replay buffer
- **Implementation**: Placeholder with extensible neural network structure

### Policy Gradient
**Status**: 🔄 Framework Ready
- **Method**: Direct policy optimization using REINFORCE algorithm
- **Network**: Policy network outputting action probabilities
- **Features**: Episode-based learning, natural exploration
- **Advantages**: Better for continuous action spaces (future expansion)

### Actor-Critic
**Status**: 🔄 Framework Ready
- **Method**: Combines value function estimation with policy gradient
- **Networks**: Separate actor (policy) and critic (value) networks
- **Features**: Lower variance than pure policy gradient
- **Benefits**: More stable training than policy gradient alone

### Genetic Algorithm
**Status**: 🔄 Framework Ready
- **Method**: Evolution-based approach with neural network population
- **Process**: Mutation, crossover, and selection of best performers
- **Features**: Population-based training, no gradient computation required
- **Use Case**: Alternative approach when gradient-based methods struggle

## Controls

### Menu Navigation
- **↑/↓**: Navigate menu options
- **Enter**: Select current option
- **ESC**: Return to previous menu

### In-Game Controls
- **Arrow Keys/WASD**: Move snake (Single Player mode only)
- **Mouse Click**: Place apple (Agent vs Player mode)
- **ESC**: Open pause menu
- **+/-**: Adjust game speed (0.5-3.0 blocks/sec)
- **F1**: Open leaderboard
- **F2**: Change AI agent (AI modes only)

### Pause Menu
- **↑/↓**: Navigate pause options
- **Enter**: Select option
- **Resume Game**: Continue playing
- **Speed Settings**: Adjust game speed with +/- keys
- **Agent Info**: View current AI agent details
- **Restart Episode**: Reset current game
- **Main Menu**: Return to main menu

### Agent Selection
- **↑/↓**: Browse available AI agents
- **Enter**: Select agent (if implemented)
- **ESC**: Return to main menu
- **Agent Status**: Green "READY" or Yellow "COMING SOON"

## Leaderboard System

### Features
- **Persistent Storage**: Scores saved to `leaderboard.json`
- **Player Names**: Custom username entry for human players
- **Agent Tags**: Automatic tagging by AI agent type
- **Sorting**: Ranked by score, then by efficiency (score/episode ratio)
- **Display**: Top 10 scores with timestamps and performance metrics

### Entry Process
1. Achieve qualifying score (>5 points or human player)
2. Enter custom username (human players only)
3. AI agents get automatic names (Q-Agent, DQN-Agent, etc.)
4. Score recorded with agent type, episode count, and efficiency

### Leaderboard Format
```
Rank  Player Name              Agent Type      Score  Episode  Efficiency
#1    AlphaGamer               Human          25     1        25.00
#2    Q-Agent                  Q-Learning     22     15       1.47
#3    Sarah_M                  Human          18     1        18.00
```

## State Representation

### Basic State (8 dimensions) - Q-Learning
```cpp
struct AgentState {
    bool dangerStraight, dangerLeft, dangerRight;  // Collision detection
    Direction currentDirection;                     // Current heading
    bool foodLeft, foodRight, foodUp, foodDown;    // Food direction
};
```

### Enhanced State (20 dimensions) - Neural Networks
```cpp
struct EnhancedState {
    AgentState basic;                    // Original 8 features
    float distanceToFood;                // Euclidean distance to apple
    float distanceToWall[4];            // Distance to walls (up/down/left/right)
    float bodyDensity[4];               // Snake density by quadrant
    int snakeLength;                    // Current snake size
    int emptySpaces;                    // Available grid spaces
    float pathToFood;                   // A* pathfinding distance
};
```

### Reward System
```cpp
EAT_FOOD = +10.0f           // Successfully eat apple
DEATH = -10.0f              // Collision with wall/self
MOVE_TOWARDS_FOOD = +1.0f   // Move closer to apple
MOVE_AWAY_FROM_FOOD = -1.0f // Move farther from apple
MOVE_PENALTY = -0.1f        // Small penalty per move
EFFICIENCY_BONUS = +2.0f    // Bonus for optimal path
```

## Development

### Build Commands
```bash
# Debug build
cmake --preset windows-default
cmake --build out/build/windows-default

# Release build  
cmake --preset windows-default -DCMAKE_BUILD_TYPE=Release
cmake --build out/build/windows-default --config Release

# Clean rebuild
rm -rf out/
cmake --preset windows-default
cmake --build out/build/windows-default
```

### Required Assets
- `assets/fonts/arial.ttf` - UI font for menus and text display

### Adding New AI Agents
1. **Implement IAgent Interface**: Create new agent class inheriting from `IAgent`
2. **Register in AgentFactory**: Add case in `createAgent()` method
3. **Update AgentSelection**: Add to `initializeAgents()` with description
4. **Set Implementation Status**: Mark `isImplemented = true` when ready
5. **Test Integration**: Verify agent works in all game modes

### Project Architecture
```
Game.cpp (Main Controller)
├── Menu System
│   ├── MainMenu
│   ├── AgentSelection
│   ├── PauseMenu
│   └── Leaderboard
├── Game Engine
│   ├── Snake Logic
│   ├── Apple Management
│   ├── Collision Detection
│   └── State Generation
├── AI Framework
│   ├── Agent Interface
│   ├── Multiple Implementations
│   ├── Model Persistence
│   └── Performance Tracking
└── Data Pipeline
    ├── Episode Recording
    ├── Metrics Collection
    ├── Leaderboard Management
    └── Model Checkpointing
```

## Docker

### Build Container
```bash
docker build -f docker/Dockerfile -t snakeai-mlops .
```

### Run Container
```bash
# Standard run
docker run --rm snakeai-mlops

# Interactive mode
docker run -it --rm snakeai-mlops /bin/bash

# With port mapping (if web interface added)
docker run -p 8080:8080 snakeai-mlops
```

## MLOps Features

### Experiment Tracking
- **Training Metrics**: Episode rewards, scores, exploration rates
- **Model Checkpoints**: Automatic saving of best-performing agents
- **Performance Dashboards**: Real-time training visualization
- **Comparative Analysis**: Multi-agent performance comparison

### Model Management
```bash
# Train new model (example for future CLI)
.\SnakeAI-MLOps.exe --train --agent qlearning --episodes 1000

# Evaluate model performance
.\SnakeAI-MLOps.exe --evaluate --model models/qtable.json

# Compare multiple agents
.\SnakeAI-MLOps.exe --compare --agents qlearning,dqn,policy
```

### Data Pipeline
- **Episode Data**: Complete game state transitions and rewards
- **Performance Metrics**: Score trends, learning curves, efficiency
- **Model Artifacts**: Serialized agent parameters and configurations
- **Leaderboard Analytics**: Player performance and agent effectiveness

## Verification Checklist

### ✅ Core Gameplay
- [x] Snake movement and growth mechanics
- [x] Apple spawning and collection
- [x] Collision detection (walls and self)
- [x] Score tracking and display

### ✅ AI Agents
- [x] Q-Learning agent with model persistence
- [x] Agent selection interface
- [x] Multiple agent framework
- [x] State representation for neural networks

### ✅ User Interface
- [x] Enhanced main menu
- [x] Pause menu with speed control
- [x] Leaderboard with username entry
- [x] Real-time game statistics

### ✅ Data & Analytics
- [x] Episode data collection
- [x] Training metrics logging
- [x] Performance tracking
- [x] Model checkpoint system

### 🔄 Future Enhancements
- [ ] Neural network agent implementations
- [ ] Web-based training dashboard
- [ ] Multi-agent tournaments
- [ ] Advanced visualization tools

## Troubleshooting

### Build Issues
```bash
# vcpkg integration problems
.\vcpkg integrate install
.\vcpkg list

# CMake cache issues
rm -rf out/
cmake --preset windows-default

# Missing dependencies
.\vcpkg install sfml:x64-windows nlohmann-json:x64-windows spdlog:x64-windows
```

### Runtime Issues
- **Font not found**: Ensure `arial.ttf` exists in `assets/fonts/`
- **Agent selection empty**: Check that agent `isImplemented` flags are set correctly
- **Leaderboard not saving**: Verify write permissions in project directory
- **Models not loading**: Check that `models/` directory exists and is writable

### Game Issues
- **Snake won't move**: Check if game is paused (ESC toggles pause)
- **No apple spawning**: Verify apple placement logic for current game mode
- **Agent not learning**: Confirm training mode is enabled and epsilon > 0
- **Performance slow**: Reduce speed or check for debug logging overhead

## Contributing

### Development Priorities
1. **Neural Network Agents**: Implement DQN, Policy Gradient, and Actor-Critic
2. **Advanced Features**: Curriculum learning, multi-agent scenarios
3. **Visualization**: Training dashboards, performance graphs
4. **Optimization**: Performance improvements, memory efficiency

### Code Standards
- Follow existing naming conventions and structure
- Add comprehensive logging for debugging
- Include unit tests for new features
- Update documentation for API changes

### Beginner Contributions
- **Bug Reports**: Test different scenarios and report issues
- **Documentation**: Improve README sections or add tutorials
- **Asset Creation**: Design better graphics or sound effects
- **Configuration**: Add new game settings or customization options

### Advanced Contributions
- **ML Implementation**: Complete placeholder agent implementations
- **Performance**: Optimize training speed and memory usage
- **Features**: Add new game modes or AI techniques
- **Infrastructure**: Improve build system or CI/CD pipeline

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Implement changes with tests
4. Update documentation
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push branch: `git push origin feature/amazing-feature`
7. Open Pull Request with detailed description

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Ready to Train AI Agents?** Start with the Q-Learning agent, experiment with different game modes, and watch your AI learn to master Snake! 🐍🤖