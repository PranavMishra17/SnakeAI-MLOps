# Machine Learning Implementation Guide - SnakeAI-MLOps

## Project Summary: Reinforcement Learning Snake AI

SnakeAI-MLOps is a comprehensive reinforcement learning platform implementing multiple RL algorithms to train AI agents for the classic Snake game. Features GPU-accelerated training, production MLOps pipelines, model evaluation, and both Python training with C++ game integration.

**Core Techniques**: Q-Learning (tabular), DQN (deep Q-learning), PPO (policy optimization), Actor-Critic (value-policy hybrid)  
**Key Features**: GPU acceleration, model comparison, C++ game integration, comprehensive evaluation  
**Use Cases**: RL research, algorithm comparison, educational tool, production ML pipeline

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Mathematical Foundation](#mathematical-foundation)
4. [State Representation](#state-representation)
5. [Training Implementation](#training-implementation)
6. [Training Profiles](#training-profiles)
7. [Model Evaluation](#model-evaluation)
8. [Performance Benchmarks](#performance-benchmarks)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)
11. [Extension Points](#extension-points)
12. [Research Applications](#research-applications)
13. [API Reference](#api-reference)
14. [Contributing](#contributing)

---

## Overview

SnakeAI-MLOps is a comprehensive machine learning platform that implements multiple reinforcement learning algorithms for training AI agents to play Snake. The project features GPU-accelerated training, production MLOps pipelines, and extensive model evaluation capabilities.

### Implemented Techniques

1. **Q-Learning** (Tabular) - ✅ Fully Implemented (Python + C++)
2. **Deep Q-Network (DQN)** - ✅ Fully Implemented (Python training, C++ placeholder)
3. **PPO (Proximal Policy Optimization)** - ✅ Fully Implemented (Python training, C++ placeholder) 
4. **Actor-Critic (A2C)** - ✅ Fully Implemented (Python training, C++ placeholder)

*Note: PPO is an advanced policy gradient method that improves upon REINFORCE with clipped objective functions, reducing policy update variance and improving training stability.*

### GPU Acceleration

All neural network training is GPU-accelerated using PyTorch with CUDA support, providing significant speedup over CPU-only training.

---

## Quick Start

### Prerequisites

```bash
# Python environment
python -m pip install torch numpy matplotlib tqdm tensorboard

# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# C++ dependencies (for game)
# Visual Studio 2022, vcpkg, SFML, nlohmann-json, spdlog
```

### Train All Models

```bash
# Complete training pipeline (all techniques)
python train_models.py --technique all

# Train specific technique
python train_models.py --technique ppo --profile balanced

# Evaluate all trained models
python train_models.py --evaluate
```

### Run Game

```bash
# Build and run C++ game
cmake --build out/build/windows-default
.\out\build\windows-default\SnakeAI-MLOps.exe
```

---

## Model Architectures Summary

### Q-Learning
- **Type**: Lookup table (no neural network)
- **Input**: 8D state → 9-bit binary encoding → 512 possible states
- **Output**: 4 Q-values per state
- **Storage**: JSON format with binary state keys

### DQN (SimpleDQN)
- **Architecture**: 8 → 64 → 64 → 4
- **Input**: 8D float tensor
- **Output**: 4 Q-values (raw logits)
- **Files**: `dqn_*.pt`

### PPO (Two Networks)
- **Policy**: 8 → 64 → 64 → 4 + Softmax
- **Value**: 8 → 64 → 64 → 1
- **Input**: 8D float tensor
- **Output**: Policy = 4 action probabilities, Value = scalar
- **Files**: `ppo_*_policy.pt`, `ppo_*_value.pt`

### Actor-Critic (Two Networks)
- **Actor**: 8 → 64 → 64 → 4 + Softmax  
- **Critic**: 8 → 64 → 64 → 1
- **Input**: 8D float tensor
- **Output**: Actor = 4 action probabilities, Critic = scalar
- **Files**: `ac_*_actor.pt`, `ac_*_critic.pt`

### Universal 8D State Vector
```
[danger_straight, danger_left, danger_right, direction/3.0, 
 food_left, food_right, food_up, food_down]
```
- All boolean flags as 0.0/1.0
- Direction normalized: 0/3, 1/3, 2/3, 1.0

### Universal Action Space
```
[UP, DOWN, LEFT, RIGHT] = [0, 1, 2, 3]
```

All neural networks use ReLU activation and Xavier initialization.

## Mathematical Foundation

### Q-Learning

**Bellman Equation**:
```
Q*(s,a) = E[R(s,a) + γ max_{a'} Q*(s',a')]
```

**Q-Learning Update Rule**:
```
Q(s,a) ← Q(s,a) + α[R + γ max_{a'} Q(s',a') - Q(s,a)]
```

Where:
- `α ∈ (0,1]`: Learning rate
- `γ ∈ [0,1]`: Discount factor  
- `R`: Immediate reward
- `s,s'`: Current and next states
- `a,a'`: Current and next actions

### Deep Q-Network (DQN)

**Loss Function**:
```
L(θ) = E[(y - Q(s,a;θ))²]
```

**Target Value**:
```
y = r + γ max_{a'} Q(s',a';θ⁻)
```

**Double DQN** (reduces overestimation):
```
y = r + γ Q(s', argmax_{a'} Q(s',a';θ), θ⁻)
```

### PPO (Proximal Policy Optimization)

PPO is an advanced policy gradient method that improves training stability through clipped objective functions.

**Objective Function**:
```
L^CLIP(θ) = E_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
```

**Policy Ratio**:
```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

**Combined Objective** (with value function and entropy):
```
L^CLIP+VF+S(θ) = E_t[L^CLIP(θ) - c₁L^VF(θ) + c₂S[π_θ](s_t)]
```

Where:
- `ε`: Clipping parameter (typically 0.2)
- `Â_t`: Advantage estimate
- `c₁, c₂`: Value function and entropy coefficients

### Actor-Critic

**Actor Update** (policy):
```
θ ← θ + α_θ ∇_θ log π_θ(a|s) δ
```

**Critic Update** (value function):
```
w ← w + α_w δ ∇_w V(s;w)
```

**TD Error**:
```
δ = r + γV(s';w) - V(s;w)
```

---

## State Representation

### Basic State (8D) - Q-Learning

```cpp
struct AgentState {
    bool dangerStraight;     // Binary collision detection
    bool dangerLeft;         // Binary collision detection  
    bool dangerRight;        // Binary collision detection
    Direction currentDir;    // 4 discrete values {UP, DOWN, LEFT, RIGHT}
    bool foodLeft;           // Binary food direction
    bool foodRight;          // Binary food direction
    bool foodUp;             // Binary food direction
    bool foodDown;           // Binary food direction
};
```

**State Space Size**: 2³ × 4 × 2⁴ = **512 possible states**

### Enhanced State (8D) - Neural Networks (Fixed)

```python
# Consistent 8D state for all neural networks
neural_state = [
    danger_straight,         # Binary collision detection
    danger_left,            # Binary collision detection
    danger_right,           # Binary collision detection
    current_direction,      # Normalized 0-1 (direction/3.0)
    food_left,              # Binary food direction
    food_right,             # Binary food direction  
    food_up,                # Binary food direction
    food_down               # Binary food direction
]
```

### Reward System

```python
REWARDS = {
    'EAT_FOOD': +10.0,          # Successfully eat apple
    'DEATH': -10.0,             # Collision with wall/self
    'MOVE_TOWARDS_FOOD': +1.0,  # Move closer to apple  
    'MOVE_AWAY_FROM_FOOD': -1.0, # Move farther from apple
    'MOVE_PENALTY': -0.1,       # Small penalty per move
    'EFFICIENCY_BONUS': +2.0    # Bonus for optimal path
}
```

---

## Training Implementation

### Directory Structure

```
models/
├── qlearning/              # Q-Learning models (.json)
│   ├── qtable_aggressive.json
│   ├── qtable_balanced.json
│   └── qtable_conservative.json
├── dqn/                     # Deep Q-Network models (.pth)
│   ├── dqn_aggressive.pth
│   ├── dqn_balanced.pth
│   └── dqn_conservative.pth
├── ppo/                     # PPO models (.pth)
│   ├── ppo_aggressive.pth
│   ├── ppo_balanced.pth
│   └── ppo_conservative.pth
├── actor_critic/           # Actor-Critic models (.pth)
│   ├── ac_aggressive.pth
│   ├── ac_balanced.pth
│   └── ac_conservative.pth
└── checkpoints/            # Training checkpoints
    ├── dqn/
    ├── ppo/
    └── actor_critic/
```

### 1. Q-Learning Training

**Implementation**: `qlearning_trainer.py`

```python
# Train Q-Learning model
python qlearning_trainer.py

# Configuration
class TrainingConfig:
    profile_name: str = "balanced"
    max_episodes: int = 2000        # Reduced for efficiency
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon_start: float = 0.2
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.995
    grid_size: int = 10             # Configurable grid size
```

**GPU Features**:
- Q-table stored as GPU tensor for fast lookup
- Batch Q-value updates
- GPU-accelerated state encoding

**Output**:
- `models/qlearning/qtable_{profile}.json` - Final model
- `models/qlearning/qtable_{profile}_report.json` - Training metrics
- `models/qlearning/training_curves_{profile}.png` - Learning curves

### 2. Deep Q-Network Training

**Implementation**: `dqn_trainer.py`

```python
# Train DQN model
python dqn_trainer.py

# Configuration (FIXED)
class DQNConfig:
    max_episodes: int = 1500        # Reduced for better convergence
    learning_rate: float = 0.001
    epsilon_start: float = 0.9
    epsilon_decay: float = 0.995
    batch_size: int = 32
    memory_capacity: int = 10000
    target_update_freq: int = 100
    hidden_size: int = 64           # Simplified architecture
    grid_size: int = 10             # Smaller grid for better learning
    double_dqn: bool = True
```

**Architecture**:
```python
class SimpleDQN(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, output_size=4):
        # Input: 8D consistent state
        # Hidden: [64, 64] fully connected layers
        # Output: 4 Q-values (one per action)
```

**Features**:
- Experience replay buffer (10K capacity)
- Target network (updated every 100 steps)
- Gradient clipping for stability
- Double DQN for reduced overestimation

**Output**:
- `models/dqn/dqn_{profile}.pth` - Final model
- `models/dqn/dqn_{profile}_best.pth` - Best performing checkpoint
- `models/dqn/dqn_{profile}_metrics.json` - Training metrics

### 3. PPO Training

**Implementation**: `ppo_trainer.py`

```python
# Train PPO model
python ppo_trainer.py

# Configuration (FIXED)
class PPOConfig:
    max_episodes: int = 1200        # Reduced for efficiency
    learning_rate: float = 0.001
    clip_epsilon: float = 0.2       # PPO clipping parameter
    entropy_coeff: float = 0.02     # Exploration bonus
    value_coeff: float = 0.5        # Value function weight
    update_epochs: int = 4          # PPO update epochs
    trajectory_length: int = 128    # Trajectory collection length
    batch_size: int = 32
    hidden_size: int = 64           # Simplified architecture
    grid_size: int = 10             # Smaller grid
```

**Architecture**:
```python
class SimplePolicyNetwork(nn.Module):
    # Input: 8D consistent state
    # Hidden: [64, 64] fully connected layers  
    # Output: 4 action probabilities (softmax)

class SimpleValueNetwork(nn.Module):
    # Input: 8D consistent state
    # Hidden: [64, 64] fully connected layers
    # Output: Single state value
```

**Algorithm**: PPO with clipped objective
```python
# Collect trajectory
for step in trajectory:
    action, log_prob, value = agent.get_action_and_value(state)
    state, reward, done = env.step(action)

# Compute advantages using GAE
advantages, returns = compute_gae(rewards, values, dones, last_value)

# PPO update with clipping
for epoch in update_epochs:
    # Compute policy ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value function loss
    value_loss = F.mse_loss(values, returns)
    
    # Combined loss with entropy bonus
    total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
```

**Output**:
- `models/ppo/ppo_{profile}.pth` - Final model
- Includes both policy and value networks

### 4. Actor-Critic Training

**Implementation**: `actor_critic_trainer.py`

```python
# Train Actor-Critic model
python actor_critic_trainer.py

# Configuration (FIXED)
class ActorCriticConfig:
    max_episodes: int = 1500        # Reduced for efficiency
    actor_lr: float = 0.001
    critic_lr: float = 0.002        # Typically higher than actor
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    hidden_size: int = 64           # Simplified architecture
    grid_size: int = 10             # Smaller grid
```

**Architecture**:
```python
# Separate networks for actor and critic (8D input)
actor = SimplePolicyNetwork(8, 64, 4)    # π(a|s)
critic = SimpleValueNetwork(8, 64)       # V(s)
```

**Algorithm**: Advantage Actor-Critic (A2C)
```python
# Single step update
state_value = critic(state)
next_state_value = critic(next_state)

# TD target and advantage
td_target = reward + gamma * next_state_value * (1 - done)
advantage = td_target - state_value

# Update actor (policy)
actor_loss = -log_prob * advantage

# Update critic (value function)
critic_loss = mse_loss(state_value, td_target)
```

**Output**:
- `models/actor_critic/ac_{profile}.pth` - Final model
- Contains both actor and critic networks

---

## Training Profiles

Each technique supports three training profiles optimized for the fixed configurations:

### Aggressive Profile
- **Goal**: Fast learning, high exploration
- **Use Case**: Quick prototyping, short training time
- **Grid Size**: 8x8 (small for fast learning)

```python
aggressive = {
    'learning_rate': 0.002,         # Higher for faster learning
    'epsilon_start': 0.3,           # High exploration
    'epsilon_decay': 0.99,          # Fast decay
    'max_episodes': 1200,           # Shorter training
    'target_score': 6,              # Achievable target
    'grid_size': 8                  # Small grid
}
```

### Balanced Profile  
- **Goal**: Stable learning, good performance
- **Use Case**: General purpose, best overall results
- **Grid Size**: 10x10 (optimal balance)

```python
balanced = {
    'learning_rate': 0.001,         # Moderate learning rate
    'epsilon_start': 0.2,           # Balanced exploration
    'epsilon_decay': 0.995,         # Gradual decay
    'max_episodes': 1500,           # Sufficient training
    'target_score': 8,              # Good target
    'grid_size': 10                 # Balanced grid
}
```

### Conservative Profile
- **Goal**: Careful learning, maximum stability
- **Use Case**: Best final performance, very stable behavior
- **Grid Size**: 12x12 (larger for challenge)

```python
conservative = {
    'learning_rate': 0.0005,        # Lower for stability
    'epsilon_start': 0.15,          # Conservative exploration
    'epsilon_decay': 0.997,         # Slow decay
    'max_episodes': 2000,           # Longer training
    'target_score': 10,             # Higher target
    'grid_size': 12                 # Larger grid
}
```

---

## Model Evaluation

### Python Evaluation (All Techniques)

**Implementation**: `model_evaluator.py`

```python
# Evaluate all models
python model_evaluator.py

# Evaluate specific model type
evaluator = EnhancedModelEvaluator()
results = evaluator.compare_all_models(episodes=100)

# Individual model evaluation
result = evaluator.evaluate_model(model_path, model_type, episodes=50)
```

**Metrics Computed**:
- Average score over N episodes
- Maximum score achieved
- Score standard deviation (consistency)
- Average episode length
- Action entropy (behavioral diversity)
- Performance rankings
- Food efficiency (steps per food)
- Behavioral stability
- Death cause analysis

**Output**:
- `models/enhanced_comparison_fixed.png` - Visual comparison
- `models/performance_heatmap_fixed.png` - Multi-metric heatmap
- `models/enhanced_evaluation_report_fixed.json` - Detailed metrics
- Console summary with rankings

### C++ Game Integration

**Q-Learning Models**: Fully supported
- Load `.json` models directly
- Real-time inference in game
- All training profiles available

**Neural Network Models**: Placeholder support
- Models detected and listed in game
- Placeholder agents for menu compatibility  
- Full functionality requires Python evaluation

```cpp
// C++ model loading (Q-Learning only)
TrainedModelManager manager;
auto models = manager.getAvailableModels();

for (const auto& model : models) {
    if (model.modelType == "qlearning") {
        // Fully functional
        auto agent = AgentFactory::createTrainedAgent(model.name);
    } else {
        // Placeholder (shows in menu but limited functionality)
        spdlog::warn("Neural network model requires Python for full functionality");
    }
}
```

---

## Performance Benchmarks

### Training Times (GPU: RTX 3080) - UPDATED

| Technique | Aggressive | Balanced | Conservative |
|-----------|------------|----------|--------------|
| Q-Learning | 3 min | 4 min | 6 min |
| DQN | 8 min | 12 min | 18 min |
| PPO | 10 min | 15 min | 22 min |
| Actor-Critic | 9 min | 13 min | 20 min |

### Memory Usage

| Technique | Model Size | GPU Memory | Training Memory |
|-----------|------------|------------|-----------------|
| Q-Learning | ~50 KB | ~10 MB | ~50 MB |
| DQN | ~500 KB | ~200 MB | ~500 MB |
| PPO | ~500 KB | ~200 MB | ~400 MB |
| Actor-Critic | ~1 MB | ~250 MB | ~500 MB |

### Performance Results (Average Scores) - FIXED

| Profile | Q-Learning | DQN | PPO | Actor-Critic |
|---------|------------|-----|-----|--------------|
| Aggressive | 8-12 | 6-10 | 6-10 | 6-10 |
| Balanced | 10-15 | 8-12 | 8-12 | 8-12 |
| Conservative | 12-18 | 10-15 | 10-15 | 10-15 |

---

## Advanced Features

### Hyperparameter Optimization

```python
# Automated hyperparameter search
from hyperopt import hp, fmin, tpe

search_space = {
    'learning_rate': hp.loguniform('lr', -5, -2),
    'epsilon_decay': hp.uniform('decay', 0.99, 0.999),
    'hidden_size': hp.choice('hidden', [32, 64, 128]),
    'grid_size': hp.choice('grid', [8, 10, 12])
}

def objective(params):
    config = DQNConfig(**params)
    train_dqn(config)
    return -evaluate_model(config.model_path)

best = fmin(objective, search_space, algo=tpe.suggest, max_evals=50)
```

### Multi-Agent Training

```python
# Self-play training setup
class SelfPlayTrainer:
    def __init__(self, num_agents=4):
        self.agents = [create_agent() for _ in range(num_agents)]
        self.leaderboard = []
    
    def train_round(self):
        # Each agent plays against others
        # Update based on relative performance
        # Maintain diverse population
```

### Curriculum Learning

```python
# Progressive difficulty
class CurriculumTrainer:
    stages = [
        {'grid_size': 8, 'max_steps': 200},   # Easy
        {'grid_size': 10, 'max_steps': 500},  # Medium  
        {'grid_size': 12, 'max_steps': 1000}  # Hard
    ]
    
    def advance_stage(self, performance):
        if performance > self.stage_threshold:
            self.current_stage += 1
```

---

## Troubleshooting

### GPU Issues

```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Memory issues
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Reduce batch size in configs
config.batch_size = 16  # Instead of 32
```

### Training Issues

```python
# Gradient explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Learning instability  
config.learning_rate *= 0.5  # Reduce learning rate

# Poor exploration
config.epsilon_start = 0.5    # Increase exploration
config.entropy_coeff = 0.02   # Increase entropy bonus
```

### Model Loading Issues

```bash
# C++ game
# Ensure models directory structure is correct
models/
├── qlearning/
│   └── qtable_balanced.json
└── ppo/
    └── ppo_balanced.pth

# Check file permissions
chmod +r models/**/*

# Verify JSON format (Q-Learning)
python -c "import json; json.load(open('models/qlearning/qtable_balanced.json'))"
```

### DQN Performance Issues (FIXED)

```python
# If DQN still underperforming:
config.grid_size = 8          # Use smaller grid
config.hidden_size = 32       # Simplify network further
config.learning_rate = 0.002  # Increase learning rate
config.max_episodes = 2000    # Train longer
```

---

## Extension Points

### Adding New Techniques

1. **Create trainer file**: `new_technique_trainer.py`
2. **Implement base classes**: Inherit from neural network utilities
3. **Add to orchestrator**: Update `train_models.py`
4. **Add C++ placeholder**: Update `MLAgents.cpp` for menu integration

### Custom Reward Functions

```python
class CustomRewardEnvironment(SnakeEnvironment):
    def calculate_reward(self, old_state, action, new_state):
        # Custom reward logic
        base_reward = super().calculate_reward(old_state, action, new_state)
        
        # Add exploration bonus
        exploration_bonus = self.calculate_exploration_bonus(new_state)
        
        # Add efficiency penalty
        efficiency_penalty = self.calculate_efficiency_penalty(action)
        
        return base_reward + exploration_bonus - efficiency_penalty
```

### Neural Network Architectures

```python
# Convolutional architecture for grid input
class ConvDQN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: Grid representation instead of feature vector
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
```

---

## Research Applications

### Academic Research

The platform supports research in:
- Reinforcement learning algorithm comparison
- State representation learning
- Multi-agent systems
- Curriculum learning
- Transfer learning between game variants
- Policy optimization methods (PPO vs other policy gradients)

### Industry Applications

Production ML pipeline features:
- Model versioning and tracking
- A/B testing framework
- Performance monitoring
- Automated retraining
- Model deployment pipelines

### Example Research Questions

1. **Which RL algorithm performs best for discrete control tasks?**
   - Compare Q-Learning vs DQN vs PPO vs Actor-Critic
   - Analyze sample efficiency and final performance

2. **How does state representation affect learning?**
   - Compare 8D discrete vs enhanced continuous state spaces
   - Analyze feature importance and learning curves

3. **What is the effect of PPO vs traditional policy gradients?**
   - Compare PPO clipping vs REINFORCE
   - Analyze training stability and sample efficiency

4. **Grid size impact on learning efficiency?**
   - Compare 8x8 vs 10x10 vs 12x12 grids
   - Analyze convergence time vs final performance trade-offs

---

## API Reference

### Python Training API

```python
# Train any technique
from train_models import train_single_technique
train_single_technique('ppo', 'balanced', episodes=1500)

# Evaluate models
from model_evaluator import EnhancedModelEvaluator
evaluator = EnhancedModelEvaluator()
results = evaluator.evaluate_model(model_path, model_type, episodes=100)

# Load and use trained models
import torch

# PPO model loading
checkpoint = torch.load('models/ppo/ppo_balanced.pth')
policy_net = SimplePolicyNetwork(8, 64, 4)
policy_net.load_state_dict(checkpoint['policy_network'])

# DQN model loading
checkpoint = torch.load('models/dqn/dqn_balanced.pth')
dqn_net = SimpleDQN(8, 64, 4)
dqn_net.load_state_dict(checkpoint['q_network'])
```

### C++ Game API

```cpp
// Load trained model (Q-Learning only)
TrainedModelManager manager;
auto models = manager.getAvailableModels();
auto agent = AgentFactory::createTrainedAgent(model_name);

// Generate state and get action
EnhancedState state = StateGenerator::generateState(snake, apple, grid);
Direction action = agent->getAction(state, false);  // No training

// Update agent (if training)
agent->updateAgent(state, action, reward, nextState);
```

---

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/username/SnakeAI-MLOps.git
cd SnakeAI-MLOps

# Python setup
pip install -r src/requirements.txt

# C++ setup (Windows)
vcpkg install sfml:x64-windows nlohmann-json:x64-windows spdlog:x64-windows
cmake --preset windows-default
```

### Adding New Features

1. **New RL Algorithm**:
   - Create `src/new_algorithm_trainer.py`
   - Follow existing patterns from DQN/PPO
   - Add to `train_models.py` orchestrator
   - Add C++ placeholder in `MLAgents.cpp`

2. **New Evaluation Metrics**:
   - Update `model_evaluator.py`
   - Add visualization to comparison plots
   - Update evaluation reports

3. **C++ Neural Network Support**:
   - Integrate ONNX Runtime or LibTorch
   - Convert trained PyTorch models to ONNX format
   - Implement inference in C++ agents

### Code Standards

- Python: PEP 8, type hints, docstrings
- C++: Google style guide, RAII, smart pointers
- Git: Feature branches, descriptive commits
- Testing: Unit tests for critical functions

---

## Conclusion

SnakeAI-MLOps provides a comprehensive platform for reinforcement learning research and development. With GPU-accelerated training, multiple RL algorithms including the advanced PPO method, and production-ready MLOps features, it serves as both an educational tool and a research platform.

The modular architecture allows easy extension with new algorithms, and the multi-language implementation (Python for training, C++ for real-time inference) provides flexibility for different use cases.

**Key Improvements**: Fixed state representation consistency (8D), optimized grid sizes (8-12), simplified network architectures (64 units), and working Actor-Critic evaluation make this a robust, high-performing RL platform.

For questions or contributions, please refer to the project repository and documentation.