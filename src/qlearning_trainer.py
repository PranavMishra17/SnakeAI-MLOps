#!/usr/bin/env python3
"""
Updated Q-Learning trainer for SnakeAI-MLOps with configurable grid sizes
GPU-accelerated with consistent grid options for fair comparison
"""
import torch
import numpy as np
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

# GPU Verification
def verify_gpu():
    """Verify GPU availability and setup"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        
        # Test GPU with small tensor
        test_tensor = torch.randn(1000, 1000).cuda()
        result = torch.mm(test_tensor, test_tensor)
        print(f"✅ GPU Test: {result.shape} tensor computed successfully")
        return device
    else:
        print("❌ No GPU available, using CPU")
        return torch.device('cpu')

@dataclass
class TrainingConfig:
    """Updated training configuration with grid size options"""
    profile_name: str = "balanced"
    max_episodes: int = 5000
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon_start: float = 0.2
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.995
    target_score: int = 20
    checkpoint_interval: int = 100
    grid_size: int = 10  # Added configurable grid size
    device: str = "cuda"

class SnakeEnvironment:
    """Updated Snake environment with configurable grid size"""
    def __init__(self, grid_size=10, device='cuda'):
        self.grid_size = grid_size
        self.device = torch.device(device)
        print(f"✅ Snake Environment: {grid_size}x{grid_size} grid")
        self.reset()
    
    def reset(self):
        """Reset environment state"""
        center = self.grid_size // 2
        self.snake = [(center, center), (center, center-1)]  # Head, body
        self.direction = 3  # RIGHT
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        return self._get_state()
    
    def _place_food(self):
        """Place food randomly avoiding snake"""
        while True:
            pos = (np.random.randint(0, self.grid_size), 
                   np.random.randint(0, self.grid_size))
            if pos not in self.snake:
                return pos
    
    def _get_state(self):
        """Get 8D state vector as in C++ version"""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Danger detection
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        current_dir = directions[self.direction]
        
        # Check straight, left, right relative to current direction
        straight_pos = (head_x + current_dir[0], head_y + current_dir[1])
        danger_straight = self._is_collision(straight_pos)
        
        # Left turn from current direction
        left_dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # LEFT of UP,DOWN,LEFT,RIGHT
        left_dir = left_dirs[self.direction]
        left_pos = (head_x + left_dir[0], head_y + left_dir[1])
        danger_left = self._is_collision(left_pos)
        
        # Right turn from current direction  
        right_dirs = [(1, 0), (-1, 0), (0, -1), (0, 1)]  # RIGHT of UP,DOWN,LEFT,RIGHT
        right_dir = right_dirs[self.direction]
        right_pos = (head_x + right_dir[0], head_y + right_dir[1])
        danger_right = self._is_collision(right_pos)
        
        # Food direction
        food_left = food_x < head_x
        food_right = food_x > head_x
        food_up = food_y < head_y
        food_down = food_y > head_y
        
        state = torch.tensor([
            int(danger_straight), int(danger_left), int(danger_right),
            self.direction,
            int(food_left), int(food_right), int(food_up), int(food_down)
        ], dtype=torch.float32, device=self.device)
        
        return state
    
    def _is_collision(self, pos):
        """Check if position causes collision"""
        x, y = pos
        return (x < 0 or x >= self.grid_size or 
                y < 0 or y >= self.grid_size or 
                pos in self.snake)
    
    def step(self, action):
        """Take action and return next state, reward, done"""
        self.direction = action
        head_x, head_y = self.snake[0]
        
        # Move snake
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        dx, dy = directions[action]
        new_head = (head_x + dx, head_y + dy)
        
        # Check collision
        if self._is_collision(new_head):
            return self._get_state(), -10.0, True  # Death penalty
        
        self.snake.insert(0, new_head)
        
        # Check food
        ate_food = False
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            ate_food = True
            reward = 10.0
        else:
            self.snake.pop()  # Remove tail if no food
            # Distance-based reward
            old_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = 1.0 if new_dist < old_dist else -1.0
        
        self.steps += 1
        # Adjust max steps based on grid size
        max_steps = self.grid_size * 50  # Scales with grid size
        done = self.steps >= max_steps
        
        return self._get_state(), reward, done

class QLearningAgent:
    """GPU-accelerated Q-Learning agent with improved performance"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Q-table as GPU tensor for fast lookup
        # State encoding: 9 bits -> 512 possible states, 4 actions
        self.q_table = torch.zeros((512, 4), device=self.device)
        self.epsilon = config.epsilon_start
        
        print(f"✅ Q-Learning Agent initialized on {self.device}")
        print(f"   Grid size: {config.grid_size}x{config.grid_size}")
    
    def encode_state(self, state):
        """Convert 8D state to single index for Q-table"""
        # Convert each element to binary representation
        binary_parts = []
        
        # Danger flags (3 bits)
        binary_parts.append(str(int(state[0].item())))  # danger_straight
        binary_parts.append(str(int(state[1].item())))  # danger_left
        binary_parts.append(str(int(state[2].item())))  # danger_right
        
        # Direction (2 bits for 4 values: 00, 01, 10, 11)
        direction = int(state[3].item())
        direction_binary = format(direction, '02b')  # Convert 0-3 to 2-bit binary
        binary_parts.append(direction_binary)
        
        # Food flags (4 bits)
        binary_parts.append(str(int(state[4].item())))  # food_left
        binary_parts.append(str(int(state[5].item())))  # food_right
        binary_parts.append(str(int(state[6].item())))  # food_up
        binary_parts.append(str(int(state[7].item())))  # food_down
        
        # Combine into single binary string (9 bits total)
        binary_str = ''.join(binary_parts)
        
        # Convert to integer index (max 511 for 9 bits)
        return int(binary_str, 2)
    
    def get_action(self, state, training=True):
        """Get action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        
        state_idx = self.encode_state(state)
        return torch.argmax(self.q_table[state_idx]).item()
    
    def update(self, state, action, reward, next_state):
        """Update Q-values using GPU tensors"""
        state_idx = self.encode_state(state)
        next_state_idx = self.encode_state(next_state)
        
        current_q = self.q_table[state_idx, action]
        max_next_q = torch.max(self.q_table[next_state_idx])
        
        target_q = reward + self.config.discount_factor * max_next_q
        self.q_table[state_idx, action] += self.config.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
    
    def load_model(self, filepath):
        """Load Q-table from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Clear existing Q-table
            self.q_table.zero_()
            
            # Load Q-values
            if "qTable" in data:
                for state_str, actions in data["qTable"].items():
                    if len(state_str) == 9:  # Valid 9-bit state
                        state_idx = int(state_str, 2)
                        if state_idx < 512:
                            self.q_table[state_idx] = torch.tensor(actions, device=self.device)
            
            # Load hyperparameters
            if "hyperparameters" in data:
                params = data["hyperparameters"]
                if "epsilon" in params:
                    self.epsilon = params["epsilon"]
                if "learningRate" in params:
                    self.config.learning_rate = params["learningRate"]
                if "discountFactor" in params:
                    self.config.discount_factor = params["discountFactor"]
            
            print(f"✅ Model loaded from: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def save_model(self, filepath):
        """Save Q-table as JSON (compatible with C++)"""
        # Convert GPU tensor to CPU numpy for serialization
        q_table_cpu = self.q_table.cpu().numpy()
        
        # Convert to C++ compatible format
        q_table_dict = {}
        for state_idx in range(512):  # 9-bit states
            if torch.any(self.q_table[state_idx] != 0):  # Only save non-zero entries
                # Convert index back to state string for C++ compatibility
                state_str = format(state_idx, '09b')  # 9-bit binary string
                q_table_dict[state_str] = q_table_cpu[state_idx].tolist()
        
        model_data = {
            "qTable": q_table_dict,
            "hyperparameters": {
                "learningRate": self.config.learning_rate,
                "discountFactor": self.config.discount_factor,
                "epsilon": self.epsilon,
                "gridSize": self.config.grid_size  # Added grid size
            },
            "metadata": {
                "profile": self.config.profile_name,
                "gridSize": self.config.grid_size,
                "totalStates": len(q_table_dict)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=4)
        
        print(f"✅ Model saved: {filepath}")

def train_qlearning(config: TrainingConfig):
    """Main training function with configurable grid size"""
    device = verify_gpu()
    config.device = str(device)
    
    # Setup paths with proper directory structure
    model_dir = Path("models/qlearning")
    checkpoint_dir = model_dir / f"{config.profile_name}_checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment and agent with configurable grid
    env = SnakeEnvironment(grid_size=config.grid_size, device=str(device))
    agent = QLearningAgent(config)
    
    # Training metrics
    scores = []
    epsilons = []
    
    print(f"🚀 Starting Q-Learning training: {config.profile_name}")
    print(f"   Grid: {config.grid_size}x{config.grid_size}")
    print(f"   Target score: {config.target_score}")
    
    # Training loop
    for episode in tqdm(range(config.max_episodes), desc="Training Q-Learning"):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.get_action(state, training=True)
            next_state, reward, done = env.step(action)
            
            agent.update(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.decay_epsilon()
        scores.append(env.score)
        epsilons.append(agent.epsilon)
        
        # Progress logging
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}: Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
            
            # Save checkpoint
            if episode % config.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"qtable_{config.profile_name}_ep{episode}.json"
                agent.save_model(str(checkpoint_path))
        
        # Early stopping
        if episode >= 100 and np.mean(scores[-100:]) >= config.target_score:
            print(f"✅ Target score reached at episode {episode}")
            break
    
    # Save final model
    final_path = model_dir / f"qtable_{config.profile_name}.json"
    agent.save_model(str(final_path))
    
    # Generate training report
    report = {
        "profile": config.profile_name,
        "episodes": len(scores),
        "final_avg_score": float(np.mean(scores[-100:])),
        "max_score": int(max(scores)),
        "final_epsilon": agent.epsilon,
        "grid_size": config.grid_size,
        "config": config.__dict__
    }
    
    report_path = model_dir / f"qtable_{config.profile_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title(f'Q-Learning Scores - {config.profile_name} ({config.grid_size}x{config.grid_size})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(epsilons)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig(model_dir / f"training_curves_{config.profile_name}.png")
    print(f"✅ Q-Learning training complete!")
    print(f"📁 Final model: {final_path}")
    print(f"📁 Checkpoints: {checkpoint_dir}/")
    print(f"📊 Report: {report_path}")

if __name__ == "__main__":
    # Train different profiles with varying grid sizes
    profiles = {
        "aggressive": TrainingConfig(
            profile_name="aggressive",
            learning_rate=0.2,
            epsilon_start=0.3,
            epsilon_end=0.05,
            max_episodes=2000,
            grid_size=8,  # Smaller grid for faster learning
            target_score=8
        ),
        "balanced": TrainingConfig(
            profile_name="balanced", 
            learning_rate=0.1,
            epsilon_start=0.2,
            epsilon_end=0.02,
            max_episodes=3000,
            grid_size=10,  # Medium grid
            target_score=12
        ),
        "conservative": TrainingConfig(
            profile_name="conservative",
            learning_rate=0.05,
            epsilon_start=0.1,
            epsilon_end=0.01,
            max_episodes=4000,
            grid_size=12,  # Larger grid
            target_score=15
        )
    }
    
    for name, config in profiles.items():
        print(f"\n{'='*50}")
        print(f"Training {name} model")
        print(f"{'='*50}")
        train_qlearning(config)

"""
USAGE EXAMPLES:
===============

# Import and train Q-Learning model
from qlearning_trainer import train_qlearning, TrainingConfig

# Quick training with default settings
config = TrainingConfig(profile_name="test", max_episodes=1000)
train_qlearning(config)

# Custom configuration with small grid for fast learning
config = TrainingConfig(
    profile_name="custom",
    learning_rate=0.15,
    grid_size=8,           # Small grid for faster convergence
    max_episodes=2000,
    target_score=8,
    epsilon_start=0.3,
    epsilon_decay=0.99
)
train_qlearning(config)

# Large grid for challenging environment
config = TrainingConfig(
    profile_name="challenge",
    grid_size=15,          # Large grid
    max_episodes=5000,
    target_score=20,
    learning_rate=0.1
)
train_qlearning(config)

# Load and evaluate existing model
from qlearning_trainer import QLearningAgent, SnakeEnvironment

config = TrainingConfig(profile_name="evaluation", grid_size=10)
agent = QLearningAgent(config)
agent.load_model("models/qlearning/qtable_balanced.json")

env = SnakeEnvironment(grid_size=10)
state = env.reset()
action = agent.get_action(state, training=False)  # No exploration

# Train from command line
python qlearning_trainer.py

# Key features:
# - Configurable grid sizes (8x8 to 20x20)
# - GPU acceleration for Q-table operations
# - C++ compatible model format
# - Automatic scaling of episode length with grid size
# - Comprehensive training metrics and plots

# Performance expectations by grid size:
# - 8x8 grid: ~8-12 average score, fast convergence
# - 10x10 grid: ~10-15 average score, good balance
# - 12x12+ grid: ~12-20+ average score, slower learning
"""