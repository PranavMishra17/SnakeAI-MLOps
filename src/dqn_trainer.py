#!/usr/bin/env python3
"""
Deep Q-Network (DQN) trainer for SnakeAI-MLOps
GPU-accelerated implementation with experience replay and target networks
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import time

from neural_network_utils import (
    DQNNetwork, NetworkConfig, ExperienceReplay, 
    verify_gpu, create_directories, save_model, 
    TrainingMetrics, encode_state_for_dqn
)

@dataclass
class DQNConfig:
    """DQN training configuration"""
    profile_name: str = "balanced"
    max_episodes: int = 2000
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # DQN specific
    batch_size: int = 32
    memory_capacity: int = 10000
    target_update_freq: int = 100
    min_memory_size: int = 1000
    
    # Network architecture
    hidden_layers: list = None
    dueling: bool = True
    double_dqn: bool = True
    
    # Training settings
    device: str = "cuda"
    checkpoint_interval: int = 200
    target_score: int = 15
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 128]

class SnakeEnvironmentDQN:
    """Snake environment optimized for DQN training"""
    
    def __init__(self, grid_size=20, device='cuda'):
        self.grid_size = grid_size
        self.device = torch.device(device)
        self.reset()
    
    def reset(self):
        """Reset environment and return initial state"""
        self.snake = [(10, 10), (10, 9)]
        self.direction = 3  # RIGHT
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        return self._get_enhanced_state()
    
    def _place_food(self):
        """Place food randomly avoiding snake"""
        while True:
            pos = (np.random.randint(0, self.grid_size), 
                   np.random.randint(0, self.grid_size))
            if pos not in self.snake:
                return pos
    
    def _get_enhanced_state(self):
        """Get 20D enhanced state for neural networks"""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Basic 8D state (like Q-Learning)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        current_dir = directions[self.direction]
        
        # Danger detection
        straight_pos = (head_x + current_dir[0], head_y + current_dir[1])
        danger_straight = self._is_collision(straight_pos)
        
        left_dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        left_dir = left_dirs[self.direction]
        left_pos = (head_x + left_dir[0], head_y + left_dir[1])
        danger_left = self._is_collision(left_pos)
        
        right_dirs = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        right_dir = right_dirs[self.direction]
        right_pos = (head_x + right_dir[0], head_y + right_dir[1])
        danger_right = self._is_collision(right_pos)
        
        # Food direction
        food_left = food_x < head_x
        food_right = food_x > head_x
        food_up = food_y < head_y
        food_down = food_y > head_y
        
        # Enhanced features for neural networks
        state_20d = torch.zeros(20, dtype=torch.float32, device=self.device)
        
        # Basic features (8D)
        state_20d[0] = float(danger_straight)
        state_20d[1] = float(danger_left)
        state_20d[2] = float(danger_right)
        state_20d[3] = float(self.direction / 3.0)  # Normalize direction
        state_20d[4] = float(food_left)
        state_20d[5] = float(food_right)
        state_20d[6] = float(food_up)
        state_20d[7] = float(food_down)
        
        # Enhanced features
        # Distance to food (normalized)
        food_distance = abs(head_x - food_x) + abs(head_y - food_y)
        state_20d[8] = food_distance / (2 * self.grid_size)
        
        # Distance to walls (normalized)
        state_20d[9] = head_y / self.grid_size  # Distance to top
        state_20d[10] = (self.grid_size - 1 - head_y) / self.grid_size  # Distance to bottom
        state_20d[11] = head_x / self.grid_size  # Distance to left
        state_20d[12] = (self.grid_size - 1 - head_x) / self.grid_size  # Distance to right
        
        # Body density in quadrants
        quadrant_counts = [0, 0, 0, 0]
        half_size = self.grid_size // 2
        for seg_x, seg_y in self.snake[1:]:  # Exclude head
            quadrant = 0
            if seg_x >= half_size:
                quadrant += 1
            if seg_y >= half_size:
                quadrant += 2
            quadrant_counts[quadrant] += 1
        
        quadrant_size = half_size * half_size
        for i in range(4):
            state_20d[13 + i] = quadrant_counts[i] / quadrant_size
        
        # Snake length (normalized)
        state_20d[17] = len(self.snake) / (self.grid_size * self.grid_size)
        
        # Empty spaces (normalized)
        empty_spaces = self.grid_size * self.grid_size - len(self.snake) - 1
        state_20d[18] = empty_spaces / (self.grid_size * self.grid_size)
        
        # Path complexity (simple heuristic)
        path_complexity = food_distance + len(self.snake) * 0.1
        state_20d[19] = path_complexity / (2 * self.grid_size + 10)
        
        return state_20d
    
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
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = directions[action]
        new_head = (head_x + dx, head_y + dy)
        
        # Check collision
        if self._is_collision(new_head):
            return self._get_enhanced_state(), -10.0, True
        
        self.snake.insert(0, new_head)
        
        # Check food
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            reward = 10.0
        else:
            self.snake.pop()
            # Distance-based reward
            old_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = 1.0 if new_dist < old_dist else -0.5
        
        self.steps += 1
        done = self.steps >= 1000  # Max steps per episode
        
        return self._get_enhanced_state(), reward, done

class DQNAgent:
    """Deep Q-Network Agent with experience replay and target network"""
    
    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        net_config = NetworkConfig(
            input_size=20,
            hidden_layers=config.hidden_layers,
            output_size=4,
            dropout=0.1
        )
        
        self.q_network = DQNNetwork(net_config, dueling=config.dueling).to(self.device)
        self.target_network = DQNNetwork(net_config, dueling=config.dueling).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Experience replay
        self.memory = ExperienceReplay(config.memory_capacity, self.device)
        
        # Training state
        self.epsilon = config.epsilon_start
        self.steps_done = 0
        
        print(f"✅ DQN Agent initialized on {self.device}")
        print(f"   Network: {sum(p.numel() for p in self.q_network.parameters())} parameters")
    
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        
        with torch.no_grad():
            state_batch = state.unsqueeze(0)  # Add batch dimension
            q_values = self.q_network(state_batch)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Single training step using experience replay"""
        if len(self.memory) < self.config.min_memory_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (self.config.discount_factor * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
    
    def save_model(self, filepath, metadata=None):
        """Save DQN model"""
        model_data = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'metadata': metadata or {}
        }
        torch.save(model_data, filepath)
        print(f"✅ DQN model saved: {filepath}")

def train_dqn(config: DQNConfig):
    """Main DQN training loop"""
    device = verify_gpu()
    config.device = str(device)
    
    # Setup directories
    base_dir = Path("models")
    model_dir = base_dir / "dqn"
    checkpoint_dir = base_dir / "checkpoints" / "dqn"
    create_directories(str(base_dir))
    
    # Initialize environment and agent
    env = SnakeEnvironmentDQN(device=str(device))
    agent = DQNAgent(config)
    metrics = TrainingMetrics()
    
    print(f"🚀 Starting DQN training: {config.profile_name}")
    print(f"   Target score: {config.target_score}")
    print(f"   Max episodes: {config.max_episodes}")
    
    best_score = 0
    best_path = None  # FIXED: Initialize best_path
    training_start = time.time()
    
    for episode in tqdm(range(config.max_episodes), desc="Training DQN"):
        state = env.reset()
        total_reward = 0
        episode_loss = 0
        steps = 0
        
        while True:
            action = agent.get_action(state, training=True)
            next_state, reward, done = env.step(action)
            
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            episode_loss += loss
            total_reward += reward
            steps += 1
            
            state = next_state
            
            if done:
                break
        
        # Update target network
        if episode % config.target_update_freq == 0:
            agent.update_target_network()
        
        agent.decay_epsilon()
        metrics.add_episode(env.score, episode_loss / steps, agent.epsilon, steps, total_reward)
        
        # Progress logging
        if episode % 100 == 0:
            avg_score = metrics.get_recent_average('scores', 100)
            avg_loss = metrics.get_recent_average('losses', 100)
            print(f"Episode {episode}: Avg Score: {avg_score:.2f}, Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.3f}")
            
            # Save checkpoint
            if episode % config.checkpoint_interval == 0 and episode > 0:
                checkpoint_path = checkpoint_dir / f"dqn_{config.profile_name}_ep{episode}.pth"
                agent.save_model(checkpoint_path, {
                    'episode': episode,
                    'avg_score': avg_score,
                    'training_time': time.time() - training_start
                })
        
        # Save best model
        if env.score > best_score:
            best_score = env.score
            best_path = model_dir / f"dqn_{config.profile_name}_best.pth"
            agent.save_model(best_path, {
                'episode': episode,
                'best_score': best_score,
                'training_time': time.time() - training_start
            })
        
        # Early stopping
        if episode >= 100 and metrics.get_recent_average('scores', 100) >= config.target_score:
            print(f"✅ Target score reached at episode {episode}")
            break
    
    # Save final model
    final_path = model_dir / f"dqn_{config.profile_name}.pth"
    agent.save_model(final_path, {
        'final_episode': episode,
        'final_avg_score': metrics.get_recent_average('scores', 100),
        'total_training_time': time.time() - training_start,
        'total_episodes': len(metrics.scores)
    })
    

    # Save metrics
    metrics_path = model_dir / f"dqn_{config.profile_name}_metrics.json"
    metrics.save_metrics(str(metrics_path))
    
    # Generate training plots
    plot_training_curves(metrics, config.profile_name, str(model_dir))
    
    # Training report
    report = {
        "profile": config.profile_name,
        "episodes": len(metrics.scores),
        "final_avg_score": float(metrics.get_recent_average('scores', 100)),
        "best_score": int(max(metrics.scores)) if metrics.scores else 0,
        "final_epsilon": agent.epsilon,
        "training_time": time.time() - training_start,
        "config": config.__dict__
    }
    
    report_path = model_dir / f"dqn_{config.profile_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"✅ DQN training complete!")
    print(f"📁 Final model: {final_path}")
    if best_path:  # FIXED: Check if best_path exists
        print(f"📁 Best model: {best_path}")
    
    print(f"📊 Report: {report_path}")

def plot_training_curves(metrics: TrainingMetrics, profile_name: str, save_dir: str):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Scores
    axes[0,0].plot(metrics.scores)
    axes[0,0].set_title('Episode Scores')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Score')
    axes[0,0].grid(True)
    
    # Running average scores
    window = 100
    if len(metrics.scores) >= window:
        running_avg = [np.mean(metrics.scores[max(0, i-window):i+1]) for i in range(len(metrics.scores))]
        axes[0,1].plot(running_avg)
        axes[0,1].set_title(f'Running Average Scores (window={window})')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Average Score')
        axes[0,1].grid(True)
    
    # Training loss
    axes[1,0].plot(metrics.losses)
    axes[1,0].set_title('Training Loss')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].grid(True)
    
    # Epsilon decay
    axes[1,1].plot(metrics.epsilons)
    axes[1,1].set_title('Epsilon Decay')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Epsilon')
    axes[1,1].grid(True)
    
    plt.suptitle(f'DQN Training Curves - {profile_name}')
    plt.tight_layout()
    
    plot_path = Path(save_dir) / f"dqn_training_curves_{profile_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved: {plot_path}")

if __name__ == "__main__":
    # Train different DQN profiles
    profiles = {
        "aggressive": DQNConfig(
            profile_name="aggressive",
            learning_rate=0.001,
            epsilon_start=1.0,
            epsilon_decay=0.99,
            max_episodes=1500,
            target_score=12
        ),
        "balanced": DQNConfig(
            profile_name="balanced",
            learning_rate=0.0005,
            epsilon_start=0.8,
            epsilon_decay=0.995,
            max_episodes=2000,
            target_score=15
        ),
        "conservative": DQNConfig(
            profile_name="conservative",
            learning_rate=0.0003,
            epsilon_start=0.5,
            epsilon_decay=0.997,
            max_episodes=2500,
            target_score=18
        )
    }
    
    for name, config in profiles.items():
        print(f"\n{'='*60}")
        print(f"🚀 Training DQN {name.upper()} model")
        print(f"{'='*60}")
        train_dqn(config)