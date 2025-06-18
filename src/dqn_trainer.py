#!/usr/bin/env python3
"""
Fixed Deep Q-Network (DQN) trainer for SnakeAI-MLOps
Simplified and robust implementation based on proven fundamentals
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import time
from collections import deque
import random

from neural_network_utils import (
    verify_gpu, create_directories, save_model,
    TrainingMetrics
)

@dataclass
class DQNConfig:
    """Simplified DQN training configuration"""
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
    
    # Simplified features (removed complex ones that cause instability)
    double_dqn: bool = True
    dueling: bool = False  # Simplified for now
    
    # Network architecture - simpler
    hidden_size: int = 128
    
    # Training settings
    device: str = "cuda"
    checkpoint_interval: int = 200
    target_score: int = 10  # More realistic target
    
class SimpleReplayBuffer:
    """Simple experience replay buffer"""
    
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack(states),
            torch.tensor(actions, device=self.device, dtype=torch.long),
            torch.tensor(rewards, device=self.device, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, device=self.device, dtype=torch.bool)
        )
    
    def __len__(self):
        return len(self.buffer)

class SimpleDQN(nn.Module):
    """Simplified DQN Network"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SnakeEnvironmentDQN:
    """Fixed Snake environment for DQN training"""
    
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
        self.prev_distance = self._get_food_distance()
        return self._get_state()
    
    def _place_food(self):
        """Place food randomly avoiding snake"""
        while True:
            pos = (np.random.randint(0, self.grid_size), 
                   np.random.randint(0, self.grid_size))
            if pos not in self.snake:
                return pos
    
    def _get_food_distance(self):
        """Get Manhattan distance to food"""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return abs(head_x - food_x) + abs(head_y - food_y)
    
    def _get_state(self):
        """Get enhanced state representation"""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Direction vectors
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        
        # Current direction
        current_dir = directions[self.direction]
        
        # Check dangers in all directions
        danger_straight = self._is_collision((head_x + current_dir[0], head_y + current_dir[1]))
        danger_left = self._is_collision((head_x + directions[(self.direction - 1) % 4][0], 
                                         head_y + directions[(self.direction - 1) % 4][1]))
        danger_right = self._is_collision((head_x + directions[(self.direction + 1) % 4][0], 
                                          head_y + directions[(self.direction + 1) % 4][1]))
        
        # Food direction relative to head
        food_left = food_x < head_x
        food_right = food_x > head_x
        food_up = food_y < head_y
        food_down = food_y > head_y
        
        # Enhanced features
        food_distance = self._get_food_distance()
        normalized_distance = food_distance / (2 * self.grid_size)
        
        # Wall distances (normalized)
        wall_distances = [
            head_y / self.grid_size,  # distance to top
            (self.grid_size - 1 - head_y) / self.grid_size,  # distance to bottom
            head_x / self.grid_size,  # distance to left
            (self.grid_size - 1 - head_x) / self.grid_size,  # distance to right
        ]
        
        # Snake length and empty spaces
        snake_length = len(self.snake) / (self.grid_size * self.grid_size)
        empty_spaces = (self.grid_size * self.grid_size - len(self.snake) - 1) / (self.grid_size * self.grid_size)
        
        # Create state vector (11 features)
        state = torch.tensor([
            float(danger_straight),
            float(danger_left), 
            float(danger_right),
            float(self.direction / 3.0),  # normalized direction
            float(food_left),
            float(food_right),
            float(food_up),
            float(food_down),
            normalized_distance,
            snake_length,
            empty_spaces
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
        # Update direction
        self.direction = action
        head_x, head_y = self.snake[0]
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        dx, dy = directions[action]
        new_head = (head_x + dx, head_y + dy)
        
        # Check collision
        if self._is_collision(new_head):
            return self._get_state(), -100.0, True  # Large death penalty
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Calculate reward
        reward = 0
        
        # Check if food eaten
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            reward = 100.0  # Large food reward
        else:
            self.snake.pop()  # Remove tail if no food eaten
            
            # Distance-based reward (encouraging movement toward food)
            current_distance = self._get_food_distance()
            if current_distance < self.prev_distance:
                reward = 1.0  # Moving closer to food
            elif current_distance > self.prev_distance:
                reward = -1.0  # Moving away from food
            else:
                reward = -0.1  # Small penalty for not making progress
            
            self.prev_distance = current_distance
        
        self.steps += 1
        
        # Episode ends if too many steps without progress
        done = self.steps >= 1000
        
        return self._get_state(), reward, done

class SimpleDQNAgent:
    """Simplified DQN Agent focusing on fundamentals"""
    
    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks - simplified architecture
        self.q_network = SimpleDQN(11, config.hidden_size, 4).to(self.device)  # 11 input features
        self.target_network = SimpleDQN(11, config.hidden_size, 4).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Experience replay
        self.memory = SimpleReplayBuffer(config.memory_capacity, self.device)
        
        # Training state
        self.epsilon = config.epsilon_start
        self.steps_done = 0
        
        print(f"✅ Simplified DQN Agent initialized on {self.device}")
        print(f"   Network: {sum(p.numel() for p in self.q_network.parameters())} parameters")
        print(f"   Input features: 11")
    
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        
        with torch.no_grad():
            state_batch = state.unsqueeze(0)
            q_values = self.q_network(state_batch)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Single training step"""
        if len(self.memory) < self.config.min_memory_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (self.config.discount_factor * 
                                                      next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network (hard update)"""
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
        print(f"✅ Simplified DQN model saved: {filepath}")

def train_dqn(config: DQNConfig):
    """Main DQN training loop - simplified and robust"""
    device = verify_gpu()
    config.device = str(device)
    
    # Setup directories
    base_dir = Path("models")
    model_dir = base_dir / "dqn"
    checkpoint_dir = base_dir / "checkpoints" / "dqn"
    create_directories(str(base_dir))
    
    # Initialize environment and agent
    env = SnakeEnvironmentDQN(device=str(device))
    agent = SimpleDQNAgent(config)
    metrics = TrainingMetrics()
    
    print(f"🚀 Starting Simplified DQN training: {config.profile_name}")
    print(f"   Target score: {config.target_score}")
    print(f"   Max episodes: {config.max_episodes}")
    
    best_score = 0
    best_path = None
    training_start = time.time()
    recent_scores = deque(maxlen=100)
    
    for episode in tqdm(range(config.max_episodes), desc="Training DQN"):
        state = env.reset()
        total_reward = 0
        episode_loss = 0
        steps = 0
        train_steps = 0
        
        while True:
            action = agent.get_action(state, training=True)
            next_state, reward, done = env.step(action)
            
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train every step once we have enough experience
            if len(agent.memory) >= config.min_memory_size:
                loss = agent.train_step()
                episode_loss += loss
                train_steps += 1
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Update target network periodically
        if episode % config.target_update_freq == 0:
            agent.update_target_network()
        
        # Update parameters
        agent.decay_epsilon()
        
        # Metrics
        recent_scores.append(env.score)
        avg_loss = episode_loss / max(train_steps, 1)
        metrics.add_episode(env.score, avg_loss, agent.epsilon, steps, total_reward)
        
        # Progress logging
        if episode % 100 == 0:
            avg_score = np.mean(recent_scores) if recent_scores else 0
            print(f"Episode {episode}: Avg Score: {avg_score:.2f}, Loss: {avg_loss:.4f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Memory: {len(agent.memory)}")
            
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
        if len(recent_scores) >= 100 and np.mean(recent_scores) >= config.target_score:
            print(f"✅ Target score reached at episode {episode}")
            break
    
    # Save final model
    final_path = model_dir / f"dqn_{config.profile_name}.pth"
    agent.save_model(final_path, {
        'final_episode': episode,
        'final_avg_score': np.mean(recent_scores) if recent_scores else 0,
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
        "final_avg_score": float(np.mean(recent_scores)) if recent_scores else 0,
        "best_score": int(max(metrics.scores)) if metrics.scores else 0,
        "final_epsilon": agent.epsilon,
        "training_time": time.time() - training_start,
        "config": config.__dict__
    }
    
    report_path = model_dir / f"dqn_{config.profile_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"✅ Simplified DQN training complete!")
    print(f"📁 Final model: {final_path}")
    if best_path:
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
    
    plt.suptitle(f'Simplified DQN Training Curves - {profile_name}')
    plt.tight_layout()
    
    plot_path = Path(save_dir) / f"dqn_training_curves_{profile_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved: {plot_path}")

if __name__ == "__main__":
    # Train different DQN profiles with simplified approach
    profiles = {
        "aggressive": DQNConfig(
            profile_name="aggressive",
            learning_rate=0.001,
            epsilon_start=1.0,
            epsilon_decay=0.99,
            max_episodes=1500,
            target_score=8,
            hidden_size=64
        ),
        "balanced": DQNConfig(
            profile_name="balanced",
            learning_rate=0.0005,
            epsilon_start=0.9,
            epsilon_decay=0.995,
            max_episodes=2000,
            target_score=10,
            hidden_size=128
        ),
        "conservative": DQNConfig(
            profile_name="conservative",
            learning_rate=0.0003,
            epsilon_start=0.7,
            epsilon_decay=0.997,
            max_episodes=2500,
            target_score=12,
            hidden_size=256
        )
    }
    
    for name, config in profiles.items():
        print(f"\n{'='*60}")
        print(f"🚀 Training Simplified DQN {name.upper()} model")
        print(f"{'='*60}")
        train_dqn(config)