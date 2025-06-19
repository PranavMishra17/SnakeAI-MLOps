#!/usr/bin/env python3
"""
Fixed PPO (Proximal Policy Optimization) trainer for SnakeAI-MLOps
Consistent 8D state representation with configurable grid sizes
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

from neural_network_utils import (
    verify_gpu, create_directories, save_model,
    TrainingMetrics
)

@dataclass
class PPOConfig:
    """Fixed PPO training configuration"""
    profile_name: str = "balanced"
    max_episodes: int = 1500
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    
    # PPO specific parameters
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.02
    value_coeff: float = 0.5
    
    # Training parameters
    update_epochs: int = 4  # Reduced for stability
    batch_size: int = 32
    trajectory_length: int = 128  # Reduced for smaller grids
    
    # Network architecture - simplified
    hidden_size: int = 64  # Reduced from 256
    
    # Training settings with configurable grid
    grid_size: int = 10  # Configurable grid size
    device: str = "cuda"
    checkpoint_interval: int = 150
    target_score: int = 8

class SimplePolicyNetwork(nn.Module):
    """Simple policy network for discrete actions"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimplePolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights properly
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class SimpleValueNetwork(nn.Module):
    """Simple value network for baseline"""
    
    def __init__(self, input_size, hidden_size):
        super(SimpleValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)

class SnakeEnvironmentPPO:
    """Fixed Snake environment using 8D state representation"""
    
    def __init__(self, grid_size=10, device='cuda'):
        self.grid_size = grid_size
        self.device = torch.device(device)
        print(f"✅ Snake Environment PPO: {grid_size}x{grid_size} grid")
        self.reset()
    
    def reset(self):
        """Reset environment and return initial state"""
        center = self.grid_size // 2
        self.snake = [(center, center), (center, center-1)]
        self.direction = 3  # RIGHT
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        self.prev_distance = self._get_food_distance()
        self.steps_without_food = 0
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
        """Get 8D state representation (consistent with Q-Learning)"""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Direction vectors
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        current_dir = directions[self.direction]
        
        # Check dangers in all directions
        danger_straight = self._is_collision((head_x + current_dir[0], head_y + current_dir[1]))
        
        # Left and right relative to current direction
        left_dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # LEFT of UP,DOWN,LEFT,RIGHT
        left_dir = left_dirs[self.direction]
        danger_left = self._is_collision((head_x + left_dir[0], head_y + left_dir[1]))
        
        right_dirs = [(1, 0), (-1, 0), (0, -1), (0, 1)]  # RIGHT of UP,DOWN,LEFT,RIGHT
        right_dir = right_dirs[self.direction]
        danger_right = self._is_collision((head_x + right_dir[0], head_y + right_dir[1]))
        
        # Food direction relative to head
        food_left = food_x < head_x
        food_right = food_x > head_x
        food_up = food_y < head_y
        food_down = food_y > head_y
        
        # Create 8D state vector (same as Q-Learning)
        state = torch.tensor([
            float(danger_straight),
            float(danger_left), 
            float(danger_right),
            float(self.direction / 3.0),  # normalized direction
            float(food_left),
            float(food_right),
            float(food_up),
            float(food_down),
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
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = directions[action]
        new_head = (head_x + dx, head_y + dy)
        
        # Check collision
        if self._is_collision(new_head):
            return self._get_state(), -10.0, True
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Calculate reward
        reward = 0
        
        # Check if food eaten
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            reward = 10.0
            self.steps_without_food = 0
        else:
            self.snake.pop()
            self.steps_without_food += 1
            
            # Distance-based reward (same as other models)
            current_distance = self._get_food_distance()
            if current_distance < self.prev_distance:
                reward = 1.0
            else:
                reward = -1.0
            
            self.prev_distance = current_distance
        
        self.steps += 1
        
        # Scale episode length with grid size
        max_steps = self.grid_size * 50
        done = (self.steps >= max_steps or self.steps_without_food > max_steps // 5)
        
        return self._get_state(), reward, done

class PPOAgent:
    """Fixed PPO agent with consistent state handling"""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks using 8D input
        self.policy_network = SimplePolicyNetwork(8, config.hidden_size, 4).to(self.device)
        self.value_network = SimpleValueNetwork(8, config.hidden_size).to(self.device)
        
        # Shared optimizer
        self.optimizer = optim.Adam(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            lr=config.learning_rate
        )
        
        # Trajectory storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        print(f"✅ Fixed PPO Agent initialized on {self.device}")
        print(f"   Policy Network: {sum(p.numel() for p in self.policy_network.parameters())} parameters")
        print(f"   Value Network: {sum(p.numel() for p in self.value_network.parameters())} parameters")
        print(f"   Input: 8D state (consistent with Q-Learning)")
        print(f"   Grid: {config.grid_size}x{config.grid_size}")
    
    def get_action_and_value(self, state):
        """Get action and value for given state"""
        with torch.no_grad():
            action_probs = self.policy_network(state.unsqueeze(0))
            value = self.value_network(state.unsqueeze(0))
            
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def get_action(self, state):
        """Get action for evaluation (greedy)"""
        with torch.no_grad():
            action_probs = self.policy_network(state.unsqueeze(0))
            return torch.argmax(action_probs).item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in trajectory buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, rewards, values, dones, last_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        returns = []
        gae = 0
        
        values = values + [last_value]
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.discount_factor * next_value * next_non_terminal - values[t]
            gae = delta + self.config.discount_factor * 0.95 * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self):
        """PPO update using collected trajectory"""
        if len(self.states) < self.config.batch_size:
            return 0.0, 0.0, 0.0
        
        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, device=self.device, dtype=torch.long)
        old_log_probs = torch.tensor(self.log_probs, device=self.device)
        
        # Get last value for GAE
        with torch.no_grad():
            if self.dones[-1]:
                last_value = 0.0
            else:
                last_value = self.value_network(states[-1].unsqueeze(0)).item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(self.rewards, self.values, self.dones, last_value)
        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO updates
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # Create mini-batches
        batch_size = self.config.batch_size
        indices = torch.randperm(len(states))
        
        for epoch in range(self.config.update_epochs):
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                if len(batch_indices) < 8:
                    continue
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                action_probs = self.policy_network(batch_states)
                action_dist = torch.distributions.Categorical(action_probs)
                new_log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy().mean()
                
                values = self.value_network(batch_states)
                
                # PPO policy loss with clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Total loss
                total_loss = (policy_loss + 
                             self.config.value_coeff * value_loss - 
                             self.config.entropy_coeff * entropy)
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_network.parameters()) + list(self.value_network.parameters()), 
                    0.5
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Clear trajectory
        self.clear_trajectory()
        
        num_updates = self.config.update_epochs * max(1, (len(states) // batch_size))
        return (total_policy_loss / num_updates, 
                total_value_loss / num_updates, 
                total_entropy / num_updates)
    
    def clear_trajectory(self):
        """Clear stored trajectory"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def save_model(self, filepath, metadata=None):
        """Save PPO model"""
        model_data = {
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'metadata': metadata or {}
        }
        
        torch.save(model_data, filepath)
        print(f"✅ Fixed PPO model saved: {filepath}")

def train_ppo(config: PPOConfig):
    """Main PPO training loop with improvements"""
    device = verify_gpu()
    config.device = str(device)
    
    # Setup directories
    base_dir = Path("models")
    model_dir = base_dir / "ppo"
    checkpoint_dir = base_dir / "checkpoints" / "ppo"
    create_directories(str(base_dir))
    
    # Initialize environment and agent
    env = SnakeEnvironmentPPO(grid_size=config.grid_size, device=str(device))
    agent = PPOAgent(config)
    metrics = TrainingMetrics()
    
    print(f"🚀 Starting Fixed PPO training: {config.profile_name}")
    print(f"   Grid: {config.grid_size}x{config.grid_size}")
    print(f"   Target score: {config.target_score}")
    print(f"   Max episodes: {config.max_episodes}")
    
    best_score = 0
    best_path = None
    training_start = time.time()
    recent_scores = deque(maxlen=100)
    
    episode = 0
    
    pbar = tqdm(total=config.max_episodes, desc="Training Fixed PPO")
    
    while episode < config.max_episodes:
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        # Collect trajectory
        while True:
            action, log_prob, value = agent.get_action_and_value(state)
            next_state, reward, done = env.step(action)
            
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            # Update when we have enough data or episode ends
            if len(agent.states) >= config.trajectory_length or done:
                policy_loss, value_loss, entropy = agent.update()
                break
        
        # Metrics
        recent_scores.append(env.score)
        metrics.add_episode(env.score, policy_loss, entropy, steps, episode_reward)
        
        episode += 1
        pbar.update(1)
        
        # Progress logging
        if episode % 100 == 0:
            avg_score = np.mean(recent_scores) if recent_scores else 0
            print(f"Episode {episode}: Avg Score: {avg_score:.2f}, "
                  f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
            
            # Save checkpoint
            if episode % config.checkpoint_interval == 0 and episode > 0:
                checkpoint_path = checkpoint_dir / f"ppo_{config.profile_name}_ep{episode}.pth"
                agent.save_model(checkpoint_path, {
                    'episode': episode,
                    'avg_score': avg_score,
                    'training_time': time.time() - training_start
                })
        
        # Save best model
        if env.score > best_score:
            best_score = env.score
            best_path = model_dir / f"ppo_{config.profile_name}_best.pth"
            agent.save_model(best_path, {
                'episode': episode,
                'best_score': best_score,
                'training_time': time.time() - training_start
            })
        
    
    pbar.close()
    
    # Save final model
    final_path = model_dir / f"ppo_{config.profile_name}.pth"
    agent.save_model(final_path, {
        'final_episode': episode,
        'final_avg_score': np.mean(recent_scores) if recent_scores else 0,
        'total_training_time': time.time() - training_start,
        'total_episodes': len(metrics.scores)
    })
    
    # Save metrics
    metrics_path = model_dir / f"ppo_{config.profile_name}_metrics.json"
    metrics.save_metrics(str(metrics_path))
    
    # Generate plots
    plot_training_curves(metrics, config.profile_name, str(model_dir))
    
    # Training report
    report = {
        "profile": config.profile_name,
        "episodes": len(metrics.scores),
        "final_avg_score": float(np.mean(recent_scores)) if recent_scores else 0,
        "best_score": int(max(metrics.scores)) if metrics.scores else 0,
        "training_time": time.time() - training_start,
        "grid_size": config.grid_size,
        "config": config.__dict__
    }
    
    report_path = model_dir / f"ppo_{config.profile_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"✅ Fixed PPO training complete!")
    print(f"📁 Final model: {final_path}")
    if best_path:
        print(f"📁 Best model: {best_path}")

def plot_training_curves(metrics: TrainingMetrics, profile_name: str, save_dir: str):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Scores
    axes[0,0].plot(metrics.scores)
    axes[0,0].set_title('Episode Scores')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Score')
    axes[0,0].grid(True)
    
    # Running average
    window = 100
    if len(metrics.scores) >= window:
        running_avg = [np.mean(metrics.scores[max(0, i-window):i+1]) for i in range(len(metrics.scores))]
        axes[0,1].plot(running_avg)
        axes[0,1].set_title(f'Running Average Scores (window={window})')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Average Score')
        axes[0,1].grid(True)
    
    # Policy loss
    axes[1,0].plot(metrics.losses)
    axes[1,0].set_title('Policy Loss')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].grid(True)
    
    # Episode lengths
    axes[1,1].plot(metrics.episode_lengths)
    axes[1,1].set_title('Episode Lengths')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Steps')
    axes[1,1].grid(True)
    
    plt.suptitle(f'Fixed PPO Training Curves - {profile_name}')
    plt.tight_layout()
    
    plot_path = Path(save_dir) / f"ppo_training_curves_{profile_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved: {plot_path}")

if __name__ == "__main__":
    # Train fixed PPO profiles with varying grid sizes
    profiles = {
        "aggressive": PPOConfig(
            profile_name="aggressive",
            learning_rate=0.002,
            max_episodes=1200,
            target_score=6,
            hidden_size=64,
            clip_epsilon=0.2,
            entropy_coeff=0.03,
            grid_size=8,  # Small grid
            trajectory_length=64
        ),
        "balanced": PPOConfig(
            profile_name="balanced",
            learning_rate=0.001,
            max_episodes=1500,
            target_score=8,
            hidden_size=64,
            clip_epsilon=0.2,
            entropy_coeff=0.02,
            grid_size=10,  # Medium grid
            trajectory_length=128
        ),
        "conservative": PPOConfig(
            profile_name="conservative",
            learning_rate=0.0005,
            max_episodes=2000,
            target_score=10,
            hidden_size=64,
            clip_epsilon=0.15,
            entropy_coeff=0.01,
            grid_size=12,  # Larger grid
            trajectory_length=256
        )
    }
    
    for name, config in profiles.items():
        print(f"\n{'='*60}")
        print(f"🚀 Training Fixed PPO {name.upper()} model")
        print(f"{'='*60}")
        train_ppo(config)

"""
USAGE EXAMPLES:
===============

# Import and train PPO model
from ppo_trainer import train_ppo, PPOConfig

# Quick training with default settings
config = PPOConfig(profile_name="test", max_episodes=1000)
train_ppo(config)

# Custom configuration for small grid
config = PPOConfig(
    profile_name="custom",
    learning_rate=0.002,
    grid_size=8,           # Small grid for faster learning
    max_episodes=1200,
    target_score=6,
    hidden_size=64,        # Simple architecture
    clip_epsilon=0.2,
    entropy_coeff=0.025,
    trajectory_length=64   # Shorter trajectories for small grid
)
train_ppo(config)

# Large grid configuration
config = PPOConfig(
    profile_name="challenge",
    grid_size=15,          # Large grid
    max_episodes=2500,
    target_score=12,
    hidden_size=128,
    trajectory_length=256
)
train_ppo(config)

# Load and evaluate model
checkpoint = torch.load("models/ppo/ppo_balanced.pth")
policy_state = checkpoint['policy_network']
# Model loading handled by evaluator

# Train from command line
python ppo_trainer.py

# Key improvements:
# - 8D state representation (consistent with Q-Learning)
# - Configurable grid sizes (8x8 to 15x15)
# - Simplified network architecture for better performance
# - Proper PPO clipping and GAE computation
# - Trajectory-based learning with configurable lengths

# Performance expectations:
# - 8x8 grid: ~6-8 average score, fast convergence
# - 10x10 grid: ~8-12 average score, good balance  
# - 12x12+ grid: ~10-15+ average score, more challenging

# Model file structure:
# - policy_network: Policy network weights
# - value_network: Value network weights
# - config: Training configuration
# - metadata: Additional training info

# PPO advantages:
# - More stable than policy gradient
# - Better sample efficiency than vanilla PG
# - Clipping prevents large policy updates
# - Works well with continuous learning
"""