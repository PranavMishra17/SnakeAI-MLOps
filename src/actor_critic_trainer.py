#!/usr/bin/env python3
"""
Fixed Actor-Critic trainer for SnakeAI-MLOps with smaller grid support
Improved performance with consistent 8D state representation
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
class ActorCriticConfig:
    """Fixed Actor-Critic training configuration"""
    profile_name: str = "balanced"
    max_episodes: int = 2000  # Reduced for better performance
    actor_lr: float = 0.001
    critic_lr: float = 0.002
    discount_factor: float = 0.99
    
    # Simplified A2C features
    entropy_coeff: float = 0.01
    entropy_decay: float = 0.999
    min_entropy_coeff: float = 0.001
    value_coeff: float = 0.5
    max_grad_norm: float = 1.0
    
    # Simplified architecture
    hidden_size: int = 64  # Reduced from 128
    
    # Training settings with smaller grid
    grid_size: int = 10  # Configurable grid size
    device: str = "cuda"
    checkpoint_interval: int = 200
    target_score: int = 8  # More realistic target

class SimplePolicyNetwork(nn.Module):
    """Simplified policy network for better performance"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimplePolicyNetwork, self).__init__()
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
        x = self.fc3(x)
        # FIXED: Add numerical stability
        x = torch.clamp(x, -10, 10)  # Prevent extreme values
        return F.softmax(x, dim=-1)
    

class SimpleValueNetwork(nn.Module):
    """Simplified value network"""
    
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

class SnakeEnvironmentAC:
    """Fixed Snake environment using 8D state representation"""
    
    def __init__(self, grid_size=10, device='cuda'):
        self.grid_size = grid_size
        self.device = torch.device(device)
        print(f"✅ Snake Environment AC: {grid_size}x{grid_size} grid")
        self.reset()
    
    def reset(self):
        """Reset environment and return initial state"""
        center = self.grid_size // 2
        self.snake = [(center, center), (center, center-1)]
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
        """Get 8D state representation (consistent with Q-Learning)"""
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
        
        # Create 8D state vector (same as Q-Learning)
        state = torch.tensor([
            float(danger_straight), float(danger_left), float(danger_right),
            float(self.direction / 3.0),  # normalized direction
            float(food_left), float(food_right), float(food_up), float(food_down)
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
            reward = 1.0 if new_dist < old_dist else -1.0
        
        self.steps += 1
        # Scale max steps with grid size
        max_steps = self.grid_size * 50
        done = self.steps >= max_steps
        
        return self._get_state(), reward, done

class SimpleActorCriticAgent:
    """Simplified A2C agent for better performance"""
    
    def __init__(self, config: ActorCriticConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks using 8D input
        self.actor = SimplePolicyNetwork(8, config.hidden_size, 4).to(self.device)
        self.critic = SimpleValueNetwork(8, config.hidden_size).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        # Storage for trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Current entropy coefficient
        self.entropy_coeff = config.entropy_coeff
        
        print(f"✅ Simple Actor-Critic Agent initialized on {self.device}")
        print(f"   Actor: {sum(p.numel() for p in self.actor.parameters())} parameters")
        print(f"   Critic: {sum(p.numel() for p in self.critic.parameters())} parameters")
        print(f"   Input: 8D state (consistent with Q-Learning)")
        print(f"   Grid: {config.grid_size}x{config.grid_size}")
    
    def get_action_and_value(self, state):
        """Get action from actor and value from critic"""
        state_batch = state.unsqueeze(0)
        
        # Actor
        action_probs = self.actor(state_batch)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        # Critic
        value = self.critic(state_batch)
        
        return action.item(), log_prob.item(), value.item()
    
    def get_action(self, state):
        """Get action for evaluation (no gradients)"""
        with torch.no_grad():
            action_probs = self.actor(state.unsqueeze(0))
            action = torch.argmax(action_probs).item()
            return action
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in trajectory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    # FIX 2: Safe advantage normalization
    def compute_returns_and_advantages(self, last_value):
        returns = []
        advantages = []
        
        R = last_value
        for t in reversed(range(len(self.rewards))):
            R = self.rewards[t] + self.config.discount_factor * R * (1 - self.dones[t])
            returns.insert(0, R)
            advantages.insert(0, R - self.values[t])
        
        return (
            torch.tensor(returns, dtype=torch.float32, device=self.device),
            torch.tensor(advantages, dtype=torch.float32, device=self.device)
        )
    

    # FIX 3: Safe training step
    def train_on_trajectory(self):
        """FIXED: Safe training with numerical stability"""
        if len(self.states) == 0:
            return 0.0, 0.0, 0.0
        
        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, device=self.device, dtype=torch.long)
        log_probs = torch.tensor(self.log_probs, device=self.device)
        
        # Compute last value
        with torch.no_grad():
            if self.dones[-1]:
                last_value = 0
            else:
                last_value = self.critic(states[-1].unsqueeze(0)).item()
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(last_value)
        
        # FIXED: Safe advantage normalization
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()  # No std for single sample
        
        # Recompute with clamping
        action_probs = self.actor(states)
        # FIXED: Check for NaN and clamp
        action_probs = torch.clamp(action_probs, 1e-8, 1.0)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        action_dist = torch.distributions.Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        new_values = self.critic(states)
        
        # Actor loss
        actor_loss = -(new_log_probs * advantages.detach()).mean()
        
        # Entropy bonus
        entropy_loss = -self.entropy_coeff * entropy
        
        # Critic loss
        critic_loss = F.mse_loss(new_values, returns.detach())
        
        # Combined actor loss
        total_actor_loss = actor_loss + entropy_loss
        
        # FIXED: Check for NaN before backward
        if torch.isnan(total_actor_loss) or torch.isnan(critic_loss):
            print("⚠️ NaN detected, skipping update")
            self.clear_trajectory()
            return 0.0, 0.0, 0.0
        
        # Backward pass
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()
        
        # Clear trajectory
        self.clear_trajectory()
        
        return actor_loss.item(), critic_loss.item(), entropy.item()

    def clear_trajectory(self):
        """Clear stored trajectory"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def update_entropy_coeff(self):
        """Decay entropy coefficient"""
        self.entropy_coeff = max(
            self.config.min_entropy_coeff,
            self.entropy_coeff * self.config.entropy_decay
        )
    
    def save_model(self, filepath, metadata=None):
        """Save Actor-Critic model"""
        model_data = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config.__dict__,
            'entropy_coeff': self.entropy_coeff,
            'metadata': metadata or {}
        }
        
        torch.save(model_data, filepath)
        print(f"✅ Fixed Actor-Critic model saved: {filepath}")

def train_actor_critic(config: ActorCriticConfig):
    """Main Actor-Critic training loop with improvements"""
    device = verify_gpu()
    config.device = str(device)
    
    # Setup directories
    base_dir = Path("models")
    model_dir = base_dir / "actor_critic"
    checkpoint_dir = base_dir / "checkpoints" / "actor_critic"
    create_directories(str(base_dir))
    
    # Initialize environment and agent
    env = SnakeEnvironmentAC(grid_size=config.grid_size, device=str(device))
    agent = SimpleActorCriticAgent(config)
    metrics = TrainingMetrics()
    
    print(f"🚀 Starting Fixed Actor-Critic training: {config.profile_name}")
    print(f"   Grid: {config.grid_size}x{config.grid_size}")
    print(f"   Target score: {config.target_score}")
    print(f"   Max episodes: {config.max_episodes}")
    
    best_score = 0
    best_path = None
    training_start = time.time()
    recent_scores = deque(maxlen=100)
    
    for episode in tqdm(range(config.max_episodes), desc="Training Fixed Actor-Critic"):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Get action and value
            action, log_prob, value = agent.get_action_and_value(state)
            next_state, reward, done = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Train on collected trajectory
        actor_loss, critic_loss, entropy = agent.train_on_trajectory()
        
        # Update parameters
        agent.update_entropy_coeff()
        
        # Metrics
        recent_scores.append(env.score)
        metrics.add_episode(env.score, actor_loss + critic_loss, agent.entropy_coeff, steps, total_reward)
        
        # Progress logging
        if episode % 100 == 0:
            avg_score = np.mean(recent_scores) if recent_scores else 0
            
            print(f"Episode {episode}: Avg Score: {avg_score:.2f}, "
                  f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, "
                  f"Entropy: {agent.entropy_coeff:.4f}")
            
            # Save checkpoint
            if episode % config.checkpoint_interval == 0 and episode > 0:
                checkpoint_path = checkpoint_dir / f"ac_{config.profile_name}_ep{episode}.pth"
                agent.save_model(checkpoint_path, {
                    'episode': episode,
                    'avg_score': avg_score,
                    'training_time': time.time() - training_start
                })
        
        # Save best model
        if env.score > best_score:
            best_score = env.score
            best_path = model_dir / f"ac_{config.profile_name}_best.pth"
            agent.save_model(best_path, {
                'episode': episode,
                'best_score': best_score,
                'training_time': time.time() - training_start
            })

    
    # Save final model
    final_path = model_dir / f"ac_{config.profile_name}.pth"
    agent.save_model(final_path, {
        'final_episode': episode,
        'final_avg_score': np.mean(recent_scores) if recent_scores else 0,
        'total_training_time': time.time() - training_start,
        'total_episodes': len(metrics.scores)
    })
    
    # Save metrics
    metrics_path = model_dir / f"ac_{config.profile_name}_metrics.json"
    metrics.save_metrics(str(metrics_path))
    
    # Generate plots
    plot_training_curves(metrics, config.profile_name, str(model_dir))
    
    # Training report
    report = {
        "profile": config.profile_name,
        "episodes": len(metrics.scores),
        "final_avg_score": float(np.mean(recent_scores)) if recent_scores else 0,
        "best_score": int(max(metrics.scores)) if metrics.scores else 0,
        "final_entropy_coeff": agent.entropy_coeff,
        "training_time": time.time() - training_start,
        "grid_size": config.grid_size,
        "config": config.__dict__
    }
    
    report_path = model_dir / f"ac_{config.profile_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"✅ Fixed Actor-Critic training complete!")
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
    
    # Running average
    window = 100
    if len(metrics.scores) >= window:
        running_avg = [np.mean(metrics.scores[max(0, i-window):i+1]) for i in range(len(metrics.scores))]
        axes[0,1].plot(running_avg)
        axes[0,1].set_title(f'Running Average Scores (window={window})')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Average Score')
        axes[0,1].grid(True)
    
    # Total loss
    axes[1,0].plot(metrics.losses)
    axes[1,0].set_title('Total Loss (Actor + Critic)')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].grid(True)
    
    # Episode lengths
    axes[1,1].plot(metrics.episode_lengths)
    axes[1,1].set_title('Episode Lengths')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Steps')
    axes[1,1].grid(True)
    
    plt.suptitle(f'Fixed Actor-Critic Training Curves - {profile_name}')
    plt.tight_layout()
    
    plot_path = Path(save_dir) / f"ac_training_curves_{profile_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved: {plot_path}")

if __name__ == "__main__":
    # Train different Actor-Critic profiles with varying grid sizes
    profiles = {
        "aggressive": ActorCriticConfig(
            profile_name="aggressive",
            actor_lr=0.002,
            critic_lr=0.004,
            entropy_coeff=0.02,
            max_episodes=1500,
            target_score=6,
            grid_size=8,  # Small grid
            hidden_size=64
        ),
        "balanced": ActorCriticConfig(
            profile_name="balanced",
            actor_lr=0.001,
            critic_lr=0.002,
            entropy_coeff=0.01,
            max_episodes=2000,
            target_score=8,
            grid_size=10,  # Medium grid
            hidden_size=64
        ),
        "conservative": ActorCriticConfig(
            profile_name="conservative",
            actor_lr=0.0005,
            critic_lr=0.001,
            entropy_coeff=0.005,
            max_episodes=2500,
            target_score=10,
            grid_size=12,  # Larger grid
            hidden_size=64
        )
    }
    
    for name, config in profiles.items():
        print(f"\n{'='*60}")
        print(f"🚀 Training Fixed Actor-Critic {name.upper()} model")
        print(f"{'='*60}")
        train_actor_critic(config)

"""
USAGE EXAMPLES:
===============

# Import and train Actor-Critic model
from actor_critic_trainer import train_actor_critic, ActorCriticConfig

# Quick training with default settings
config = ActorCriticConfig(profile_name="test", max_episodes=1000)
train_actor_critic(config)

# Custom configuration for small grid
config = ActorCriticConfig(
    profile_name="custom",
    actor_lr=0.002,
    critic_lr=0.003,
    grid_size=8,          # Small grid for faster learning
    max_episodes=1500,
    target_score=6,
    hidden_size=64,       # Simple architecture
    entropy_coeff=0.015
)
train_actor_critic(config)

# Large grid configuration
config = ActorCriticConfig(
    profile_name="challenge",
    grid_size=15,         # Large grid
    max_episodes=3000,
    target_score=12,
    hidden_size=128
)
train_actor_critic(config)

# Load and evaluate model
checkpoint = torch.load("models/actor_critic/ac_balanced.pth")
actor_state = checkpoint['actor_state_dict']
# Model loading handled by evaluator

# Train from command line
python actor_critic_trainer.py

# Key improvements:
# - 8D state representation (consistent with Q-Learning)
# - Configurable grid sizes (8x8 to 15x15)
# - Simplified network architecture
# - Better trajectory-based training
# - Proper actor-critic separation for evaluation

# Performance expectations:
# - 8x8 grid: ~6-8 average score, fast convergence
# - 10x10 grid: ~8-12 average score, good balance
# - 12x12+ grid: ~10-15+ average score, more challenging

# Model file structure:
# - actor_state_dict: Policy network weights
# - critic_state_dict: Value network weights  
# - config: Training configuration
# - metadata: Additional training info
"""