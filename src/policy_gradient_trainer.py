#!/usr/bin/env python3
"""
Policy Gradient (REINFORCE) trainer for SnakeAI-MLOps
GPU-accelerated implementation with baseline variance reduction
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
    PolicyNetwork, ValueNetwork, NetworkConfig,
    verify_gpu, create_directories, save_model,
    TrainingMetrics
)

@dataclass
class PolicyGradientConfig:
    """Policy Gradient training configuration"""
    profile_name: str = "balanced"
    max_episodes: int = 3000
    learning_rate: float = 0.001
    baseline_lr: float = 0.005
    discount_factor: float = 0.99
    
    # Policy gradient specific
    use_baseline: bool = True
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    
    # Network architecture
    hidden_layers: list = None
    
    # Training settings
    device: str = "cuda"
    checkpoint_interval: int = 300
    target_score: int = 12
    batch_episodes: int = 1  # Train after each episode
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64]

class SnakeEnvironmentPG:
    """Snake environment for Policy Gradient training"""
    
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
        
        # Basic danger detection
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        current_dir = directions[self.direction]
        
        # Check straight, left, right
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
        
        # Enhanced 20D state
        state = torch.zeros(20, dtype=torch.float32, device=self.device)
        
        # Basic features (8D)
        state[0] = float(danger_straight)
        state[1] = float(danger_left)
        state[2] = float(danger_right)
        state[3] = float(self.direction / 3.0)
        state[4] = float(food_left)
        state[5] = float(food_right)
        state[6] = float(food_up)
        state[7] = float(food_down)
        
        # Enhanced features
        food_distance = abs(head_x - food_x) + abs(head_y - food_y)
        state[8] = food_distance / (2 * self.grid_size)
        
        # Wall distances
        state[9] = head_y / self.grid_size
        state[10] = (self.grid_size - 1 - head_y) / self.grid_size
        state[11] = head_x / self.grid_size
        state[12] = (self.grid_size - 1 - head_x) / self.grid_size
        
        # Body density in quadrants
        quadrant_counts = [0, 0, 0, 0]
        half_size = self.grid_size // 2
        for seg_x, seg_y in self.snake[1:]:
            quadrant = 0
            if seg_x >= half_size:
                quadrant += 1
            if seg_y >= half_size:
                quadrant += 2
            quadrant_counts[quadrant] += 1
        
        quadrant_size = half_size * half_size
        for i in range(4):
            state[13 + i] = quadrant_counts[i] / quadrant_size
        
        # Additional features
        state[17] = len(self.snake) / (self.grid_size * self.grid_size)
        empty_spaces = self.grid_size * self.grid_size - len(self.snake) - 1
        state[18] = empty_spaces / (self.grid_size * self.grid_size)
        state[19] = (food_distance + len(self.snake) * 0.1) / (2 * self.grid_size + 10)
        
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
            reward = 0.5 if new_dist < old_dist else -0.1
        
        self.steps += 1
        done = self.steps >= 1000
        
        return self._get_enhanced_state(), reward, done

class PolicyGradientAgent:
    """REINFORCE agent with optional baseline"""
    
    def __init__(self, config: PolicyGradientConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        net_config = NetworkConfig(
            input_size=20,
            hidden_layers=config.hidden_layers,
            output_size=4,
            dropout=0.1
        )
        
        self.policy_network = PolicyNetwork(net_config).to(self.device)
        
        if config.use_baseline:
            self.value_network = ValueNetwork(net_config).to(self.device)
            self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=config.baseline_lr)
        
        # Optimizer
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)
        
        # Episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
        print(f"✅ Policy Gradient Agent initialized on {self.device}")
        print(f"   Policy Network: {sum(p.numel() for p in self.policy_network.parameters())} parameters")
        if config.use_baseline:
            print(f"   Value Network: {sum(p.numel() for p in self.value_network.parameters())} parameters")
    
    def get_action(self, state):
        """Sample action from policy"""
        with torch.no_grad():
            action_probs = self.policy_network(state.unsqueeze(0))
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item()
    
    def store_transition(self, state, action, reward, log_prob):
        """Store transition for episode"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_log_probs.append(log_prob)
    
    def compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.config.discount_factor * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)
    
    def train_episode(self):
        """Train on collected episode data"""
        if not self.episode_rewards:
            return 0.0, 0.0
        
        # Compute returns
        returns = self.compute_returns(self.episode_rewards)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Convert to tensors
        states = torch.stack(self.episode_states)
        log_probs = torch.tensor(self.episode_log_probs, device=self.device)
        
        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0
        
        # Compute baseline if used
        if self.config.use_baseline:
            values = self.value_network(states).squeeze()
            advantages = returns - values.detach()
            
            # Value network loss
            value_loss = F.mse_loss(values, returns.detach())
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        else:
            advantages = returns
        
        # Policy network loss
        policy_loss = -(log_probs * advantages).mean()
        
        # Entropy regularization
        action_probs = self.policy_network(states)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -self.config.entropy_coeff * entropy
        
        total_loss = policy_loss + entropy_loss
        
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Clear episode data
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
        
        return policy_loss.item(), value_loss if self.config.use_baseline else 0.0
    
    def save_model(self, filepath, metadata=None):
        """Save policy gradient model"""
        model_data = {
            'policy_network': self.policy_network.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'config': self.config.__dict__,
            'metadata': metadata or {}
        }
        
        if self.config.use_baseline:
            model_data['value_network'] = self.value_network.state_dict()
            model_data['value_optimizer'] = self.value_optimizer.state_dict()
        
        torch.save(model_data, filepath)
        print(f"✅ Policy Gradient model saved: {filepath}")

def train_policy_gradient(config: PolicyGradientConfig):
    """Main Policy Gradient training loop"""
    device = verify_gpu()
    config.device = str(device)
    
    # Setup directories
    base_dir = Path("models")
    model_dir = base_dir / "policy_gradient"
    checkpoint_dir = base_dir / "checkpoints" / "policy_gradient"
    create_directories(str(base_dir))
    
    # Initialize environment and agent
    env = SnakeEnvironmentPG(device=str(device))
    agent = PolicyGradientAgent(config)
    metrics = TrainingMetrics()
    
    print(f"🚀 Starting Policy Gradient training: {config.profile_name}")
    print(f"   Target score: {config.target_score}")
    print(f"   Max episodes: {config.max_episodes}")
    print(f"   Using baseline: {config.use_baseline}")
    
    best_score = 0
    training_start = time.time()
    recent_scores = deque(maxlen=100)
    
    for episode in tqdm(range(config.max_episodes), desc="Training Policy Gradient"):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # Collect episode
        while True:
            action, log_prob = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            agent.store_transition(state, action, reward, log_prob)
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Train on episode
        policy_loss, value_loss = agent.train_episode()
        
        # Metrics
        recent_scores.append(env.score)
        metrics.add_episode(env.score, policy_loss, 0.0, steps, total_reward)
        
        # Progress logging
        if episode % 100 == 0:
            avg_score = np.mean(recent_scores) if recent_scores else 0
            avg_loss = metrics.get_recent_average('losses', 100)
            print(f"Episode {episode}: Avg Score: {avg_score:.2f}, Policy Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if episode % config.checkpoint_interval == 0 and episode > 0:
                checkpoint_path = checkpoint_dir / f"pg_{config.profile_name}_ep{episode}.pth"
                agent.save_model(checkpoint_path, {
                    'episode': episode,
                    'avg_score': avg_score,
                    'training_time': time.time() - training_start
                })
        
        # Save best model
        if env.score > best_score:
            best_score = env.score
            best_path = model_dir / f"pg_{config.profile_name}_best.pth"
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
    final_path = model_dir / f"pg_{config.profile_name}.pth"
    agent.save_model(final_path, {
        'final_episode': episode,
        'final_avg_score': np.mean(recent_scores) if recent_scores else 0,
        'total_training_time': time.time() - training_start,
        'total_episodes': len(metrics.scores)
    })
    
    # Save metrics
    metrics_path = model_dir / f"pg_{config.profile_name}_metrics.json"
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
        "config": config.__dict__
    }
    
    report_path = model_dir / f"pg_{config.profile_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"✅ Policy Gradient training complete!")
    print(f"📁 Final model: {final_path}")
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
    
    plt.suptitle(f'Policy Gradient Training Curves - {profile_name}')
    plt.tight_layout()
    
    plot_path = Path(save_dir) / f"pg_training_curves_{profile_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved: {plot_path}")

if __name__ == "__main__":
    # Train different Policy Gradient profiles
    profiles = {
        "aggressive": PolicyGradientConfig(
            profile_name="aggressive",
            learning_rate=0.003,
            baseline_lr=0.01,
            entropy_coeff=0.02,
            max_episodes=2000,
            target_score=10
        ),
        "balanced": PolicyGradientConfig(
            profile_name="balanced",
            learning_rate=0.001,
            baseline_lr=0.005,
            entropy_coeff=0.01,
            max_episodes=3000,
            target_score=12
        ),
        "conservative": PolicyGradientConfig(
            profile_name="conservative",
            learning_rate=0.0005,
            baseline_lr=0.002,
            entropy_coeff=0.005,
            max_episodes=4000,
            target_score=15
        )
    }
    
    for name, config in profiles.items():
        print(f"\n{'='*60}")
        print(f"🚀 Training Policy Gradient {name.upper()} model")
        print(f"{'='*60}")
        train_policy_gradient(config)