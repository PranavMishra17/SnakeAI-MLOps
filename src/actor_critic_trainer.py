#!/usr/bin/env python3
"""
Actor-Critic trainer for SnakeAI-MLOps
GPU-accelerated implementation with Advantage Actor-Critic (A2C)
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
class ActorCriticConfig:
    """Actor-Critic training configuration"""
    profile_name: str = "balanced"
    max_episodes: int = 2500
    actor_lr: float = 0.001
    critic_lr: float = 0.002
    discount_factor: float = 0.99
    
    # A2C specific
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 1.0
    
    # Network architecture
    hidden_layers: list = None
    
    # Training settings
    device: str = "cuda"
    checkpoint_interval: int = 250
    target_score: int = 13
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64]

class SnakeEnvironmentAC:
    """Snake environment for Actor-Critic training"""
    
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
        
        # Danger detection
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        current_dir = directions[self.direction]
        
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
        
        # Basic features
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
        
        # Wall distances (normalized)
        state[9] = head_y / self.grid_size
        state[10] = (self.grid_size - 1 - head_y) / self.grid_size
        state[11] = head_x / self.grid_size
        state[12] = (self.grid_size - 1 - head_x) / self.grid_size
        
        # Body density
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
            reward = 0.2 if new_dist < old_dist else -0.1
        
        self.steps += 1
        done = self.steps >= 1000
        
        return self._get_enhanced_state(), reward, done

class ActorCriticAgent:
    """Advantage Actor-Critic (A2C) agent"""
    
    def __init__(self, config: ActorCriticConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        net_config = NetworkConfig(
            input_size=20,
            hidden_layers=config.hidden_layers,
            output_size=4,
            dropout=0.1
        )
        
        self.actor = PolicyNetwork(net_config).to(self.device)
        self.critic = ValueNetwork(net_config).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        print(f"✅ Actor-Critic Agent initialized on {self.device}")
        print(f"   Actor: {sum(p.numel() for p in self.actor.parameters())} parameters")
        print(f"   Critic: {sum(p.numel() for p in self.critic.parameters())} parameters")
    
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
        
        return action.item(), log_prob, value.squeeze()
    
    def get_action(self, state):
        """Get action for evaluation (no gradients)"""
        with torch.no_grad():
            action, _, _ = self.get_action_and_value(state)
            return action
    
    def train_step(self, states, actions, rewards, next_states, dones, log_probs):
        """Single training step using collected transitions"""
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        log_probs = torch.stack(log_probs)
        
        # Compute values
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        
        # Compute returns (TD target)
        returns = rewards + self.config.discount_factor * next_values * ~dones
        
        # Compute advantages
        advantages = returns - values
        
        # Actor loss (policy gradient with advantage)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Entropy bonus for exploration
        action_probs = self.actor(states)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -self.config.entropy_coeff * entropy
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(values, returns.detach())
        
        # Total actor loss
        total_actor_loss = actor_loss + entropy_loss
        
        # Update actor
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), entropy.item()
    
    def save_model(self, filepath, metadata=None):
        """Save Actor-Critic model"""
        model_data = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config.__dict__,
            'metadata': metadata or {}
        }
        
        torch.save(model_data, filepath)
        print(f"✅ Actor-Critic model saved: {filepath}")

def train_actor_critic(config: ActorCriticConfig):
    """Main Actor-Critic training loop"""
    device = verify_gpu()
    config.device = str(device)
    
    # Setup directories
    base_dir = Path("models")
    model_dir = base_dir / "actor_critic"
    checkpoint_dir = base_dir / "checkpoints" / "actor_critic"
    create_directories(str(base_dir))
    
    # Initialize environment and agent
    env = SnakeEnvironmentAC(device=str(device))
    agent = ActorCriticAgent(config)
    metrics = TrainingMetrics()
    
    print(f"🚀 Starting Actor-Critic training: {config.profile_name}")
    print(f"   Target score: {config.target_score}")
    print(f"   Max episodes: {config.max_episodes}")
    
    best_score = 0
    training_start = time.time()
    recent_scores = deque(maxlen=100)
    
    for episode in tqdm(range(config.max_episodes), desc="Training Actor-Critic"):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # Episode storage
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_dones = []
        episode_log_probs = []
        
        # Collect episode
        while True:
            action, log_prob, value = agent.get_action_and_value(state)
            next_state, reward, done = env.step(action)
            
            # Store transition
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_next_states.append(next_state)
            episode_dones.append(done)
            episode_log_probs.append(log_prob)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Train on episode
        actor_loss, critic_loss, entropy = agent.train_step(
            episode_states, episode_actions, episode_rewards,
            episode_next_states, episode_dones, episode_log_probs
        )
        
        # Metrics
        recent_scores.append(env.score)
        total_loss = actor_loss + critic_loss
        metrics.add_episode(env.score, total_loss, 0.0, steps, total_reward)
        
        # Progress logging
        if episode % 100 == 0:
            avg_score = np.mean(recent_scores) if recent_scores else 0
            print(f"Episode {episode}: Avg Score: {avg_score:.2f}, "
                  f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, "
                  f"Entropy: {entropy:.4f}")
            
            # Save checkpoint
            if episode % config.checkpoint_interval == 0 and episode > 0:
                checkpoint_path = checkpoint_dir / f"ac_{config.profile_name}_ep{episode}.pth"
                agent.save_model(checkpoint_path, {
                    'episode': episode,
                    'avg_score': avg_score,
                    'actor_loss': actor_loss,
                    'critic_loss': critic_loss,
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
        
        # Early stopping
        if len(recent_scores) >= 100 and np.mean(recent_scores) >= config.target_score:
            print(f"✅ Target score reached at episode {episode}")
            break
    
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
        "training_time": time.time() - training_start,
        "config": config.__dict__
    }
    
    report_path = model_dir / f"ac_{config.profile_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"✅ Actor-Critic training complete!")
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
    
    # Total loss (actor + critic)
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
    
    plt.suptitle(f'Actor-Critic Training Curves - {profile_name}')
    plt.tight_layout()
    
    plot_path = Path(save_dir) / f"ac_training_curves_{profile_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved: {plot_path}")

if __name__ == "__main__":
    # Train different Actor-Critic profiles
    profiles = {
        "aggressive": ActorCriticConfig(
            profile_name="aggressive",
            actor_lr=0.003,
            critic_lr=0.006,
            entropy_coeff=0.02,
            max_episodes=2000,
            target_score=10
        ),
        "balanced": ActorCriticConfig(
            profile_name="balanced",
            actor_lr=0.001,
            critic_lr=0.002,
            entropy_coeff=0.01,
            max_episodes=2500,
            target_score=13
        ),
        "conservative": ActorCriticConfig(
            profile_name="conservative",
            actor_lr=0.0005,
            critic_lr=0.001,
            entropy_coeff=0.005,
            max_episodes=3000,
            target_score=16
        )
    }
    
    for name, config in profiles.items():
        print(f"\n{'='*60}")
        print(f"🚀 Training Actor-Critic {name.upper()} model")
        print(f"{'='*60}")
        train_actor_critic(config)