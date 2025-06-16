#!/usr/bin/env python3
"""
Enhanced Actor-Critic trainer for SnakeAI-MLOps
Implements: N-step returns, entropy scheduling, improved advantage estimation
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
    """Enhanced Actor-Critic training configuration"""
    profile_name: str = "balanced"
    max_episodes: int = 2500
    actor_lr: float = 0.001
    critic_lr: float = 0.002
    discount_factor: float = 0.99
    
    # Enhanced A2C features
    n_step: int = 5  # N-step returns
    entropy_coeff: float = 0.01
    entropy_decay: float = 0.999
    min_entropy_coeff: float = 0.001
    value_coeff: float = 0.5
    max_grad_norm: float = 1.0
    
    # Advanced features
    use_gae: bool = True  # Generalized Advantage Estimation
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    gradient_accumulation_steps: int = 4
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_decay_step: int = 300
    lr_decay_gamma: float = 0.9
    
    # Network architecture
    hidden_layers: list = None
    
    # Training settings
    device: str = "cuda"
    checkpoint_interval: int = 250
    target_score: int = 13
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64]

class NStepStorage:
    """N-step experience storage for Actor-Critic"""
    
    def __init__(self, n_step, gamma, device):
        self.n_step = n_step
        self.gamma = gamma
        self.device = device
        self.reset()
        
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def compute_returns_and_advantages(self, last_value, use_gae=True, gae_lambda=0.95):
        """Compute n-step returns and advantages"""
        returns = []
        advantages = []
        
        if use_gae:
            # GAE computation
            gae = 0
            for t in reversed(range(len(self.rewards))):
                if t == len(self.rewards) - 1:
                    next_value = last_value
                else:
                    next_value = self.values[t + 1]
                
                delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
                gae = delta + self.gamma * gae_lambda * (1 - self.dones[t]) * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + self.values[t])
        else:
            # Standard n-step returns
            R = last_value
            for t in reversed(range(len(self.rewards))):
                R = self.rewards[t] + self.gamma * R * (1 - self.dones[t])
                returns.insert(0, R)
                advantages.insert(0, R - self.values[t])
        
        return (
            torch.tensor(returns, dtype=torch.float32, device=self.device),
            torch.tensor(advantages, dtype=torch.float32, device=self.device)
        )
    
    def get_batch(self):
        """Get all stored data as batch"""
        return (
            torch.stack(self.states),
            torch.tensor(self.actions, dtype=torch.long, device=self.device),
            torch.tensor(self.rewards, dtype=torch.float32, device=self.device),
            torch.tensor(self.values, dtype=torch.float32, device=self.device),
            torch.tensor(self.log_probs, dtype=torch.float32, device=self.device),
            torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        )

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
        
        # Wall distances
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
            reward = 0.1 if new_dist < old_dist else -0.05
        
        self.steps += 1
        done = self.steps >= 1000
        
        return self._get_enhanced_state(), reward, done

class EnhancedActorCriticAgent:
    """Enhanced A2C agent with n-step returns and GAE"""
    
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
        
        # Learning rate schedulers
        if config.use_lr_scheduler:
            self.actor_scheduler = optim.lr_scheduler.StepLR(
                self.actor_optimizer, 
                step_size=config.lr_decay_step,
                gamma=config.lr_decay_gamma
            )
            self.critic_scheduler = optim.lr_scheduler.StepLR(
                self.critic_optimizer,
                step_size=config.lr_decay_step,
                gamma=config.lr_decay_gamma
            )
        
        # N-step storage
        self.storage = NStepStorage(config.n_step, config.discount_factor, self.device)
        
        # Gradient accumulation
        self.gradient_accumulation_counter = 0
        
        # Current entropy coefficient
        self.entropy_coeff = config.entropy_coeff
        
        print(f"✅ Enhanced Actor-Critic Agent initialized on {self.device}")
        print(f"   Actor: {sum(p.numel() for p in self.actor.parameters())} parameters")
        print(f"   Critic: {sum(p.numel() for p in self.critic.parameters())} parameters")
        print(f"   Features: N-step={config.n_step}, GAE={config.use_gae}")
    
    def get_action_and_value(self, state):
        """Get action from actor and value from critic"""
        state_batch = state.unsqueeze(0)
        
        # Actor
        action_probs = self.actor(state_batch)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        # Critic
        value = self.critic(state_batch).squeeze()
        
        return action.item(), log_prob.item(), value.item()
    
    def get_action(self, state):
        """Get action for evaluation (no gradients)"""
        with torch.no_grad():
            action_probs = self.actor(state.unsqueeze(0))
            action = torch.argmax(action_probs).item()
            return action
    
    def train_on_rollout(self):
        """Train on collected n-step rollout"""
        # Get batch data
        states, actions, rewards, values, log_probs, dones = self.storage.get_batch()
        
        # Compute last value for bootstrapping
        with torch.no_grad():
            if dones[-1]:
                last_value = 0
            else:
                # Use critic to estimate value of last state
                last_value = self.critic(states[-1].unsqueeze(0)).squeeze().item()
        
        # Compute returns and advantages
        returns, advantages = self.storage.compute_returns_and_advantages(
            last_value, 
            use_gae=self.config.use_gae,
            gae_lambda=self.config.gae_lambda
        )
        
        # Normalize advantages
        if self.config.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Recompute action probabilities and values for current parameters
        action_probs = self.actor(states)
        action_dist = torch.distributions.Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        new_values = self.critic(states).squeeze()
        
        # Actor loss (policy gradient with advantages)
        actor_loss = -(new_log_probs * advantages.detach()).mean()
        
        # Entropy bonus
        entropy_loss = -self.entropy_coeff * entropy
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(new_values, returns.detach())
        
        # Total actor loss
        total_actor_loss = actor_loss + entropy_loss
        
        # Gradient accumulation
        self.gradient_accumulation_counter += 1
        
        # Scale losses by accumulation steps
        total_actor_loss = total_actor_loss / self.config.gradient_accumulation_steps
        critic_loss = critic_loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        total_actor_loss.backward()
        critic_loss.backward()
        
        # Update parameters if accumulation complete
        if self.gradient_accumulation_counter >= self.config.gradient_accumulation_steps:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            
            # Update
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            # Zero gradients
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            # Reset counter
            self.gradient_accumulation_counter = 0
        
        # Clear storage
        self.storage.reset()
        
        return actor_loss.item(), critic_loss.item(), entropy.item()
    
    def update_entropy_coeff(self):
        """Decay entropy coefficient"""
        self.entropy_coeff = max(
            self.config.min_entropy_coeff,
            self.entropy_coeff * self.config.entropy_decay
        )
    
    def update_schedulers(self):
        """Update learning rate schedulers"""
        if self.config.use_lr_scheduler:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
    
    def save_model(self, filepath, metadata=None):
        """Save enhanced Actor-Critic model"""
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
        print(f"✅ Enhanced Actor-Critic model saved: {filepath}")

def train_actor_critic(config: ActorCriticConfig):
    """Main Actor-Critic training loop with enhancements"""
    device = verify_gpu()
    config.device = str(device)
    
    # Setup directories
    base_dir = Path("models")
    model_dir = base_dir / "actor_critic"
    checkpoint_dir = base_dir / "checkpoints" / "actor_critic"
    create_directories(str(base_dir))
    
    # Initialize environment and agent
    env = SnakeEnvironmentAC(device=str(device))
    agent = EnhancedActorCriticAgent(config)
    metrics = TrainingMetrics()
    
    print(f"🚀 Starting Enhanced Actor-Critic training: {config.profile_name}")
    print(f"   Target score: {config.target_score}")
    print(f"   Max episodes: {config.max_episodes}")
    
    best_score = 0
    best_path = None
    training_start = time.time()
    recent_scores = deque(maxlen=100)
    
    for episode in tqdm(range(config.max_episodes), desc="Training Enhanced Actor-Critic"):
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_actor_loss = 0
        episode_critic_loss = 0
        update_count = 0
        
        while True:
            # Get action and value
            action, log_prob, value = agent.get_action_and_value(state)
            next_state, reward, done = env.step(action)
            
            # Store transition
            agent.storage.add(state, action, reward, value, log_prob, done)
            
            total_reward += reward
            steps += 1
            
            # Train on n-step rollout
            if len(agent.storage.states) >= config.n_step or done:
                actor_loss, critic_loss, entropy = agent.train_on_rollout()
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
                update_count += 1
            
            state = next_state
            
            if done:
                # Process any remaining transitions
                if len(agent.storage.states) > 0:
                    actor_loss, critic_loss, entropy = agent.train_on_rollout()
                    episode_actor_loss += actor_loss
                    episode_critic_loss += critic_loss
                    update_count += 1
                break
        
        # Update parameters
        agent.update_entropy_coeff()
        agent.update_schedulers()
        
        # Metrics
        recent_scores.append(env.score)
        avg_actor_loss = episode_actor_loss / max(update_count, 1)
        avg_critic_loss = episode_critic_loss / max(update_count, 1)
        metrics.add_episode(env.score, avg_actor_loss + avg_critic_loss, agent.entropy_coeff, steps, total_reward)
        
        # Progress logging
        if episode % 100 == 0:
            avg_score = np.mean(recent_scores) if recent_scores else 0
            current_actor_lr = agent.actor_scheduler.get_last_lr()[0] if config.use_lr_scheduler else config.actor_lr
            current_critic_lr = agent.critic_scheduler.get_last_lr()[0] if config.use_lr_scheduler else config.critic_lr
            
            print(f"Episode {episode}: Avg Score: {avg_score:.2f}, "
                  f"Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}, "
                  f"Entropy: {agent.entropy_coeff:.4f}, "
                  f"LR: A={current_actor_lr:.5f} C={current_critic_lr:.5f}")
            
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
        "final_entropy_coeff": agent.entropy_coeff,
        "training_time": time.time() - training_start,
        "config": config.__dict__
    }
    
    report_path = model_dir / f"ac_{config.profile_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"✅ Enhanced Actor-Critic training complete!")
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
    
    plt.suptitle(f'Enhanced Actor-Critic Training Curves - {profile_name}')
    plt.tight_layout()
    
    plot_path = Path(save_dir) / f"ac_training_curves_{profile_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved: {plot_path}")

if __name__ == "__main__":
    # Train different Actor-Critic profiles with enhancements
    profiles = {
        "aggressive": ActorCriticConfig(
            profile_name="aggressive",
            actor_lr=0.003,
            critic_lr=0.006,
            entropy_coeff=0.02,
            max_episodes=2000,
            target_score=10,
            n_step=3,
            gae_lambda=0.9
        ),
        "balanced": ActorCriticConfig(
            profile_name="balanced",
            actor_lr=0.001,
            critic_lr=0.002,
            entropy_coeff=0.01,
            max_episodes=2500,
            target_score=13,
            n_step=5,
            gae_lambda=0.95
        ),
        "conservative": ActorCriticConfig(
            profile_name="conservative",
            actor_lr=0.0005,
            critic_lr=0.001,
            entropy_coeff=0.005,
            max_episodes=3000,
            target_score=16,
            n_step=8,
            gae_lambda=0.98
        )
    }
    
    for name, config in profiles.items():
        print(f"\n{'='*60}")
        print(f"🚀 Training Enhanced Actor-Critic {name.upper()} model")
        print(f"{'='*60}")
        train_actor_critic(config)