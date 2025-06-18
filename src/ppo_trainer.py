#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) trainer for SnakeAI-MLOps
Clean PPO implementation with proper model saving to models/ppo/
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
    """PPO training configuration"""
    profile_name: str = "balanced"
    max_episodes: int = 1500
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    
    # PPO specific parameters
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.02
    value_coeff: float = 0.5
    
    # Training parameters
    update_epochs: int = 6
    batch_size: int = 32
    trajectory_length: int = 256
    
    # Network architecture
    hidden_size: int = 256
    
    # Training settings
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
    """Snake environment optimized for PPO training"""
    
    def __init__(self, grid_size=15, device='cuda'):
        self.grid_size = grid_size
        self.device = torch.device(device)
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
        """Get 11D state representation"""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Direction vectors
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
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
        
        # Snake length and empty spaces
        snake_length = len(self.snake) / (self.grid_size * self.grid_size)
        empty_spaces = (self.grid_size * self.grid_size - len(self.snake) - 1) / (self.grid_size * self.grid_size)
        
        # Create 11D state vector
        state = torch.tensor([
            float(danger_straight),
            float(danger_left), 
            float(danger_right),
            float(self.direction / 3.0),
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
        self.direction = action
        head_x, head_y = self.snake[0]
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = directions[action]
        new_head = (head_x + dx, head_y + dy)
        
        # Check collision
        if self._is_collision(new_head):
            return self._get_state(), -100.0, True
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Calculate reward
        reward = 0
        
        # Check if food eaten
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            reward = 50.0
            self.steps_without_food = 0
        else:
            self.snake.pop()
            self.steps_without_food += 1
            
            # Distance-based reward
            current_distance = self._get_food_distance()
            if current_distance < self.prev_distance:
                reward = 2.0
            elif current_distance > self.prev_distance:
                reward = -1.0
            else:
                reward = -0.2
            
            self.prev_distance = current_distance
        
        # Living bonus
        reward += 0.1
        
        # Penalty for taking too long
        if self.steps_without_food > 100:
            reward -= 1.0
        
        self.steps += 1
        done = (self.steps >= 2000 or self.steps_without_food > 200)
        
        return self._get_state(), reward, done

class PPOAgent:
    """PPO agent with clipped surrogate objective"""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        self.policy_network = SimplePolicyNetwork(11, config.hidden_size, 4).to(self.device)
        self.value_network = SimpleValueNetwork(11, config.hidden_size).to(self.device)
        
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
        
        print(f"✅ PPO Agent initialized on {self.device}")
        print(f"   Policy Network: {sum(p.numel() for p in self.policy_network.parameters())} parameters")
        print(f"   Value Network: {sum(p.numel() for p in self.value_network.parameters())} parameters")
    
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
                
                values = self.value_network(batch_states).squeeze()
                
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
        """Save PPO model to models/ppo/ directory"""
        model_data = {
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'metadata': metadata or {}
        }
        
        torch.save(model_data, filepath)
        print(f"✅ PPO model saved: {filepath}")

def train_ppo(config: PPOConfig):
    """Main PPO training loop"""
    device = verify_gpu()
    config.device = str(device)
    
    # Setup directories - FIXED: Use models/ppo/ directory
    base_dir = Path("models")
    model_dir = base_dir / "ppo"
    checkpoint_dir = base_dir / "checkpoints" / "ppo"
    create_directories(str(base_dir))
    
    # Initialize environment and agent
    env = SnakeEnvironmentPPO(device=str(device))
    agent = PPOAgent(config)
    metrics = TrainingMetrics()
    
    print(f"🚀 Starting PPO training: {config.profile_name}")
    print(f"   Target score: {config.target_score}")
    print(f"   Max episodes: {config.max_episodes}")
    
    best_score = 0
    best_path = None
    training_start = time.time()
    recent_scores = deque(maxlen=100)
    
    episode = 0
    
    pbar = tqdm(total=config.max_episodes, desc="Training PPO")
    
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
            print(f"Episode {episode}: Avg Score: {avg_score:.2f}")
            
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
        
        # Early stopping
        if len(recent_scores) >= 100 and np.mean(recent_scores) >= config.target_score:
            print(f"✅ Target score reached at episode {episode}")
            break
    
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
    
    # Training report
    report = {
        "profile": config.profile_name,
        "episodes": len(metrics.scores),
        "final_avg_score": float(np.mean(recent_scores)) if recent_scores else 0,
        "best_score": int(max(metrics.scores)) if metrics.scores else 0,
        "training_time": time.time() - training_start,
        "config": config.__dict__
    }
    
    report_path = model_dir / f"ppo_{config.profile_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"✅ PPO training complete!")
    print(f"📁 Final model: {final_path}")
    if best_path:
        print(f"📁 Best model: {best_path}")

if __name__ == "__main__":
    # Train PPO balanced profile
    config = PPOConfig(
        profile_name="balanced",
        learning_rate=0.001,
        max_episodes=1500,
        target_score=8,
        hidden_size=256,
        clip_epsilon=0.2,
        entropy_coeff=0.02,
        trajectory_length=256,
        update_epochs=6,
        batch_size=32
    )
    
    print(f"🚀 Training PPO BALANCED model")
    train_ppo(config)