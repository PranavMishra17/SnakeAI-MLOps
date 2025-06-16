#!/usr/bin/env python3
"""
Enhanced Deep Q-Network (DQN) trainer for SnakeAI-MLOps
Implements: Prioritized Experience Replay, Soft Updates, N-step returns
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
from collections import deque

from neural_network_utils import (
    DQNNetwork, NetworkConfig, 
    verify_gpu, create_directories, save_model, 
    TrainingMetrics, encode_state_for_dqn
)

@dataclass
class DQNConfig:
    """Enhanced DQN training configuration"""
    profile_name: str = "balanced"
    max_episodes: int = 2000
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # DQN specific
    batch_size: int = 32
    memory_capacity: int = 50000  # Increased from 10000
    target_update_freq: int = 100
    min_memory_size: int = 1000
    
    # Enhanced features
    soft_update_tau: float = 0.005  # Soft target update
    n_step: int = 3  # N-step returns
    use_per: bool = True  # Prioritized Experience Replay
    per_alpha: float = 0.6  # PER prioritization
    per_beta: float = 0.4  # PER importance sampling
    per_beta_increment: float = 0.001  # Beta annealing
    
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

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, device='cuda'):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.device = torch.device(device)
        
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done):
        """Add experience with max priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample batch with prioritized replay"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
            
        # Calculate sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        
        # Get batch
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack(states).to(self.device),
            torch.tensor(actions, device=self.device, dtype=torch.long),
            torch.tensor(rewards, device=self.device, dtype=torch.float32),
            torch.stack(next_states).to(self.device),
            torch.tensor(dones, device=self.device, dtype=torch.bool),
            indices,
            weights
        )
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        priorities = np.abs(td_errors.cpu().numpy()) + 1e-6
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class NStepBuffer:
    """N-step return buffer"""
    
    def __init__(self, n_step, gamma, device):
        self.n_step = n_step
        self.gamma = gamma
        self.device = device
        self.buffer = deque(maxlen=n_step)
        
    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def get(self):
        """Get n-step transition"""
        if len(self.buffer) < self.n_step:
            return None
            
        # Calculate n-step return
        n_step_return = 0
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            n_step_return += (self.gamma ** i) * r
            if d and i < self.n_step - 1:
                # Episode ended early
                return (
                    self.buffer[0][0],  # state
                    self.buffer[0][1],  # action
                    n_step_return,
                    self.buffer[i][3],  # next_state where it ended
                    True
                )
        
        return (
            self.buffer[0][0],  # state
            self.buffer[0][1],  # action
            n_step_return,
            self.buffer[-1][3],  # next_state
            self.buffer[-1][4]  # done
        )
    
    def clear(self):
        self.buffer.clear()

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
        
        # Basic 8D state
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
        
        food_left = food_x < head_x
        food_right = food_x > head_x
        food_up = food_y < head_y
        food_down = food_y > head_y
        
        # Enhanced 20D state
        state_20d = torch.zeros(20, dtype=torch.float32, device=self.device)
        
        # Basic features
        state_20d[0] = float(danger_straight)
        state_20d[1] = float(danger_left)
        state_20d[2] = float(danger_right)
        state_20d[3] = float(self.direction / 3.0)
        state_20d[4] = float(food_left)
        state_20d[5] = float(food_right)
        state_20d[6] = float(food_up)
        state_20d[7] = float(food_down)
        
        # Enhanced features
        food_distance = abs(head_x - food_x) + abs(head_y - food_y)
        state_20d[8] = food_distance / (2 * self.grid_size)
        
        state_20d[9] = head_y / self.grid_size
        state_20d[10] = (self.grid_size - 1 - head_y) / self.grid_size
        state_20d[11] = head_x / self.grid_size
        state_20d[12] = (self.grid_size - 1 - head_x) / self.grid_size
        
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
            state_20d[13 + i] = quadrant_counts[i] / quadrant_size
        
        state_20d[17] = len(self.snake) / (self.grid_size * self.grid_size)
        empty_spaces = self.grid_size * self.grid_size - len(self.snake) - 1
        state_20d[18] = empty_spaces / (self.grid_size * self.grid_size)
        state_20d[19] = (food_distance + len(self.snake) * 0.1) / (2 * self.grid_size + 10)
        
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
            # Improved reward shaping
            old_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = 0.1 if new_dist < old_dist else -0.1
        
        self.steps += 1
        done = self.steps >= 1000
        
        return self._get_enhanced_state(), reward, done

class EnhancedDQNAgent:
    """Enhanced DQN Agent with PER, soft updates, and n-step returns"""
    
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
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer with learning rate scheduling
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        
        # Experience replay
        if config.use_per:
            self.memory = PrioritizedReplayBuffer(
                config.memory_capacity, 
                config.per_alpha, 
                config.per_beta,
                self.device
            )
        else:
            from neural_network_utils import ExperienceReplay
            self.memory = ExperienceReplay(config.memory_capacity, self.device)
        
        # N-step buffer
        self.n_step_buffer = NStepBuffer(config.n_step, config.discount_factor, self.device)
        
        # Training state
        self.epsilon = config.epsilon_start
        self.steps_done = 0
        self.beta = config.per_beta if config.use_per else 1.0
        
        print(f"✅ Enhanced DQN Agent initialized on {self.device}")
        print(f"   Network: {sum(p.numel() for p in self.q_network.parameters())} parameters")
        print(f"   Features: PER={config.use_per}, N-step={config.n_step}, Soft Updates")
    
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        
        with torch.no_grad():
            state_batch = state.unsqueeze(0)
            q_values = self.q_network(state_batch)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience with n-step returns"""
        self.n_step_buffer.append(state, action, reward, next_state, done)
        
        # Get n-step transition
        n_step_transition = self.n_step_buffer.get()
        if n_step_transition:
            if self.config.use_per:
                self.memory.push(*n_step_transition)
            else:
                self.memory.push(*n_step_transition)
        
        # Clear buffer on episode end
        if done:
            # Process remaining transitions
            while len(self.n_step_buffer.buffer) > 0:
                n_step_transition = self.n_step_buffer.get()
                if n_step_transition:
                    if self.config.use_per:
                        self.memory.push(*n_step_transition)
                    else:
                        self.memory.push(*n_step_transition)
                self.n_step_buffer.buffer.popleft()
            self.n_step_buffer.clear()
    
    def train_step(self):
        """Single training step with enhanced features"""
        if len(self.memory) < self.config.min_memory_size:
            return 0.0
        
        # Sample batch
        if self.config.use_per:
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.config.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
            weights = torch.ones(self.config.batch_size, device=self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            # N-step returns already calculated in buffer
            target_q_values = rewards.unsqueeze(1) + (self.config.discount_factor ** self.config.n_step * 
                                                      next_q_values * ~dones.unsqueeze(1))
        
        # TD errors for PER
        td_errors = target_q_values - current_q_values
        
        # Weighted loss
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities
        if self.config.use_per:
            self.memory.update_priorities(indices, td_errors.detach().abs().squeeze())
        
        return loss.item()
    
    def soft_update_target_network(self):
        """Soft update of target network parameters"""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                self.config.soft_update_tau * param.data + 
                (1.0 - self.config.soft_update_tau) * target_param.data
            )
    
    def update_target_network(self):
        """Update target network (hard or soft)"""
        if self.config.soft_update_tau < 1.0:
            # Soft update every step
            self.soft_update_target_network()
        else:
            # Hard update periodically
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
    
    def update_beta(self):
        """Update importance sampling beta for PER"""
        if self.config.use_per:
            self.beta = min(1.0, self.beta + self.config.per_beta_increment)
            if hasattr(self.memory, 'beta'):
                self.memory.beta = self.beta
    
    def save_model(self, filepath, metadata=None):
        """Save enhanced DQN model"""
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
        print(f"✅ Enhanced DQN model saved: {filepath}")

def train_dqn(config: DQNConfig):
    """Main DQN training loop with enhancements"""
    device = verify_gpu()
    config.device = str(device)
    
    # Setup directories
    base_dir = Path("models")
    model_dir = base_dir / "dqn"
    checkpoint_dir = base_dir / "checkpoints" / "dqn"
    create_directories(str(base_dir))
    
    # Initialize environment and agent
    env = SnakeEnvironmentDQN(device=str(device))
    agent = EnhancedDQNAgent(config)
    metrics = TrainingMetrics()
    
    print(f"🚀 Starting Enhanced DQN training: {config.profile_name}")
    print(f"   Target score: {config.target_score}")
    print(f"   Max episodes: {config.max_episodes}")
    
    best_score = 0
    best_path = None
    training_start = time.time()
    recent_scores = deque(maxlen=100)
    
    for episode in tqdm(range(config.max_episodes), desc="Training Enhanced DQN"):
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
            
            # Soft update every step
            if config.soft_update_tau < 1.0:
                agent.update_target_network()
            
            state = next_state
            
            if done:
                break
        
        # Hard update periodically (if not using soft updates)
        if config.soft_update_tau >= 1.0 and episode % config.target_update_freq == 0:
            agent.update_target_network()
        
        # Update parameters
        agent.decay_epsilon()
        agent.update_beta()
        agent.scheduler.step()
        
        # Metrics
        recent_scores.append(env.score)
        metrics.add_episode(env.score, episode_loss / max(steps, 1), agent.epsilon, steps, total_reward)
        
        # Progress logging
        if episode % 100 == 0:
            avg_score = np.mean(recent_scores) if recent_scores else 0
            avg_loss = metrics.get_recent_average('losses', 100)
            current_lr = agent.scheduler.get_last_lr()[0]
            print(f"Episode {episode}: Avg Score: {avg_score:.2f}, Loss: {avg_loss:.4f}, "
                  f"Epsilon: {agent.epsilon:.3f}, LR: {current_lr:.5f}")
            
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
        "final_beta": agent.beta if config.use_per else 1.0,
        "training_time": time.time() - training_start,
        "config": config.__dict__
    }
    
    report_path = model_dir / f"dqn_{config.profile_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"✅ Enhanced DQN training complete!")
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
    
    plt.suptitle(f'Enhanced DQN Training Curves - {profile_name}')
    plt.tight_layout()
    
    plot_path = Path(save_dir) / f"dqn_training_curves_{profile_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved: {plot_path}")

if __name__ == "__main__":
    # Train different DQN profiles with enhanced features
    profiles = {
        "aggressive": DQNConfig(
            profile_name="aggressive",
            learning_rate=0.001,
            epsilon_start=1.0,
            epsilon_decay=0.99,
            max_episodes=1500,
            target_score=12,
            soft_update_tau=0.01,
            n_step=2
        ),
        "balanced": DQNConfig(
            profile_name="balanced",
            learning_rate=0.0005,
            epsilon_start=0.8,
            epsilon_decay=0.995,
            max_episodes=2000,
            target_score=15,
            soft_update_tau=0.005,
            n_step=3
        ),
        "conservative": DQNConfig(
            profile_name="conservative",
            learning_rate=0.0003,
            epsilon_start=0.5,
            epsilon_decay=0.997,
            max_episodes=2500,
            target_score=18,
            soft_update_tau=0.001,
            n_step=5
        )
    }
    
    for name, config in profiles.items():
        print(f"\n{'='*60}")
        print(f"🚀 Training Enhanced DQN {name.upper()} model")
        print(f"{'='*60}")
        train_dqn(config)