#!/usr/bin/env python3
"""
Shared neural network utilities for SnakeAI-MLOps
GPU-accelerated PyTorch implementations with comprehensive usage examples
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class NetworkConfig:
    """Neural network configuration"""
    input_size: int = 8  # Changed default to 8D for consistency
    hidden_layers: List[int] = None
    output_size: int = 4  # 4 actions
    activation: str = "relu"
    dropout: float = 0.0
    batch_norm: bool = False
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64]  # Simplified default

class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable architecture"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        layers = []
        in_size = config.input_size
        
        # Hidden layers
        for hidden_size in config.hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "tanh":
                layers.append(nn.Tanh())
            elif config.activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            in_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_size, config.output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier initialization for better convergence"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class DQNNetwork(nn.Module):
    """Deep Q-Network with dueling architecture option"""
    
    def __init__(self, config: NetworkConfig, dueling: bool = True):
        super().__init__()
        self.config = config
        self.dueling = dueling
        
        # Shared layers
        shared_layers = []
        in_size = config.input_size
        
        for hidden_size in config.hidden_layers[:-1]:  # All but last hidden layer
            shared_layers.append(nn.Linear(in_size, hidden_size))
            shared_layers.append(nn.ReLU())
            if config.dropout > 0:
                shared_layers.append(nn.Dropout(config.dropout))
            in_size = hidden_size
        
        self.shared_net = nn.Sequential(*shared_layers)
        
        if self.dueling:
            # Dueling architecture: separate value and advantage streams
            final_hidden = config.hidden_layers[-1]
            
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(in_size, final_hidden),
                nn.ReLU(),
                nn.Linear(final_hidden, 1)
            )
            
            # Advantage stream  
            self.advantage_stream = nn.Sequential(
                nn.Linear(in_size, final_hidden),
                nn.ReLU(),
                nn.Linear(final_hidden, config.output_size)
            )
        else:
            # Standard DQN
            self.q_head = nn.Sequential(
                nn.Linear(in_size, config.hidden_layers[-1]),
                nn.ReLU(),
                nn.Linear(config.hidden_layers[-1], config.output_size)
            )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        shared_features = self.shared_net(x)
        
        if self.dueling:
            value = self.value_stream(shared_features)
            advantage = self.advantage_stream(shared_features)
            
            # Dueling aggregation
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q_values
        else:
            return self.q_head(shared_features)

class PolicyNetwork(nn.Module):
    """Policy network for policy gradient methods"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        self.network = MLP(config)
        
    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits, dim=-1)
    
    def get_action_and_log_prob(self, state):
        """Sample action and return log probability"""
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

class ValueNetwork(nn.Module):
    """Value network for actor-critic methods"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        # Value network outputs single value, not action values
        value_config = NetworkConfig(
            input_size=config.input_size,
            hidden_layers=config.hidden_layers,
            output_size=1,  # Single value output
            activation=config.activation,
            dropout=config.dropout,
            batch_norm=config.batch_norm
        )
        self.network = MLP(value_config)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)  # Remove last dimension

class ExperienceReplay:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample batch of experiences"""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        
        return (
            torch.stack(states).to(self.device),
            torch.tensor(actions, device=self.device, dtype=torch.long),
            torch.tensor(rewards, device=self.device, dtype=torch.float32),
            torch.stack(next_states).to(self.device),
            torch.tensor(dones, device=self.device, dtype=torch.bool)
        )
    
    def __len__(self):
        return len(self.buffer)

def save_model(model: nn.Module, config: dict, filepath: str, metadata: dict = None):
    """Save PyTorch model with configuration"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'metadata': metadata or {}
    }
    torch.save(checkpoint, filepath)
    print(f"✅ Model saved: {filepath}")

def load_model(model_class, filepath: str, device: torch.device):
    """Load PyTorch model from checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create model from config
    config = checkpoint['config']
    model = model_class(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('metadata', {})

def verify_gpu():
    """Verify GPU availability and setup"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        
        # Test GPU with small computation
        test_tensor = torch.randn(1000, 1000, device=device)
        result = torch.mm(test_tensor, test_tensor)
        print(f"✅ GPU Test: {result.shape} tensor computed successfully")
        return device
    else:
        print("❌ No GPU available, using CPU")
        return torch.device('cpu')

def create_directories(base_path: str):
    """Create necessary directories for training"""
    base = Path(base_path)
    
    directories = [
        base / "qlearning",
        base / "dqn", 
        base / "ppo",
        base / "actor_critic",
        base / "checkpoints" / "dqn",
        base / "checkpoints" / "ppo", 
        base / "checkpoints" / "actor_critic",
        base / "logs",
        base / "plots"
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created directory structure in {base}")

def encode_state_for_dqn(state_8d):
    """Convert 8D discrete state to enhanced state for neural networks (DEPRECATED)"""
    # This function is kept for backward compatibility but not recommended
    # Use the fixed state preparation in the evaluator instead
    print("⚠️  encode_state_for_dqn is deprecated. Use evaluator's prepare_neural_state instead.")
    
    state_enhanced = torch.zeros(11, dtype=torch.float32)
    
    # Basic features (8D -> first 8 elements)
    state_enhanced[:8] = state_8d.float()
    
    # Engineered features (simulate enhanced state)
    food_features = state_8d[4:8]  # food_left, food_right, food_up, food_down
    food_distance = torch.sum(food_features).float()
    state_enhanced[8] = food_distance / 4.0  # Normalize
    
    # Additional placeholders
    state_enhanced[9] = 0.1   # snake length placeholder
    state_enhanced[10] = 0.9  # empty spaces placeholder
    
    return state_enhanced

class TrainingMetrics:
    """Track training metrics across episodes"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.scores = []
        self.losses = []
        self.epsilons = []
        self.episode_lengths = []
        self.rewards = []
    
    def add_episode(self, score: float, loss: float = 0.0, epsilon: float = 0.0, 
                   length: int = 0, total_reward: float = 0.0):
        self.scores.append(score)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        self.episode_lengths.append(length)
        self.rewards.append(total_reward)
    
    def get_recent_average(self, key: str, window: int = 100):
        """Get recent average for specified metric"""
        data = getattr(self, key, [])
        if not data:
            return 0.0
        return np.mean(data[-window:])
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON"""
        metrics = {
            'scores': self.scores,
            'losses': self.losses, 
            'epsilons': self.epsilons,
            'episode_lengths': self.episode_lengths,
            'rewards': self.rewards
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Metrics saved: {filepath}")

if __name__ == "__main__":
    # Test utilities
    device = verify_gpu()
    
    # Test network architectures
    config = NetworkConfig(input_size=8, hidden_layers=[64, 64], output_size=4)
    
    dqn = DQNNetwork(config, dueling=True).to(device)
    policy = PolicyNetwork(config).to(device)
    value = ValueNetwork(config).to(device)
    
    # Test forward pass
    test_state = torch.randn(1, 8, device=device)
    
    q_values = dqn(test_state)
    action_probs = policy(test_state)
    state_value = value(test_state)
    
    print(f"✅ DQN output shape: {q_values.shape}")
    print(f"✅ Policy output shape: {action_probs.shape}")
    print(f"✅ Value output shape: {state_value.shape}")
    
    # Test directory creation
    create_directories("models")

"""
COMPREHENSIVE USAGE EXAMPLES:
=============================

# 1. BASIC NETWORK CREATION
from neural_network_utils import NetworkConfig, DQNNetwork, PolicyNetwork, ValueNetwork

# Simple 8D input configuration (recommended)
config = NetworkConfig(
    input_size=8,           # 8D state (consistent across all models)
    hidden_layers=[64, 64], # Two hidden layers with 64 units each
    output_size=4,          # 4 actions (UP, DOWN, LEFT, RIGHT)
    activation="relu",      # ReLU activation
    dropout=0.1            # 10% dropout for regularization
)

# Create different network types
dqn_net = DQNNetwork(config, dueling=True)    # DQN with dueling architecture
policy_net = PolicyNetwork(config)            # Policy network for PPO/AC
value_net = ValueNetwork(config)              # Value network for AC

# 2. GPU SETUP AND VERIFICATION
from neural_network_utils import verify_gpu

device = verify_gpu()  # Automatically detects and tests GPU
# Output: ✅ GPU Available: NVIDIA GeForce RTX 3080 (10.0GB)

# Move networks to GPU
dqn_net = dqn_net.to(device)
policy_net = policy_net.to(device)

# 3. DIRECTORY STRUCTURE CREATION
from neural_network_utils import create_directories

create_directories("models")
# Creates: models/qlearning/, models/dqn/, models/ppo/, models/actor_critic/
#          models/checkpoints/, models/logs/, models/plots/

# 4. TRAINING METRICS TRACKING
from neural_network_utils import TrainingMetrics

metrics = TrainingMetrics()

# During training loop
for episode in range(1000):
    # ... training code ...
    score = 10.5
    loss = 0.25
    epsilon = 0.1
    steps = 150
    total_reward = 45.2
    
    metrics.add_episode(score, loss, epsilon, steps, total_reward)

# Save metrics
metrics.save_metrics("models/dqn/training_metrics.json")

# Get recent performance
recent_avg_score = metrics.get_recent_average('scores', window=100)
print(f"Recent average score: {recent_avg_score:.2f}")

# 5. EXPERIENCE REPLAY BUFFER (for DQN)
from neural_network_utils import ExperienceReplay

replay_buffer = ExperienceReplay(capacity=10000, device=device)

# Store experiences during training
state = torch.randn(8, device=device)
action = 2
reward = 1.0
next_state = torch.randn(8, device=device)
done = False

replay_buffer.push(state, action, reward, next_state, done)

# Sample batch for training
if len(replay_buffer) >= 32:
    states, actions, rewards, next_states, dones = replay_buffer.sample(32)
    # Use for DQN training...

# 6. MODEL SAVING AND LOADING
from neural_network_utils import save_model, load_model

# Save a trained model
config_dict = {
    'input_size': 8,
    'hidden_layers': [64, 64],
    'output_size': 4
}

metadata = {
    'episode': 1000,
    'best_score': 15.2,
    'training_time': 3600
}

save_model(dqn_net, config_dict, "models/dqn/my_model.pth", metadata)

# Load a saved model
loaded_model, loaded_metadata = load_model(DQNNetwork, "models/dqn/my_model.pth", device)
print(f"Loaded model trained for {loaded_metadata['training_time']} seconds")

# 7. NETWORK FORWARD PASS EXAMPLES

# DQN forward pass
state_batch = torch.randn(32, 8, device=device)  # Batch of 32 states
q_values = dqn_net(state_batch)  # Shape: [32, 4]
best_actions = torch.argmax(q_values, dim=1)  # Shape: [32]

# Policy network forward pass
action_probs = policy_net(state_batch)  # Shape: [32, 4] (probabilities)
action_dist = torch.distributions.Categorical(action_probs)
sampled_actions = action_dist.sample()  # Shape: [32]

# Value network forward pass
state_values = value_net(state_batch)  # Shape: [32]

# 8. CUSTOM NETWORK CONFIGURATION EXAMPLES

# Small network for fast training
small_config = NetworkConfig(
    input_size=8,
    hidden_layers=[32, 32],  # Smaller hidden layers
    output_size=4,
    dropout=0.0              # No dropout for small network
)

# Large network for complex tasks
large_config = NetworkConfig(
    input_size=8,
    hidden_layers=[128, 128, 64],  # Three hidden layers
    output_size=4,
    dropout=0.2,                   # Higher dropout
    batch_norm=True                # Batch normalization
)

# Network with different activation
tanh_config = NetworkConfig(
    input_size=8,
    hidden_layers=[64, 64],
    output_size=4,
    activation="tanh"              # Tanh activation instead of ReLU
)

# 9. INTEGRATION WITH TRAINING LOOPS

# Example DQN training integration
import torch.optim as optim
import torch.nn.functional as F

# Setup
config = NetworkConfig(input_size=8, hidden_layers=[64, 64], output_size=4)
q_network = DQNNetwork(config).to(device)
target_network = DQNNetwork(config).to(device)
target_network.load_state_dict(q_network.state_dict())

optimizer = optim.Adam(q_network.parameters(), lr=0.001)
replay_buffer = ExperienceReplay(10000, device)

# Training step
def train_step():
    if len(replay_buffer) < 32:
        return
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(32)
    
    current_q_values = q_network(states).gather(1, actions.unsqueeze(1))
    next_q_values = target_network(next_states).max(1)[0].detach()
    target_q_values = rewards + 0.99 * next_q_values * ~dones
    
    loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# 10. POLICY GRADIENT INTEGRATION

# Example PPO training integration
config = NetworkConfig(input_size=8, hidden_layers=[64, 64], output_size=4)
policy_network = PolicyNetwork(config).to(device)
value_network = ValueNetwork(config).to(device)

policy_optimizer = optim.Adam(policy_network.parameters(), lr=0.001)
value_optimizer = optim.Adam(value_network.parameters(), lr=0.002)

# PPO training step
def ppo_train_step(states, actions, advantages, returns, old_log_probs):
    # Policy loss
    new_probs = policy_network(states)
    new_dist = torch.distributions.Categorical(new_probs)
    new_log_probs = new_dist.log_prob(actions)
    
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss
    values = value_network(states)
    value_loss = F.mse_loss(values, returns)
    
    # Update
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

# 11. EVALUATION MODE EXAMPLES

# Set networks to evaluation mode
dqn_net.eval()
policy_net.eval()

# Disable gradients for inference
with torch.no_grad():
    state = torch.randn(1, 8, device=device)
    
    # DQN action selection
    q_values = dqn_net(state)
    action = torch.argmax(q_values).item()
    
    # Policy network action selection
    action_probs = policy_net(state)
    action = torch.argmax(action_probs).item()  # Greedy selection

# 12. COMMON TROUBLESHOOTING

# Check tensor shapes and devices
def debug_tensors():
    state = torch.randn(8)  # CPU tensor
    print(f"State shape: {state.shape}, device: {state.device}")
    
    # Move to GPU if available
    state = state.to(device)
    print(f"State shape: {state.shape}, device: {state.device}")
    
    # Check network output shapes
    output = dqn_net(state.unsqueeze(0))  # Add batch dimension
    print(f"DQN output shape: {output.shape}")

# Check model parameters
def model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

model_info(dqn_net)

# 13. PERFORMANCE OPTIMIZATION TIPS

# Use mixed precision training (if supported)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def optimized_train_step():
    with autocast():
        # Forward pass with automatic mixed precision
        q_values = q_network(states)
        loss = F.mse_loss(q_values, targets)
    
    # Scaled backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Enable benchmark mode for consistent input sizes
torch.backends.cudnn.benchmark = True

# 14. STATE REPRESENTATION CONSISTENCY

# Always use 8D state representation for consistency across all models
def ensure_state_consistency(state):
    if state.shape[-1] != 8:
        print(f"⚠️  Warning: Expected 8D state, got {state.shape[-1]}D")
        # Convert or pad as needed
    return state

# The neural_network_utils module provides:
# - Consistent network architectures across all RL algorithms
# - GPU acceleration and memory management
# - Proper weight initialization and training utilities
# - Comprehensive metrics tracking and model persistence
# - Integration helpers for DQN, PPO, and Actor-Critic training
# - Debugging and optimization utilities

# Best practices:
# 1. Always use 8D state representation for consistency
# 2. Use NetworkConfig for reproducible network architectures
# 3. Leverage GPU acceleration with verify_gpu()
# 4. Track training metrics with TrainingMetrics
# 5. Save models with metadata for better organization
# 6. Use appropriate network sizes (64-128 hidden units work well)
# 7. Apply gradient clipping and proper initialization
# 8. Set networks to eval() mode during evaluation
"""