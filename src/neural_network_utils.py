#!/usr/bin/env python3
"""
Shared neural network utilities for SnakeAI-MLOps
GPU-accelerated PyTorch implementations
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
    input_size: int = 20  # Enhanced state features
    hidden_layers: List[int] = None
    output_size: int = 4  # 4 actions
    activation: str = "relu"
    dropout: float = 0.0
    batch_norm: bool = False
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 128]

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
        base / "policy_gradient",
        base / "actor_critic",
        base / "checkpoints" / "dqn",
        base / "checkpoints" / "policy_gradient", 
        base / "checkpoints" / "actor_critic",
        base / "logs",
        base / "plots"
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created directory structure in {base}")

def encode_state_for_dqn(state_8d):
    """Convert 8D discrete state to 20D continuous for neural networks"""
    state_20d = torch.zeros(20, dtype=torch.float32)
    
    # Basic features (8D -> first 8 elements)
    state_20d[:8] = state_8d.float()
    
    # Engineered features (simulate enhanced state)
    # Distance to food (normalized)
    food_features = state_8d[4:8]  # food_left, food_right, food_up, food_down
    food_distance = torch.sum(food_features).float()  # Simple distance approximation
    state_20d[8] = food_distance / 4.0  # Normalize
    
    # Distance to walls (based on current position - simulated)
    state_20d[9:13] = torch.tensor([0.5, 0.5, 0.5, 0.5])  # Placeholder wall distances
    
    # Body density (simulated quadrant density)
    state_20d[13:17] = torch.tensor([0.1, 0.1, 0.1, 0.1])  # Placeholder body density
    
    # Snake length (normalized)
    state_20d[17] = 0.1  # Placeholder snake length
    
    # Empty spaces (normalized)  
    state_20d[18] = 0.9  # Placeholder empty spaces
    
    # Path to food (A* distance)
    state_20d[19] = food_distance / 4.0  # Use same as distance for now
    
    return state_20d

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
    config = NetworkConfig(input_size=20, hidden_layers=[128, 64], output_size=4)
    
    dqn = DQNNetwork(config, dueling=True).to(device)
    policy = PolicyNetwork(config).to(device)
    value = ValueNetwork(config).to(device)
    
    # Test forward pass
    test_state = torch.randn(1, 20, device=device)
    
    q_values = dqn(test_state)
    action_probs = policy(test_state)
    state_value = value(test_state)
    
    print(f"✅ DQN output shape: {q_values.shape}")
    print(f"✅ Policy output shape: {action_probs.shape}")
    print(f"✅ Value output shape: {state_value.shape}")
    
    # Test directory creation
    create_directories("models")