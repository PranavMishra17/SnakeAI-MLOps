#!/usr/bin/env python3
"""
Convert PyTorch .pth models to TorchScript .pt format for C++ LibTorch compatibility
Processes all trained RL models and saves them as .pt files in the same directories
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
from typing import Dict, Any

# Define network architectures (same as in training scripts)
class SimpleDQN(nn.Module):
    """DQN Network for conversion"""
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SimplePolicyNetwork(nn.Module):
    """Policy Network for PPO/Actor-Critic"""
    def __init__(self, input_size, hidden_size, output_size):
        super(SimplePolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class SimpleValueNetwork(nn.Module):
    """Value Network for PPO/Actor-Critic"""
    def __init__(self, input_size, hidden_size):
        super(SimpleValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)

def get_network_config(checkpoint: Dict[str, Any]) -> Dict[str, int]:
    """Extract network configuration from checkpoint"""
    config = checkpoint.get('config', {})
    
    # Determine input size from saved weights
    if 'q_network' in checkpoint:
        first_layer_weight = checkpoint['q_network']['fc1.weight']
    elif 'policy_network' in checkpoint:
        first_layer_weight = checkpoint['policy_network']['fc1.weight']
    elif 'actor_state_dict' in checkpoint:
        first_layer_weight = checkpoint['actor_state_dict']['fc1.weight']
    else:
        raise ValueError("Cannot determine network architecture")
    
    input_size = first_layer_weight.shape[1]
    hidden_size = config.get('hidden_size', 64)
    
    return {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': 4  # Always 4 actions
    }

def convert_dqn_model(pth_path: Path, device: torch.device) -> bool:
    """Convert DQN .pth to .pt"""
    try:
        print(f"Converting DQN: {pth_path.name}")
        
        # Load checkpoint
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
        config = get_network_config(checkpoint)
        
        # Create and load model
        model = SimpleDQN(config['input_size'], config['hidden_size'], config['output_size'])
        model.load_state_dict(checkpoint['q_network'], strict=False)
        model.eval()
        
        # Create example input for tracing
        example_input = torch.randn(1, config['input_size'])
        
        # Convert to TorchScript
        traced_model = torch.jit.trace(model, example_input)
        
        # Save as .pt
        pt_path = pth_path.with_suffix('.pt')
        traced_model.save(str(pt_path))
        
        print(f"✅ DQN saved: {pt_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to convert DQN {pth_path.name}: {e}")
        return False

def convert_ppo_model(pth_path: Path, device: torch.device) -> bool:
    """Convert PPO .pth to .pt (saves policy and value separately)"""
    try:
        print(f"Converting PPO: {pth_path.name}")
        
        # Load checkpoint
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
        config = get_network_config(checkpoint)
        
        # Create and load policy network
        policy_model = SimplePolicyNetwork(config['input_size'], config['hidden_size'], config['output_size'])
        policy_model.load_state_dict(checkpoint['policy_network'], strict=False)
        policy_model.eval()
        
        # Create and load value network
        value_model = SimpleValueNetwork(config['input_size'], config['hidden_size'])
        value_model.load_state_dict(checkpoint['value_network'], strict=False)
        value_model.eval()
        
        # Create example input
        example_input = torch.randn(1, config['input_size'])
        
        # Convert to TorchScript
        traced_policy = torch.jit.trace(policy_model, example_input)
        traced_value = torch.jit.trace(value_model, example_input)
        
        # Save policy and value separately
        policy_path = pth_path.with_name(pth_path.stem + '_policy.pt')
        value_path = pth_path.with_name(pth_path.stem + '_value.pt')
        
        traced_policy.save(str(policy_path))
        traced_value.save(str(value_path))
        
        print(f"✅ PPO saved: {policy_path.name}, {value_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to convert PPO {pth_path.name}: {e}")
        return False

def convert_actor_critic_model(pth_path: Path, device: torch.device) -> bool:
    """Convert Actor-Critic .pth to .pt (saves actor and critic separately)"""
    try:
        print(f"Converting Actor-Critic: {pth_path.name}")
        
        # Load checkpoint
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
        config = get_network_config(checkpoint)
        
        # Create and load actor network
        actor_model = SimplePolicyNetwork(config['input_size'], config['hidden_size'], config['output_size'])
        actor_model.load_state_dict(checkpoint['actor_state_dict'], strict=False)
        actor_model.eval()
        
        # Create and load critic network
        critic_model = SimpleValueNetwork(config['input_size'], config['hidden_size'])
        critic_model.load_state_dict(checkpoint['critic_state_dict'], strict=False)
        critic_model.eval()
        
        # Create example input
        example_input = torch.randn(1, config['input_size'])
        
        # Convert to TorchScript
        traced_actor = torch.jit.trace(actor_model, example_input)
        traced_critic = torch.jit.trace(critic_model, example_input)
        
        # Save actor and critic separately
        actor_path = pth_path.with_name(pth_path.stem + '_actor.pt')
        critic_path = pth_path.with_name(pth_path.stem + '_critic.pt')
        
        traced_actor.save(str(actor_path))
        traced_critic.save(str(critic_path))
        
        print(f"✅ Actor-Critic saved: {actor_path.name}, {critic_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to convert Actor-Critic {pth_path.name}: {e}")
        return False

def convert_all_models():
    """Convert all .pth models to .pt format"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_dir = Path("../models")
    
    if not models_dir.exists():
        print("❌ Models directory not found")
        return
    
    print(f"🔄 Converting PyTorch models to TorchScript format")
    print(f"Device: {device}")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    # Convert DQN models
    dqn_dir = models_dir / "dqn"
    if dqn_dir.exists():
        for pth_file in dqn_dir.glob("*.pth"):
            if "checkpoint" not in pth_file.name:  # Skip checkpoints
                total_count += 1
                if convert_dqn_model(pth_file, device):
                    success_count += 1
    
    # Convert PPO models
    ppo_dir = models_dir / "ppo"
    if ppo_dir.exists():
        for pth_file in ppo_dir.glob("*.pth"):
            if "checkpoint" not in pth_file.name:
                total_count += 1
                if convert_ppo_model(pth_file, device):
                    success_count += 1
    
    # Convert Actor-Critic models
    ac_dir = models_dir / "actor_critic"
    if ac_dir.exists():
        for pth_file in ac_dir.glob("*.pth"):
            if "checkpoint" not in pth_file.name:
                total_count += 1
                if convert_actor_critic_model(pth_file, device):
                    success_count += 1
    
    print("\n" + "=" * 60)
    print(f"🎉 Conversion complete: {success_count}/{total_count} models converted")
    
    if success_count > 0:
        print(f"\n📁 TorchScript models saved in respective directories:")
        print(f"   • DQN: models/dqn/*.pt")
        print(f"   • PPO: models/ppo/*_policy.pt, *_value.pt")
        print(f"   • Actor-Critic: models/actor_critic/*_actor.pt, *_critic.pt")
        
        print(f"\n🚀 Ready for C++ LibTorch integration!")
        print(f"   Example C++ loading:")
        print(f"   torch::jit::script::Module model = torch::jit::load(\"model.pt\");")

def list_convertible_models():
    """List all .pth models that can be converted"""
    models_dir = Path("../models")
    
    if not models_dir.exists():
        print("❌ Models directory not found")
        return
    
    print("🔍 Available .pth models for conversion:")
    print("=" * 50)
    
    total = 0
    
    for technique in ["dqn", "ppo", "actor_critic"]:
        tech_dir = models_dir / technique
        if tech_dir.exists():
            pth_files = [f for f in tech_dir.glob("*.pth") if "checkpoint" not in f.name]
            if pth_files:
                print(f"\n{technique.upper()}:")
                for pth_file in pth_files:
                    print(f"   • {pth_file.name}")
                    total += 1
    
    print(f"\nTotal: {total} models")

def verify_conversion(models_dir: Path = None):
    """Verify that .pt models can be loaded"""
    if models_dir is None:
        models_dir = Path("../models")
    
    print("🔍 Verifying converted .pt models:")
    print("=" * 40)
    
    for technique in ["dqn", "ppo", "actor_critic"]:
        tech_dir = models_dir / technique
        if tech_dir.exists():
            pt_files = list(tech_dir.glob("*.pt"))
            if pt_files:
                print(f"\n{technique.upper()}:")
                for pt_file in pt_files:
                    try:
                        model = torch.jit.load(str(pt_file))
                        print(f"   ✅ {pt_file.name}")
                        
                        # Test inference
                        if "_policy" in pt_file.name or "_actor" in pt_file.name or technique == "dqn":
                            test_input = torch.randn(1, 8)  # Assuming 8D input
                            with torch.no_grad():
                                output = model(test_input)
                            print(f"      Output shape: {output.shape}")
                        
                    except Exception as e:
                        print(f"   ❌ {pt_file.name}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert RL models to TorchScript")
    parser.add_argument("--list", action="store_true", help="List convertible models")
    parser.add_argument("--verify", action="store_true", help="Verify converted models")
    parser.add_argument("--convert", action="store_true", help="Convert all models")
    
    args = parser.parse_args()
    
    if args.list:
        list_convertible_models()
    elif args.verify:
        verify_conversion()
    elif args.convert:
        convert_all_models()
    else:
        # Default: convert all
        convert_all_models()

        """
        
        Key features:

Converts DQN, PPO, and Actor-Critic models automatically
Handles different network architectures and input sizes
Saves .pt files in same directories as .pth files
For multi-network models: PPO saves _policy.pt + _value.pt, Actor-Critic saves _actor.pt + _critic.pt

Usage:
bash# Convert all models
python convert_models_to_torchscript.py

# List available models first
python convert_models_to_torchscript.py --list

# Verify converted models work
python convert_models_to_torchscript.py --verify
What you'll get:

models/dqn/dqn_balanced.pt
models/ppo/ppo_balanced_policy.pt + ppo_balanced_value.pt
models/actor_critic/ac_balanced_actor.pt + ac_balanced_critic.pt

C++ loading example:
cpptorch::jit::script::Module model = torch::jit::load("dqn_balanced.pt");
torch::Tensor input = torch::randn({1, 8});  // 8D state
torch::Tensor output = model.forward({input}).toTensor();
The script automatically detects input sizes from your saved weights and handles the conversion safely.


        
        """