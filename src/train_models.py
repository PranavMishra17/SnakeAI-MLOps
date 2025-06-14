#!/usr/bin/env python3
"""
Main training orchestrator for SnakeAI-MLOps models
GPU-accelerated training for all ML techniques
"""
import argparse
import torch
import time
from pathlib import Path
from neural_network_utils import verify_gpu, create_directories

# Import trainers
from qlearning_trainer import train_qlearning, TrainingConfig as QConfig
from dqn_trainer import train_dqn, DQNConfig  
from policy_gradient_trainer import train_policy_gradient, PolicyGradientConfig
from actor_critic_trainer import train_actor_critic, ActorCriticConfig
from model_evaluator import ModelEvaluator

def check_gpu_requirements():
    """Verify GPU setup and requirements"""
    print("🔍 Checking GPU requirements...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device = verify_gpu()
    
    # Memory check
    total_memory = torch.cuda.get_device_properties(0).total_memory
    
    if total_memory < 2e9:  # Less than 2GB
        print(f"⚠️  Warning: Low GPU memory ({total_memory/1e9:.1f}GB)")
    
    print(f"✅ GPU Ready: {total_memory/1e9:.1f}GB total")
    return True

def train_all_models():
    """Train all ML model types with GPU acceleration"""
    if not check_gpu_requirements():
        print("❌ GPU requirements not met")
        return
    
    # Create directory structure
    create_directories("models")
    
    total_start = time.time()
    
    print(f"\n{'='*70}")
    print("🚀 STARTING COMPLETE ML PIPELINE TRAINING")
    print(f"{'='*70}")
    
    # 1. Q-Learning Models
    print(f"\n{'='*50}")
    print("📊 TRAINING Q-LEARNING MODELS")
    print(f"{'='*50}")
    
    qlearning_profiles = {
        "aggressive": QConfig(
            profile_name="aggressive",
            learning_rate=0.2,
            epsilon_start=0.3,
            epsilon_end=0.05,
            max_episodes=3000
        ),
        "balanced": QConfig(
            profile_name="balanced", 
            learning_rate=0.1,
            epsilon_start=0.2,
            epsilon_end=0.02,
            max_episodes=5000
        ),
        "conservative": QConfig(
            profile_name="conservative",
            learning_rate=0.05,
            epsilon_start=0.1,
            epsilon_end=0.01,
            max_episodes=7000
        )
    }
    
    for name, config in qlearning_profiles.items():
        print(f"\n🎯 Training Q-Learning {name}")
        start = time.time()
        train_qlearning(config)
        print(f"✅ Q-Learning {name} completed in {time.time() - start:.1f}s")
    
    # 2. DQN Models
    print(f"\n{'='*50}")
    print("🧠 TRAINING DEEP Q-NETWORK MODELS")
    print(f"{'='*50}")
    
    dqn_profiles = {
        "aggressive": DQNConfig(
            profile_name="aggressive",
            learning_rate=0.001,
            epsilon_start=1.0,
            epsilon_decay=0.99,
            max_episodes=1500,
            target_score=12
        ),
        "balanced": DQNConfig(
            profile_name="balanced",
            learning_rate=0.0005,
            epsilon_start=0.8,
            epsilon_decay=0.995,
            max_episodes=2000,
            target_score=15
        ),
        "conservative": DQNConfig(
            profile_name="conservative",
            learning_rate=0.0003,
            epsilon_start=0.5,
            epsilon_decay=0.997,
            max_episodes=2500,
            target_score=18
        )
    }
    
    for name, config in dqn_profiles.items():
        print(f"\n🎯 Training DQN {name}")
        start = time.time()
        train_dqn(config)
        print(f"✅ DQN {name} completed in {time.time() - start:.1f}s")
    
    # 3. Policy Gradient Models
    print(f"\n{'='*50}")
    print("🎭 TRAINING POLICY GRADIENT MODELS")
    print(f"{'='*50}")
    
    pg_profiles = {
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
    
    for name, config in pg_profiles.items():
        print(f"\n🎯 Training Policy Gradient {name}")
        start = time.time()
        train_policy_gradient(config)
        print(f"✅ Policy Gradient {name} completed in {time.time() - start:.1f}s")
    
    # 4. Actor-Critic Models
    print(f"\n{'='*50}")
    print("🎪 TRAINING ACTOR-CRITIC MODELS")
    print(f"{'='*50}")
    
    ac_profiles = {
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
    
    for name, config in ac_profiles.items():
        print(f"\n🎯 Training Actor-Critic {name}")
        start = time.time()
        train_actor_critic(config)
        print(f"✅ Actor-Critic {name} completed in {time.time() - start:.1f}s")
    
    total_time = time.time() - total_start
    print(f"\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
    
    # Comprehensive evaluation
    print(f"\n{'='*50}")
    print("📊 EVALUATING ALL TRAINED MODELS")
    print(f"{'='*50}")
    evaluate_all_models()

def train_single_technique(technique: str, profile: str = "balanced", episodes: int = None):
    """Train single ML technique"""
    if not check_gpu_requirements():
        return
    
    create_directories("models")
    
    if technique == "qlearning":
        config = QConfig(
            profile_name=profile,
            learning_rate=0.2 if profile == "aggressive" else 0.1 if profile == "balanced" else 0.05,
            epsilon_start=0.3 if profile == "aggressive" else 0.2 if profile == "balanced" else 0.1,
            max_episodes=episodes or (3000 if profile == "aggressive" else 5000 if profile == "balanced" else 7000)
        )
        train_qlearning(config)
        
    elif technique == "dqn":
        config = DQNConfig(
            profile_name=profile,
            learning_rate=0.001 if profile == "aggressive" else 0.0005 if profile == "balanced" else 0.0003,
            max_episodes=episodes or (1500 if profile == "aggressive" else 2000 if profile == "balanced" else 2500)
        )
        train_dqn(config)
        
    elif technique == "policy_gradient":
        config = PolicyGradientConfig(
            profile_name=profile,
            learning_rate=0.003 if profile == "aggressive" else 0.001 if profile == "balanced" else 0.0005,
            max_episodes=episodes or (2000 if profile == "aggressive" else 3000 if profile == "balanced" else 4000)
        )
        train_policy_gradient(config)
        
    elif technique == "actor_critic":
        config = ActorCriticConfig(
            profile_name=profile,
            actor_lr=0.003 if profile == "aggressive" else 0.001 if profile == "balanced" else 0.0005,
            critic_lr=0.006 if profile == "aggressive" else 0.002 if profile == "balanced" else 0.001,
            max_episodes=episodes or (2000 if profile == "aggressive" else 2500 if profile == "balanced" else 3000)
        )
        train_actor_critic(config)
        
    else:
        print(f"❌ Unknown technique: {technique}")
        print("Available: qlearning, dqn, policy_gradient, actor_critic")

def evaluate_all_models():
    """Evaluate all available models"""
    if not check_gpu_requirements():
        return
    
    evaluator = ModelEvaluator()
    model_dir = Path("models")
    
    if not model_dir.exists():
        print("❌ No models directory found")
        return
    
    # Collect all final models (not checkpoints)
    model_files = []
    
    # Q-Learning models
    for qfile in (model_dir / "qlearning").glob("qtable_*.json"):
        if "report" not in qfile.name and "checkpoint" not in qfile.name:
            model_files.append(str(qfile))
    
    # Neural network models
    for technique in ["dqn", "policy_gradient", "actor_critic"]:
        tech_dir = model_dir / technique
        if tech_dir.exists():
            for model_file in tech_dir.glob(f"{technique[:2]}*_*.pth"):
                if "best" not in model_file.name and "checkpoint" not in model_file.name:
                    model_files.append(str(model_file))
    
    if not model_files:
        print("❌ No trained models found")
        return
    
    print(f"📊 Evaluating {len(model_files)} models...")
    evaluator.compare_models(model_files, episodes=50)

def list_available_models():
    """List all available trained models"""
    model_dir = Path("models")
    if not model_dir.exists():
        print("❌ No models directory found")
        return
    
    print("📋 Available Models:")
    print("=" * 50)
    
    # Q-Learning models
    qlearning_dir = model_dir / "qlearning"
    if qlearning_dir.exists():
        print("\n🎯 Q-Learning Models:")
        for qfile in qlearning_dir.glob("qtable_*.json"):
            if "report" not in qfile.name:
                print(f"   • {qfile.name}")
    
    # Neural network models
    for technique, emoji in [("dqn", "🧠"), ("policy_gradient", "🎭"), ("actor_critic", "🎪")]:
        tech_dir = model_dir / technique
        if tech_dir.exists():
            print(f"\n{emoji} {technique.replace('_', ' ').title()} Models:")
            for model_file in tech_dir.glob("*.pth"):
                if "checkpoint" not in model_file.name:
                    print(f"   • {model_file.name}")

def main():
    parser = argparse.ArgumentParser(description="Train SnakeAI ML Models")
    parser.add_argument("--technique", 
                       choices=["qlearning", "dqn", "policy_gradient", "actor_critic", "all"],
                       default="all", help="ML technique to train")
    parser.add_argument("--profile", 
                       choices=["aggressive", "balanced", "conservative"],
                       default="balanced", help="Training profile")
    parser.add_argument("--episodes", type=int, help="Number of episodes")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate existing models")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--gpu-test", action="store_true", help="Test GPU functionality")
    
    args = parser.parse_args()
    
    if args.gpu_test:
        check_gpu_requirements()
        return
    
    if args.list:
        list_available_models()
        return
    
    if args.evaluate:
        evaluate_all_models()
        return
    
    if args.technique == "all":
        train_all_models()
    else:
        train_single_technique(args.technique, args.profile, args.episodes)

if __name__ == "__main__":
    main()

"""
Usage Examples:

# Train all models (complete pipeline)
python train_models.py --technique all

# Train specific technique
python train_models.py --technique dqn --profile balanced
python train_models.py --technique policy_gradient --profile aggressive --episodes 2000

# Evaluate all models
python train_models.py --evaluate

# List available models
python train_models.py --list

# Test GPU setup
python train_models.py --gpu-test

Individual Training:
python qlearning_trainer.py
python dqn_trainer.py  
python policy_gradient_trainer.py
python actor_critic_trainer.py

Output Structure:
models/
├── qlearning/
│   ├── qtable_aggressive.json
│   ├── qtable_balanced.json
│   └── qtable_conservative.json
├── dqn/
│   ├── dqn_aggressive.pth
│   ├── dqn_balanced.pth
│   └── dqn_conservative.pth
├── policy_gradient/
│   ├── pg_aggressive.pth
│   ├── pg_balanced.pth
│   └── pg_conservative.pth
├── actor_critic/
│   ├── ac_aggressive.pth
│   ├── ac_balanced.pth
│   └── ac_conservative.pth
└── checkpoints/
    ├── dqn/
    ├── policy_gradient/
    └── actor_critic/
"""