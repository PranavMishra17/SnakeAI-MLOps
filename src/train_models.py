#!/usr/bin/env python3
"""
Updated training orchestrator for SnakeAI-MLOps models
Focus on balanced training profiles only (removing aggressive/conservative)
"""
import argparse
import torch
import time
from pathlib import Path
from neural_network_utils import verify_gpu, create_directories

# Import trainers
from qlearning_trainer import train_qlearning, TrainingConfig as QConfig

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
    """Train all ML model types with balanced profiles only"""
    if not check_gpu_requirements():
        print("❌ GPU requirements not met")
        return
    
    # Create directory structure
    create_directories("models")
    
    total_start = time.time()
    
    print(f"\n{'='*70}")
    print("🚀 STARTING FOCUSED ML PIPELINE TRAINING")
    print("🎯 Training BALANCED profiles only for better performance")
    print(f"{'='*70}")
    
    # Import here to avoid circular imports
    try:
        from dqn_trainer import train_dqn, DQNConfig
        from policy_gradient_trainer import train_ppo, PPOConfig  # Now uses PPO
        # Note: Actor-Critic removed as requested, can be added back if needed
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the trainer files are available")
        return
    
    # 1. Q-Learning Model (Balanced only)
    print(f"\n{'='*50}")
    print("📊 TRAINING Q-LEARNING MODEL")
    print(f"{'='*50}")
    
    qlearning_config = QConfig(
        profile_name="balanced", 
        learning_rate=0.1,
        epsilon_start=0.2,
        epsilon_end=0.02,
        max_episodes=3000,
        target_score=15
    )
    
    print(f"🎯 Training Q-Learning balanced")
    start = time.time()
    train_qlearning(qlearning_config)
    print(f"✅ Q-Learning completed in {time.time() - start:.1f}s")
    
    # 2. DQN Model (Balanced only)
    print(f"\n{'='*50}")
    print("🧠 TRAINING DEEP Q-NETWORK MODEL")
    print(f"{'='*50}")
    
    dqn_config = DQNConfig(
        profile_name="balanced",
        learning_rate=0.0005,
        epsilon_start=0.9,
        epsilon_decay=0.995,
        max_episodes=1500,
        target_score=10,
        hidden_size=128
    )
    
    print(f"🎯 Training DQN balanced")
    start = time.time()
    train_dqn(dqn_config)
    print(f"✅ DQN completed in {time.time() - start:.1f}s")
    
    # 3. PPO Model (Balanced only)
    print(f"\n{'='*50}")
    print("🎭 TRAINING PPO MODEL")
    print(f"{'='*50}")
    
    ppo_config = PPOConfig(
        profile_name="balanced",
        learning_rate=0.001,  # Increased from 0.0003
        max_episodes=1500,  # Reduced from 1500
        target_score=8,  # Reduced from 12
        hidden_size=256,  # Increased from 128
        clip_epsilon=0.2,
        entropy_coeff=0.02,  # Increased from 0.01
        trajectory_length=256,  # Increased from 128
        update_epochs=6,  # Increased from 4
        batch_size=32  # Reduced from 64
    )
    
    print(f"🎯 Training PPO balanced")
    start = time.time()
    train_ppo(ppo_config)
    print(f"✅ PPO completed in {time.time() - start:.1f}s")
    
    total_time = time.time() - total_start
    print(f"\n🎉 ALL BALANCED MODELS TRAINED SUCCESSFULLY!")
    print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
    
    # Comprehensive evaluation
    print(f"\n{'='*50}")
    print("📊 EVALUATING ALL TRAINED MODELS")
    print(f"{'='*50}")
    evaluate_all_models()

def train_single_technique(technique: str, profile: str = "balanced", episodes: int = None):
    """Train single ML technique (balanced profile only)"""
    if profile != "balanced":
        print(f"⚠️  Only 'balanced' profile is supported. Switching to balanced.")
        profile = "balanced"
    
    if not check_gpu_requirements():
        return
    
    create_directories("models")
    
    if technique == "qlearning":
        config = QConfig(
            profile_name=profile,
            learning_rate=0.1,
            epsilon_start=0.2,
            epsilon_end=0.02,
            max_episodes=episodes or 3000,
            target_score=15
        )
        train_qlearning(config)
        
    elif technique == "dqn":
        try:
            from dqn_trainer import train_dqn, DQNConfig
            config = DQNConfig(
                profile_name=profile,
                learning_rate=0.0005,
                epsilon_start=0.9,
                epsilon_decay=0.995,
                max_episodes=episodes or 1500,
                target_score=10,
                hidden_size=128
            )
            train_dqn(config)
        except ImportError:
            print("❌ Fixed DQN trainer not found. Please ensure fixed_dqn_trainer.py is available.")
        
    elif technique == "ppo":
        try:
            from policy_gradient_trainer import train_ppo, PPOConfig
            config = PPOConfig(
                profile_name=profile,
                learning_rate=0.001,  # Improved settings
                max_episodes=episodes or 1500,
                target_score=8,
                hidden_size=256,
                clip_epsilon=0.2,
                entropy_coeff=0.02,
                trajectory_length=256,
                update_epochs=6,
                batch_size=32
            )
            train_ppo(config)
        except ImportError:
            print("❌ PPO trainer not found. Please ensure policy_gradient_trainer.py has PPO implementation.")
        
    else:
        print(f"❌ Unknown technique: {technique}")
        print("Available: qlearning, dqn, ppo")

def evaluate_all_models():
    """Evaluate all available models using fixed evaluator"""
    if not check_gpu_requirements():
        return
    
    try:
        from fixed_model_evaluator import FixedModelEvaluator
        evaluator = FixedModelEvaluator()
    except ImportError:
        print("❌ Fixed model evaluator not found. Please ensure fixed_model_evaluator.py is available.")
        return
    
    model_dir = Path("models")
    
    if not model_dir.exists():
        print("❌ No models directory found")
        return
    
    # Collect all final models (balanced profiles only)
    model_files = []
    
    # Q-Learning models
    qlearning_dir = model_dir / "qlearning"
    if qlearning_dir.exists():
        for qfile in qlearning_dir.glob("qtable_balanced.json"):
            model_files.append(str(qfile))
    
    # DQN models
    dqn_dir = model_dir / "dqn"
    if dqn_dir.exists():
        for dqn_file in dqn_dir.glob("dqn_balanced.pth"):
            model_files.append(str(dqn_file))
    
    # PPO models
    ppo_dir = model_dir / "ppo"
    if ppo_dir.exists():
        for ppo_file in ppo_dir.glob("ppo_balanced.pth"):
            model_files.append(str(ppo_file))
    
    if not model_files:
        print("❌ No balanced models found")
        print("Available models:")
        for technique_dir in ["qlearning", "dqn", "policy_gradient"]:
            tech_path = model_dir / technique_dir
            if tech_path.exists():
                models = list(tech_path.glob("*"))
                if models:
                    print(f"  {technique_dir}: {[m.name for m in models]}")
        return
    
    print(f"📊 Evaluating {len(model_files)} balanced models...")
    evaluator.compare_models(model_files, episodes=50)

def list_available_models():
    """List all available trained models"""
    model_dir = Path("models")
    if not model_dir.exists():
        print("❌ No models directory found")
        return
    
    print("📋 Available Models (Balanced Profiles):")
    print("=" * 50)
    
    # Q-Learning models
    qlearning_dir = model_dir / "qlearning"
    if qlearning_dir.exists():
        print("\n🎯 Q-Learning Models:")
        for qfile in qlearning_dir.glob("qtable_*.json"):
            if "report" not in qfile.name:
                print(f"   • {qfile.name}")
    
    # Neural network models
    for technique, emoji in [("dqn", "🧠"), ("ppo", "🎭")]:
        tech_dir = model_dir / technique
        if tech_dir.exists():
            print(f"\n{emoji} {technique.replace('_', ' ').title()} Models:")
            for model_file in tech_dir.glob("*.pth"):
                if "checkpoint" not in model_file.name:
                    print(f"   • {model_file.name}")

def main():
    parser = argparse.ArgumentParser(description="Train SnakeAI ML Models (Balanced Focus)")
    parser.add_argument("--technique", 
                       choices=["qlearning", "dqn", "ppo", "all"],
                       default="all", help="ML technique to train")
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
        train_single_technique(args.technique, "balanced", args.episodes)

if __name__ == "__main__":
    main()

"""
Usage Examples (Focused on Balanced Profiles):

# Train all models (balanced profiles only)
python train_models.py --technique all

# Train specific technique (balanced only)
python train_models.py --technique dqn
python train_models.py --technique ppo --episodes 2000

# Evaluate all models
python train_models.py --evaluate

# List available models
python train_models.py --list

# Test GPU setup
python train_models.py --gpu-test

Key Improvements:
1. Removed aggressive/conservative profiles (always performed worse)
2. Focused on balanced parameters that work well
3. Simplified training pipeline
4. Fixed state representations and reward structures
5. Integrated with fixed evaluator for proper assessment
6. Reduced training times while maintaining performance

Expected Performance (Balanced Profiles):
- Q-Learning: 10-20 average score
- DQN: 8-15 average score  
- PPO: 10-18 average score (better than Policy Gradient)

Output Structure:
models/
├── qlearning/
│   └── qtable_balanced.json
├── dqn/
│   └── dqn_balanced.pth
└── ppo/
    └── ppo_balanced.pth
"""