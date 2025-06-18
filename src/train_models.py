#!/usr/bin/env python3
"""
Main training orchestrator for SnakeAI-MLOps models
Trains Q-Learning, DQN, PPO, and Actor-Critic models
"""
import argparse
import torch
import time
from pathlib import Path
from neural_network_utils import verify_gpu, create_directories

# Import trainers
from qlearning_trainer import train_qlearning, TrainingConfig as QConfig

def check_gpu_requirements():
    """Verify GPU setup"""
    print("🔍 Checking GPU requirements...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device = verify_gpu()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    
    if total_memory < 2e9:
        print(f"⚠️  Warning: Low GPU memory ({total_memory/1e9:.1f}GB)")
    
    print(f"✅ GPU Ready: {total_memory/1e9:.1f}GB total")
    return True

def train_all_models():
    """Train all ML model types with balanced profiles"""
    if not check_gpu_requirements():
        print("❌ GPU requirements not met")
        return
    
    create_directories("models")
    total_start = time.time()
    
    print(f"\n{'='*70}")
    print("🚀 STARTING UNIFIED ML PIPELINE TRAINING")
    print("🎯 Training BALANCED profiles: Q-Learning, DQN, PPO, Actor-Critic")
    print(f"{'='*70}")
    
    # Import here to avoid circular imports
    try:
        from dqn_trainer import train_dqn, DQNConfig
        from ppo_trainer import train_ppo, PPOConfig  # Updated import
        from actor_critic_trainer import train_actor_critic, ActorCriticConfig
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # 1. Q-Learning Model
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
    
    start = time.time()
    train_qlearning(qlearning_config)
    print(f"✅ Q-Learning completed in {time.time() - start:.1f}s")
    
    # 2. DQN Model
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
    
    start = time.time()
    train_dqn(dqn_config)
    print(f"✅ DQN completed in {time.time() - start:.1f}s")
    
    # 3. PPO Model
    print(f"\n{'='*50}")
    print("🎭 TRAINING PPO MODEL")
    print(f"{'='*50}")
    
    ppo_config = PPOConfig(
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
    
    start = time.time()
    train_ppo(ppo_config)
    print(f"✅ PPO completed in {time.time() - start:.1f}s")
    
    # 4. Actor-Critic Model
    print(f"\n{'='*50}")
    print("🎪 TRAINING ACTOR-CRITIC MODEL")
    print(f"{'='*50}")
    
    ac_config = ActorCriticConfig(
        profile_name="balanced",
        actor_lr=0.001,
        critic_lr=0.002,
        entropy_coeff=0.01,
        max_episodes=2500,
        target_score=13,
        n_step=5,
        gae_lambda=0.95
    )
    
    start = time.time()
    train_actor_critic(ac_config)
    print(f"✅ Actor-Critic completed in {time.time() - start:.1f}s")
    
    total_time = time.time() - total_start
    print(f"\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
    
    # Evaluation
    print(f"\n{'='*50}")
    print("📊 EVALUATING ALL TRAINED MODELS")
    print(f"{'='*50}")
    evaluate_all_models()

def train_single_technique(technique: str, profile: str = "balanced", episodes: int = None):
    """Train single ML technique"""
    if profile != "balanced":
        print(f"⚠️  Only 'balanced' profile supported. Using balanced.")
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
            print("❌ DQN trainer not found")
        
    elif technique == "ppo":
        try:
            from ppo_trainer import train_ppo, PPOConfig
            config = PPOConfig(
                profile_name=profile,
                learning_rate=0.001,
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
            print("❌ PPO trainer not found")
        
    elif technique == "actor_critic":
        try:
            from actor_critic_trainer import train_actor_critic, ActorCriticConfig
            config = ActorCriticConfig(
                profile_name=profile,
                actor_lr=0.001,
                critic_lr=0.002,
                entropy_coeff=0.01,
                max_episodes=episodes or 2500,
                target_score=13
            )
            train_actor_critic(config)
        except ImportError:
            print("❌ Actor-Critic trainer not found")
        
    else:
        print(f"❌ Unknown technique: {technique}")
        print("Available: qlearning, dqn, ppo, actor_critic")

def evaluate_all_models():
    """Evaluate all available models using unified evaluator"""
    if not check_gpu_requirements():
        return
    
    try:
        from evaluator import UnifiedModelEvaluator
        evaluator = UnifiedModelEvaluator()
    except ImportError:
        print("❌ Unified evaluator not found")
        return
    
    model_dir = Path("models")
    
    if not model_dir.exists():
        print("❌ No models directory found")
        return
    
    # Find all balanced models
    model_files = []
    
    # Q-Learning models
    qlearning_dir = model_dir / "qlearning"
    if qlearning_dir.exists():
        for qfile in qlearning_dir.glob("qtable_balanced.json"):
            model_files.append(str(qfile))
    
    # Neural network models
    for technique in ["dqn", "ppo", "actor_critic"]:
        tech_dir = model_dir / technique
        if tech_dir.exists():
            for model_file in tech_dir.glob(f"*_balanced.pth"):
                model_files.append(str(model_file))
    
    if not model_files:
        print("❌ No balanced models found")
        return
    
    print(f"📊 Evaluating {len(model_files)} balanced models...")
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
    for technique, emoji in [("dqn", "🧠"), ("ppo", "🎭"), ("actor_critic", "🎪")]:
        tech_dir = model_dir / technique
        if tech_dir.exists():
            print(f"\n{emoji} {technique.upper()} Models:")
            for model_file in tech_dir.glob("*.pth"):
                if "checkpoint" not in model_file.name:
                    print(f"   • {model_file.name}")

def main():
    parser = argparse.ArgumentParser(description="Train SnakeAI ML Models")
    parser.add_argument("--technique", 
                       choices=["qlearning", "dqn", "ppo", "actor_critic", "all"],
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
Usage Examples:

# Train all models (balanced profiles)
python train_models.py --technique all

# Train specific technique
python train_models.py --technique dqn
python train_models.py --technique ppo --episodes 2000

# Evaluate all models
python train_models.py --evaluate

# List available models
python train_models.py --list

# Test GPU setup
python train_models.py --gpu-test

Expected Performance:
- Q-Learning: 10-20 average score
- DQN: 8-15 average score  
- PPO: 10-18 average score
- Actor-Critic: 12-20 average score

Output Structure:
models/
├── qlearning/
│   └── qtable_balanced.json
├── dqn/
│   └── dqn_balanced.pth
├── ppo/
│   └── ppo_balanced.pth
└── actor_critic/
    └── ac_balanced.pth
"""