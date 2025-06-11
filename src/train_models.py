#!/usr/bin/env python3
"""
Main training orchestrator for SnakeAI-MLOps models
Replaces all C++ training functionality with GPU acceleration
"""
import argparse
import torch
import time
from pathlib import Path
from qlearning_trainer import train_qlearning, TrainingConfig, verify_gpu
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
    free_memory = torch.cuda.memory_reserved(0)
    
    if total_memory < 2e9:  # Less than 2GB
        print(f"⚠️  Warning: Low GPU memory ({total_memory/1e9:.1f}GB)")
    
    print(f"✅ GPU Ready: {total_memory/1e9:.1f}GB total")
    return True

def train_all_profiles():
    """Train all Q-learning profiles with GPU acceleration"""
    if not check_gpu_requirements():
        print("❌ GPU requirements not met")
        return
    
    profiles = {
        "aggressive": TrainingConfig(
            profile_name="aggressive",
            learning_rate=0.2,
            epsilon_start=0.3,
            epsilon_end=0.05,
            max_episodes=3000
        ),
        "balanced": TrainingConfig(
            profile_name="balanced", 
            learning_rate=0.1,
            epsilon_start=0.2,
            epsilon_end=0.02,
            max_episodes=5000
        ),
        "conservative": TrainingConfig(
            profile_name="conservative",
            learning_rate=0.05,
            epsilon_start=0.1,
            epsilon_end=0.01,
            max_episodes=7000
        )
    }
    
    total_start = time.time()
    
    for name, config in profiles.items():
        print(f"\n{'='*60}")
        print(f"🚀 Training {name.upper()} Q-Learning Agent")
        print(f"{'='*60}")
        
        start_time = time.time()
        train_qlearning(config)
        training_time = time.time() - start_time
        
        print(f"✅ {name} training completed in {training_time:.1f}s")
    
    total_time = time.time() - total_start
    print(f"\n🎉 All models trained in {total_time:.1f}s")
    
    # Auto-evaluate after training
    print("\n📊 Evaluating trained models...")
    evaluator = ModelEvaluator()
    model_paths = [f"models/qtable_{name}.json" for name in profiles.keys()]
    evaluator.compare_models(model_paths, episodes=50)

def train_single_profile(profile_name: str, episodes: int = None):
    """Train a single profile"""
    if not check_gpu_requirements():
        return
    
    configs = {
        "aggressive": TrainingConfig(
            profile_name="aggressive", learning_rate=0.2, 
            epsilon_start=0.3, max_episodes=episodes or 3000
        ),
        "balanced": TrainingConfig(
            profile_name="balanced", learning_rate=0.1,
            epsilon_start=0.2, max_episodes=episodes or 5000
        ),
        "conservative": TrainingConfig(
            profile_name="conservative", learning_rate=0.05,
            epsilon_start=0.1, max_episodes=episodes or 7000
        )
    }
    
    if profile_name not in configs:
        print(f"❌ Invalid profile: {profile_name}")
        print(f"Available: {list(configs.keys())}")
        return
    
    config = configs[profile_name]
    print(f"🚀 Training {profile_name} Q-Learning Agent")
    train_qlearning(config)

def evaluate_models():
    """Evaluate all available models"""
    if not check_gpu_requirements():
        return
    
    evaluator = ModelEvaluator()
    model_dir = Path("models")
    
    if not model_dir.exists():
        print("❌ No models directory found")
        return
    
    # Get final models only (not checkpoints or reports)
    model_files = []
    for model_file in model_dir.glob("qtable_*.json"):
        if "report" not in model_file.name and "checkpoint" not in model_file.name:
            model_files.append(model_file)
    
    if not model_files:
        print("❌ No trained models found")
        print("💡 Available files:")
        for f in model_dir.iterdir():
            print(f"   {f.name}")
        return
    
    print(f"📊 Evaluating {len(model_files)} final models...")
    evaluator.compare_models([str(f) for f in model_files], episodes=100)

def main():
    parser = argparse.ArgumentParser(description="Train SnakeAI Q-Learning Models")
    parser.add_argument("--profile", choices=["aggressive", "balanced", "conservative", "all"],
                       default="all", help="Training profile")
    parser.add_argument("--episodes", type=int, help="Number of episodes")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate existing models")
    parser.add_argument("--gpu-test", action="store_true", help="Test GPU functionality")
    
    args = parser.parse_args()
    
    if args.gpu_test:
        check_gpu_requirements()
        return
    
    if args.evaluate:
        evaluate_models()
        return
    
    if args.profile == "all":
        train_all_profiles()
    else:
        train_single_profile(args.profile, args.episodes)

if __name__ == "__main__":
    main()

    """
    Setup
bash# Install dependencies
pip install -r requirements.txt

# Verify GPU
python train_models.py --gpu-test
Training
bash# Train all models (aggressive, balanced, conservative)
python train_models.py --profile all

# Train specific model
python train_models.py --profile balanced --episodes 3000

# Evaluate existing models
python train_models.py --evaluate
Individual Scripts
bash# Direct training
python qlearning_trainer.py

# Model evaluation only
python model_evaluator.py
📁 Output Files for C++ Game:

models/qtable_aggressive.json - Fast learning model
models/qtable_balanced.json - Stable model
models/qtable_conservative.json - Careful model

These JSON files are automatically loaded by existing C++ MLAgents.cpp code.
    
    
    """