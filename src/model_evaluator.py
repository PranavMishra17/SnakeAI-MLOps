#!/usr/bin/env python3
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from qlearning_trainer import SnakeEnvironment, verify_gpu
from typing import List, Dict

class ModelEvaluator:
    """GPU-accelerated model evaluation and comparison"""
    
    def __init__(self):
        self.device = verify_gpu()
        print(f"✅ ModelEvaluator initialized on {self.device}")
    
    def load_model(self, model_path: str) -> torch.Tensor:
        """Load Q-table model from JSON"""
        with open(model_path, 'r') as f:
            data = json.load(f)
        
        # Initialize empty Q-table on GPU (9-bit states = 512 possible states)
        q_table = torch.zeros((512, 4), device=self.device)
        
        # Load Q-values
        for state_str, actions in data["qTable"].items():
            state_idx = int(state_str, 2)  # Binary string to int (supports both 8-bit and 9-bit)
            if state_idx < 512:  # Ensure valid index for 512-state Q-table
                q_table[state_idx] = torch.tensor(actions, device=self.device)
        
        print(f"✅ Model loaded: {model_path}")
        return q_table
    
    def encode_state(self, state):
        """Convert 8D state to Q-table index (matches training encoding)"""
        # Convert each element to binary representation
        binary_parts = []
        
        # Danger flags (3 bits)
        binary_parts.append(str(int(state[0].item())))  # danger_straight
        binary_parts.append(str(int(state[1].item())))  # danger_left
        binary_parts.append(str(int(state[2].item())))  # danger_right
        
        # Direction (2 bits for 4 values: 00, 01, 10, 11)
        direction = int(state[3].item())
        direction_binary = format(direction, '02b')  # Convert 0-3 to 2-bit binary
        binary_parts.append(direction_binary)
        
        # Food flags (4 bits)
        binary_parts.append(str(int(state[4].item())))  # food_left
        binary_parts.append(str(int(state[5].item())))  # food_right
        binary_parts.append(str(int(state[6].item())))  # food_up
        binary_parts.append(str(int(state[7].item())))  # food_down
        
        # Combine into single binary string (9 bits total)
        binary_str = ''.join(binary_parts)
        
        # Convert to integer index (max 511 for 9 bits)
        return int(binary_str, 2)
    
    def evaluate_model(self, model_path: str, episodes: int = 100) -> Dict:
        """Evaluate single model performance"""
        q_table = self.load_model(model_path)
        env = SnakeEnvironment(device=str(self.device))
        
        scores = []
        episode_lengths = []
        
        print(f"🧪 Evaluating {model_path} over {episodes} episodes...")
        
        for episode in range(episodes):
            state = env.reset()
            steps = 0
            
            while True:
                # Get best action (no exploration)
                state_idx = self.encode_state(state)
                action = torch.argmax(q_table[state_idx]).item()
                
                state, reward, done = env.step(action)
                steps += 1
                
                if done:
                    break
            
            scores.append(env.score)
            episode_lengths.append(steps)
        
        results = {
            "model": Path(model_path).name,
            "episodes": episodes,
            "avg_score": float(np.mean(scores)),
            "max_score": int(max(scores)),
            "min_score": int(min(scores)),
            "std_score": float(np.std(scores)),
            "avg_length": float(np.mean(episode_lengths)),
            "scores": scores
        }
        
        print(f"✅ {results['model']}: Avg={results['avg_score']:.2f}, Max={results['max_score']}")
        return results
    
    def compare_models(self, model_paths: List[str], episodes: int = 100):
        """Compare multiple models with GPU acceleration"""
        results = []
        
        for path in model_paths:
            if Path(path).exists():
                result = self.evaluate_model(path, episodes)
                results.append(result)
        
        # Generate comparison plots
        self._plot_comparison(results)
        self._save_comparison_report(results)
        
        return results
    
    def list_available_models(self, include_checkpoints=False):
        """List all available models in proper directory structure"""
        model_dir = Path("models")
        models = []
        
        if not model_dir.exists():
            return models
            
        # Final models
        for model_file in model_dir.glob("qtable_*.json"):
            if "report" not in model_file.name and "checkpoint" not in model_file.name:
                models.append({
                    "type": "final",
                    "path": str(model_file),
                    "profile": model_file.stem.replace("qtable_", "")
                })
        
        # Checkpoints if requested
        if include_checkpoints:
            for checkpoint_dir in model_dir.glob("*_checkpoints"):
                for checkpoint_file in checkpoint_dir.glob("*.json"):
                    models.append({
                        "type": "checkpoint", 
                        "path": str(checkpoint_file),
                        "profile": checkpoint_dir.name.replace("_checkpoints", "")
                    })
        
        return models
    
    def _plot_comparison(self, results: List[Dict]):
        """Generate comparison visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        models = [r["model"] for r in results]
        avg_scores = [r["avg_score"] for r in results]
        max_scores = [r["max_score"] for r in results]
        std_scores = [r["std_score"] for r in results]
        
        # Average scores
        axes[0,0].bar(models, avg_scores, color=['red', 'blue', 'green'][:len(models)])
        axes[0,0].set_title('Average Scores')
        axes[0,0].set_ylabel('Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Max scores
        axes[0,1].bar(models, max_scores, color=['red', 'blue', 'green'][:len(models)])
        axes[0,1].set_title('Maximum Scores')
        axes[0,1].set_ylabel('Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Score distributions
        for i, result in enumerate(results):
            axes[1,0].hist(result["scores"], alpha=0.6, label=result["model"], bins=20)
        axes[1,0].set_title('Score Distributions')
        axes[1,0].set_xlabel('Score')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # Standard deviations
        axes[1,1].bar(models, std_scores, color=['red', 'blue', 'green'][:len(models)])
        axes[1,1].set_title('Score Variability (Std Dev)')
        axes[1,1].set_ylabel('Standard Deviation')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig("models/model_comparison.png", dpi=300, bbox_inches='tight')
        print("✅ Comparison plots saved: models/model_comparison.png")
    
    def _save_comparison_report(self, results: List[Dict]):
        """Save detailed comparison report"""
        report = {
            "comparison_summary": {
                "total_models": len(results),
                "evaluation_date": str(np.datetime64('now')),
                "gpu_device": str(self.device)
            },
            "model_rankings": sorted(results, key=lambda x: x["avg_score"], reverse=True),
            "detailed_results": results
        }
        
        with open("models/comparison_report.json", 'w') as f:
            json.dump(report, f, indent=4)
        
        print("✅ Comparison report saved: models/comparison_report.json")

def benchmark_gpu_performance():
    """Benchmark GPU vs CPU performance for Q-learning"""
    print("🚀 Benchmarking GPU vs CPU performance...")
    
    # Test GPU
    device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q_table_gpu = torch.randn(256, 4, device=device_gpu)
    
    # Time GPU operations
    import time
    start = time.time()
    for _ in range(10000):
        state_idx = torch.randint(0, 256, (1,), device=device_gpu)
        action = torch.argmax(q_table_gpu[state_idx])
        q_table_gpu[state_idx, action] += 0.1 * torch.randn(1, device=device_gpu)
    gpu_time = time.time() - start
    
    # Test CPU
    q_table_cpu = torch.randn(256, 4, device='cpu')
    start = time.time()
    for _ in range(10000):
        state_idx = torch.randint(0, 256, (1,), device='cpu')
        action = torch.argmax(q_table_cpu[state_idx])
        q_table_cpu[state_idx, action] += 0.1 * torch.randn(1, device='cpu')
    cpu_time = time.time() - start
    
    speedup = cpu_time / gpu_time
    print(f"✅ GPU Time: {gpu_time:.3f}s")
    print(f"✅ CPU Time: {cpu_time:.3f}s") 
    print(f"✅ GPU Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    # Benchmark performance
    benchmark_gpu_performance()
    
    # Evaluate models
    evaluator = ModelEvaluator()
    
    model_paths = [
        "models/qtable_aggressive.json",
        "models/qtable_balanced.json", 
        "models/qtable_conservative.json"
    ]
    
    # Filter existing models
    existing_models = [p for p in model_paths if Path(p).exists()]
    
    if existing_models:
        print(f"\n📊 Comparing {len(existing_models)} models...")
        results = evaluator.compare_models(existing_models, episodes=100)
        
        # Print summary
        print("\n🏆 Model Rankings:")
        for i, result in enumerate(sorted(results, key=lambda x: x["avg_score"], reverse=True)):
            print(f"{i+1}. {result['model']}: {result['avg_score']:.2f} avg score")
    else:
        print("❌ No trained models found. Run qlearning_trainer.py first.")