"""
Fixed Model evaluator with proper state representation for DQN
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from collections import defaultdict
import pandas as pd

from neural_network_utils import verify_gpu
from qlearning_trainer import SnakeEnvironment

class SimpleDQN(torch.nn.Module):
    """Copy of SimpleDQN for loading models"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleDQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class FixedModelEvaluator:
    """Fixed model evaluation with proper state handling"""
    
    def __init__(self):
        self.device = verify_gpu()
        print(f"✅ Fixed ModelEvaluator initialized on {self.device}")
    
    def load_qlearning_model(self, model_path: str) -> torch.Tensor:
        """Load Q-Learning model from JSON"""
        with open(model_path, 'r') as f:
            data = json.load(f)
        
        # Initialize Q-table on GPU (9-bit states = 512 possible states)
        q_table = torch.zeros((512, 4), device=self.device)
        
        # Load Q-values
        for state_str, actions in data["qTable"].items():
            state_idx = int(state_str, 2)  # Binary string to int
            if state_idx < 512:
                q_table[state_idx] = torch.tensor(actions, device=self.device)
        
        print(f"✅ Q-Learning model loaded: {model_path}")
        return q_table
    
    def load_dqn_model(self, model_path: str):
        """Load DQN model with proper architecture detection"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config_dict = checkpoint['config']
        
        # Get network parameters
        hidden_size = config_dict.get('hidden_size', 128)
        
        # Create model with correct architecture
        # Check the saved state dict to determine input size
        saved_state = checkpoint['q_network']
        input_size = saved_state['fc1.weight'].shape[1]  # Get actual input size from saved weights
        
        model = SimpleDQN(input_size, hidden_size, 4)
        model.load_state_dict(saved_state)
        model.to(self.device)
        model.eval()
        
        print(f"✅ DQN model loaded: {model_path} (input_size: {input_size})")
        return model, checkpoint.get('metadata', {}), input_size
    
    def load_policy_gradient_model(self, model_path: str):
        """Load Policy Gradient model"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config_dict = checkpoint['config']
        
        # Get network parameters
        hidden_size = config_dict.get('hidden_size', 128)
        
        # Check the saved state dict to determine input size
        saved_state = checkpoint['policy_network']
        input_size = saved_state['fc1.weight'].shape[1]
        
        # Simple policy network for evaluation
        class SimplePolicyNetwork(torch.nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(SimplePolicyNetwork, self).__init__()
                self.fc1 = torch.nn.Linear(input_size, hidden_size)
                self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
                self.fc3 = torch.nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return F.softmax(x, dim=-1)
        
        model = SimplePolicyNetwork(input_size, hidden_size, 4)
        model.load_state_dict(saved_state)
        model.to(self.device)
        model.eval()
        
        print(f"✅ Policy Gradient model loaded: {model_path} (input_size: {input_size})")
        return model, checkpoint.get('metadata', {}), input_size
    
    def encode_state_qlearning(self, state):
        """Convert 8D state to Q-table index for Q-Learning"""
        binary_parts = []
        
        # Danger flags (3 bits)
        binary_parts.append(str(int(state[0].item())))
        binary_parts.append(str(int(state[1].item())))
        binary_parts.append(str(int(state[2].item())))
        
        # Direction (2 bits)
        direction = int(state[3].item())
        direction_binary = format(direction, '02b')
        binary_parts.append(direction_binary)
        
        # Food flags (4 bits)
        binary_parts.append(str(int(state[4].item())))
        binary_parts.append(str(int(state[5].item())))
        binary_parts.append(str(int(state[6].item())))
        binary_parts.append(str(int(state[7].item())))
        
        binary_str = ''.join(binary_parts)
        return int(binary_str, 2)
    
    def convert_to_dqn_state(self, basic_state, env):
        """Convert 8D basic state to DQN format (11D)"""
        head_x, head_y = env.snake[0]
        food_x, food_y = env.food
        
        # Get food distance
        food_distance = abs(head_x - food_x) + abs(head_y - food_y)
        normalized_distance = food_distance / (2 * env.grid_size)
        
        # Wall distances (normalized)
        wall_distances = [
            head_y / env.grid_size,  # distance to top
            (env.grid_size - 1 - head_y) / env.grid_size,  # distance to bottom
            head_x / env.grid_size,  # distance to left
            (env.grid_size - 1 - head_x) / env.grid_size,  # distance to right
        ]
        
        # Snake length and empty spaces
        snake_length = len(env.snake) / (env.grid_size * env.grid_size)
        empty_spaces = (env.grid_size * env.grid_size - len(env.snake) - 1) / (env.grid_size * env.grid_size)
        
        # Create 11D state vector to match DQN training
        dqn_state = torch.tensor([
            basic_state[0].item(),  # danger_straight
            basic_state[1].item(),  # danger_left
            basic_state[2].item(),  # danger_right
            basic_state[3].item() / 3.0,  # normalized direction
            basic_state[4].item(),  # food_left
            basic_state[5].item(),  # food_right
            basic_state[6].item(),  # food_up
            basic_state[7].item(),  # food_down
            normalized_distance,
            snake_length,
            empty_spaces
        ], dtype=torch.float32, device=self.device)
        
        return dqn_state
    
    def evaluate_model(self, model_path: str, model_type: str, episodes: int = 100) -> Dict:
        """Evaluate single model with fixed state handling"""
        env = SnakeEnvironment(device=str(self.device))
        
        # Load model
        if model_type == "qlearning":
            model = self.load_qlearning_model(model_path)
            metadata = {}
            input_size = None
        elif model_type == "dqn":
            model, metadata, input_size = self.load_dqn_model(model_path)
        elif model_type == "policy_gradient" or model_type == "ppo":
            model, metadata, input_size = self.load_ppo_model(model_path)
        else:
            print(f"❌ Model type {model_type} not supported in fixed evaluator")
            return {}
        
        # Evaluation metrics
        scores = []
        episode_lengths = []
        action_distributions = [0, 0, 0, 0]
        death_causes = defaultdict(int)
        
        print(f"🧪 Evaluating {model_type} model over {episodes} episodes...")
        
        for episode in range(episodes):
            state = env.reset()
            steps = 0
            
            while True:
                # Get action based on model type
                if model_type == "qlearning":
                    state_idx = self.encode_state_qlearning(state)
                    action = torch.argmax(model[state_idx]).item()
                    
                elif model_type == "dqn":
                    # Convert to DQN state format
                    dqn_state = self.convert_to_dqn_state(state, env)
                    with torch.no_grad():
                        q_values = model(dqn_state.unsqueeze(0))
                        action = torch.argmax(q_values).item()
                
                elif model_type == "policy_gradient" or model_type == "ppo":
                    # Convert to PPO state format (same as DQN)
                    ppo_state = self.convert_to_dqn_state(state, env)
                    with torch.no_grad():
                        action_probs = model(ppo_state.unsqueeze(0))
                        action = torch.argmax(action_probs).item()  # Greedy action for evaluation
                
                action_distributions[action] += 1
                state, reward, done = env.step(action)
                steps += 1
                
                if done:
                    # Determine death cause
                    if steps >= 1000:
                        death_causes['timeout'] += 1
                    elif env.score == 0:
                        death_causes['early_death'] += 1
                    else:
                        death_causes['collision'] += 1
                    break
            
            scores.append(env.score)
            episode_lengths.append(steps)
        
        # Calculate statistics
        results = {
            "model_path": model_path,
            "model_type": model_type,
            "episodes": episodes,
            
            # Basic metrics
            "avg_score": float(np.mean(scores)),
            "max_score": int(max(scores)),
            "min_score": int(min(scores)),
            "std_score": float(np.std(scores)),
            "median_score": float(np.median(scores)),
            
            # Performance metrics
            "avg_length": float(np.mean(episode_lengths)),
            "max_length": int(max(episode_lengths)),
            
            # Behavioral metrics
            "action_distribution": action_distributions,
            "action_entropy": self._calculate_action_entropy(action_distributions),
            
            # Death analysis
            "death_causes": dict(death_causes),
            "collision_rate": death_causes['collision'] / episodes,
            "timeout_rate": death_causes['timeout'] / episodes,
            
            # Consistency metrics
            "performance_consistency": 1.0 / (1.0 + float(np.std(scores))),
            
            # Raw data
            "scores": scores,
            "metadata": metadata
        }
        
        print(f"✅ {model_type}: Avg={results['avg_score']:.2f}, Max={results['max_score']}, "
              f"Consistency={results['performance_consistency']:.3f}")
        
        return results
    
    def _calculate_action_entropy(self, action_counts):
        """Calculate entropy of action distribution"""
        total = sum(action_counts)
        if total == 0:
            return 0.0
        
        probs = [count / total for count in action_counts]
        entropy = -sum(p * np.log2(p + 1e-8) for p in probs if p > 0)
        return entropy
    
    def compare_models(self, model_files: List[str], episodes: int = 100):
        """Compare multiple models"""
        results = []
        
        for model_file in model_files:
            model_path = Path(model_file)
            
            # Determine model type from path
            if "qlearning" in model_path.parts:
                model_type = "qlearning"
            elif "dqn" in model_path.parts:
                model_type = "dqn"
            elif "policy_gradient" in model_path.parts:
                model_type = "policy_gradient"
            elif "ppo" in model_path.parts:
                model_type = "ppo"
            else:
                print(f"❌ Cannot determine model type for: {model_file}")
                continue
            
            try:
                result = self.evaluate_model(model_file, model_type, episodes)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ Failed to evaluate {model_file}: {e}")
        
        if results:
            self._generate_comparison_report(results)
            self._save_comparison_results(results)
        
        return results
    
    def _generate_comparison_report(self, results: List[Dict]):
        """Generate comparison visualization"""
        if len(results) < 2:
            print("⚠️  Need at least 2 models for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Prepare data
        models = [f"{r['model_type']}-{Path(r['model_path']).stem}" for r in results]
        avg_scores = [r['avg_score'] for r in results]
        max_scores = [r['max_score'] for r in results]
        consistency = [r['performance_consistency'] for r in results]
        
        # Colors by technique
        colors = []
        for r in results:
            if r['model_type'] == 'qlearning':
                colors.append('#FF6B6B')
            elif r['model_type'] == 'dqn':
                colors.append('#4ECDC4')
            else:
                colors.append('#45B7D1')
        
        # 1. Average scores
        axes[0,0].bar(range(len(models)), avg_scores, color=colors)
        axes[0,0].set_title('Average Scores by Model', fontsize=14, weight='bold')
        axes[0,0].set_ylabel('Score')
        axes[0,0].set_xticks(range(len(models)))
        axes[0,0].set_xticklabels(models, rotation=45, ha='right')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Score distribution (box plot)
        score_data = [r['scores'] for r in results]
        axes[0,1].boxplot(score_data, tick_labels=models)
        axes[0,1].set_title('Score Distribution', fontsize=14, weight='bold')
        axes[0,1].set_ylabel('Score')
        axes[0,1].set_xticklabels(models, rotation=45, ha='right')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Performance consistency
        axes[1,0].bar(range(len(models)), consistency, color=colors)
        axes[1,0].set_title('Performance Consistency', fontsize=14, weight='bold')
        axes[1,0].set_ylabel('Consistency Score')
        axes[1,0].set_xticks(range(len(models)))
        axes[1,0].set_xticklabels(models, rotation=45, ha='right')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Action distribution heatmap
        action_data = []
        action_labels = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        for r in results:
            action_dist = r['action_distribution']
            total = sum(action_dist)
            if total > 0:
                normalized_dist = [count / total for count in action_dist]
            else:
                normalized_dist = [0.25, 0.25, 0.25, 0.25]
            action_data.append(normalized_dist)
        
        action_data = np.array(action_data).T
        
        im = axes[1,1].imshow(action_data, cmap='YlOrRd', aspect='auto')
        axes[1,1].set_title('Action Distribution Heatmap', fontsize=14, weight='bold')
        axes[1,1].set_yticks(range(len(action_labels)))
        axes[1,1].set_yticklabels(action_labels)
        axes[1,1].set_xticks(range(len(models)))
        axes[1,1].set_xticklabels(models, rotation=45, ha='right')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1,1], label='Action Probability')
        
        plt.suptitle('Fixed Model Comparison - SnakeAI MLOps', fontsize=18, weight='bold')
        plt.tight_layout()
        
        plot_path = Path("models") / "fixed_model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Fixed model comparison saved: {plot_path}")
    
    def _save_comparison_results(self, results: List[Dict]):
        """Save detailed comparison results"""
        # Rank models by performance
        ranked_results = sorted(results, key=lambda x: x['avg_score'], reverse=True)
        
        comparison_data = {
            "evaluation_summary": {
                "total_models": len(results),
                "evaluation_date": str(np.datetime64('now')),
                "gpu_device": str(self.device),
                "episodes_per_model": results[0]['episodes'] if results else 0
            },
            "model_rankings": [
                {
                    "rank": i + 1,
                    "model": Path(r['model_path']).stem,
                    "type": r['model_type'],
                    "avg_score": r['avg_score'],
                    "max_score": r['max_score'],
                    "consistency": r['performance_consistency']
                }
                for i, r in enumerate(ranked_results)
            ],
            "best_performer": {
                "model": Path(ranked_results[0]['model_path']).stem,
                "type": ranked_results[0]['model_type'],
                "avg_score": ranked_results[0]['avg_score']
            } if ranked_results else None,
            "detailed_results": results
        }
        
        report_path = Path("models") / "fixed_evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(comparison_data, f, indent=4, default=lambda x: float(x) if isinstance(x, np.number) else x)
        
        print(f"✅ Fixed evaluation report saved: {report_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("🏆 FIXED MODEL EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n📊 Model Rankings (by average score):")
        for i, result in enumerate(ranked_results):
            model_name = Path(result['model_path']).stem
            print(f"{i+1:2d}. {result['model_type']:12s} {model_name:25s} "
                  f"Avg: {result['avg_score']:6.2f} Max: {result['max_score']:3d}")

# Test function to verify fixes
def test_fixed_evaluator():
    """Test the fixed evaluator"""
    print("🧪 Testing Fixed Model Evaluator...")
    
    evaluator = FixedModelEvaluator()
    
    # Find available models
    model_dir = Path("models")
    test_models = []
    
    # Q-Learning models
    qlearning_dir = model_dir / "qlearning"
    if qlearning_dir.exists():
        for qfile in qlearning_dir.glob("qtable_*.json"):
            if "report" not in qfile.name:
                test_models.append((str(qfile), "qlearning"))
    
    # DQN models  
    dqn_dir = model_dir / "dqn"
    if dqn_dir.exists():
        for dqn_file in dqn_dir.glob("dqn_*.pth"):
            if "best" not in dqn_file.name and "checkpoint" not in dqn_file.name:
                test_models.append((str(dqn_file), "dqn"))
    
    if not test_models:
        print("❌ No models found for testing")
        return False
    
    print(f"Found {len(test_models)} models to test:")
    for model_path, model_type in test_models:
        print(f"   • {model_type}: {Path(model_path).name}")
    
    # Test evaluation
    for model_path, model_type in test_models[:2]:  # Test first 2 models
        try:
            result = evaluator.evaluate_model(model_path, model_type, episodes=10)
            print(f"✅ {model_type} evaluation successful: avg_score={result['avg_score']:.2f}")
        except Exception as e:
            print(f"❌ {model_type} evaluation failed: {e}")
            return False
    
    # Test comparison if we have multiple models
    if len(test_models) >= 2:
        try:
            model_files = [path for path, _ in test_models[:2]]
            evaluator.compare_models(model_files, episodes=5)
            print("✅ Model comparison successful")
        except Exception as e:
            print(f"❌ Model comparison failed: {e}")
            return False
    
    print("✅ Fixed evaluator test completed successfully!")
    return True

if __name__ == "__main__":
    test_fixed_evaluator()