#!/usr/bin/env python3
"""
Unified Model Evaluator for SnakeAI-MLOps
Evaluates Q-Learning, DQN, PPO, and Actor-Critic models
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
    """Simple DQN for loading models"""
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleDQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SimplePolicyNetwork(torch.nn.Module):
    """Simple policy network for PPO/AC"""
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

class UnifiedModelEvaluator:
    """Unified evaluator for all ML model types"""
    
    def __init__(self):
        self.device = verify_gpu()
        print(f"✅ Unified Model Evaluator initialized on {self.device}")
    
    def load_qlearning_model(self, model_path: str) -> torch.Tensor:
        """Load Q-Learning model from JSON"""
        with open(model_path, 'r') as f:
            data = json.load(f)
        
        q_table = torch.zeros((512, 4), device=self.device)
        
        for state_str, actions in data["qTable"].items():
            state_idx = int(state_str, 2)
            if state_idx < 512:
                q_table[state_idx] = torch.tensor(actions, device=self.device)
        
        print(f"✅ Q-Learning model loaded: {model_path}")
        return q_table
    
    def load_dqn_model(self, model_path: str):
        """Load DQN model"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config_dict = checkpoint['config']
        
        hidden_size = config_dict.get('hidden_size', 128)
        saved_state = checkpoint['q_network']
        input_size = saved_state['fc1.weight'].shape[1]
        
        model = SimpleDQN(input_size, hidden_size, 4)
        model.load_state_dict(saved_state)
        model.to(self.device)
        model.eval()
        
        print(f"✅ DQN model loaded: {model_path}")
        return model, checkpoint.get('metadata', {}), input_size
    
    def load_ppo_model(self, model_path: str):
        """Load PPO model"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config_dict = checkpoint['config']
        
        hidden_size = config_dict.get('hidden_size', 256)
        saved_state = checkpoint['policy_network']
        input_size = saved_state['fc1.weight'].shape[1]
        
        model = SimplePolicyNetwork(input_size, hidden_size, 4)
        model.load_state_dict(saved_state)
        model.to(self.device)
        model.eval()
        
        print(f"✅ PPO model loaded: {model_path}")
        return model, checkpoint.get('metadata', {}), input_size
    
    def load_actor_critic_model(self, model_path: str):
        """Load Actor-Critic model"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config_dict = checkpoint['config']
        
        hidden_size = config_dict.get('hidden_size', 128)
        actor_state = checkpoint['actor_state_dict']
        input_size = actor_state['fc1.weight'].shape[1]
        
        model = SimplePolicyNetwork(input_size, hidden_size, 4)
        model.load_state_dict(actor_state)
        model.to(self.device)
        model.eval()
        
        print(f"✅ Actor-Critic model loaded: {model_path}")
        return model, checkpoint.get('metadata', {}), input_size
    
    def encode_state_qlearning(self, state):
        """Convert state to Q-table index"""
        binary_parts = []
        
        binary_parts.append(str(int(state[0].item())))
        binary_parts.append(str(int(state[1].item())))
        binary_parts.append(str(int(state[2].item())))
        
        direction = int(state[3].item())
        direction_binary = format(direction, '02b')
        binary_parts.append(direction_binary)
        
        binary_parts.append(str(int(state[4].item())))
        binary_parts.append(str(int(state[5].item())))
        binary_parts.append(str(int(state[6].item())))
        binary_parts.append(str(int(state[7].item())))
        
        binary_str = ''.join(binary_parts)
        return int(binary_str, 2)
    
    def convert_to_neural_state(self, basic_state, env):
        """Convert 8D basic state to neural network format"""
        head_x, head_y = env.snake[0]
        food_x, food_y = env.food
        
        food_distance = abs(head_x - food_x) + abs(head_y - food_y)
        normalized_distance = food_distance / (2 * env.grid_size)
        
        wall_distances = [
            head_y / env.grid_size,
            (env.grid_size - 1 - head_y) / env.grid_size,
            head_x / env.grid_size,
            (env.grid_size - 1 - head_x) / env.grid_size,
        ]
        
        snake_length = len(env.snake) / (env.grid_size * env.grid_size)
        empty_spaces = (env.grid_size * env.grid_size - len(env.snake) - 1) / (env.grid_size * env.grid_size)
        
        # Create 11D state vector for neural networks
        neural_state = torch.tensor([
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
        
        return neural_state
    
    def evaluate_model(self, model_path: str, model_type: str, episodes: int = 100) -> Dict:
        """Evaluate single model"""
        env = SnakeEnvironment(device=str(self.device))
        
        # Load model based on type
        if model_type == "qlearning":
            model = self.load_qlearning_model(model_path)
            metadata = {}
            input_size = None
        elif model_type == "dqn":
            model, metadata, input_size = self.load_dqn_model(model_path)
        elif model_type == "ppo":
            model, metadata, input_size = self.load_ppo_model(model_path)
        elif model_type == "actor_critic":
            model, metadata, input_size = self.load_actor_critic_model(model_path)
        else:
            print(f"❌ Model type {model_type} not supported")
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
                    
                elif model_type in ["dqn", "ppo", "actor_critic"]:
                    neural_state = self.convert_to_neural_state(state, env)
                    with torch.no_grad():
                        if model_type == "dqn":
                            q_values = model(neural_state.unsqueeze(0))
                            action = torch.argmax(q_values).item()
                        else:  # PPO or Actor-Critic
                            action_probs = model(neural_state.unsqueeze(0))
                            action = torch.argmax(action_probs).item()
                
                action_distributions[action] += 1
                state, reward, done = env.step(action)
                steps += 1
                
                if done:
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
            
            "avg_score": float(np.mean(scores)),
            "max_score": int(max(scores)),
            "min_score": int(min(scores)),
            "std_score": float(np.std(scores)),
            "median_score": float(np.median(scores)),
            
            "avg_length": float(np.mean(episode_lengths)),
            "max_length": int(max(episode_lengths)),
            
            "action_distribution": action_distributions,
            "action_entropy": self._calculate_action_entropy(action_distributions),
            
            "death_causes": dict(death_causes),
            "collision_rate": death_causes['collision'] / episodes,
            "timeout_rate": death_causes['timeout'] / episodes,
            
            "performance_consistency": 1.0 / (1.0 + float(np.std(scores))),
            
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
    
    def compare_models(self, model_files: List[str] = None, episodes: int = 100):
        """Compare multiple models or all available models"""
        if model_files is None:
            model_files = self._find_all_models()
        
        if not model_files:
            print("❌ No models found for comparison")
            return []
        
        results = []
        
        for model_file in model_files:
            model_path = Path(model_file)
            
            # Determine model type from path
            if "qlearning" in model_path.parts:
                model_type = "qlearning"
            elif "dqn" in model_path.parts:
                model_type = "dqn"
            elif "ppo" in model_path.parts:
                model_type = "ppo"
            elif "actor_critic" in model_path.parts:
                model_type = "actor_critic"
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
    
    def _find_all_models(self):
        """Find all available models"""
        model_files = []
        model_dir = Path("models")
        
        if not model_dir.exists():
            return model_files
        
        # Q-Learning models
        qlearning_dir = model_dir / "qlearning"
        if qlearning_dir.exists():
            for qfile in qlearning_dir.glob("qtable_*.json"):
                if "report" not in qfile.name:
                    model_files.append(str(qfile))
        
        # Neural network models
        for technique in ["dqn", "ppo", "actor_critic"]:
            tech_dir = model_dir / technique
            if tech_dir.exists():
                for model_file in tech_dir.glob("*.pth"):
                    if "checkpoint" not in model_file.name:
                        model_files.append(str(model_file))
        
        return model_files
    
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
            elif r['model_type'] == 'ppo':
                colors.append('#45B7D1')
            elif r['model_type'] == 'actor_critic':
                colors.append('#F9CA24')
        
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
        
        plt.colorbar(im, ax=axes[1,1], label='Action Probability')
        
        plt.suptitle('Unified Model Comparison - SnakeAI MLOps', fontsize=18, weight='bold')
        plt.tight_layout()
        
        plot_path = Path("models") / "unified_model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Unified model comparison saved: {plot_path}")
    
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
        
        report_path = Path("models") / "unified_evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(comparison_data, f, indent=4, default=lambda x: float(x) if isinstance(x, np.number) else x)
        
        print(f"✅ Unified evaluation report saved: {report_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("🏆 UNIFIED MODEL EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n📊 Model Rankings (by average score):")
        for i, result in enumerate(ranked_results):
            model_name = Path(result['model_path']).stem
            print(f"{i+1:2d}. {result['model_type']:12s} {model_name:25s} "
                  f"Avg: {result['avg_score']:6.2f} Max: {result['max_score']:3d}")

if __name__ == "__main__":
    # Run unified evaluation
    evaluator = UnifiedModelEvaluator()
    
    # Find and evaluate all models
    results = evaluator.compare_models(episodes=50)
    
    if results:
        print(f"\n🎉 Unified evaluation complete! {len(results)} models compared.")
        print("📁 Check models/ directory for detailed plots and reports")
    else:
        print("❌ No models found. Run training first:")
        print("   python src/train_models.py --technique all")