#!/usr/bin/env python3
"""
Model evaluator for all SnakeAI-MLOps ML techniques
GPU-accelerated evaluation and comparison
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import time

from neural_network_utils import (
    DQNNetwork, PolicyNetwork, ValueNetwork, NetworkConfig,
    verify_gpu, encode_state_for_dqn
)
from qlearning_trainer import SnakeEnvironment

class UnifiedModelEvaluator:
    """GPU-accelerated model evaluation for all ML techniques"""
    
    def __init__(self):
        self.device = verify_gpu()
        print(f"✅ UnifiedModelEvaluator initialized on {self.device}")
    
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
    
    def load_neural_model(self, model_path: str, model_type: str):
        """Load neural network model (DQN, Policy Gradient, Actor-Critic)"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config_dict = checkpoint['config']
        
        # Create network config
        net_config = NetworkConfig(
            input_size=20,
            hidden_layers=config_dict.get('hidden_layers', [128, 64]),
            output_size=4,
            dropout=0.0  # No dropout during evaluation
        )
        
        if model_type == "dqn":
            model = DQNNetwork(net_config, dueling=config_dict.get('dueling', True))
            model.load_state_dict(checkpoint['q_network'])
            
        elif model_type == "policy_gradient":
            model = PolicyNetwork(net_config)
            model.load_state_dict(checkpoint['policy_network'])
            
        elif model_type == "actor_critic":
            # Load both actor and critic, return actor for action selection
            actor = PolicyNetwork(net_config)
            actor.load_state_dict(checkpoint['actor_state_dict'])
            
            critic = ValueNetwork(net_config)
            critic.load_state_dict(checkpoint['critic_state_dict'])
            
            model = (actor, critic)  # Return both
        
        model.to(self.device) if not isinstance(model, tuple) else [m.to(self.device) for m in model]
        
        if isinstance(model, tuple):
            model[0].eval()
            model[1].eval()
        else:
            model.eval()
        
        print(f"✅ {model_type.upper()} model loaded: {model_path}")
        return model, checkpoint.get('metadata', {})
    
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
    
    def enhance_state_for_neural(self, basic_state):
        """Convert 8D basic state to 20D enhanced state for neural networks"""
        enhanced_state = torch.zeros(20, dtype=torch.float32, device=self.device)
        
        # Copy basic features
        enhanced_state[:8] = basic_state.float()
        
        # Add enhanced features (simplified for evaluation)
        food_features = basic_state[4:8]
        food_distance = torch.sum(food_features).float()
        enhanced_state[8] = food_distance / 4.0
        
        # Wall distances (placeholder)
        enhanced_state[9:13] = torch.tensor([0.5, 0.5, 0.5, 0.5], device=self.device)
        
        # Body density (placeholder)
        enhanced_state[13:17] = torch.tensor([0.1, 0.1, 0.1, 0.1], device=self.device)
        
        # Additional features
        enhanced_state[17] = 0.1  # Snake length
        enhanced_state[18] = 0.9  # Empty spaces
        enhanced_state[19] = food_distance / 4.0  # Path complexity
        
        return enhanced_state
    
    def evaluate_model(self, model_path: str, model_type: str, episodes: int = 100) -> Dict:
        """Evaluate single model performance"""
        env = SnakeEnvironment(device=str(self.device))
        
        # Load model based on type
        if model_type == "qlearning":
            model = self.load_qlearning_model(model_path)
            metadata = {}
        else:
            model, metadata = self.load_neural_model(model_path, model_type)
        
        scores = []
        episode_lengths = []
        action_distributions = [0, 0, 0, 0]  # Count actions taken
        
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
                    enhanced_state = self.enhance_state_for_neural(state)
                    with torch.no_grad():
                        q_values = model(enhanced_state.unsqueeze(0))
                        action = torch.argmax(q_values).item()
                
                elif model_type == "policy_gradient":
                    enhanced_state = self.enhance_state_for_neural(state)
                    with torch.no_grad():
                        action_probs = model(enhanced_state.unsqueeze(0))
                        action = torch.argmax(action_probs).item()  # Greedy for evaluation
                
                elif model_type == "actor_critic":
                    actor, critic = model
                    enhanced_state = self.enhance_state_for_neural(state)
                    with torch.no_grad():
                        action_probs = actor(enhanced_state.unsqueeze(0))
                        action = torch.argmax(action_probs).item()  # Greedy for evaluation
                
                action_distributions[action] += 1
                state, reward, done = env.step(action)
                steps += 1
                
                if done:
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
            "avg_length": float(np.mean(episode_lengths)),
            "action_distribution": action_distributions,
            "action_entropy": self._calculate_action_entropy(action_distributions),
            "scores": scores,
            "metadata": metadata
        }
        
        print(f"✅ {model_type}: Avg={results['avg_score']:.2f}, Max={results['max_score']}")
        return results
    
    def _calculate_action_entropy(self, action_counts):
        """Calculate entropy of action distribution"""
        total = sum(action_counts)
        if total == 0:
            return 0.0
        
        probs = [count / total for count in action_counts]
        entropy = -sum(p * np.log2(p + 1e-8) for p in probs if p > 0)
        return entropy
    
    def compare_all_models(self, episodes: int = 100):
        """Compare all available models across all techniques"""
        model_dir = Path("models")
        if not model_dir.exists():
            print("❌ No models directory found")
            return []
        
        results = []
        
        # Q-Learning models
        qlearning_dir = model_dir / "qlearning"
        if qlearning_dir.exists():
            for qfile in qlearning_dir.glob("qtable_*.json"):
                if "report" not in qfile.name and "checkpoint" not in qfile.name:
                    result = self.evaluate_model(str(qfile), "qlearning", episodes)
                    results.append(result)
        
        # Neural network models
        for technique in ["dqn", "policy_gradient", "actor_critic"]:
            tech_dir = model_dir / technique
            if tech_dir.exists():
                prefix = technique[:2]  # dq, pg, ac
                for model_file in tech_dir.glob(f"{prefix}_*.pth"):
                    if "best" not in model_file.name and "checkpoint" not in model_file.name:
                        result = self.evaluate_model(str(model_file), technique, episodes)
                        results.append(result)
        
        if results:
            self._generate_comprehensive_comparison(results)
            self._save_comparison_report(results)
        
        return results
    
    def _generate_comprehensive_comparison(self, results: List[Dict]):
        """Generate comprehensive comparison visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Prepare data
        models = [f"{r['model_type']}-{Path(r['model_path']).stem}" for r in results]
        avg_scores = [r['avg_score'] for r in results]
        max_scores = [r['max_score'] for r in results]
        std_scores = [r['std_score'] for r in results]
        avg_lengths = [r['avg_length'] for r in results]
        action_entropies = [r['action_entropy'] for r in results]
        
        # Colors by technique
        colors = []
        for r in results:
            if r['model_type'] == 'qlearning':
                colors.append('red')
            elif r['model_type'] == 'dqn':
                colors.append('blue')
            elif r['model_type'] == 'policy_gradient':
                colors.append('green')
            elif r['model_type'] == 'actor_critic':
                colors.append('orange')
        
        # Average scores
        axes[0,0].bar(range(len(models)), avg_scores, color=colors)
        axes[0,0].set_title('Average Scores by Model')
        axes[0,0].set_ylabel('Score')
        axes[0,0].set_xticks(range(len(models)))
        axes[0,0].set_xticklabels(models, rotation=45, ha='right')
        axes[0,0].grid(True, alpha=0.3)
        
        # Max scores
        axes[0,1].bar(range(len(models)), max_scores, color=colors)
        axes[0,1].set_title('Maximum Scores by Model')
        axes[0,1].set_ylabel('Score')
        axes[0,1].set_xticks(range(len(models)))
        axes[0,1].set_xticklabels(models, rotation=45, ha='right')
        axes[0,1].grid(True, alpha=0.3)
        
        # Score variability (std dev)
        axes[0,2].bar(range(len(models)), std_scores, color=colors)
        axes[0,2].set_title('Score Variability (Std Dev)')
        axes[0,2].set_ylabel('Standard Deviation')
        axes[0,2].set_xticks(range(len(models)))
        axes[0,2].set_xticklabels(models, rotation=45, ha='right')
        axes[0,2].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[1,0].bar(range(len(models)), avg_lengths, color=colors)
        axes[1,0].set_title('Average Episode Lengths')
        axes[1,0].set_ylabel('Steps')
        axes[1,0].set_xticks(range(len(models)))
        axes[1,0].set_xticklabels(models, rotation=45, ha='right')
        axes[1,0].grid(True, alpha=0.3)
        
        # Action entropy (behavioral diversity)
        axes[1,1].bar(range(len(models)), action_entropies, color=colors)
        axes[1,1].set_title('Action Entropy (Behavioral Diversity)')
        axes[1,1].set_ylabel('Entropy (bits)')
        axes[1,1].set_xticks(range(len(models)))
        axes[1,1].set_xticklabels(models, rotation=45, ha='right')
        axes[1,1].grid(True, alpha=0.3)
        
        # Score distributions (violin plot)
        all_scores = []
        all_labels = []
        for i, result in enumerate(results):
            all_scores.extend(result['scores'])
            all_labels.extend([models[i]] * len(result['scores']))
        
        # Group by technique for violin plot
        technique_scores = {}
        for result in results:
            tech = result['model_type']
            if tech not in technique_scores:
                technique_scores[tech] = []
            technique_scores[tech].extend(result['scores'])
        
        if technique_scores:
            violin_data = list(technique_scores.values())
            violin_labels = list(technique_scores.keys())
            axes[1,2].violinplot(violin_data, positions=range(len(violin_labels)))
            axes[1,2].set_title('Score Distribution by Technique')
            axes[1,2].set_ylabel('Score')
            axes[1,2].set_xticks(range(len(violin_labels)))
            axes[1,2].set_xticklabels(violin_labels)
            axes[1,2].grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Model Comparison - SnakeAI MLOps', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_path = Path("models") / "comprehensive_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive comparison saved: {plot_path}")
        
        # Generate technique-specific comparison
        self._generate_technique_comparison(results)
    
    def _generate_technique_comparison(self, results: List[Dict]):
        """Generate comparison within each technique"""
        techniques = {}
        for result in results:
            tech = result['model_type']
            if tech not in techniques:
                techniques[tech] = []
            techniques[tech].append(result)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (tech, tech_results) in enumerate(techniques.items()):
            if i >= 4:
                break
                
            profiles = [Path(r['model_path']).stem.split('_')[-1] for r in tech_results]
            scores = [r['avg_score'] for r in tech_results]
            
            axes[i].bar(profiles, scores, 
                       color=['red', 'blue', 'green'][:len(profiles)])
            axes[i].set_title(f'{tech.replace("_", " ").title()} Profiles')
            axes[i].set_ylabel('Average Score')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(techniques), 4):
            axes[i].set_visible(False)
        
        plt.suptitle('Profile Comparison Within Each Technique')
        plt.tight_layout()
        
        profile_plot_path = Path("models") / "profile_comparison.png"
        plt.savefig(profile_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Profile comparison saved: {profile_plot_path}")
    
    def _save_comparison_report(self, results: List[Dict]):
        """Save detailed comparison report"""
        # Rank models by average score
        ranked_results = sorted(results, key=lambda x: x['avg_score'], reverse=True)
        
        report = {
            "evaluation_summary": {
                "total_models": len(results),
                "evaluation_date": str(np.datetime64('now')),
                "gpu_device": str(self.device),
                "episodes_per_model": results[0]['episodes'] if results else 0
            },
            "model_rankings": ranked_results,
            "technique_summary": self._generate_technique_summary(results),
            "best_performers": {
                "overall_best": ranked_results[0] if ranked_results else None,
                "best_by_technique": self._get_best_by_technique(results)
            }
        }
        
        report_path = Path("models") / "evaluation_report.json"
        with open(report_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json.dump(report, f, indent=4, default=lambda x: float(x) if isinstance(x, np.number) else x)
        
        print(f"✅ Evaluation report saved: {report_path}")
        
        # Print summary to console
        self._print_summary(ranked_results)
    
    def _generate_technique_summary(self, results: List[Dict]):
        """Generate summary statistics by technique"""
        techniques = {}
        for result in results:
            tech = result['model_type']
            if tech not in techniques:
                techniques[tech] = []
            techniques[tech].append(result['avg_score'])
        
        summary = {}
        for tech, scores in techniques.items():
            summary[tech] = {
                "count": len(scores),
                "avg_score": float(np.mean(scores)),
                "max_score": float(max(scores)),
                "min_score": float(min(scores)),
                "std_score": float(np.std(scores))
            }
        
        return summary
    
    def _get_best_by_technique(self, results: List[Dict]):
        """Get best performing model for each technique"""
        techniques = {}
        for result in results:
            tech = result['model_type']
            if tech not in techniques or result['avg_score'] > techniques[tech]['avg_score']:
                techniques[tech] = result
        
        return techniques
    
    def _print_summary(self, ranked_results: List[Dict]):
        """Print evaluation summary to console"""
        print(f"\n{'='*60}")
        print("🏆 MODEL EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n📊 Top 5 Performing Models:")
        for i, result in enumerate(ranked_results[:5]):
            model_name = Path(result['model_path']).stem
            print(f"{i+1:2d}. {result['model_type']:15s} {model_name:20s} "
                  f"Avg: {result['avg_score']:6.2f} Max: {result['max_score']:3d}")
        
        # Technique comparison
        technique_summary = self._generate_technique_summary(ranked_results)
        print(f"\n🎯 Average Performance by Technique:")
        for tech, stats in sorted(technique_summary.items(), 
                                 key=lambda x: x[1]['avg_score'], reverse=True):
            print(f"   {tech:15s}: {stats['avg_score']:6.2f} ± {stats['std_score']:5.2f}")

def benchmark_gpu_performance():
    """Benchmark GPU vs CPU performance"""
    print("🚀 Benchmarking GPU vs CPU performance...")
    
    device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test neural network inference
    net_config = NetworkConfig(input_size=20, hidden_layers=[128, 128], output_size=4)
    model_gpu = DQNNetwork(net_config).to(device_gpu)
    model_cpu = DQNNetwork(net_config).to('cpu')
    
    # Generate test data
    test_data_gpu = torch.randn(1000, 20, device=device_gpu)
    test_data_cpu = torch.randn(1000, 20, device='cpu')
    
    # Time GPU inference
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model_gpu(test_data_gpu)
    gpu_time = time.time() - start
    
    # Time CPU inference
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model_cpu(test_data_cpu)
    cpu_time = time.time() - start
    
    speedup = cpu_time / gpu_time
    print(f"✅ GPU Inference Time: {gpu_time:.3f}s")
    print(f"✅ CPU Inference Time: {cpu_time:.3f}s")
    print(f"✅ GPU Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    # Benchmark performance
    benchmark_gpu_performance()
    
    # Evaluate all models
    evaluator = UnifiedModelEvaluator()
    results = evaluator.compare_all_models(episodes=50)
    
    if results:
        print(f"\n🎉 Evaluation complete! {len(results)} models compared.")
    else:
        print("❌ No models found. Run training first:")
        print("   python train_models.py --technique all")