#!/usr/bin/env python3
"""
Enhanced Model evaluator for all SnakeAI-MLOps ML techniques
Fixed Actor-Critic evaluation with comprehensive metrics and visualization
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

class SimpleValueNetwork(torch.nn.Module):
    """Simple value network"""
    def __init__(self, input_size, hidden_size):
        super(SimpleValueNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)

class EnhancedModelEvaluator:
    """Enhanced model evaluation with comprehensive metrics - FIXED"""
    
    def __init__(self):
        self.device = verify_gpu()
        print(f"✅ Enhanced Model Evaluator initialized on {self.device}")
    
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
        """Load neural network model - FIXED for all architectures"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config_dict = checkpoint['config']
        
        # Determine input size and hidden size from saved model
        if model_type == "dqn":
            saved_state = checkpoint['q_network']
            input_size = saved_state['fc1.weight'].shape[1]
            hidden_size = config_dict.get('hidden_size', 64)
            
            model = SimpleDQN(input_size, hidden_size, 4)
            model.load_state_dict(saved_state, strict=False)
            
        elif model_type == "ppo":
            saved_state = checkpoint['policy_network']
            input_size = saved_state['fc1.weight'].shape[1]
            hidden_size = config_dict.get('hidden_size', 64)
            
            model = SimplePolicyNetwork(input_size, hidden_size, 4)
            model.load_state_dict(saved_state, strict=False)
            
        elif model_type == "actor_critic":
            # Load both actor and critic - FIXED
            actor_state = checkpoint['actor_state_dict']
            input_size = actor_state['fc1.weight'].shape[1]
            hidden_size = config_dict.get('hidden_size', 64)
            
            actor = SimplePolicyNetwork(input_size, hidden_size, 4)
            actor.load_state_dict(actor_state, strict=False)
            
            critic = SimpleValueNetwork(input_size, hidden_size)
            if 'critic_state_dict' in checkpoint:
                critic.load_state_dict(checkpoint['critic_state_dict'], strict=False)
            
            model = (actor, critic)
        
        # Move to device and set eval mode
        if isinstance(model, tuple):
            model[0].to(self.device)
            model[1].to(self.device)
            model[0].eval()
            model[1].eval()
        else:
            model.to(self.device)
            model.eval()
        
        print(f"✅ {model_type.upper()} model loaded: {model_path} (input_size: {input_size})")
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
    
    def prepare_neural_state(self, basic_state, model_input_size):
        """Prepare state for neural networks - FIXED to handle any input size"""
        if model_input_size == 8:
            # Model expects 8D input (same as basic state)
            return basic_state
        elif model_input_size == 11:
            # Model expects 11D input
            enhanced_state = torch.zeros(11, dtype=torch.float32, device=self.device)
            enhanced_state[:8] = basic_state
            
            # Add simple enhanced features
            food_features = basic_state[4:8]
            food_distance = torch.sum(food_features).float()
            enhanced_state[8] = food_distance / 4.0
            enhanced_state[9] = 0.1  # snake length placeholder
            enhanced_state[10] = 0.9  # empty spaces placeholder
            
            return enhanced_state
        elif model_input_size == 20:
            # Model expects 20D input
            enhanced_state = torch.zeros(20, dtype=torch.float32, device=self.device)
            enhanced_state[:8] = basic_state
            
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
        else:
            # Unknown input size - try to adapt
            print(f"⚠️  Unknown input size {model_input_size}, using basic 8D state")
            return basic_state
    
    def evaluate_model(self, model_path: str, model_type: str, episodes: int = 100, 
                      detailed_metrics: bool = True) -> Dict:
        """Evaluate single model with comprehensive metrics - FIXED"""
        # Use consistent grid size for fair evaluation
        env = SnakeEnvironment(grid_size=10, device=str(self.device))
        
        # Load model
        if model_type == "qlearning":
            model = self.load_qlearning_model(model_path)
            metadata = {}
            input_size = None
        else:
            model, metadata, input_size = self.load_neural_model(model_path, model_type)
        
        # Evaluation metrics
        scores = []
        episode_lengths = []
        action_distributions = [0, 0, 0, 0]
        death_causes = defaultdict(int)
        
        # Detailed metrics
        food_efficiency = []
        survival_time = []
        max_snake_length = []
        action_changes = []
        
        print(f"🧪 Evaluating {model_type} model over {episodes} episodes...")
        
        for episode in range(episodes):
            state = env.reset()
            steps = 0
            prev_action = None
            action_change_count = 0
            foods_eaten = 0
            
            while True:
                # Get action based on model type
                if model_type == "qlearning":
                    state_idx = self.encode_state_qlearning(state)
                    action = torch.argmax(model[state_idx]).item()
                    
                elif model_type == "dqn":
                    neural_state = self.prepare_neural_state(state, input_size)
                    with torch.no_grad():
                        q_values = model(neural_state.unsqueeze(0))
                        action = torch.argmax(q_values).item()
                
                elif model_type == "ppo":
                    neural_state = self.prepare_neural_state(state, input_size)
                    with torch.no_grad():
                        action_probs = model(neural_state.unsqueeze(0))
                        action = torch.argmax(action_probs).item()  # Greedy for evaluation
                
                elif model_type == "actor_critic":
                    actor, critic = model
                    neural_state = self.prepare_neural_state(state, input_size)
                    with torch.no_grad():
                        action_probs = actor(neural_state.unsqueeze(0))
                        action = torch.argmax(action_probs).item()  # Greedy for evaluation
                
                # Track action changes
                if prev_action is not None and action != prev_action:
                    action_change_count += 1
                prev_action = action
                
                action_distributions[action] += 1
                state, reward, done = env.step(action)
                steps += 1
                
                # Track food consumption
                if reward > 5:  # Food eaten
                    foods_eaten += 1
                
                if done:
                    # Determine death cause
                    if steps >= 500:  # Timeout for 10x10 grid
                        death_causes['timeout'] += 1
                    elif env.score == 0:
                        death_causes['early_death'] += 1
                    else:
                        death_causes['collision'] += 1
                    break
            
            scores.append(env.score)
            episode_lengths.append(steps)
            max_snake_length.append(len(env.snake))
            survival_time.append(steps)
            action_changes.append(action_change_count / max(steps - 1, 1))
            
            if foods_eaten > 0:
                food_efficiency.append(steps / foods_eaten)
            else:
                food_efficiency.append(steps)
        
        # Calculate comprehensive statistics
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
            "avg_survival_time": float(np.mean(survival_time)),
            "max_snake_length": int(max(max_snake_length)),
            "avg_food_efficiency": float(np.mean(food_efficiency)),
            
            # Behavioral metrics
            "action_distribution": action_distributions,
            "action_entropy": self._calculate_action_entropy(action_distributions),
            "avg_action_changes": float(np.mean(action_changes)),
            "behavioral_stability": 1.0 - float(np.mean(action_changes)),
            
            # Death analysis
            "death_causes": dict(death_causes),
            "collision_rate": death_causes['collision'] / episodes,
            "timeout_rate": death_causes['timeout'] / episodes,
            
            # Consistency metrics
            "score_variance": float(np.var(scores)),
            "performance_consistency": 1.0 / (1.0 + float(np.std(scores))),
            
            # Raw data
            "scores": scores,
            "metadata": metadata
        }
        
        # Calculate percentiles
        results["score_percentiles"] = {
            "25th": float(np.percentile(scores, 25)),
            "50th": float(np.percentile(scores, 50)),
            "75th": float(np.percentile(scores, 75)),
            "90th": float(np.percentile(scores, 90))
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
    
    def compare_all_models(self, episodes: int = 100, save_individual_plots: bool = True):
        """Compare all available models with enhanced visualizations"""
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
                    try:
                        result = self.evaluate_model(str(qfile), "qlearning", episodes)
                        results.append(result)
                    except Exception as e:
                        print(f"❌ Failed to evaluate {qfile}: {e}")
        
        # Neural network models - FIXED: Include best models, exclude checkpoints
        for technique in ["dqn", "ppo", "actor_critic"]:
            tech_dir = model_dir / technique
            if tech_dir.exists():
                for model_file in tech_dir.glob("*.pth"):
                    # Only exclude checkpoint files, keep best models
                    if "checkpoint" not in model_file.name:
                        try:
                            result = self.evaluate_model(str(model_file), technique, episodes)
                            results.append(result)
                        except Exception as e:
                            print(f"❌ Failed to evaluate {model_file}: {e}")
        
        if results:
            self._generate_comprehensive_comparison(results)
            self._generate_detailed_analysis(results)
            self._save_comparison_report(results)
            
            if save_individual_plots:
                self._generate_individual_model_plots(results)
        
        return results
    
    def _generate_comprehensive_comparison(self, results: List[Dict]):
        """Generate comprehensive comparison visualizations"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Prepare data
        models = [f"{r['model_type']}-{Path(r['model_path']).stem}" for r in results]
        avg_scores = [r['avg_score'] for r in results]
        max_scores = [r['max_score'] for r in results]
        consistency = [r['performance_consistency'] for r in results]
        food_efficiency = [r['avg_food_efficiency'] for r in results]
        survival_time = [r['avg_survival_time'] for r in results]
        action_entropy = [r['action_entropy'] for r in results]
        
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
        axes[0,2].bar(range(len(models)), consistency, color=colors)
        axes[0,2].set_title('Performance Consistency', fontsize=14, weight='bold')
        axes[0,2].set_ylabel('Consistency Score')
        axes[0,2].set_xticks(range(len(models)))
        axes[0,2].set_xticklabels(models, rotation=45, ha='right')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Food efficiency
        axes[1,0].bar(range(len(models)), food_efficiency, color=colors)
        axes[1,0].set_title('Food Efficiency (Lower is Better)', fontsize=14, weight='bold')
        axes[1,0].set_ylabel('Steps per Food')
        axes[1,0].set_xticks(range(len(models)))
        axes[1,0].set_xticklabels(models, rotation=45, ha='right')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Survival time
        axes[1,1].bar(range(len(models)), survival_time, color=colors)
        axes[1,1].set_title('Average Survival Time', fontsize=14, weight='bold')
        axes[1,1].set_ylabel('Steps')
        axes[1,1].set_xticks(range(len(models)))
        axes[1,1].set_xticklabels(models, rotation=45, ha='right')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Behavioral diversity (action entropy)
        axes[1,2].bar(range(len(models)), action_entropy, color=colors)
        axes[1,2].set_title('Behavioral Diversity (Action Entropy)', fontsize=14, weight='bold')
        axes[1,2].set_ylabel('Entropy (bits)')
        axes[1,2].set_xticks(range(len(models)))
        axes[1,2].set_xticklabels(models, rotation=45, ha='right')
        axes[1,2].grid(True, alpha=0.3)
        
        # 7. Death cause analysis
        death_data = []
        death_labels = ['Collision', 'Timeout', 'Early Death']
        for r in results:
            death_causes = r['death_causes']
            total = sum(death_causes.values())
            if total > 0:
                death_data.append([
                    death_causes.get('collision', 0) / total,
                    death_causes.get('timeout', 0) / total,
                    death_causes.get('early_death', 0) / total
                ])
            else:
                death_data.append([0, 0, 0])
        
        death_data = np.array(death_data).T
        bottom = np.zeros(len(models))
        
        for i, label in enumerate(death_labels):
            axes[2,0].bar(range(len(models)), death_data[i], bottom=bottom, label=label)
            bottom += death_data[i]
        
        axes[2,0].set_title('Death Cause Distribution', fontsize=14, weight='bold')
        axes[2,0].set_ylabel('Proportion')
        axes[2,0].set_xticks(range(len(models)))
        axes[2,0].set_xticklabels(models, rotation=45, ha='right')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)
        
        # 8. Performance radar chart
        ax_radar = plt.subplot(3, 3, 8, projection='polar')
        self._create_radar_chart(ax_radar, results[:4])  # Top 4 models
        
        # 9. Technique comparison
        technique_scores = {}
        for result in results:
            tech = result['model_type']
            if tech not in technique_scores:
                technique_scores[tech] = []
            technique_scores[tech].append(result['avg_score'])
        
        tech_names = list(technique_scores.keys())
        tech_avgs = [np.mean(scores) for scores in technique_scores.values()]
        tech_stds = [np.std(scores) for scores in technique_scores.values()]
        
        x_pos = np.arange(len(tech_names))
        axes[2,2].bar(x_pos, tech_avgs, yerr=tech_stds, capsize=5)
        axes[2,2].set_title('Average Performance by Technique', fontsize=14, weight='bold')
        axes[2,2].set_ylabel('Average Score')
        axes[2,2].set_xticks(x_pos)
        axes[2,2].set_xticklabels(tech_names)
        axes[2,2].grid(True, alpha=0.3)
        
        plt.suptitle('Enhanced Model Comparison - SnakeAI MLOps (FIXED)', fontsize=18, weight='bold')
        plt.tight_layout()
        
        plot_path = Path("models") / "enhanced_comparison_fixed.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Enhanced comprehensive comparison saved: {plot_path}")
    
    def _create_radar_chart(self, ax, results):
        """Create radar chart for top models"""
        categories = ['Score', 'Consistency', 'Efficiency', 'Survival', 'Diversity']
        
        # Normalize metrics to 0-1 scale
        max_score = max(r['avg_score'] for r in results) if results else 1
        max_survival = max(r['avg_survival_time'] for r in results) if results else 1
        min_efficiency = min(r['avg_food_efficiency'] for r in results) if results else 1
        max_efficiency = max(r['avg_food_efficiency'] for r in results) if results else 1
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        
        for result in results[:4]:  # Top 4
            values = [
                result['avg_score'] / max_score,
                result['performance_consistency'],
                1.0 - (result['avg_food_efficiency'] - min_efficiency) / max(max_efficiency - min_efficiency, 1),
                result['avg_survival_time'] / max_survival,
                result['action_entropy'] / 2.0  # Max entropy is 2
            ]
            values += values[:1]
            
            model_name = Path(result['model_path']).stem
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Metric Performance Comparison', fontsize=12, weight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax.grid(True)
    
    def _generate_detailed_analysis(self, results: List[Dict]):
        """Generate detailed analysis plots"""
        # Create heatmap of model performance across different metrics
        metrics = ['avg_score', 'performance_consistency', 'behavioral_stability', 
                  'action_entropy', 'avg_survival_time']
        metric_names = ['Avg Score', 'Consistency', 'Stability', 'Diversity', 'Survival']
        
        model_names = [Path(r['model_path']).stem for r in results]
        
        # Normalize metrics
        data = []
        for metric in metrics:
            values = [r[metric] for r in results]
            if metric == 'avg_food_efficiency':  # Lower is better
                normalized = 1 - (np.array(values) - min(values)) / (max(values) - min(values))
            else:
                normalized = (np.array(values) - min(values)) / (max(values) - min(values) + 1e-8)
            data.append(normalized)
        
        data = np.array(data)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(data, 
                    xticklabels=model_names,
                    yticklabels=metric_names,
                    annot=True,
                    fmt='.2f',
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Normalized Score'})
        
        plt.title('Multi-Metric Performance Heatmap (FIXED)', fontsize=16, weight='bold')
        plt.xlabel('Models')
        plt.ylabel('Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        heatmap_path = Path("models") / "performance_heatmap_fixed.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Performance heatmap saved: {heatmap_path}")
    
    def _generate_individual_model_plots(self, results: List[Dict]):
        """Generate individual plots for each model"""
        for result in results:
            model_name = Path(result['model_path']).stem
            scores = result['scores']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Score histogram
            axes[0,0].hist(scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[0,0].axvline(result['avg_score'], color='red', linestyle='dashed', linewidth=2, label=f'Mean: {result["avg_score"]:.2f}')
            axes[0,0].set_title('Score Distribution')
            axes[0,0].set_xlabel('Score')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Learning curve (moving average)
            window = min(20, len(scores) // 5)
            if window > 1:
                moving_avg = [np.mean(scores[max(0, i-window):i+1]) for i in range(len(scores))]
                axes[0,1].plot(moving_avg)
                axes[0,1].set_title('Performance Over Time (Moving Average)')
                axes[0,1].set_xlabel('Episode')
                axes[0,1].set_ylabel('Average Score')
                axes[0,1].grid(True, alpha=0.3)
            
            # Action distribution
            action_dist = result['action_distribution']
            actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            axes[1,0].bar(actions, action_dist, color=['red', 'blue', 'green', 'orange'])
            axes[1,0].set_title('Action Distribution')
            axes[1,0].set_ylabel('Count')
            axes[1,0].grid(True, alpha=0.3)
            
            # Death causes
            death_causes = result['death_causes']
            if death_causes:
                causes = list(death_causes.keys())
                counts = list(death_causes.values())
                axes[1,1].pie(counts, labels=causes, autopct='%1.1f%%')
                axes[1,1].set_title('Death Causes')
            
            plt.suptitle(f'Detailed Analysis: {model_name} (FIXED)', fontsize=16, weight='bold')
            plt.tight_layout()
            
            plot_path = Path("models") / f"analysis_{model_name}_fixed.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_comparison_report(self, results: List[Dict]):
        """Save detailed comparison report"""
        # Rank models by multiple criteria
        rankings = {
            'avg_score': sorted(results, key=lambda x: x['avg_score'], reverse=True),
            'performance_consistency': sorted(results, key=lambda x: x['performance_consistency'], reverse=True),
            'avg_food_efficiency': sorted(results, key=lambda x: x['avg_food_efficiency']),
            'avg_survival_time': sorted(results, key=lambda x: x['avg_survival_time'], reverse=True)
        }
        
        report = {
            "evaluation_summary": {
                "total_models": len(results),
                "evaluation_date": str(np.datetime64('now')),
                "gpu_device": str(self.device),
                "episodes_per_model": results[0]['episodes'] if results else 0,
                "evaluator_version": "Enhanced Fixed"
            },
            "model_rankings": {
                criterion: [
                    {
                        "rank": i + 1,
                        "model": Path(r['model_path']).stem,
                        "type": r['model_type'],
                        "score": r[criterion if criterion != 'efficiency' else 'avg_food_efficiency']
                    }
                    for i, r in enumerate(ranking[:5])
                ]
                for criterion, ranking in rankings.items()
            },
            "technique_summary": self._generate_technique_summary(results),
            "best_performers": {
                "overall_best": Path(rankings['avg_score'][0]['model_path']).stem,
                "most_consistent": Path(rankings['performance_consistency'][0]['model_path']).stem,
                "most_efficient": Path(rankings['avg_food_efficiency'][0]['model_path']).stem,
                "best_survivor": Path(rankings['avg_survival_time'][0]['model_path']).stem
            } if rankings['avg_score'] else {},
            "detailed_results": results
        }
        
        report_path = Path("models") / "enhanced_evaluation_report_fixed.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=lambda x: float(x) if isinstance(x, np.number) else x)
        
        print(f"✅ Enhanced evaluation report saved: {report_path}")
        
        # Print summary
        self._print_enhanced_summary(rankings)
    
    def _generate_technique_summary(self, results: List[Dict]):
        """Generate enhanced summary statistics by technique"""
        techniques = {}
        for result in results:
            tech = result['model_type']
            if tech not in techniques:
                techniques[tech] = {
                    'scores': [],
                    'consistency': [],
                    'efficiency': [],
                    'survival': []
                }
            techniques[tech]['scores'].append(result['avg_score'])
            techniques[tech]['consistency'].append(result['performance_consistency'])
            techniques[tech]['efficiency'].append(result['avg_food_efficiency'])
            techniques[tech]['survival'].append(result['avg_survival_time'])
        
        summary = {}
        for tech, metrics in techniques.items():
            summary[tech] = {
                "count": len(metrics['scores']),
                "avg_score": {
                    "mean": float(np.mean(metrics['scores'])),
                    "std": float(np.std(metrics['scores'])),
                    "max": float(max(metrics['scores'])),
                    "min": float(min(metrics['scores']))
                },
                "consistency": {
                    "mean": float(np.mean(metrics['consistency'])),
                    "std": float(np.std(metrics['consistency']))
                },
                "efficiency": {
                    "mean": float(np.mean(metrics['efficiency'])),
                    "std": float(np.std(metrics['efficiency']))
                },
                "survival": {
                    "mean": float(np.mean(metrics['survival'])),
                    "std": float(np.std(metrics['survival']))
                }
            }
        
        return summary
    
    def _print_enhanced_summary(self, rankings: Dict[str, List[Dict]]):
        """Print enhanced evaluation summary"""
        print(f"\n{'='*80}")
        print("🏆 ENHANCED MODEL EVALUATION SUMMARY (FIXED)")
        print(f"{'='*80}")
        
        criteria = {
            'avg_score': '📊 Highest Average Score',
            'performance_consistency': '🎯 Most Consistent',
            'avg_food_efficiency': '⚡ Most Efficient',
            'avg_survival_time': '🛡️ Best Survivor'
        }
        
        for criterion, title in criteria.items():
            print(f"\n{title}:")
            for i, result in enumerate(rankings[criterion][:3]):
                model_name = Path(result['model_path']).stem
                value = result[criterion]
                print(f"{i+1:2d}. {result['model_type']:15s} {model_name:25s} "
                      f"{value:8.2f}")

def benchmark_enhanced_features():
    """Benchmark the impact of enhanced features"""
    print("🚀 Benchmarking Enhanced Features...")
    
    evaluator = EnhancedModelEvaluator()
    
    # Quick evaluation on subset of models
    model_dir = Path("models")
    test_models = []
    
    # Get one model of each type
    for technique in ["qlearning", "dqn", "ppo", "actor_critic"]:
        tech_dir = model_dir / technique
        if tech_dir.exists():
            if technique == "qlearning":
                models = list(tech_dir.glob("qtable_balanced.json"))
            else:
                models = list(tech_dir.glob(f"*_balanced.pth"))
            
            if models:
                test_models.append((str(models[0]), technique))
    
    if test_models:
        print(f"Found {len(test_models)} models to benchmark")
        
        # Compare evaluation times
        for model_path, model_type in test_models:
            start_time = time.time()
            result = evaluator.evaluate_model(model_path, model_type, episodes=20)
            eval_time = time.time() - start_time
            
            print(f"{model_type:15s}: {eval_time:.2f}s for 20 episodes "
                  f"(Avg Score: {result['avg_score']:.2f})")

if __name__ == "__main__":
    # Benchmark enhanced features
    benchmark_enhanced_features()
    
    # Evaluate all models
    evaluator = EnhancedModelEvaluator()
    results = evaluator.compare_all_models(episodes=100)
    
    if results:
        print(f"\n🎉 Enhanced evaluation complete! {len(results)} models compared.")
        print("📁 Check models/ directory for detailed plots and reports")
    else:
        print("❌ No models found. Run training first:")
        print("   python train_models.py --technique all")

"""
USAGE EXAMPLES:
===============

# Import and use the enhanced evaluator
from model_evaluator import EnhancedModelEvaluator

# Create evaluator instance
evaluator = EnhancedModelEvaluator()

# Evaluate single model with detailed metrics
result = evaluator.evaluate_model("models/dqn/dqn_balanced.pth", "dqn", episodes=100, detailed_metrics=True)
print(f"Average score: {result['avg_score']:.2f}")
print(f"Performance consistency: {result['performance_consistency']:.3f}")
print(f"Behavioral stability: {result['behavioral_stability']:.3f}")

# Compare all available models with comprehensive analysis
results = evaluator.compare_all_models(episodes=100, save_individual_plots=True)

# Benchmark evaluation performance
benchmark_enhanced_features()

# Run from command line
python model_evaluator.py

# Key features:
# - Fixed Actor-Critic model loading with separate actor/critic networks
# - Dynamic state preparation for different input sizes (8D, 11D, 20D)
# - Comprehensive metrics: efficiency, stability, diversity, survival
# - Advanced visualizations: radar charts, heatmaps, individual plots
# - Detailed ranking by multiple criteria
# - Technique-wise performance summaries

# Generated outputs:
# - enhanced_comparison_fixed.png: Comprehensive 9-panel comparison
# - performance_heatmap_fixed.png: Multi-metric heatmap
# - analysis_[model]_fixed.png: Individual model analysis
# - enhanced_evaluation_report_fixed.json: Detailed JSON report

# Model compatibility:
# - Q-Learning: 8D state, Q-table format
# - DQN: 8D state, neural network
# - PPO: 8D state, policy + value networks
# - Actor-Critic: 8D state, separate actor + critic networks

# Performance metrics explained:
# - avg_score: Mean score across episodes
# - performance_consistency: 1/(1+std), higher is more consistent
# - behavioral_stability: 1-action_change_rate, measures action consistency
# - action_entropy: Diversity of action selection
# - food_efficiency: Steps per food eaten (lower is better)
"""