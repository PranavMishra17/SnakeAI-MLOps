#!/usr/bin/env python3
"""
SnakeAI-MLOps Example Usage Scripts
Demonstrates how to use various components of the system
"""
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from neural_network_utils import verify_gpu, create_directories
from qlearning_trainer import train_qlearning, TrainingConfig as QConfig
from dqn_trainer import train_dqn, DQNConfig
from ppo_trainer import train_policy_gradient, PolicyGradientConfig
from actor_critic_trainer import train_actor_critic, ActorCriticConfig
from model_evaluator import UnifiedModelEvaluator

def example_01_basic_training():
    """Example 1: Basic Q-Learning training"""
    print("🎯 Example 1: Basic Q-Learning Training")
    print("=" * 50)
    
    # Verify GPU
    device = verify_gpu()
    
    # Create directories
    create_directories("models")
    
    # Quick Q-Learning training
    config = QConfig(
        profile_name="example_basic",
        max_episodes=1000,  # Short training for demo
        learning_rate=0.1,
        epsilon_start=0.2,
        epsilon_end=0.05,
        target_score=10
    )
    
    print(f"Training Q-Learning agent for {config.max_episodes} episodes...")
    start_time = time.time()
    
    train_qlearning(config)
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.1f} seconds")
    print(f"📁 Model saved to: models/qlearning/qtable_example_basic.json")

def example_02_neural_network_training():
    """Example 2: Neural network training comparison"""
    print("\n🧠 Example 2: Neural Network Training")
    print("=" * 50)
    
    # Short training configs for demonstration
    
    # DQN training
    dqn_config = DQNConfig(
        profile_name="example_dqn",
        max_episodes=500,
        learning_rate=0.001,
        batch_size=32,
        target_score=8
    )
    
    print("Training DQN agent...")
    start_time = time.time()
    train_dqn(dqn_config)
    dqn_time = time.time() - start_time
    print(f"✅ DQN training completed in {dqn_time:.1f} seconds")
    
    # Policy Gradient training
    pg_config = PolicyGradientConfig(
        profile_name="example_pg",
        max_episodes=600,
        learning_rate=0.002,
        target_score=8
    )
    
    print("Training Policy Gradient agent...")
    start_time = time.time()
    train_policy_gradient(pg_config)
    pg_time = time.time() - start_time
    print(f"✅ Policy Gradient training completed in {pg_time:.1f} seconds")
    
    print(f"\n📊 Training Time Comparison:")
    print(f"   DQN: {dqn_time:.1f}s")
    print(f"   Policy Gradient: {pg_time:.1f}s")

def example_03_model_evaluation():
    """Example 3: Model evaluation and comparison"""
    print("\n📊 Example 3: Model Evaluation")
    print("=" * 50)
    
    evaluator = UnifiedModelEvaluator()
    
    # Check available models
    model_dir = Path("models")
    available_models = []
    
    # Find Q-Learning models
    qlearning_dir = model_dir / "qlearning"
    if qlearning_dir.exists():
        for model_file in qlearning_dir.glob("qtable_*.json"):
            if "report" not in model_file.name:
                available_models.append((str(model_file), "qlearning"))
    
    # Find neural network models
    for technique in ["dqn", "policy_gradient", "actor_critic"]:
        tech_dir = model_dir / technique
        if tech_dir.exists():
            for model_file in tech_dir.glob("*.pth"):
                if "checkpoint" not in model_file.name and "best" not in model_file.name:
                    available_models.append((str(model_file), technique))
    
    if not available_models:
        print("❌ No models found. Please run training examples first.")
        return
    
    print(f"Found {len(available_models)} models to evaluate:")
    for model_path, model_type in available_models:
        model_name = Path(model_path).stem
        print(f"   • {model_type}: {model_name}")
    
    # Evaluate each model
    results = []
    for model_path, model_type in available_models[:3]:  # Limit to first 3 for demo
        print(f"\nEvaluating {model_type} model...")
        result = evaluator.evaluate_model(model_path, model_type, episodes=20)  # Short evaluation
        results.append(result)
    
    # Print comparison
    print(f"\n🏆 Evaluation Results:")
    sorted_results = sorted(results, key=lambda x: x['avg_score'], reverse=True)
    for i, result in enumerate(sorted_results):
        model_name = Path(result['model_path']).stem
        print(f"{i+1}. {result['model_type']:15s} {model_name:20s} "
              f"Avg: {result['avg_score']:6.2f} Max: {result['max_score']:3d}")

def example_04_hyperparameter_comparison():
    """Example 4: Hyperparameter comparison"""
    print("\n⚙️ Example 4: Hyperparameter Comparison")
    print("=" * 50)
    
    # Compare different learning rates for Q-Learning
    learning_rates = [0.05, 0.1, 0.2]
    results = []
    
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        
        config = QConfig(
            profile_name=f"example_lr_{lr}",
            max_episodes=800,
            learning_rate=lr,
            epsilon_start=0.2,
            target_score=8
        )
        
        start_time = time.time()
        train_qlearning(config)
        training_time = time.time() - start_time
        
        # Quick evaluation
        evaluator = UnifiedModelEvaluator()
        model_path = f"models/qlearning/qtable_example_lr_{lr}.json"
        result = evaluator.evaluate_model(model_path, "qlearning", episodes=15)
        
        results.append({
            'learning_rate': lr,
            'avg_score': result['avg_score'],
            'training_time': training_time
        })
    
    # Print comparison
    print(f"\n📈 Learning Rate Comparison:")
    print("LR     | Avg Score | Training Time")
    print("-------|-----------|-------------")
    for result in results:
        print(f"{result['learning_rate']:<6} | {result['avg_score']:8.2f}  | {result['training_time']:8.1f}s")
    
    # Find best learning rate
    best_result = max(results, key=lambda x: x['avg_score'])
    print(f"\n🎯 Best learning rate: {best_result['learning_rate']} "
          f"(avg score: {best_result['avg_score']:.2f})")

def example_05_custom_training_loop():
    """Example 5: Custom training loop with monitoring"""
    print("\n🔄 Example 5: Custom Training Loop")
    print("=" * 50)
    
    from qlearning_trainer import QLearningAgent, SnakeEnvironment
    
    # Create custom environment and agent
    env = SnakeEnvironment(device='cuda' if verify_gpu().type == 'cuda' else 'cpu')
    
    custom_config = QConfig(
        profile_name="custom_example",
        learning_rate=0.15,
        epsilon_start=0.3,
        epsilon_end=0.05,
        epsilon_decay=0.99
    )
    
    agent = QLearningAgent(custom_config)
    
    # Custom training loop with detailed monitoring
    episodes = 500
    scores = []
    
    print(f"Running custom training loop for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.get_action(state, training=True)
            next_state, reward, done = env.step(action)
            
            agent.update(state, action, reward, next_state)
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        agent.decay_epsilon()
        scores.append(env.score)
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_score = sum(scores[-100:]) / min(len(scores), 100)
            print(f"Episode {episode+1:3d}: Avg Score: {avg_score:5.2f}, "
                  f"Last Score: {env.score:2d}, Epsilon: {agent.epsilon:.3f}")
    
    # Final statistics
    final_avg = sum(scores[-100:]) / min(len(scores), 100)
    max_score = max(scores)
    
    print(f"\n📊 Custom Training Results:")
    print(f"   Final Average Score (last 100): {final_avg:.2f}")
    print(f"   Maximum Score: {max_score}")
    print(f"   Final Epsilon: {agent.epsilon:.3f}")
    
    # Save custom model
    model_path = "models/qlearning/qtable_custom_example.json"
    agent.save_model(model_path)
    print(f"   Model saved to: {model_path}")

def example_06_advanced_evaluation():
    """Example 6: Advanced model evaluation with visualizations"""
    print("\n📈 Example 6: Advanced Evaluation")
    print("=" * 50)
    
    # This example shows how to create custom evaluation metrics
    
    import json
    import matplotlib.pyplot as plt
    from collections import defaultdict
    
    evaluator = UnifiedModelEvaluator()
    
    # Find all Q-Learning models for comparison
    qlearning_dir = Path("models/qlearning")
    if not qlearning_dir.exists():
        print("❌ No Q-Learning models found. Run training examples first.")
        return
    
    model_files = list(qlearning_dir.glob("qtable_*.json"))
    if len(model_files) < 2:
        print("❌ Need at least 2 models for comparison. Run more training examples.")
        return
    
    # Evaluate all models
    all_results = []
    for model_file in model_files[:4]:  # Limit to first 4
        print(f"Evaluating {model_file.stem}...")
        result = evaluator.evaluate_model(str(model_file), "qlearning", episodes=30)
        all_results.append(result)
    
    # Create custom visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Score distributions
    plt.subplot(2, 2, 1)
    for result in all_results:
        model_name = Path(result['model_path']).stem.replace('qtable_', '')
        plt.hist(result['scores'], alpha=0.6, label=model_name, bins=10)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distributions')
    plt.legend()
    
    # Plot 2: Performance comparison
    plt.subplot(2, 2, 2)
    models = [Path(r['model_path']).stem.replace('qtable_', '') for r in all_results]
    avg_scores = [r['avg_score'] for r in all_results]
    max_scores = [r['max_score'] for r in all_results]
    
    x = range(len(models))
    plt.bar([i - 0.2 for i in x], avg_scores, 0.4, label='Average', alpha=0.8)
    plt.bar([i + 0.2 for i in x], max_scores, 0.4, label='Maximum', alpha=0.8)
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    
    # Plot 3: Consistency analysis
    plt.subplot(2, 2, 3)
    std_scores = [r['std_score'] for r in all_results]
    plt.bar(models, std_scores, alpha=0.8, color='orange')
    plt.xlabel('Models')
    plt.ylabel('Standard Deviation')
    plt.title('Score Consistency (Lower = More Consistent)')
    plt.xticks(rotation=45)
    
    # Plot 4: Action diversity
    plt.subplot(2, 2, 4)
    entropies = [r['action_entropy'] for r in all_results]
    plt.bar(models, entropies, alpha=0.8, color='green')
    plt.xlabel('Models')
    plt.ylabel('Action Entropy')
    plt.title('Behavioral Diversity')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('models/example_advanced_evaluation.png', dpi=300, bbox_inches='tight')
    print("✅ Advanced evaluation plot saved to: models/example_advanced_evaluation.png")
    
    # Print detailed analysis
    print(f"\n🔍 Detailed Analysis:")
    for result in all_results:
        model_name = Path(result['model_path']).stem.replace('qtable_', '')
        consistency = "High" if result['std_score'] < 3 else "Medium" if result['std_score'] < 6 else "Low"
        diversity = "High" if result['action_entropy'] > 1.5 else "Medium" if result['action_entropy'] > 1.0 else "Low"
        
        print(f"   {model_name}:")
        print(f"     Performance: {result['avg_score']:.2f} ± {result['std_score']:.2f}")
        print(f"     Consistency: {consistency}")
        print(f"     Behavioral Diversity: {diversity}")

def run_all_examples():
    """Run all examples in sequence"""
    print("🚀 Running All SnakeAI-MLOps Examples")
    print("=" * 60)
    
    examples = [
        example_01_basic_training,
        example_02_neural_network_training,
        example_03_model_evaluation,
        example_04_hyperparameter_comparison,
        example_05_custom_training_loop,
        example_06_advanced_evaluation
    ]
    
    start_time = time.time()
    
    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\n{'='*20} Running Example {i}/{len(examples)} {'='*20}")
            example_func()
            print(f"✅ Example {i} completed successfully")
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            print("Continuing with next example...")
    
    total_time = time.time() - start_time
    print(f"\n🎉 All examples completed in {total_time/60:.1f} minutes")
    print(f"📁 Check the models/ directory for generated models and plots")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SnakeAI-MLOps Examples")
    parser.add_argument("--example", type=int, choices=range(1, 7), 
                       help="Run specific example (1-6)")
    parser.add_argument("--all", action="store_true", 
                       help="Run all examples")
    
    args = parser.parse_args()
    
    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_01_basic_training,
            2: example_02_neural_network_training,
            3: example_03_model_evaluation,
            4: example_04_hyperparameter_comparison,
            5: example_05_custom_training_loop,
            6: example_06_advanced_evaluation
        }
        examples[args.example]()
    else:
        print("SnakeAI-MLOps Examples")
        print("=" * 30)
        print("Available examples:")
        print("1. Basic Q-Learning training")
        print("2. Neural network training comparison")
        print("3. Model evaluation")
        print("4. Hyperparameter comparison")
        print("5. Custom training loop")
        print("6. Advanced evaluation with visualizations")
        print("\nUsage:")
        print("  python examples.py --example 1")
        print("  python examples.py --all")

if __name__ == "__main__":
    main()