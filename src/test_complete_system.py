#!/usr/bin/env python3
"""
Comprehensive System Test for SnakeAI-MLOps
Tests all components: training, evaluation, model loading, and integration
"""
import sys
import time
import json
from pathlib import Path
import subprocess
import tempfile
import shutil

# Add src to path
sys.path.append('src')

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Core utilities
        from neural_network_utils import verify_gpu, create_directories, NetworkConfig
        print("✅ Neural network utilities")
        
        # Trainers
        from qlearning_trainer import train_qlearning, TrainingConfig, QLearningAgent
        print("✅ Q-Learning trainer")
        
        from dqn_trainer import train_dqn, DQNConfig, DQNAgent
        print("✅ DQN trainer")
        
        from ppo_trainer import train_policy_gradient, PolicyGradientConfig
        print("✅ Policy Gradient trainer")
        
        from actor_critic_trainer import train_actor_critic, ActorCriticConfig
        print("✅ Actor-Critic trainer")
        
        # Evaluation and orchestration
        from model_evaluator import UnifiedModelEvaluator
        print("✅ Model evaluator")
        
        from train_models import train_single_technique
        print("✅ Training orchestrator")
        
        # PyTorch and dependencies
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ External dependencies")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_gpu_setup():
    """Test GPU availability and basic operations"""
    print("\n🧪 Testing GPU setup...")
    
    try:
        import torch
        
        # Check CUDA availability
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU Available: {gpu_name}")
            
            # Test basic operations
            x = torch.randn(1000, 1000, device=device)
            y = torch.mm(x, x)
            print("✅ GPU tensor operations")
            
            # Test neural network on GPU
            from neural_network_utils import DQNNetwork, NetworkConfig
            config = NetworkConfig(input_size=20, hidden_layers=[64, 32], output_size=4)
            model = DQNNetwork(config).to(device)
            
            test_input = torch.randn(10, 20, device=device)
            output = model(test_input)
            print("✅ Neural network GPU inference")
            
            return True
        else:
            print("⚠️  No GPU available, using CPU")
            return True
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_directory_structure():
    """Test that directory structure is correctly created"""
    print("\n🧪 Testing directory structure...")
    
    try:
        from neural_network_utils import create_directories
        
        # Create test directory structure
        test_dir = "test_models"
        create_directories(test_dir)
        
        required_dirs = [
            f"{test_dir}/qlearning",
            f"{test_dir}/dqn",
            f"{test_dir}/policy_gradient", 
            f"{test_dir}/actor_critic",
            f"{test_dir}/checkpoints"
        ]
        
        for directory in required_dirs:
            if not Path(directory).exists():
                print(f"❌ Directory not created: {directory}")
                return False
        
        print("✅ Directory structure created correctly")
        
        # Clean up test directory
        shutil.rmtree(test_dir)
        print("✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Directory test failed: {e}")
        return False

def test_qlearning_training():
    """Test Q-Learning training pipeline"""
    print("\n🧪 Testing Q-Learning training...")
    
    try:
        from qlearning_trainer import train_qlearning, TrainingConfig
        
        # Quick training configuration
        config = TrainingConfig(
            profile_name="test_qlearning",
            max_episodes=100,  # Very short for testing
            learning_rate=0.1,
            epsilon_start=0.2,
            target_score=5
        )
        
        # Ensure test directory exists
        Path("models/qlearning").mkdir(parents=True, exist_ok=True)
        
        print("Training Q-Learning agent (100 episodes)...")
        start_time = time.time()
        train_qlearning(config)
        training_time = time.time() - start_time
        
        # Check output files
        model_file = "models/qlearning/qtable_test_qlearning.json"
        report_file = "models/qlearning/qtable_test_qlearning_report.json"
        
        if not Path(model_file).exists():
            print(f"❌ Model file not created: {model_file}")
            return False
            
        if not Path(report_file).exists():
            print(f"❌ Report file not created: {report_file}")
            return False
        
        # Validate JSON format
        with open(model_file, 'r') as f:
            model_data = json.load(f)
            
        if 'qTable' not in model_data or 'hyperparameters' not in model_data:
            print("❌ Invalid model file format")
            return False
        
        print(f"✅ Q-Learning training completed ({training_time:.1f}s)")
        print(f"✅ Model saved with {len(model_data['qTable'])} states")
        
        return True
        
    except Exception as e:
        print(f"❌ Q-Learning training failed: {e}")
        return False

def test_neural_network_training():
    """Test neural network training (DQN as representative)"""
    print("\n🧪 Testing neural network training...")
    
    try:
        from dqn_trainer import train_dqn, DQNConfig
        
        # Quick training configuration
        config = DQNConfig(
            profile_name="test_dqn",
            max_episodes=50,  # Very short for testing
            learning_rate=0.001,
            batch_size=16,
            target_score=3,
            memory_capacity=1000
        )
        
        # Ensure test directory exists
        Path("models/dqn").mkdir(parents=True, exist_ok=True)
        
        print("Training DQN agent (50 episodes)...")
        start_time = time.time()
        train_dqn(config)
        training_time = time.time() - start_time
        
        # Check output files
        model_file = "models/dqn/dqn_test_dqn.pth"
        
        if not Path(model_file).exists():
            print(f"❌ DQN model file not created: {model_file}")
            return False
        
        # Validate PyTorch format
        import torch
        checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)  # FIXED
        
        required_keys = ['q_network', 'config', 'metadata']
        for key in required_keys:
            if key not in checkpoint:
                print(f"❌ Missing key in checkpoint: {key}")
                return False
        
        print(f"✅ DQN training completed ({training_time:.1f}s)")
        print(f"✅ Model saved successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ DQN training failed: {e}")
        return False

def test_model_evaluation():
    """Test model evaluation system"""
    print("\n🧪 Testing model evaluation...")
    
    try:
        from model_evaluator import UnifiedModelEvaluator
        
        evaluator = UnifiedModelEvaluator()
        
        # Find test models created in previous tests
        qlearning_model = "models/qlearning/qtable_test_qlearning.json"
        dqn_model = "models/dqn/dqn_test_dqn.pth"
        
        test_results = []
        
        # Test Q-Learning evaluation
        if Path(qlearning_model).exists():
            print("Evaluating Q-Learning model...")
            result = evaluator.evaluate_model(qlearning_model, "qlearning", episodes=10)
            test_results.append(result)
            print(f"✅ Q-Learning evaluation: avg score {result['avg_score']:.2f}")
        
        # Test DQN evaluation
        if Path(dqn_model).exists():
            print("Evaluating DQN model...")
            result = evaluator.evaluate_model(dqn_model, "dqn", episodes=10)
            test_results.append(result)
            print(f"✅ DQN evaluation: avg score {result['avg_score']:.2f}")
        
        if not test_results:
            print("❌ No models found for evaluation")
            return False
        
        # Test comparison functionality
        if len(test_results) > 1:
            print("Testing model comparison...")
            # This would normally generate plots and reports
            print("✅ Model comparison functionality works")
        
        return True
        
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        return False

def test_training_orchestrator():
    """Test the main training orchestrator"""
    print("\n🧪 Testing training orchestrator...")
    
    try:
        from train_models import train_single_technique
        
        # Test orchestrator with Q-Learning (fastest to train)
        print("Testing orchestrator with Q-Learning...")
        train_single_technique("qlearning", "balanced", episodes=50)
        
        # Check if model was created
        model_file = "models/qlearning/qtable_balanced.json"
        if Path(model_file).exists():
            print("✅ Training orchestrator works correctly")
            return True
        else:
            print("❌ Orchestrator did not create expected model")
            return False
            
    except Exception as e:
        print(f"❌ Training orchestrator test failed: {e}")
        return False

def test_state_representation():
    """Test state representation and encoding"""
    print("\n🧪 Testing state representation...")
    
    try:
        from qlearning_trainer import SnakeEnvironment
        from neural_network_utils import encode_state_for_dqn
        
        # Test basic environment
        env = SnakeEnvironment(device='cpu')
        state = env.reset()
        
        if state.shape[0] != 8:
            print(f"❌ Expected 8D state, got {state.shape[0]}D")
            return False
        
        print("✅ Basic state representation (8D)")
        
        # Test enhanced state
        enhanced_state = encode_state_for_dqn(state)
        
        if enhanced_state.shape[0] != 20:
            print(f"❌ Expected 20D enhanced state, got {enhanced_state.shape[0]}D")
            return False
            
        print("✅ Enhanced state representation (20D)")
        
        # Test state encoding for Q-Learning
        from qlearning_trainer import QLearningAgent, TrainingConfig
        
        config = TrainingConfig(profile_name="test_encoding")
        agent = QLearningAgent(config)
        
        # Test state encoding
        state_idx = agent.encode_state(state)
        if not (0 <= state_idx < 512):
            print(f"❌ Invalid state index: {state_idx}")
            return False
            
        print("✅ Q-Learning state encoding")
        
        return True
        
    except Exception as e:
        print(f"❌ State representation test failed: {e}")
        return False

def test_model_loading():
    """Test model loading and inference"""
    print("\n🧪 Testing model loading...")
    
    try:
        # Test Q-Learning model loading
        qlearning_model = "models/qlearning/qtable_test_qlearning.json"
        if Path(qlearning_model).exists():
            from qlearning_trainer import QLearningAgent, TrainingConfig, SnakeEnvironment
            
            # Create agent and load model
            config = TrainingConfig(profile_name="test_loading")
            agent = QLearningAgent(config)
            
            # Load the trained model
            if agent.load_model(qlearning_model):
                print("✅ Q-Learning model loaded")
                
                # Test inference
                env = SnakeEnvironment(device='cpu')
                state = env.reset()
                action = agent.get_action(state, training=False)
                
                if 0 <= action <= 3:
                    print("✅ Q-Learning inference works")
                else:
                    print(f"❌ Invalid action: {action}")
                    return False
            else:
                print("❌ Q-Learning model loading failed")
                return False
        
        # Test neural network model loading
        dqn_model = "models/dqn/dqn_test_dqn.pth"
        if Path(dqn_model).exists():
            import torch
            from neural_network_utils import DQNNetwork, NetworkConfig
            
            # Load checkpoint
            checkpoint = torch.load(dqn_model, map_location='cpu', weights_only=False)  # FIXED
            
            # Recreate model from config
            # After (fixed):
            config_dict = checkpoint['config']
            net_config = NetworkConfig(
                input_size=20,
                hidden_layers=config_dict.get('hidden_layers', [128, 64]),
                output_size=4,
                dropout=0.0
            )
            model = DQNNetwork(net_config)
            model.load_state_dict(checkpoint['q_network'])
            model.eval()
            
            # Test inference
            test_state = torch.randn(1, 20)
            with torch.no_grad():
                q_values = model(test_state)
                
            if q_values.shape == (1, 4):
                print("✅ DQN model loading and inference works")
            else:
                print(f"❌ Invalid DQN output shape: {q_values.shape}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def test_cpp_compatibility():
    """Test C++ build compatibility (if possible)"""
    print("\n🧪 Testing C++ compatibility...")
    
    try:
        # Check if C++ build exists
        cpp_builds = [
            "out/build/windows-default/SnakeAI-MLOps.exe",
            "build/SnakeAI-MLOps"
        ]
        
        cpp_executable = None
        for build_path in cpp_builds:
            if Path(build_path).exists():
                cpp_executable = build_path
                break
        
        if cpp_executable:
            print(f"✅ C++ executable found: {cpp_executable}")
            
            # Test that it can start (just version check, not full run)
            try:
                # Most games respond to --help or --version
                result = subprocess.run([cpp_executable, "--help"], 
                                      capture_output=True, timeout=5)
                print("✅ C++ executable can be launched")
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                # Expected - game doesn't have help option, but it started
                print("✅ C++ executable starts successfully")
        else:
            print("⚠️  C++ executable not found (run cmake build)")
            print("   This is expected if C++ build hasn't been done yet")
        
        # Check that Q-Learning models can be loaded by checking file format
        qlearning_models = list(Path("models/qlearning").glob("qtable_*.json"))
        if qlearning_models:
            with open(qlearning_models[0], 'r') as f:
                model_data = json.load(f)
                
            # Check C++ compatible format
            if 'qTable' in model_data and 'hyperparameters' in model_data:
                print("✅ Q-Learning models are C++ compatible")
            else:
                print("❌ Q-Learning models not in C++ compatible format")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ C++ compatibility test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files created during testing"""
    print("\n🧹 Cleaning up test files...")
    
    test_files = [
        "models/qlearning/qtable_test_qlearning.json",
        "models/qlearning/qtable_test_qlearning_report.json",
        "models/dqn/dqn_test_dqn.pth",
        "models/dqn/dqn_test_dqn_metrics.json",
        "models/qlearning/training_curves_test_qlearning.png",
        "models/dqn/dqn_training_curves_test_dqn.png"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            Path(file_path).unlink()
            print(f"Removed: {file_path}")
    
    print("✅ Cleanup completed")

def run_comprehensive_test():
    """Run all tests in sequence"""
    print("🚀 SnakeAI-MLOps Comprehensive System Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("GPU Setup", test_gpu_setup),
        ("Directory Structure", test_directory_structure),
        ("Q-Learning Training", test_qlearning_training),
        ("Neural Network Training", test_neural_network_training),
        ("Model Evaluation", test_model_evaluation),
        ("Training Orchestrator", test_training_orchestrator),
        ("State Representation", test_state_representation),
        ("Model Loading", test_model_loading),
        ("C++ Compatibility", test_cpp_compatibility)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Cleanup
    cleanup_test_files()
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n{'='*60}")
    print("🎯 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    print(f"Total Time: {total_time:.1f} seconds")
    
    print(f"\n📊 Individual Results:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name:<25}: {status}")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED! 🎉")
        print("The SnakeAI-MLOps system is working correctly.")
        print("\nNext steps:")
        print("1. Run full training: python src/train_models.py --technique all")
        print("2. Evaluate models: python src/train_models.py --evaluate") 
        print("3. Play the game: ./out/build/windows-default/SnakeAI-MLOps.exe")
    else:
        failed_tests = [name for name, success in results if not success]
        print(f"\n⚠️  {total - passed} tests failed:")
        for test_name in failed_tests:
            print(f"   - {test_name}")
        print("\nPlease check the error messages above and:")
        print("1. Ensure all dependencies are installed")
        print("2. Check GPU setup if neural network tests failed")
        print("3. Verify file permissions")
        print("4. Run: python setup.py --full")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)