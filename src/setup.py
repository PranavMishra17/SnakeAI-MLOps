#!/usr/bin/env python3
"""
SnakeAI-MLOps Setup Script
Automated installation and configuration for all components
"""
import os
import sys
import subprocess
import platform
from pathlib import Path
import argparse

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_gpu_support():
    """Check for CUDA GPU support"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("⚠️  No GPU detected, using CPU (training will be slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet, GPU check will be done after installation")
        return None

def install_python_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "src/requirements.txt"
        ])
        print("✅ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False

def create_directory_structure():
    """Create necessary directory structure"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "models/qlearning",
        "models/dqn", 
        "models/policy_gradient",
        "models/actor_critic",
        "models/checkpoints/dqn",
        "models/checkpoints/policy_gradient",
        "models/checkpoints/actor_critic",
        "data",
        "logs",
        "assets/fonts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    print("✅ Directory structure created")
    return True

def setup_git_hooks():
    """Setup git hooks for development"""
    print("\n🔧 Setting up git hooks...")
    
    hooks_dir = Path(".git/hooks")
    if not hooks_dir.exists():
        print("⚠️  Not a git repository, skipping git hooks")
        return True
    
    # Pre-commit hook for code formatting
    pre_commit_hook = hooks_dir / "pre-commit"
    hook_content = """#!/bin/sh
# Auto-format Python code before commit
black src/
flake8 src/ --max-line-length=100 --ignore=E203,W503
"""
    
    try:
        with open(pre_commit_hook, 'w') as f:
            f.write(hook_content)
        os.chmod(pre_commit_hook, 0o755)
        print("✅ Git hooks configured")
    except Exception as e:
        print(f"⚠️  Could not setup git hooks: {e}")
    
    return True

def verify_cpp_dependencies():
    """Verify C++ build dependencies"""
    print("\n🔧 Checking C++ dependencies...")
    
    required_tools = {
        'cmake': 'CMake',
        'git': 'Git'
    }
    
    missing_tools = []
    for tool, name in required_tools.items():
        try:
            subprocess.check_output([tool, '--version'], stderr=subprocess.DEVNULL)
            print(f"✅ {name} found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {name} not found")
            missing_tools.append(name)
    
    if platform.system() == "Windows":
        # Check for Visual Studio
        vs_paths = [
            "C:/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe",
            "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/MSBuild/Current/Bin/MSBuild.exe"
        ]
        
        vs_found = any(Path(path).exists() for path in vs_paths)
        if vs_found:
            print("✅ Visual Studio found")
        else:
            print("❌ Visual Studio not found")
            missing_tools.append("Visual Studio 2019/2022")
    
    if missing_tools:
        print(f"\n⚠️  Missing tools: {', '.join(missing_tools)}")
        print("Please install these tools to build the C++ game")
        return False
    
    return True

def setup_vcpkg():
    """Setup vcpkg package manager"""
    print("\n📦 Setting up vcpkg...")
    
    vcpkg_dir = Path("../vcpkg")
    if vcpkg_dir.exists():
        print("✅ vcpkg directory found")
        return True
    
    try:
        print("Cloning vcpkg...")
        subprocess.check_call([
            "git", "clone", "https://github.com/Microsoft/vcpkg.git", str(vcpkg_dir)
        ])
        
        # Bootstrap vcpkg
        if platform.system() == "Windows":
            bootstrap_script = vcpkg_dir / "bootstrap-vcpkg.bat"
            subprocess.check_call([str(bootstrap_script)])
        else:
            bootstrap_script = vcpkg_dir / "bootstrap-vcpkg.sh"
            subprocess.check_call(["sh", str(bootstrap_script)])
        
        # Integrate vcpkg
        subprocess.check_call([
            str(vcpkg_dir / "vcpkg"), "integrate", "install"
        ])
        
        print("✅ vcpkg setup complete")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ vcpkg setup failed: {e}")
        return False

def install_cpp_dependencies():
    """Install C++ dependencies via vcpkg"""
    print("\n📦 Installing C++ dependencies...")
    
    vcpkg_exe = Path("../vcpkg/vcpkg")
    if platform.system() == "Windows":
        vcpkg_exe = Path("../vcpkg/vcpkg.exe")
    
    if not vcpkg_exe.exists():
        print("❌ vcpkg not found, please run setup with --vcpkg first")
        return False
    
    dependencies = [
        "sfml:x64-windows" if platform.system() == "Windows" else "sfml",
        "nlohmann-json:x64-windows" if platform.system() == "Windows" else "nlohmann-json",
        "spdlog:x64-windows" if platform.system() == "Windows" else "spdlog"
    ]
    
    try:
        for dep in dependencies:
            print(f"Installing {dep}...")
            subprocess.check_call([str(vcpkg_exe), "install", dep])
        
        print("✅ C++ dependencies installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ C++ dependency installation failed: {e}")
        return False

def build_cpp_project():
    """Build the C++ project"""
    print("\n🔨 Building C++ project...")
    
    try:
        # Configure
        subprocess.check_call([
            "cmake", "--preset", "windows-default" if platform.system() == "Windows" else "linux-default"
        ])
        
        # Build
        build_dir = "out/build/windows-default" if platform.system() == "Windows" else "build"
        subprocess.check_call([
            "cmake", "--build", build_dir
        ])
        
        print("✅ C++ project built successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ C++ build failed: {e}")
        return False

def run_tests():
    """Run basic tests to verify installation"""
    print("\n🧪 Running verification tests...")
    
    # Test Python imports
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
        print("✅ Python imports successful")
    except ImportError as e:
        print(f"❌ Python import failed: {e}")
        return False
    
    # Test GPU (if available)
    try:
        import torch
        if torch.cuda.is_available():
            test_tensor = torch.randn(10, 10).cuda()
            result = torch.mm(test_tensor, test_tensor)
            print("✅ GPU computation test successful")
        else:
            print("⚠️  GPU not available, but CPU fallback works")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False
    
    # Test directory structure
    required_dirs = ["models", "data", "logs", "assets"]
    for directory in required_dirs:
        if not Path(directory).exists():
            print(f"❌ Directory {directory} not found")
            return False
    print("✅ Directory structure verified")
    
    return True

def print_usage_instructions():
    """Print usage instructions after successful setup"""
    print(f"\n{'='*60}")
    print("🎉 SETUP COMPLETE! 🎉")
    print(f"{'='*60}")
    
    print("\n📚 Next Steps:")
    print("1. Train models:")
    print("   python src/train_models.py --technique all")
    print("   python src/train_models.py --technique dqn --profile balanced")
    
    print("\n2. Evaluate models:")
    print("   python src/train_models.py --evaluate")
    print("   python src/model_evaluator.py")
    
    print("\n3. Run the game:")
    if platform.system() == "Windows":
        print("   .\\out\\build\\windows-default\\SnakeAI-MLOps.exe")
    else:
        print("   ./build/SnakeAI-MLOps")
    
    print("\n📖 Documentation:")
    print("   - ML.md: Comprehensive ML guide")
    print("   - README.md: General usage")
    print("   - Individual trainer files: src/*_trainer.py")
    
    print("\n🆘 Troubleshooting:")
    print("   - GPU issues: python src/train_models.py --gpu-test")
    print("   - Dependencies: python setup.py --python-only")
    print("   - Clean rebuild: rm -rf models/ && python setup.py --full")

def main():
    parser = argparse.ArgumentParser(description="SnakeAI-MLOps Setup")
    parser.add_argument("--python-only", action="store_true", 
                       help="Setup only Python environment")
    parser.add_argument("--cpp-only", action="store_true",
                       help="Setup only C++ environment")
    parser.add_argument("--vcpkg", action="store_true",
                       help="Setup vcpkg package manager")
    parser.add_argument("--no-tests", action="store_true",
                       help="Skip verification tests")
    parser.add_argument("--full", action="store_true",
                       help="Full setup (default)")
    
    args = parser.parse_args()
    
    print("🚀 SnakeAI-MLOps Setup Starting...")
    print("=" * 50)
    
    # Basic checks
    if not check_python_version():
        sys.exit(1)
    
    success = True
    
    if args.python_only or args.full or not any([args.cpp_only, args.vcpkg]):
        # Python setup
        success &= create_directory_structure()
        success &= install_python_dependencies()
        success &= setup_git_hooks()
        
        # Check GPU after PyTorch installation
        check_gpu_support()
    
    if args.vcpkg:
        success &= setup_vcpkg()
        
    if args.cpp_only or args.full:
        # C++ setup
        success &= verify_cpp_dependencies()
        if success:
            success &= install_cpp_dependencies()
            success &= build_cpp_project()
    
    if args.full and not args.no_tests:
        success &= run_tests()
    
    if success:
        print_usage_instructions()
    else:
        print("\n❌ Setup completed with errors")
        print("Please check the error messages above and retry")
        sys.exit(1)

if __name__ == "__main__":
    main()