#!/usr/bin/env python3
"""
Colab Setup Script for QGML GPU Testing
Run this first in Colab to set up the environment.
"""

import os
import subprocess
import sys

def install_packages():
    """Install required packages for GPU testing"""
    print("📦 Installing Required Packages for GPU Testing")
    print("=" * 60)
    
    packages = [
        "jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "optax matplotlib numpy scipy"
    ]
    
    for package in packages:
        print(f"Installing: {package}")
        result = subprocess.run(f"pip install {package}", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Successfully installed: {package}")
        else:
            print(f"❌ Failed to install: {package}")
            print(f"Error: {result.stderr}")
        print()
    
    print("✅ Package installation complete!")

def check_gpu():
    """Check GPU availability"""
    print("🔍 Checking GPU Availability")
    print("=" * 40)
    
    try:
        import torch
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"PyTorch CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("PyTorch not installed yet")
    
    try:
        import jax
        print(f"JAX devices: {jax.devices()}")
        print(f"JAX default backend: {jax.default_backend()}")
    except ImportError:
        print("JAX not installed yet")
    
    print()

def clone_repo():
    """Clone the QGML repository"""
    print("📁 Setting up QGML repository")
    print("=" * 40)
    
    # Check if already exists
    if os.path.exists('qgml_new'):
        print("✅ qgml_new directory already exists")
        return
    
    # Clone repository (update with your actual GitHub URL)
    repo_url = "https://github.com/your-username/qgml_new.git"
    print(f"Cloning repository: {repo_url}")
    
    result = subprocess.run(f"git clone {repo_url}", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Repository cloned successfully")
    else:
        print(f"❌ Failed to clone repository: {result.stderr}")
        print("Please update the repo_url in this script with your actual GitHub URL")

def install_qgml():
    """Install QGML package"""
    print("🔧 Installing QGML package")
    print("=" * 40)
    
    if not os.path.exists('qgml_new'):
        print("❌ qgml_new directory not found. Run clone_repo() first.")
        return
    
    # Install in development mode
    result = subprocess.run("cd qgml_new && pip install -e .", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ QGML package installed successfully")
    else:
        print(f"❌ Failed to install QGML: {result.stderr}")

def main():
    """Main setup function"""
    print("🚀 QGML Colab Setup Script")
    print("=" * 50)
    print("This script will set up your Colab environment for GPU testing")
    print()
    
    # Install packages
    install_packages()
    
    # Check GPU
    check_gpu()
    
    # Clone repository
    clone_repo()
    
    # Install QGML
    install_qgml()
    
    print("\n✅ Setup complete!")
    print("You can now run the GPU performance test script.")
    print("Next steps:")
    print("1. Update the GitHub URL in clone_repo() if needed")
    print("2. Run: python gpu_performance_test.py")

if __name__ == "__main__":
    main()
