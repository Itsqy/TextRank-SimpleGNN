# Kaggle Setup Script for GNN Text Summarization
# Run this in a Kaggle notebook cell

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")

def setup_kaggle_environment():
    """Setup the complete environment for GNN text summarization"""
    print("🚀 Setting up Kaggle environment for GNN Text Summarization")
    print("=" * 60)
    
    # List of required packages
    packages = [
        "torch>=1.9.0",
        "numpy>=1.21.0", 
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.6.0",
        "rouge-score>=0.1.2",
        "networkx>=2.6.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0"
    ]
    
    print("📦 Installing required packages...")
    for package in packages:
        install_package(package)
    
    print("\n📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
    
    print("\n🔍 Verifying installations...")
    try:
        import torch
        import numpy as np
        import pandas as pd
        import sklearn
        import nltk
        import rouge_score
        import networkx as nx
        import matplotlib.pyplot as plt
        import tqdm
        
        print("✅ All packages imported successfully!")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   NumPy version: {np.__version__}")
        print(f"   Pandas version: {pd.__version__}")
        print(f"   Scikit-learn version: {sklearn.__version__}")
        print(f"   NetworkX version: {nx.__version__}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    
    print("\n🎉 Setup completed! You can now run your GNN text summarization code.")
    print("=" * 60)

# Run the setup
if __name__ == "__main__":
    setup_kaggle_environment() 