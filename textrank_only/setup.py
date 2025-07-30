#!/usr/bin/env python3
"""
Setup script for TextRank Text Summarization (No GNN)
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages for TextRank...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def check_data_folder():
    """Check if the data folder exists and has the expected structure"""
    print("📁 Checking data folder structure...")
    
    expected_path = "../liputan6_data/canonical/train"
    if not os.path.exists(expected_path):
        print(f"❌ Expected data folder not found: {expected_path}")
        print("Please make sure your liputan6_data folder is in the parent directory.")
        return False
    
    # Check for JSON files
    json_files = [f for f in os.listdir(expected_path) if f.endswith('.json')]
    if not json_files:
        print(f"❌ No JSON files found in {expected_path}")
        return False
    
    print(f"✅ Found {len(json_files)} JSON files in {expected_path}")
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up TextRank Text Summarization (No GNN)")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check data folder
    if not check_data_folder():
        print("\n📋 Please ensure:")
        print("1. Your liputan6_data folder is in the parent directory")
        print("2. The folder structure is: ../liputan6_data/canonical/train/")
        print("3. The train folder contains JSON files")
        return
    
    print("\n✅ Setup completed successfully!")
    print("\n🎯 To run the TextRank summarization pipeline:")
    print("   python textrank.py")
    
    print("\n📊 The script will:")
    print("   - Load articles from ../liputan6_data/canonical/train/")
    print("   - Generate summaries using TextRank algorithm")
    print("   - Calculate ROUGE scores")
    print("   - Save results to textrank_summarization_results.csv")
    print("   - Create visualizations (textrank_rouge_scores.png)")
    
    print("\n🔧 Key differences from GNN version:")
    print("   - No PyTorch dependency")
    print("   - No neural network training")
    print("   - Pure graph-based algorithm")
    print("   - Faster execution")
    print("   - Configurable damping factor and similarity threshold")

if __name__ == "__main__":
    main() 