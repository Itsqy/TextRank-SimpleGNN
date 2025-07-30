#!/usr/bin/env python3
"""
Setup script for TextRank + GNN Text Summarization
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
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
    
    expected_path = "liputan6_data/canonical/train"
    if not os.path.exists(expected_path):
        print(f"❌ Expected data folder not found: {expected_path}")
        print("Please make sure your liputan6_data folder is in the correct location.")
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
    print("🚀 Setting up TextRank + GNN Text Summarization")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check data folder
    if not check_data_folder():
        print("\n📋 Please ensure:")
        print("1. Your liputan6_data folder is in the project root")
        print("2. The folder structure is: liputan6_data/canonical/train/")
        print("3. The train folder contains JSON files")
        return
    
    print("\n✅ Setup completed successfully!")
    print("\n🎯 To run the summarization pipeline:")
    print("   python gnn.py")
    
    print("\n📊 The script will:")
    print("   - Load articles from liputan6_data/canonical/train/")
    print("   - Generate summaries using TextRank + GNN")
    print("   - Calculate ROUGE scores")
    print("   - Save results to summarization_results.csv")
    print("   - Create a visualization (rouge_scores.png)")
    print("   - Export the model to ONNX format")

if __name__ == "__main__":
    main() 