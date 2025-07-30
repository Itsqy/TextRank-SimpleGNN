# Kaggle Notebook Setup Cells
# Copy and paste each cell into your Kaggle notebook

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
"""
# Install all required packages
!pip install torch>=1.9.0
!pip install numpy>=1.21.0
!pip install pandas>=1.3.0
!pip install scikit-learn>=1.0.0
!pip install nltk>=3.6.0
!pip install rouge-score>=0.1.2
!pip install networkx>=2.6.0
!pip install matplotlib>=3.4.0
!pip install tqdm>=4.62.0

print("✅ All packages installed successfully!")
"""

# ============================================================================
# CELL 2: Download NLTK Data
# ============================================================================
"""
# Download NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
print("✅ NLTK data downloaded successfully!")
"""

# ============================================================================
# CELL 3: Verify Installation
# ============================================================================
"""
# Verify all packages are working
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
"""

# ============================================================================
# CELL 4: Download Dataset (if needed)
# ============================================================================
"""
# Download your dataset from Google Drive or other sources
# Example for Google Drive:
# !pip install gdown
# !gdown --id YOUR_GOOGLE_DRIVE_FILE_ID -O liputan6_data.zip
# !unzip -q liputan6_data.zip -d /kaggle/working/

print("✅ Dataset downloaded and extracted!")
"""

# ============================================================================
# CELL 5: Quick Test
# ============================================================================
"""
# Quick test to make sure everything works
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Test NLTK
text = "Ini adalah kalimat pertama. Ini adalah kalimat kedua."
sentences = sent_tokenize(text)
print(f"Tokenized sentences: {sentences}")

# Test stopwords
stop_words = set(stopwords.words('indonesian'))
print(f"Indonesian stopwords loaded: {len(stop_words)} words")

print("✅ Quick test completed successfully!")
""" 