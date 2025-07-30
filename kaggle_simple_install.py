# Simple Kaggle Installation Script
# Copy and paste this into your Kaggle notebook cell

# ============================================================================
# CELL 1: Install All Dependencies (One-liner)
# ============================================================================
"""
!pip install numpy pandas networkx rouge-score torch scikit-learn matplotlib tqdm nltk
"""

# ============================================================================
# CELL 2: Download NLTK Data
# ============================================================================
"""
import nltk
nltk.download('punkt')
nltk.download('stopwords')
print("✅ Setup completed!")
"""

# ============================================================================
# CELL 3: Test Import
# ============================================================================
"""
import torch
import numpy as np
import pandas as pd
import networkx as nx
import rouge_score
import sklearn
import matplotlib.pyplot as plt
import tqdm
import nltk

print("✅ All packages imported successfully!")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
""" 