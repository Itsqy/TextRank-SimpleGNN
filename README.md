# TextRank + GNN Text Summarization

This project implements a hybrid approach combining TextRank algorithm with Graph Neural Networks (GNN) for Indonesian text summarization using the Liputan6 dataset.

## 🎯 Overview

The system works by:
1. **Preprocessing**: Tokenizing Indonesian text and removing stopwords
2. **Graph Construction**: Building similarity graphs between sentences using TF-IDF and cosine similarity
3. **TextRank**: Computing initial sentence importance scores using PageRank algorithm
4. **GNN Training**: Training a simple Graph Neural Network to learn sentence representations
5. **Summarization**: Selecting top-ranked sentences to create the final summary
6. **Evaluation**: Computing ROUGE scores to measure summary quality

## 📁 Project Structure

```
.
├── gnn.py                 # Main implementation
├── setup.py              # Setup script
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── liputan6_data/       # Dataset folder
    └── canonical/
        └── train/       # Training articles (JSON files)
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Run the setup script
python setup.py
```

This will:
- Install all required packages
- Check if your data folder is properly structured
- Provide instructions for running the pipeline

### 2. Run the Pipeline

```bash
python gnn.py
```

## 📊 Output Files

The script generates several output files:

- `summarization_results.csv` - Detailed results with ROUGE scores
- `rouge_scores.png` - Visualization of average ROUGE scores
- `model.pth` - Trained PyTorch model
- `input_dim.txt` - Input dimension for model export
- `gnn_model.onnx` - ONNX format model for deployment

## 🔧 Key Components

### TextRank + GNN Architecture

```python
class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, features, adj):
        x = torch.mm(adj, features)  # Graph convolution
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x.squeeze()
```

### Data Format

Each JSON file in the dataset contains:
```json
{
    "id": 1,
    "url": "https://...",
    "clean_article": [["sentence", "1", "words"], ["sentence", "2", "words"]],
    "clean_summary": [["summary", "sentence", "1"], ["summary", "sentence", "2"]],
    "extractive_summary": [1, 8]
}
```

## 📈 Performance Metrics

The system evaluates summaries using ROUGE scores:
- **ROUGE-1**: Unigram overlap between generated and reference summaries
- **ROUGE-2**: Bigram overlap between generated and reference summaries  
- **ROUGE-L**: Longest common subsequence between summaries

## 🛠️ Dependencies

- **PyTorch**: Deep learning framework
- **NetworkX**: Graph algorithms
- **NLTK**: Natural language processing
- **scikit-learn**: Machine learning utilities
- **rouge-score**: ROUGE evaluation metrics
- **pandas**: Data manipulation
- **matplotlib**: Visualization

## 🔄 Model Export

The trained model is automatically exported to ONNX format for deployment:

```python
# Export to ONNX
torch.onnx.export(
    model,
    (dummy_features, dummy_adj),
    "gnn_model.onnx",
    input_names=['features', 'adj'],
    output_names=['output'],
    dynamic_axes={'features': {0: 'batch_size'}, 'adj': {0: 'batch_size'}}
)
```

## 🐛 Troubleshooting

### Common Issues

1. **NLTK Data Missing**: The script automatically downloads required NLTK data
2. **Memory Issues**: Reduce the number of articles processed by changing `data[:10]` in `main()`
3. **CUDA Issues**: The code runs on CPU by default. For GPU acceleration, modify tensor creation

### Error Messages

- `"Folder does not exist"`: Check that `liputan6_data/canonical/train/` exists
- `"No JSON files found"`: Ensure the train folder contains `.json` files
- `"Error in summarize"`: Usually indicates issues with text preprocessing

## 📝 Customization

### Adjusting Parameters

```python
# In gnn.py, modify these parameters:
epochs = 10          # GNN training epochs
lr = 0.01           # Learning rate
hidden_dim = 16     # Hidden layer dimension
num_sentences = 5   # Number of sentences in summary
```

### Using Different Datasets

To use a different dataset, modify the `load_data()` function and ensure your data follows the expected JSON format.

## 🤝 Contributing

Feel free to contribute improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- Liputan6 dataset for Indonesian text summarization
- TextRank algorithm by Mihalcea and Tarau
- PyTorch Geometric for GNN implementations 