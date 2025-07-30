# TextRank Text Summarization (No GNN)

This project implements a pure TextRank algorithm for Indonesian text summarization using the Liputan6 dataset, without any Graph Neural Network components.

## 🎯 Overview

The system works by:
1. **Preprocessing**: Tokenizing Indonesian text and removing stopwords
2. **Similarity Matrix**: Building similarity matrix between sentences using TF-IDF and cosine similarity
3. **Graph Construction**: Creating a graph where nodes are sentences and edges represent similarity
4. **TextRank**: Computing sentence importance scores using PageRank algorithm
5. **Summarization**: Selecting top-ranked sentences to create the final summary
6. **Evaluation**: Computing ROUGE scores to measure summary quality

## 📁 Project Structure

```
textrank_only/
├── textrank.py              # Main TextRank implementation
├── setup.py                 # Setup script
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── ../liputan6_data/       # Dataset folder (parent directory)
    └── canonical/
        └── train/          # Training articles (JSON files)
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to textrank_only folder
cd textrank_only

# Run the setup script
python setup.py
```

This will:
- Install all required packages (no PyTorch needed!)
- Check if your data folder is properly structured
- Provide instructions for running the pipeline

### 2. Run the Pipeline

```bash
python textrank.py
```

## 📊 Output Files

The script generates several output files:

- `textrank_summarization_results.csv` - Detailed results with ROUGE scores
- `textrank_rouge_scores.png` - Visualization of ROUGE scores with bar charts and box plots

## 🔧 Key Components

### TextRank Algorithm

```python
def textrank_scores(graph, damping=0.85, max_iter=100, tol=1e-6):
    """Calculate TextRank scores using PageRank algorithm"""
    scores = nx.pagerank(graph, alpha=damping, max_iter=max_iter, 
                        tol=tol, weight='weight')
    return scores
```

### Graph Construction

```python
def build_graph(similarity_matrix, threshold=0.0):
    """Build graph from similarity matrix"""
    graph = nx.Graph()
    # Add nodes and edges based on similarity
    for i in range(N):
        for j in range(i + 1, N):
            if similarity_matrix[i][j] > threshold:
                graph.add_edge(i, j, weight=similarity_matrix[i][j])
    return graph
```

## 📈 Performance Metrics

The system evaluates summaries using ROUGE scores:
- **ROUGE-1**: Unigram overlap between generated and reference summaries
- **ROUGE-2**: Bigram overlap between generated and reference summaries  
- **ROUGE-L**: Longest common subsequence between summaries

## 🛠️ Dependencies

- **NetworkX**: Graph algorithms and PageRank implementation
- **NLTK**: Natural language processing
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **rouge-score**: ROUGE evaluation metrics
- **pandas**: Data manipulation
- **matplotlib**: Visualization

## ⚙️ Configuration Parameters

You can adjust these parameters in the `main()` function:

```python
num_articles = 10           # Number of articles to process
num_sentences = 5          # Number of sentences in summary
damping = 0.85            # PageRank damping factor (0.85 is standard)
similarity_threshold = 0.0 # Minimum similarity for graph edges
```

## 🔄 Comparison with GNN Version

| Feature | TextRank Only | TextRank + GNN |
|---------|---------------|----------------|
| **Dependencies** | Lightweight (no PyTorch) | Heavy (PyTorch, CUDA) |
| **Training Time** | Instant | Requires training epochs |
| **Memory Usage** | Low | High (neural network) |
| **Model Export** | Not needed | ONNX/CoreML export |
| **Customization** | Limited | Highly configurable |
| **Performance** | Baseline | Potentially better |

## 🐛 Troubleshooting

### Common Issues

1. **NLTK Data Missing**: The script automatically downloads required NLTK data
2. **Memory Issues**: TextRank uses much less memory than GNN version
3. **Path Issues**: Make sure `liputan6_data` is in the parent directory

### Error Messages

- `"Folder does not exist"`: Check that `../liputan6_data/canonical/train/` exists
- `"No JSON files found"`: Ensure the train folder contains `.json` files
- `"Error in summarize_textrank"`: Usually indicates issues with text preprocessing

## 📝 Customization

### Adjusting TextRank Parameters

```python
# In textrank.py, modify these parameters:
damping = 0.85              # PageRank damping factor (0.85 is standard)
similarity_threshold = 0.0  # Minimum similarity for graph edges
max_iter = 100             # Maximum PageRank iterations
tol = 1e-6                 # PageRank convergence tolerance
```

### Using Different Similarity Metrics

You can modify the `build_similarity_matrix()` function to use different similarity measures:

```python
# Example: Use different vectorizer
from sklearn.feature_extraction.text import CountVectorizer

def build_similarity_matrix(sentences):
    vectorizer = CountVectorizer()  # Instead of TfidfVectorizer
    matrix = vectorizer.fit_transform(sentences)
    return cosine_similarity(matrix)
```

## 🎯 Advantages of TextRank-Only Approach

1. **Simplicity**: Pure graph-based algorithm, easy to understand
2. **Speed**: No training required, instant results
3. **Lightweight**: Minimal dependencies, no GPU needed
4. **Interpretable**: Clear relationship between sentence similarity and ranking
5. **Robust**: Well-established algorithm with proven effectiveness

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
- NetworkX library for graph algorithms 