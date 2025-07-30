# 🧠 TextRank + SimpleGNN Streamlit App

A beautiful and interactive web application for text summarization using TextRank enhanced with Graph Neural Networks.

## 🚀 Features

- **Interactive Web Interface**: User-friendly Streamlit interface
- **Dual Model Support**: TextRank + SimpleGNN and TextRank-only options
- **Real-time Evaluation**: ROUGE metrics with detailed Precision, Recall, and F1-Score
- **Multiple Input Methods**: Text input, file upload, and sample texts
- **Interactive Visualizations**: Plotly charts for metrics and sentence rankings
- **Configurable Parameters**: Adjustable training epochs, learning rate, and summary length
- **Progress Tracking**: Real-time progress bars during model training

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for downloading NLTK data)

## 🛠️ Installation

### Option 1: Quick Setup (Recommended)

```bash
# Clone or download the project files
# Navigate to the project directory
cd "gnn and text rank"

# Run the setup script
python streamlit_setup.py
```

### Option 2: Manual Setup

```bash
# Install Streamlit requirements
pip install -r streamlit_requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create Streamlit config directory
mkdir -p .streamlit
```

## 🎯 Usage

### Running the App

```bash
# Start the Streamlit app
streamlit run streamlit_app.py
```

The app will be available at: **http://localhost:8501**

### Using the App

1. **Choose Model**: Select between "TextRank + SimpleGNN" or "TextRank Only"
2. **Configure Parameters**: Adjust summary length, training epochs, and learning rate
3. **Input Text**: Choose from three input methods:
   - **Text Input**: Paste your text directly
   - **File Upload**: Upload a .txt or .md file
   - **Sample Text**: Use provided sample articles
4. **Generate Summary**: Click the "Generate Summary" button
5. **Evaluate Results**: Optionally provide a reference summary for ROUGE evaluation
6. **View Metrics**: See detailed Precision, Recall, and F1-Score breakdown

## 📊 Features Explained

### Model Selection

- **TextRank + SimpleGNN**: Enhanced version with Graph Neural Network learning
- **TextRank Only**: Traditional TextRank algorithm

### Configuration Options

- **Number of Sentences**: 1-10 sentences in summary
- **Training Epochs**: 5-20 epochs for GNN training
- **Learning Rate**: 0.001, 0.01, or 0.1 for model optimization

### Evaluation Metrics

- **ROUGE-1**: Unigram overlap between generated and reference summaries
- **ROUGE-2**: Bigram overlap between generated and reference summaries  
- **ROUGE-L**: Longest common subsequence between summaries
- **Precision**: How many generated words appear in the reference
- **Recall**: How many reference words appear in the generated summary
- **F1-Score**: Harmonic mean of precision and recall

## 🎨 Interface Overview

### Sidebar Configuration
- Model selection dropdown
- Parameter sliders and selectors
- Help tooltips for each option

### Main Content Area
- **Left Column**: Text input with multiple methods
- **Right Column**: Summary generation and analysis
- **Bottom Section**: Evaluation metrics and visualizations

### Interactive Elements
- Progress bars during processing
- Real-time metric updates
- Interactive Plotly charts
- File upload functionality

## 📈 Performance Metrics

Based on the Liputan6 dataset evaluation:

| Metric | TextRank + SimpleGNN | TextRank Only | Improvement |
|--------|---------------------|---------------|-------------|
| ROUGE-1 F1 | 0.277 ± 0.141 | 0.281 ± 0.138 | -1.0% |
| ROUGE-2 F1 | 0.171 ± 0.148 | 0.151 ± 0.132 | +13.4% |
| ROUGE-L F1 | 0.207 ± 0.108 | 0.190 ± 0.105 | +8.8% |

## 🔧 Technical Details

### SimpleGNN Architecture
- **Input Layer**: TF-IDF vectors (variable dimension)
- **Graph Convolution**: Adjacency matrix multiplication
- **Hidden Layer**: 16 units with ReLU activation
- **Output Layer**: Single importance score per sentence
- **Training**: Adam optimizer with MSE loss

### Processing Pipeline
1. **Text Preprocessing**: Sentence tokenization and cleaning
2. **Graph Construction**: TF-IDF similarity matrix
3. **Model Training**: GNN training on sentence graph
4. **Sentence Ranking**: Sort by learned importance scores
5. **Summary Generation**: Select top-ranked sentences
6. **Evaluation**: Calculate ROUGE metrics if reference provided

## 🌐 Deployment Options

### Local Deployment
```bash
streamlit run streamlit_app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy automatically

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📁 File Structure

```
gnn and text rank/
├── streamlit_app.py              # Main Streamlit application
├── streamlit_requirements.txt    # Python dependencies
├── streamlit_setup.py           # Setup script
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── gnn.py                       # Original GNN implementation
├── textrank_only/               # TextRank-only implementation
├── liputan6_data/               # Dataset directory
└── STREAMLIT_README.md          # This file
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill existing process
   lsof -ti:8501 | xargs kill -9
   # Or use different port
   streamlit run streamlit_app.py --server.port=8502
   ```

2. **NLTK Data Missing**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

3. **Memory Issues**
   - Reduce training epochs
   - Use shorter texts
   - Increase system memory

4. **CUDA/GPU Issues**
   - The app uses CPU by default
   - For GPU support, ensure PyTorch CUDA is installed

### Performance Tips

- **Faster Processing**: Use fewer training epochs (5-10)
- **Better Quality**: Use more training epochs (15-20)
- **Memory Efficient**: Process shorter texts
- **Real-time**: Use TextRank-only for instant results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **TextRank Algorithm**: Mihalcea and Tarau (2004)
- **Graph Neural Networks**: PyTorch Geometric
- **ROUGE Metrics**: rouge-score library
- **Web Framework**: Streamlit
- **Visualizations**: Plotly

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the technical documentation

---

**🧠 TextRank + SimpleGNN Streamlit App** - Combining the interpretability of TextRank with the learning capability of Graph Neural Networks! ✨ 