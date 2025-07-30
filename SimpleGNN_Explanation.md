# SimpleGNN Architecture Explanation

## 🧠 What is SimpleGNN?

The `SimpleGNN` is a lightweight Graph Neural Network implementation used to enhance the TextRank algorithm for text summarization. It's designed to learn better sentence representations by incorporating graph structure information.

## 📐 Architecture Overview

```python
class SimpleGNN(nn.Module):
    """Simple Graph Neural Network for sentence ranking"""
    def __init__(self, input_dim, hidden_dim):
        super(SimpleGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First layer
        self.fc2 = nn.Linear(hidden_dim, 1)          # Output layer
        
    def forward(self, features, adj):
        x = torch.mm(adj, features)  # Graph convolution step
        x = self.fc1(x)              # First linear transformation
        x = torch.relu(x)            # Activation function
        x = self.fc2(x)              # Output transformation
        return x.squeeze()           # Remove extra dimensions
```

## 🔧 Architecture Components

### 1. **Input Layer**
- **Features**: TF-IDF vectors of sentences (shape: `[num_sentences, input_dim]`)
- **Adjacency Matrix**: Graph connectivity matrix (shape: `[num_sentences, num_sentences]`)
- **Input Dimension**: Varies based on vocabulary size (e.g., 114 in your case)

### 2. **Graph Convolution Layer**
```python
x = torch.mm(adj, features)
```
- **Purpose**: Aggregates information from neighboring sentences
- **Operation**: Matrix multiplication between adjacency matrix and features
- **Effect**: Each sentence gets updated with information from similar sentences

### 3. **Hidden Layer**
```python
self.fc1 = nn.Linear(input_dim, hidden_dim)  # hidden_dim = 16
x = self.fc1(x)
x = torch.relu(x)
```
- **Purpose**: Learns non-linear transformations
- **Size**: 16 hidden units (configurable)
- **Activation**: ReLU for non-linearity

### 4. **Output Layer**
```python
self.fc2 = nn.Linear(hidden_dim, 1)
x = self.fc2(x)
```
- **Purpose**: Produces final sentence importance scores
- **Output**: Single value per sentence (importance score)

## 🔄 Training Process

### 1. **Data Preparation**
```python
# Convert sentences to TF-IDF features
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(sentences).toarray()
features = torch.tensor(X, dtype=torch.float32)

# Create adjacency matrix from graph
adj = nx.to_numpy_array(graph, nodelist=range(N))
adj = torch.tensor(adj, dtype=torch.float32)
```

### 2. **Target Generation**
```python
# Use TextRank scores as training targets
initial_scores = textrank_scores(graph)
y = torch.tensor([initial_scores.get(i, 0) for i in range(N)], dtype=torch.float32)
```

### 3. **Training Loop**
```python
for epoch in range(epochs):  # epochs = 10
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss = loss_fn(output, y)  # MSE Loss
    loss.backward()
    optimizer.step()
```

## 🎯 How It Works

### Step 1: Graph Construction
```
Sentence 1 ←→ Sentence 2 ←→ Sentence 3
    ↓           ↓           ↓
[TF-IDF]    [TF-IDF]    [TF-IDF]
```

### Step 2: Graph Convolution
```
Adjacency Matrix × Features = Updated Features
[0.8, 0.3, 0.1]   [1,0,1,0]   [0.8,0.3,0.1]
[0.3, 0.9, 0.2] × [0,1,0,1] = [0.3,0.9,0.2]
[0.1, 0.2, 0.7]   [1,1,0,0]   [0.1,0.2,0.7]
```

### Step 3: Neural Network Processing
```
Updated Features → Linear(16) → ReLU → Linear(1) → Scores
[0.8,0.3,0.1] → [0.5,0.2,...] → [0.5,0.2,...] → [0.3]
```

## 📊 Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Input Dimension** | 114 | TF-IDF vocabulary size |
| **Hidden Dimension** | 16 | Number of hidden units |
| **Output Dimension** | 1 | Single importance score |
| **Learning Rate** | 0.01 | Adam optimizer learning rate |
| **Epochs** | 10 | Number of training iterations |
| **Loss Function** | MSE | Mean Squared Error |

## 🔍 Key Features

### 1. **Graph-Aware Learning**
- Incorporates sentence similarity relationships
- Learns from graph structure, not just individual features

### 2. **Supervised Training**
- Uses TextRank scores as ground truth
- Learns to improve upon baseline TextRank

### 3. **Lightweight Design**
- Only 2 linear layers
- Small hidden dimension (16)
- Fast training and inference

### 4. **Interpretable**
- Output directly corresponds to sentence importance
- Can be easily analyzed and debugged

## 🚀 Advantages Over Pure TextRank

1. **Learning Capability**: Can learn complex patterns beyond simple similarity
2. **Adaptive**: Adjusts to different types of text and domains
3. **Non-linear**: Captures non-linear relationships between sentences
4. **Optimized**: Trained to minimize prediction error

## 📈 Performance Impact

Based on your results:
- **ROUGE-1**: Slight decrease (-1.0%) - may need tuning
- **ROUGE-2**: Significant improvement (+13.4%) - captures better bigram patterns
- **ROUGE-L**: Good improvement (+8.8%) - better sentence ordering

## 🔧 Customization Options

### 1. **Architecture Modifications**
```python
# Deeper network
self.fc1 = nn.Linear(input_dim, 32)
self.fc2 = nn.Linear(32, 16)
self.fc3 = nn.Linear(16, 1)

# Different activation
x = torch.tanh(x)  # Instead of ReLU
```

### 2. **Training Parameters**
```python
epochs = 20        # More training
lr = 0.001         # Lower learning rate
hidden_dim = 32    # Larger hidden layer
```

### 3. **Loss Function**
```python
loss_fn = nn.HuberLoss()  # More robust to outliers
```

## 🎯 Summary

The SimpleGNN is a **lightweight, graph-aware neural network** that:
- Takes TF-IDF features and graph structure as input
- Learns to predict sentence importance scores
- Uses TextRank scores as training targets
- Provides a 7.1% average improvement over pure TextRank
- Is easily deployable and interpretable

This hybrid approach combines the interpretability of TextRank with the learning capability of neural networks! 🧠✨ 