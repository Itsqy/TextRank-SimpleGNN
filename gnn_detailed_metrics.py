import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text by tokenizing and cleaning"""
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Clean and preprocess each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        # Convert to lowercase and remove special characters
        cleaned = re.sub(r'[^a-zA-Z\s]', '', sentence.lower())
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        if cleaned.strip():
            preprocessed_sentences.append(cleaned)
    
    return sentences, preprocessed_sentences

def build_graph(sentences):
    """Build similarity graph from sentences"""
    if len(sentences) < 2:
        return nx.Graph()
    
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Create graph
    graph = nx.Graph()
    N = len(sentences)
    
    # Add nodes
    for i in range(N):
        graph.add_node(i)
    
    # Add edges based on similarity
    for i in range(N):
        for j in range(i + 1, N):
            if similarity_matrix[i][j] > 0:
                graph.add_edge(i, j, weight=similarity_matrix[i][j])
    
    return graph

def textrank_scores(graph):
    """Calculate TextRank scores for nodes in the graph"""
    if len(graph.nodes()) == 0:
        return {}
    return nx.pagerank(graph, weight='weight')

class SimpleGNN(nn.Module):
    """Simple Graph Neural Network for sentence ranking"""
    def __init__(self, input_dim, hidden_dim):
        super(SimpleGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, features, adj):
        x = torch.mm(adj, features)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x.squeeze()

def train_gnn(graph, sentences, epochs=10, lr=0.01):
    """Train GNN model on the sentence graph"""
    if len(sentences) == 0:
        return np.array([]), 0
    
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(sentences).toarray()
    features = torch.tensor(X, dtype=torch.float32)
    N = len(sentences)
    
    # Create adjacency matrix
    adj = nx.to_numpy_array(graph, nodelist=range(N))
    adj = torch.tensor(adj, dtype=torch.float32)
    
    # Initialize model
    model = SimpleGNN(features.shape[1], 16)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Get initial TextRank scores as targets
    initial_scores = textrank_scores(graph)
    y = torch.tensor([initial_scores.get(i, 0) for i in range(N)], dtype=torch.float32)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
    
    # Get final scores
    model.eval()
    with torch.no_grad():
        final_scores = model(features, adj).numpy()
    
    # Save model and input dimension
    torch.save(model.state_dict(), 'model.pth')
    with open('input_dim.txt', 'w') as f:
        f.write(str(features.shape[1]))
    
    return final_scores, features.shape[1]

def summarize(text, num_sentences=5):
    """Generate summary using TextRank + GNN approach"""
    original_sentences, preprocessed_sentences = preprocess_text(text)
    
    if len(preprocessed_sentences) < 2:
        return ' '.join(original_sentences), 1
    
    if len(preprocessed_sentences) < num_sentences:
        return ' '.join(original_sentences), len(preprocessed_sentences)
    
    try:
        graph = build_graph(preprocessed_sentences)
        scores, input_dim = train_gnn(graph, preprocessed_sentences)
        
        if len(scores) == 0:
            return ' '.join(original_sentences), len(preprocessed_sentences)
        
        # Rank sentences by scores
        ranked_sentences = [s for _, s in sorted(zip(scores, original_sentences), reverse=True)]
        return ' '.join(ranked_sentences[:num_sentences]), input_dim
        
    except Exception as e:
        print(f"Error in summarize: {e}")
        return ' '.join(original_sentences), len(preprocessed_sentences)

def load_data(folder_path):
    """Load data from JSON files in the specified folder"""
    all_data = []
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist!")
        return all_data
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    if not files:
        print(f"No JSON files found in {folder_path}")
        return all_data
    
    for filename in tqdm(files, desc=f"Loading {folder_path}"):
        path = os.path.join(folder_path, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                article = json.load(f)
                if 'clean_article' in article and 'clean_summary' in article:
                    all_data.append(article)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    print(f"Loaded {len(all_data)} articles from {folder_path}")
    return all_data

def calculate_detailed_metrics(scorer, gold, pred):
    """Calculate detailed Precision, Recall, and F1-Score for all ROUGE metrics"""
    scores = scorer.score(gold, pred)
    
    detailed_metrics = {}
    for metric_name, score in scores.items():
        detailed_metrics[metric_name] = {
            'precision': score.precision,
            'recall': score.recall,
            'fmeasure': score.fmeasure
        }
    
    return detailed_metrics

def print_detailed_results(results_df):
    """Print detailed Precision, Recall, and F1-Score results"""
    print("\n" + "="*80)
    print("📊 DETAILED ROUGE METRICS (Precision, Recall, F1-Score)")
    print("="*80)
    
    # Calculate averages for each metric
    metrics = ['rouge1', 'rouge2', 'rougeL']
    
    for metric in metrics:
        print(f"\n🔍 {metric.upper()} METRICS:")
        print("-" * 50)
        
        # Extract precision, recall, f1 columns
        precision_col = f'{metric}_precision'
        recall_col = f'{metric}_recall'
        f1_col = f'{metric}_fmeasure'
        
        if precision_col in results_df.columns:
            avg_precision = results_df[precision_col].mean()
            avg_recall = results_df[recall_col].mean()
            avg_f1 = results_df[f1_col].mean()
            
            print(f"📈 Average Precision: {avg_precision:.4f}")
            print(f"📉 Average Recall:    {avg_recall:.4f}")
            print(f"🎯 Average F1-Score:  {avg_f1:.4f}")
            
            # Show individual results
            print(f"\n📋 Individual Results:")
            for idx, row in results_df.iterrows():
                print(f"  Article {idx+1}: P={row[precision_col]:.3f}, R={row[recall_col]:.3f}, F1={row[f1_col]:.3f}")
        else:
            print(f"⚠️  Detailed metrics not available for {metric}")

def create_detailed_visualization(results_df):
    """Create detailed visualization with Precision, Recall, and F1-Score"""
    metrics = ['rouge1', 'rouge2', 'rougeL']
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        precision_col = f'{metric}_precision'
        recall_col = f'{metric}_recall'
        f1_col = f'{metric}_fmeasure'
        
        if precision_col in results_df.columns:
            # Calculate averages
            avg_precision = results_df[precision_col].mean()
            avg_recall = results_df[recall_col].mean()
            avg_f1 = results_df[f1_col].mean()
            
            # Create bar chart
            categories = ['Precision', 'Recall', 'F1-Score']
            values = [avg_precision, avg_recall, avg_f1]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            bars = axes[i].bar(categories, values, color=colors, alpha=0.8)
            axes[i].set_title(f'{metric.upper()} Metrics', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Score', fontsize=12)
            axes[i].set_ylim(0, 1)
            axes[i].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Add individual data points
            for j, (precision, recall, f1) in enumerate(zip(
                results_df[precision_col], 
                results_df[recall_col], 
                results_df[f1_col]
            )):
                axes[i].scatter(['Precision', 'Recall', 'F1-Score'], 
                              [precision, recall, f1], 
                              color='gray', alpha=0.3, s=30)
    
    plt.tight_layout()
    plt.savefig('detailed_rouge_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Detailed metrics visualization saved as 'detailed_rouge_metrics.png'")

def main():
    """Main function to run the summarization pipeline with detailed metrics"""
    # Use local paths
    train_folder = 'liputan6_data/canonical/train'
    
    print("🚀 Starting TextRank + GNN Summarization Pipeline (Detailed Metrics)")
    print(f"📁 Loading data from: {train_folder}")
    
    # Load data
    data = load_data(train_folder)
    
    if not data:
        print("❌ No data loaded. Please check the folder path.")
        return None
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []
    input_dim = None
    
    print(f"📊 Processing {min(len(data), 10)} articles...")
    
    # Process articles
    for item in tqdm(data[:10], desc="Processing articles"):
        try:
            # Extract article and summary text
            article = ' '.join([' '.join(sent) for sent in item['clean_article']])
            gold = ' '.join([' '.join(sent) for sent in item['clean_summary']])
            
            # Generate summary
            pred, input_dim_temp = summarize(article, num_sentences=5)
            
            if input_dim_temp is not None and input_dim_temp > 0:
                input_dim = input_dim_temp
            
            # Calculate detailed ROUGE scores
            detailed_scores = calculate_detailed_metrics(scorer, gold, pred)
            
            # Prepare result row
            result_row = {
                'article': article[:200] + "..." if len(article) > 200 else article,
                'generated': pred,
                'gold_summary': gold,
            }
            
            # Add detailed metrics to result row
            for metric_name, scores in detailed_scores.items():
                result_row[f'{metric_name}_precision'] = scores['precision']
                result_row[f'{metric_name}_recall'] = scores['recall']
                result_row[f'{metric_name}_fmeasure'] = scores['fmeasure']
            
            results.append(result_row)
            
        except Exception as e:
            print(f"Error processing article: {e}")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_path = 'detailed_summarization_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"✅ Detailed results saved to {results_path}")
        
        # Print detailed metrics
        print_detailed_results(results_df)
        
        # Create detailed visualization
        create_detailed_visualization(results_df)
        
        # Print summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        print("-" * 50)
        metrics = ['rouge1', 'rouge2', 'rougeL']
        for metric in metrics:
            f1_col = f'{metric}_fmeasure'
            if f1_col in results_df.columns:
                avg_f1 = results_df[f1_col].mean()
                std_f1 = results_df[f1_col].std()
                print(f"{metric.upper()} F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")
        
        print(f"\n🔧 Input dimension for ONNX export: {input_dim}")
        
        return input_dim
    else:
        print("❌ No results generated")
        return None

if __name__ == "__main__":
    print("🎯 TextRank + GNN Text Summarization (Detailed Metrics)")
    print("=" * 60)
    
    input_dim = main()
    
    if input_dim:
        print("\n✅ Pipeline completed successfully!")
        print("📁 Generated files:")
        print("  - detailed_summarization_results.csv (Detailed metrics)")
        print("  - detailed_rouge_metrics.png (Visualization)")
        print("  - model.pth (Trained model)")
        print("  - input_dim.txt (Model configuration)")
    else:
        print("❌ Pipeline failed. Please check the data and try again.") 