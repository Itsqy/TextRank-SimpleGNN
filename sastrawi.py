import nltk
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer
from tqdm import tqdm
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# === SASTRAWI for Stemming Bahasa Indonesia ===
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
stemmer = StemmerFactory().create_stemmer()

# Download NLTK data if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text by tokenizing sentences, stemming, and removing stopwords."""
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    preprocessed_sentences = []
    for sent in sentences:
        words = word_tokenize(sent.lower())
        words = [stemmer.stem(w) for w in words if w not in stop_words and w not in string.punctuation]
        preprocessed_sentences.append(' '.join(words))
    return sentences, preprocessed_sentences

def build_graph(sentences, sim_threshold=0.15):
    """Build similarity graph from sentences using TF-IDF and cosine similarity with threshold."""
    if len(sentences) < 2:
        graph = nx.Graph()
        for i in range(len(sentences)):
            graph.add_node(i)
        return graph
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    graph = nx.Graph()
    N = len(sentences)
    for i in range(N):
        graph.add_node(i)
    for i in range(N):
        for j in range(i + 1, N):
            if similarity_matrix[i][j] > sim_threshold:
                graph.add_edge(i, j, weight=similarity_matrix[i][j])
    return graph

def textrank_scores(graph):
    """Calculate TextRank scores for nodes in the graph."""
    if len(graph.nodes()) == 0:
        return {}
    return nx.pagerank(graph, weight='weight')

class SimpleGNN(nn.Module):
    """Simple Graph Neural Network for sentence ranking."""
    def __init__(self, input_dim, hidden_dim):
        super(SimpleGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, features, adj):
        x = torch.mm(adj, features)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()

def train_gnn(graph, sentences, epochs=50, lr=0.0001):
    """Train GNN model on the sentence graph, with extra features."""
    if len(sentences) == 0:
        return np.array([]), 0
    
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(sentences).toarray()
    N = len(sentences)
    # Fitur posisi & panjang kalimat
    pos_feature = np.arange(N) / N
    length_feature = np.array([len(s.split()) for s in sentences]) / (np.max([len(s.split()) for s in sentences]) + 1)
    X = np.hstack([X, pos_feature.reshape(-1,1), length_feature.reshape(-1,1)])
    features = torch.tensor(X, dtype=torch.float32)
    adj = nx.to_numpy_array(graph, nodelist=range(N))
    adj = torch.tensor(adj, dtype=torch.float32)
    
    model = SimpleGNN(features.shape[1], 64)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    initial_scores = textrank_scores(graph)
    y = torch.tensor([initial_scores.get(i, 0) for i in range(N)], dtype=torch.float32)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        final_scores = model(features, adj).numpy()
    torch.save(model.state_dict(), 'model.pth')
    with open('input_dim.txt', 'w') as f:
        f.write(str(features.shape[1]))
    return final_scores, features.shape[1]

def is_redundant(selected_sentences, candidate, threshold=0.7):
    """Check if candidate sentence is redundant compared to selected ones (Jaccard)."""
    for s in selected_sentences:
        set1 = set(candidate.split())
        set2 = set(s.split())
        if len(set1 | set2) == 0:
            continue
        sim = len(set1 & set2) / len(set1 | set2)
        if sim > threshold:
            return True
    return False

def summarize(text, num_sentences=5):
    """Generate summary using TextRank + GNN approach, with redundancy removal."""
    original_sentences, preprocessed_sentences = preprocess_text(text)
    if len(preprocessed_sentences) < 2:
        return ' '.join(original_sentences), 1
    if len(preprocessed_sentences) < num_sentences:
        return ' '.join(original_sentences), len(preprocessed_sentences)
    try:
        graph = build_graph(preprocessed_sentences, sim_threshold=0.15)
        scores, input_dim = train_gnn(graph, preprocessed_sentences)
        if len(scores) == 0:
            return ' '.join(original_sentences), len(preprocessed_sentences)
        sorted_idx = np.argsort(scores)[::-1]
        selected = []
        for idx in sorted_idx:
            if len(selected) >= num_sentences:
                break
            candidate = original_sentences[idx]
            if not is_redundant(selected, candidate, threshold=0.7):
                selected.append(candidate)
        while len(selected) < num_sentences and len(selected) < len(original_sentences):
            for idx in sorted_idx:
                if original_sentences[idx] not in selected:
                    selected.append(original_sentences[idx])
                    if len(selected) >= num_sentences:
                        break
        return ' '.join(selected), input_dim
    except Exception as e:
        print(f"Error in summarize: {e}")
        return ' '.join(original_sentences), len(preprocessed_sentences)

def load_data(folder_path):
    """Load data from JSON files in the specified folder."""
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

def export_to_onnx(hidden_dim=32):
    """Export trained model to ONNX format."""
    try:
        with open('input_dim.txt', 'r') as f:
            input_dim = int(f.read())
        model = SimpleGNN(input_dim, hidden_dim)
        model.load_state_dict(torch.load('model.pth'))
        model.eval()
        N = 3
        dummy_features = torch.randn(N, input_dim)
        dummy_adj = torch.randn(N, N)
        torch.onnx.export(
            model,
            (dummy_features, dummy_adj),
            "gnn_model.onnx",
            input_names=['features', 'adj'],
            output_names=['output'],
            dynamic_axes={
                'features': {0: 'batch_size'}, 
                'adj': {0: 'batch_size', 1: 'batch_size'}
            }
        )
        print("✅ Model exported to gnn_model.onnx")
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")

def main():
    """Main function to run the summarization pipeline."""
    train_folder = 'liputan6_data/canonical/train'
    print("🚀 Starting TextRank + GNN Summarization Pipeline")
    print(f"📁 Loading data from: {train_folder}")
    data = load_data(train_folder)
    if not data:
        print("❌ No data loaded. Please check the folder path.")
        return None
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_list, rouge2_list, rougeL_list = [], [], []
    results = []
    input_dim = None
    print(f"📊 Processing {min(len(data), 10)} articles...")
    for item in tqdm(data[:10], desc="Processing articles"):
        try:
            article = ' '.join([' '.join(sent) for sent in item['clean_article']])
            gold = ' '.join([' '.join(sent) for sent in item['clean_summary']])
            pred, input_dim_temp = summarize(article, num_sentences=5)
            if input_dim_temp is not None and input_dim_temp > 0:
                input_dim = input_dim_temp
            score = scorer.score(gold, pred)
            rouge1_list.append(score['rouge1'].fmeasure)
            rouge2_list.append(score['rouge2'].fmeasure)
            rougeL_list.append(score['rougeL'].fmeasure)
            results.append({
                'article': article,
                'generated': pred,
                'Gold_summary': gold,
                'rouge1': score['rouge1'].fmeasure,
                'rouge2': score['rouge2'].fmeasure,
                'rougeL': score['rougeL'].fmeasure
            })
        except Exception as e:
            print(f"Error processing article: {e}")
    if results:
        results_df = pd.DataFrame(results)
        results_path = 'summarization_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"✅ Results saved to {results_path}")
        print(f"\n📈 Average ROUGE Scores:")
        print(f"ROUGE-1: {np.mean(rouge1_list):.4f}")
        print(f"ROUGE-2: {np.mean(rouge2_list):.4f}")
        print(f"ROUGE-L: {np.mean(rougeL_list):.4f}")
        plt.figure(figsize=(10, 6))
        plt.bar(['ROUGE-1', 'ROUGE-2', 'ROUGE-L'], [
            np.mean(rouge1_list),
            np.mean(rouge2_list),
            np.mean(rougeL_list)
        ], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title("Average ROUGE Scores for TextRank + GNN Summarization + sastrawi")
        plt.ylim(0, 1)
        plt.ylabel("F1 Score")
        plt.grid(axis='y', alpha=0.3)
        for i, v in enumerate([np.mean(rouge1_list), np.mean(rouge2_list), np.mean(rougeL_list)]):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig('rouge_scores.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Chart saved as rouge_scores.png")
        print(f"🔧 Input dimension for ONNX export: {input_dim}")
        return input_dim
    else:
        print("❌ No results generated")
        return None

if __name__ == "__main__":
    print("🎯 TextRank + GNN Text Summarization+ sasatrawi")
    print("=" * 50)
    input_dim = main()
    if input_dim:
        print("\n🔄 Exporting model to ONNX...")
        export_to_onnx()
        print("✅ Pipeline completed successfully!")
    else:
        print("❌ Pipeline failed. Please check the data and try again.")
