import nltk
import networkx as nx
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer  # BoW instead of TF-IDF
from rouge_score import rouge_scorer
from tqdm import tqdm
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text by tokenizing sentences and removing stopwords"""
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    preprocessed_sentences = []
    for sent in sentences:
        words = word_tokenize(sent.lower())
        words = [w for w in words if w not in stop_words and w not in string.punctuation]
        preprocessed_sentences.append(' '.join(words))
    return sentences, preprocessed_sentences

def build_similarity_matrix_bow(sentences):
    """Build similarity matrix from sentences using Bag-of-Words and cosine similarity"""
    if len(sentences) < 2:
        return np.eye(len(sentences))
    
    # Use CountVectorizer (BoW) instead of TfidfVectorizer
    bow_vectorizer = CountVectorizer()
    bow_matrix = bow_vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(bow_matrix)
    return similarity_matrix

def build_similarity_matrix_tfidf(sentences):
    """Build similarity matrix from sentences using TF-IDF and cosine similarity (for comparison)"""
    if len(sentences) < 2:
        return np.eye(len(sentences))
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

def build_graph(similarity_matrix, threshold=0.0):
    """Build graph from similarity matrix"""
    N = len(similarity_matrix)
    graph = nx.Graph()
    
    # Add all nodes
    for i in range(N):
        graph.add_node(i)
    
    # Add edges based on similarity threshold
    for i in range(N):
        for j in range(i + 1, N):
            if similarity_matrix[i][j] > threshold:
                graph.add_edge(i, j, weight=similarity_matrix[i][j])
    
    return graph

def textrank_scores(graph, damping=0.85, max_iter=100, tol=1e-6):
    """Calculate TextRank scores using PageRank algorithm"""
    if len(graph.nodes()) == 0:
        return {}
    
    # Use PageRank with damping factor
    scores = nx.pagerank(graph, alpha=damping, max_iter=max_iter, tol=tol, weight='weight')
    return scores

def summarize_textrank_bow(text, num_sentences=5, damping=0.85, similarity_threshold=0.0):
    """Generate summary using TextRank algorithm with Bag-of-Words"""
    original_sentences, preprocessed_sentences = preprocess_text(text)
    
    if len(preprocessed_sentences) < 2:
        return ' '.join(original_sentences)
    
    if len(preprocessed_sentences) < num_sentences:
        return ' '.join(original_sentences)
    
    try:
        # Build similarity matrix using BoW
        similarity_matrix = build_similarity_matrix_bow(preprocessed_sentences)
        
        # Build graph
        graph = build_graph(similarity_matrix, threshold=similarity_threshold)
        
        # Calculate TextRank scores
        scores = textrank_scores(graph, damping=damping)
        
        if not scores:
            return ' '.join(original_sentences)
        
        # Rank sentences by scores
        ranked_sentences = [s for _, s in sorted(zip(scores.values(), original_sentences), reverse=True)]
        return ' '.join(ranked_sentences[:num_sentences])
        
    except Exception as e:
        print(f"Error in summarize_textrank_bow: {e}")
        return ' '.join(original_sentences)

def summarize_textrank_tfidf(text, num_sentences=5, damping=0.85, similarity_threshold=0.0):
    """Generate summary using TextRank algorithm with TF-IDF (for comparison)"""
    original_sentences, preprocessed_sentences = preprocess_text(text)
    
    if len(preprocessed_sentences) < 2:
        return ' '.join(original_sentences)
    
    if len(preprocessed_sentences) < num_sentences:
        return ' '.join(original_sentences)
    
    try:
        # Build similarity matrix using TF-IDF
        similarity_matrix = build_similarity_matrix_tfidf(preprocessed_sentences)
        
        # Build graph
        graph = build_graph(similarity_matrix, threshold=similarity_threshold)
        
        # Calculate TextRank scores
        scores = textrank_scores(graph, damping=damping)
        
        if not scores:
            return ' '.join(original_sentences)
        
        # Rank sentences by scores
        ranked_sentences = [s for _, s in sorted(zip(scores.values(), original_sentences), reverse=True)]
        return ' '.join(ranked_sentences[:num_sentences])
        
    except Exception as e:
        print(f"Error in summarize_textrank_tfidf: {e}")
        return ' '.join(original_sentences)

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

def compare_bow_vs_tfidf(data, num_articles=10, num_sentences=5):
    """Compare TextRank performance with BoW vs TF-IDF"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Results for BoW
    bow_rouge1, bow_rouge2, bow_rougeL = [], [], []
    # Results for TF-IDF
    tfidf_rouge1, tfidf_rouge2, tfidf_rougeL = [], [], []
    
    print(f"📊 Comparing BoW vs TF-IDF on {min(len(data), num_articles)} articles...")
    
    for item in tqdm(data[:num_articles], desc="Processing articles"):
        try:
            # Extract article and summary text
            article = ' '.join([' '.join(sent) for sent in item['clean_article']])
            gold = ' '.join([' '.join(sent) for sent in item['clean_summary']])
            
            # Generate summary using BoW
            pred_bow = summarize_textrank_bow(article, num_sentences=num_sentences)
            score_bow = scorer.score(gold, pred_bow)
            bow_rouge1.append(score_bow['rouge1'].fmeasure)
            bow_rouge2.append(score_bow['rouge2'].fmeasure)
            bow_rougeL.append(score_bow['rougeL'].fmeasure)
            
            # Generate summary using TF-IDF
            pred_tfidf = summarize_textrank_tfidf(article, num_sentences=num_sentences)
            score_tfidf = scorer.score(gold, pred_tfidf)
            tfidf_rouge1.append(score_tfidf['rouge1'].fmeasure)
            tfidf_rouge2.append(score_tfidf['rouge2'].fmeasure)
            tfidf_rougeL.append(score_tfidf['rougeL'].fmeasure)
            
        except Exception as e:
            print(f"Error processing article: {e}")
    
    return (bow_rouge1, bow_rouge2, bow_rougeL), (tfidf_rouge1, tfidf_rouge2, tfidf_rougeL)

def create_comparison_visualization(bow_results, tfidf_results, save_path='bow_vs_tfidf_comparison.png'):
    """Create visualization comparing BoW vs TF-IDF"""
    bow_rouge1, bow_rouge2, bow_rougeL = bow_results
    tfidf_rouge1, tfidf_rouge2, tfidf_rougeL = tfidf_results
    
    # Calculate averages
    bow_avg = [np.mean(bow_rouge1), np.mean(bow_rouge2), np.mean(bow_rougeL)]
    tfidf_avg = [np.mean(tfidf_rouge1), np.mean(tfidf_rouge2), np.mean(tfidf_rougeL)]
    
    # Create comparison chart
    x = np.arange(3)
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart comparison
    bars1 = ax1.bar(x - width/2, bow_avg, width, label='Bag-of-Words', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, tfidf_avg, width, label='TF-IDF', color='#4ECDC4', alpha=0.8)
    
    ax1.set_xlabel('ROUGE Metrics')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('TextRank: BoW vs TF-IDF Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Box plot comparison
    data_to_plot = [bow_rouge1, tfidf_rouge1, bow_rouge2, tfidf_rouge2, bow_rougeL, tfidf_rougeL]
    bp = ax2.boxplot(data_to_plot, positions=[1, 2, 4, 5, 7, 8], 
                     labels=['BoW', 'TF-IDF', 'BoW', 'TF-IDF', 'BoW', 'TF-IDF'],
                     patch_artist=True)
    
    # Color the boxes
    colors = ['#FF6B6B', '#4ECDC4'] * 3
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('ROUGE Score Distributions: BoW vs TF-IDF')
    ax2.set_ylabel('F1 Score')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add metric labels
    ax2.text(1.5, ax2.get_ylim()[1] + 0.02, 'ROUGE-1', ha='center', fontweight='bold')
    ax2.text(4.5, ax2.get_ylim()[1] + 0.02, 'ROUGE-2', ha='center', fontweight='bold')
    ax2.text(7.5, ax2.get_ylim()[1] + 0.02, 'ROUGE-L', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comparison chart saved as {save_path}")

def main():
    """Main function to compare BoW vs TF-IDF TextRank"""
    # Configuration
    train_folder = 'liputan6_data/canonical/train'
    num_articles = 100
    
    print("🎯 TextRank: Bag-of-Words vs TF-IDF Comparison")
    print("=" * 60)
    print(f"📁 Loading data from: {train_folder}")
    
    # Load data
    data = load_data(train_folder)
    
    if not data:
        print("❌ No data loaded. Please check the folder path.")
        return
    
    # Compare BoW vs TF-IDF
    bow_results, tfidf_results = compare_bow_vs_tfidf(data, num_articles)
    
    # Print results
    bow_rouge1, bow_rouge2, bow_rougeL = bow_results
    tfidf_rouge1, tfidf_rouge2, tfidf_rougeL = tfidf_results
    
    print(f"\n📈 Comparison Results:")
    print(f"{'Metric':<10} {'BoW':<10} {'TF-IDF':<10} {'Difference':<12}")
    print("-" * 45)
    print(f"{'ROUGE-1':<10} {np.mean(bow_rouge1):<10.4f} {np.mean(tfidf_rouge1):<10.4f} {np.mean(tfidf_rouge1) - np.mean(bow_rouge1):<12.4f}")
    print(f"{'ROUGE-2':<10} {np.mean(bow_rouge2):<10.4f} {np.mean(tfidf_rouge2):<10.4f} {np.mean(tfidf_rouge2) - np.mean(bow_rouge2):<12.4f}")
    print(f"{'ROUGE-L':<10} {np.mean(bow_rougeL):<10.4f} {np.mean(tfidf_rougeL):<10.4f} {np.mean(tfidf_rougeL) - np.mean(bow_rougeL):<12.4f}")
    
    # Create visualization
    create_comparison_visualization(bow_results, tfidf_results)
    
    print(f"\n✅ Comparison completed!")
    print(f"📊 TF-IDF typically performs better because it considers word importance")

if __name__ == "__main__":
    main() 