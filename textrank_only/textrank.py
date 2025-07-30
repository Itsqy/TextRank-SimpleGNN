import nltk
import networkx as nx
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

def build_similarity_matrix(sentences):
    """Build similarity matrix from sentences using TF-IDF and cosine similarity"""
    if len(sentences) < 2:
        return np.eye(len(sentences))
    
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

def summarize_textrank(text, num_sentences=5, damping=0.85, similarity_threshold=0.0):
    """Generate summary using TextRank algorithm"""
    original_sentences, preprocessed_sentences = preprocess_text(text)
    
    if len(preprocessed_sentences) < 2:
        return ' '.join(original_sentences)
    
    if len(preprocessed_sentences) < num_sentences:
        return ' '.join(original_sentences)
    
    try:
        # Build similarity matrix
        similarity_matrix = build_similarity_matrix(preprocessed_sentences)
        
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
        print(f"Error in summarize_textrank: {e}")
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

def evaluate_summaries(data, num_articles=1000, num_sentences=5, damping=0.85, similarity_threshold=0.0):
    """Evaluate TextRank summaries using ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_list, rouge2_list, rougeL_list = [], [], []
    results = []
    
    print(f"📊 Processing {min(len(data), num_articles)} articles with TextRank...")
    
    for item in tqdm(data[:num_articles], desc="Processing articles"):
        try:
            # Extract article and summary text
            article = ' '.join([' '.join(sent) for sent in item['clean_article']])
            gold = ' '.join([' '.join(sent) for sent in item['clean_summary']])
            
            # Generate summary using TextRank
            pred = summarize_textrank(article, num_sentences=num_sentences, 
                                    damping=damping, similarity_threshold=similarity_threshold)
            
            # Calculate ROUGE scores
            score = scorer.score(gold, pred)
            rouge1_list.append(score['rouge1'].fmeasure)
            rouge2_list.append(score['rouge2'].fmeasure)
            rougeL_list.append(score['rougeL'].fmeasure)
            
            results.append({
                'article': article,
                'generated': pred,
                'original': gold,
                'rouge1': score['rouge1'].fmeasure,
                'rouge2': score['rouge2'].fmeasure,
                'rougeL': score['rougeL'].fmeasure
            })
            
        except Exception as e:
            print(f"Error processing article: {e}")
    
    return results, rouge1_list, rouge2_list, rougeL_list

def create_visualization(rouge1_list, rouge2_list, rougeL_list, save_path='textrank_rouge_scores.png'):
    """Create visualization of ROUGE scores"""
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart of average scores
    avg_scores = [np.mean(rouge1_list), np.mean(rouge2_list), np.mean(rougeL_list)]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax1.bar(['ROUGE-1', 'ROUGE-2', 'ROUGE-L'], avg_scores, color=colors)
    ax1.set_title("Average ROUGE Scores - TextRank", fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("F1 Score", fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, avg_scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Box plot of score distributions
    data_to_plot = [rouge1_list, rouge2_list, rougeL_list]
    bp = ax2.boxplot(data_to_plot, tick_labels=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'], patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title("ROUGE Score Distributions - TextRank", fontsize=14, fontweight='bold')
    ax2.set_ylabel("F1 Score", fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Chart saved as {save_path}")

def main():
    """Main function to run the TextRank summarization pipeline"""
    # Configuration
    train_folder = 'liputan6_data/canonical/train'
    num_articles = 1000
    num_sentences = 5
    damping = 0.85
    similarity_threshold = 0.0
    
    print("🎯 TextRank Text Summarization (No GNN)")
    print("=" * 50)
    print(f"📁 Loading data from: {train_folder}")
    
    # Load data
    data = load_data(train_folder)
    
    if not data:
        print("❌ No data loaded. Please check the folder path.")
        return
    
    # Evaluate summaries
    results, rouge1_list, rouge2_list, rougeL_list = evaluate_summaries(
        data, num_articles, num_sentences, damping, similarity_threshold
    )
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_path = 'textrank_summarization_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"✅ Results saved to {results_path}")
        
        # Print average scores
        print(f"\n📈 Average ROUGE Scores (TextRank only):")
        print(f"ROUGE-1: {np.mean(rouge1_list):.4f}")
        print(f"ROUGE-2: {np.mean(rouge2_list):.4f}")
        print(f"ROUGE-L: {np.mean(rougeL_list):.4f}")
        
        # Create visualization
        create_visualization(rouge1_list, rouge2_list, rougeL_list)
        
        # Print detailed statistics
        print(f"\n📊 Detailed Statistics:")
        print(f"Number of articles processed: {len(results)}")
        print(f"ROUGE-1 - Mean: {np.mean(rouge1_list):.4f}, Std: {np.std(rouge1_list):.4f}")
        print(f"ROUGE-2 - Mean: {np.mean(rouge2_list):.4f}, Std: {np.std(rouge2_list):.4f}")
        print(f"ROUGE-L - Mean: {np.mean(rougeL_list):.4f}, Std: {np.std(rougeL_list):.4f}")
        
        return True
    else:
        print("❌ No results generated")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ TextRank pipeline completed successfully!")
    else:
        print("\n❌ TextRank pipeline failed. Please check the data and try again.") 