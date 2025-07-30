import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import sent_tokenize
import re
from io import StringIO
import base64

# Page configuration
st.set_page_config(
    page_title="TextRank + SimpleGNN Summarizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

download_nltk_data()

# SimpleGNN Model
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

# Text preprocessing
def preprocess_text(text):
    """Preprocess text by tokenizing and cleaning"""
    sentences = sent_tokenize(text)
    
    preprocessed_sentences = []
    for sentence in sentences:
        cleaned = re.sub(r'[^a-zA-Z\s]', '', sentence.lower())
        cleaned = ' '.join(cleaned.split())
        if cleaned.strip():
            preprocessed_sentences.append(cleaned)
    
    return sentences, preprocessed_sentences

# Graph building
def build_graph(sentences):
    """Build similarity graph from sentences"""
    if len(sentences) < 2:
        return nx.Graph()
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    graph = nx.Graph()
    N = len(sentences)
    
    for i in range(N):
        graph.add_node(i)
    
    for i in range(N):
        for j in range(i + 1, N):
            if similarity_matrix[i][j] > 0:
                graph.add_edge(i, j, weight=similarity_matrix[i][j])
    
    return graph

# TextRank scores
def textrank_scores(graph):
    """Calculate TextRank scores for nodes in the graph"""
    if len(graph.nodes()) == 0:
        return {}
    return nx.pagerank(graph, weight='weight')

# GNN training
def train_gnn(graph, sentences, epochs=10, lr=0.01):
    """Train GNN model on the sentence graph"""
    if len(sentences) == 0:
        return np.array([]), 0
    
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(sentences).toarray()
    features = torch.tensor(X, dtype=torch.float32)
    N = len(sentences)
    
    adj = nx.to_numpy_array(graph, nodelist=range(N))
    adj = torch.tensor(adj, dtype=torch.float32)
    
    model = SimpleGNN(features.shape[1], 16)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    initial_scores = textrank_scores(graph)
    y = torch.tensor([initial_scores.get(i, 0) for i in range(N)], dtype=torch.float32)
    
    # Training loop with progress bar
    progress_bar = st.progress(0)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        progress_bar.progress((epoch + 1) / epochs)
    
    model.eval()
    with torch.no_grad():
        final_scores = model(features, adj).numpy()
    
    return final_scores, features.shape[1]

# Summarization function
def summarize_text(text, num_sentences=5):
    """Generate summary using TextRank + GNN approach"""
    original_sentences, preprocessed_sentences = preprocess_text(text)
    
    if len(preprocessed_sentences) < 2:
        return ' '.join(original_sentences), 1, original_sentences
    
    if len(preprocessed_sentences) < num_sentences:
        return ' '.join(original_sentences), len(preprocessed_sentences), original_sentences
    
    try:
        graph = build_graph(preprocessed_sentences)
        scores, input_dim = train_gnn(graph, preprocessed_sentences)
        
        if len(scores) == 0:
            return ' '.join(original_sentences), len(preprocessed_sentences), original_sentences
        
        ranked_sentences = [s for _, s in sorted(zip(scores, original_sentences), reverse=True)]
        return ' '.join(ranked_sentences[:num_sentences]), input_dim, ranked_sentences[:num_sentences]
        
    except Exception as e:
        st.error(f"Error in summarization: {e}")
        return ' '.join(original_sentences), len(preprocessed_sentences), original_sentences

def summarize_text_with_params(text, num_sentences=5, epochs=10, lr=0.01):
    """Generate summary using TextRank + GNN approach with custom parameters"""
    original_sentences, preprocessed_sentences = preprocess_text(text)
    
    if len(preprocessed_sentences) < 2:
        return ' '.join(original_sentences), 1, original_sentences
    
    if len(preprocessed_sentences) < num_sentences:
        return ' '.join(original_sentences), len(preprocessed_sentences), original_sentences
    
    try:
        graph = build_graph(preprocessed_sentences)
        scores, input_dim = train_gnn(graph, preprocessed_sentences, epochs=epochs, lr=lr)
        
        if len(scores) == 0:
            return ' '.join(original_sentences), len(preprocessed_sentences), original_sentences
        
        ranked_sentences = [s for _, s in sorted(zip(scores, original_sentences), reverse=True)]
        return ' '.join(ranked_sentences[:num_sentences]), input_dim, ranked_sentences[:num_sentences]
        
    except Exception as e:
        st.error(f"Error in summarization: {e}")
        return ' '.join(original_sentences), len(preprocessed_sentences), original_sentences

# Calculate ROUGE metrics
def calculate_rouge_metrics(gold, pred):
    """Calculate detailed ROUGE metrics"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(gold, pred)
    
    detailed_metrics = {}
    for metric_name, score in scores.items():
        detailed_metrics[metric_name] = {
            'precision': score.precision,
            'recall': score.recall,
            'fmeasure': score.fmeasure
        }
    
    return detailed_metrics

# Create visualizations
def create_metrics_chart(metrics):
    """Create interactive metrics visualization"""
    metrics_data = []
    for metric_name, scores in metrics.items():
        metrics_data.append({
            'Metric': f'{metric_name.upper()} Precision',
            'Value': scores['precision'],
            'Type': 'Precision'
        })
        metrics_data.append({
            'Metric': f'{metric_name.upper()} Recall',
            'Value': scores['recall'],
            'Type': 'Recall'
        })
        metrics_data.append({
            'Metric': f'{metric_name.upper()} F1-Score',
            'Value': scores['fmeasure'],
            'Type': 'F1-Score'
        })
    
    df = pd.DataFrame(metrics_data)
    
    fig = px.bar(df, x='Metric', y='Value', color='Type',
                 title='ROUGE Metrics Breakdown',
                 color_discrete_map={
                     'Precision': '#FF6B6B',
                     'Recall': '#4ECDC4',
                     'F1-Score': '#45B7D1'
                 })
    
    fig.update_layout(
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        showlegend=True
    )
    
    return fig

def create_sentence_ranking_chart(sentences, scores):
    """Create sentence ranking visualization"""
    if not scores or len(scores) == 0:
        return None
    
    df = pd.DataFrame({
        'Sentence': [f"Sentence {i+1}" for i in range(len(sentences))],
        'Score': scores[:len(sentences)],
        'Text': sentences[:len(scores)]
    })
    
    fig = px.bar(df, x='Sentence', y='Score',
                 title='Sentence Importance Scores',
                 hover_data=['Text'])
    
    fig.update_layout(
        xaxis_title="Sentences",
        yaxis_title="Importance Score",
        showlegend=False
    )
    
    return fig

# Main Streamlit app
def main():
    st.title("🧠 TextRank + SimpleGNN Text Summarizer")
    st.markdown("---")
    

    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Choose Model",
        ["TextRank + SimpleGNN", "TextRank Only"],
        help="Select the summarization approach"
    )
    
    # Parameters
    num_sentences = st.sidebar.slider(
        "Number of Sentences in Summary",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of sentences to include in the summary"
    )
    
    if model_type == "TextRank + SimpleGNN":
        epochs = st.sidebar.slider(
            "Training Epochs",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of training epochs for GNN"
        )
        
        learning_rate = st.sidebar.selectbox(
            "Learning Rate",
            [0.001, 0.01, 0.1],
            index=1,
            help="Learning rate for GNN training"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Text")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "File Upload", "Sample Text"]
        )
        
        input_text = ""
        
        if input_method == "Text Input":
            input_text = st.text_area(
                "Enter your text here:",
                height=300,
                placeholder="Paste your article text here..."
            )
        
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader(
                "Upload a text file:",
                type=['txt', 'md']
            )
            if uploaded_file is not None:
                input_text = uploaded_file.read().decode()
        
        elif input_method == "Sample Text":
            sample_texts = {
                "News Article": """Artificial intelligence has revolutionized many industries in recent years. 
                Machine learning algorithms are being used to solve complex problems in healthcare, finance, and transportation. 
                Deep learning models have achieved remarkable success in image recognition and natural language processing. 
                Companies are investing heavily in AI research and development. 
                The future of AI looks promising with advancements in quantum computing and neural networks. 
                However, there are also concerns about job displacement and ethical implications. 
                Governments around the world are developing regulations for AI deployment. 
                Researchers continue to push the boundaries of what's possible with artificial intelligence.""",
                
                "Research Paper": """Climate change represents one of the most significant challenges facing humanity today. 
                Scientific evidence shows that global temperatures are rising at an unprecedented rate. 
                Greenhouse gas emissions from human activities are the primary driver of this warming. 
                The consequences include more frequent extreme weather events and rising sea levels. 
                Mitigation strategies include renewable energy adoption and carbon capture technologies. 
                Adaptation measures are also necessary to address unavoidable climate impacts. 
                International cooperation is essential for effective climate action. 
                The Paris Agreement provides a framework for global climate efforts."""
            }
            
            selected_sample = st.selectbox("Choose sample text:", list(sample_texts.keys()))
            input_text = sample_texts[selected_sample]
            st.text_area("Sample text:", input_text, height=200, disabled=True)
    
    with col2:
        st.subheader("📊 Summary & Analysis")
        
        if st.button("🚀 Generate Summary", type="primary"):
            if input_text.strip():
                with st.spinner("Processing text and training model..."):
                    # Generate summary
                    if model_type == "TextRank + SimpleGNN":
                        # Use the configured parameters from sidebar
                        summary, input_dim, ranked_sentences = summarize_text_with_params(input_text, num_sentences, epochs, learning_rate)
                    else:
                        # TextRank only implementation
                        original_sentences, preprocessed_sentences = preprocess_text(input_text)
                        if len(preprocessed_sentences) < 2:
                            summary = ' '.join(original_sentences)
                            ranked_sentences = original_sentences
                        else:
                            graph = build_graph(preprocessed_sentences)
                            scores = textrank_scores(graph)
                            if scores:
                                ranked_sentences = [s for _, s in sorted(zip(scores.values(), original_sentences), reverse=True)]
                                summary = ' '.join(ranked_sentences[:num_sentences])
                            else:
                                summary = ' '.join(original_sentences[:num_sentences])
                                ranked_sentences = original_sentences[:num_sentences]
                

                
                # Display summary
                st.success("✅ Summary generated successfully!")
                st.markdown("### Generated Summary:")
                st.write(summary)
                
                # Text statistics
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Original Sentences", len(sent_tokenize(input_text)))
                with col_stats2:
                    st.metric("Summary Sentences", len(sent_tokenize(summary)))
                with col_stats3:
                    compression_ratio = len(summary.split()) / len(input_text.split()) * 100
                    st.metric("Compression Ratio", f"{compression_ratio:.1f}%")
                
                # Evaluation section
                st.markdown("---")
                st.subheader("🎯 Evaluation")
                
                # Simple evaluation with button
                st.write("**To evaluate your summary, enter a reference summary below:**")
                
                reference_summary = st.text_area(
                    "Reference Summary (Optional):",
                    height=100,
                    placeholder="Paste the reference/gold summary here to calculate ROUGE metrics..."
                )
                
                if st.button("📊 Calculate ROUGE Metrics", type="primary"):
                    if reference_summary.strip():
                        try:
                            with st.spinner("Calculating ROUGE metrics..."):
                                metrics = calculate_rouge_metrics(reference_summary, summary)
                            
                            st.success("✅ ROUGE metrics calculated successfully!")
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("ROUGE-1 F1", f"{metrics['rouge1']['fmeasure']:.3f}")
                                st.caption(f"P: {metrics['rouge1']['precision']:.3f} | R: {metrics['rouge1']['recall']:.3f}")
                            
                            with col2:
                                st.metric("ROUGE-2 F1", f"{metrics['rouge2']['fmeasure']:.3f}")
                                st.caption(f"P: {metrics['rouge2']['precision']:.3f} | R: {metrics['rouge2']['recall']:.3f}")
                            
                            with col3:
                                st.metric("ROUGE-L F1", f"{metrics['rougeL']['fmeasure']:.3f}")
                                st.caption(f"P: {metrics['rougeL']['precision']:.3f} | R: {metrics['rougeL']['recall']:.3f}")
                            
                            # Metrics visualization
                            st.plotly_chart(create_metrics_chart(metrics), use_container_width=True)
                            
                            # Detailed breakdown
                            with st.expander("📊 Detailed Metrics Breakdown"):
                                st.write("**ROUGE-1 Metrics (Unigram Overlap):**")
                                st.write(f"- Precision: {metrics['rouge1']['precision']:.4f}")
                                st.write(f"- Recall: {metrics['rouge1']['recall']:.4f}")
                                st.write(f"- F1-Score: {metrics['rouge1']['fmeasure']:.4f}")
                                
                                st.write("**ROUGE-2 Metrics (Bigram Overlap):**")
                                st.write(f"- Precision: {metrics['rouge2']['precision']:.4f}")
                                st.write(f"- Recall: {metrics['rouge2']['recall']:.4f}")
                                st.write(f"- F1-Score: {metrics['rouge2']['fmeasure']:.4f}")
                                
                                st.write("**ROUGE-L Metrics (Longest Common Subsequence):**")
                                st.write(f"- Precision: {metrics['rougeL']['precision']:.4f}")
                                st.write(f"- Recall: {metrics['rougeL']['recall']:.4f}")
                                st.write(f"- F1-Score: {metrics['rougeL']['fmeasure']:.4f}")
                        
                        except Exception as e:
                            st.error(f"❌ Error calculating ROUGE metrics: {e}")
                            st.info("💡 Make sure your reference summary is valid text.")
                    else:
                        st.warning("⚠️ Please enter a reference summary first.")
                
                # Auto-generate option
                if st.button("🤖 Auto-Generate Reference & Calculate"):
                    try:
                        # Auto-generate reference from first few sentences
                        sentences = sent_tokenize(input_text)
                        auto_reference = ' '.join(sentences[:3])
                        
                        st.write("**Auto-Generated Reference:**")
                        st.write(auto_reference)
                        
                        with st.spinner("Calculating ROUGE metrics..."):
                            metrics = calculate_rouge_metrics(auto_reference, summary)
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("ROUGE-1 F1", f"{metrics['rouge1']['fmeasure']:.3f}")
                        with col2:
                            st.metric("ROUGE-2 F1", f"{metrics['rouge2']['fmeasure']:.3f}")
                        with col3:
                            st.metric("ROUGE-L F1", f"{metrics['rougeL']['fmeasure']:.3f}")
                        
                        st.plotly_chart(create_metrics_chart(metrics), use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error in auto-evaluation: {e}")
                
                # Debug mode (optional)
                if st.checkbox("🔧 Show Debug Information"):
                    st.write("**Debug Information:**")
                    st.write(f"- Input text length: {len(input_text)} characters")
                    st.write(f"- Summary length: {len(summary)} characters")
                    st.write(f"- Number of sentences in summary: {len(sent_tokenize(summary))}")
                    st.write(f"- Model type: {model_type}")
                    if model_type == "TextRank + SimpleGNN":
                        st.write(f"- Training epochs: {epochs}")
                        st.write(f"- Learning rate: {learning_rate}")
                
                # Model information
                if model_type == "TextRank + SimpleGNN":
                    st.markdown("---")
                    st.subheader("🧠 Model Information")
                    st.info(f"**SimpleGNN Architecture:**\n- Input Dimension: {input_dim if 'input_dim' in locals() else 'N/A'}\n- Hidden Dimension: 16\n- Training Epochs: {epochs}\n- Learning Rate: {learning_rate}")
            else:
                st.error("Please enter some text to summarize.")
    
    
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🧠 TextRank + SimpleGNN Text Summarizer | Built with Streamlit</p>
        <p>This app combines the interpretability of TextRank with the learning capability of Graph Neural Networks</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 