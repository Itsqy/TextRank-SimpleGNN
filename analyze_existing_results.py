import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

def analyze_existing_results():
    """Analyze existing results and show detailed metrics"""
    
    # Load existing results
    try:
        results_df = pd.read_csv('summarization_results.csv')
        print("✅ Loaded existing results from 'summarization_results.csv'")
    except FileNotFoundError:
        print("❌ 'summarization_results.csv' not found. Please run the main pipeline first.")
        return
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate detailed metrics for each row
    detailed_results = []
    
    print("🔍 Calculating detailed metrics...")
    for idx, row in results_df.iterrows():
        gold = row['original']  # The actual column name is 'original'
        pred = row['generated']
        
        # Calculate detailed scores
        scores = scorer.score(gold, pred)
        
        detailed_row = {
            'article_id': idx + 1,
            'article': row['article'],
            'generated': pred,
            'gold_summary': gold,
        }
        
        # Add detailed metrics
        for metric_name, score in scores.items():
            detailed_row[f'{metric_name}_precision'] = score.precision
            detailed_row[f'{metric_name}_recall'] = score.recall
            detailed_row[f'{metric_name}_fmeasure'] = score.fmeasure
        
        detailed_results.append(detailed_row)
    
    # Create detailed DataFrame
    detailed_df = pd.DataFrame(detailed_results)
    
    # Save detailed results
    detailed_df.to_csv('detailed_summarization_results.csv', index=False)
    print("✅ Detailed results saved to 'detailed_summarization_results.csv'")
    
    # Print detailed metrics
    print_detailed_results(detailed_df)
    
    # Create visualization
    create_detailed_visualization(detailed_df)
    
    return detailed_df

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
            
            std_precision = results_df[precision_col].std()
            std_recall = results_df[recall_col].std()
            std_f1 = results_df[f1_col].std()
            
            print(f"📈 Average Precision: {avg_precision:.4f} ± {std_precision:.4f}")
            print(f"📉 Average Recall:    {avg_recall:.4f} ± {std_recall:.4f}")
            print(f"🎯 Average F1-Score:  {avg_f1:.4f} ± {std_f1:.4f}")
            
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

def create_comparison_table(results_df):
    """Create a comparison table showing all metrics"""
    metrics = ['rouge1', 'rouge2', 'rougeL']
    
    print("\n" + "="*100)
    print("📊 COMPREHENSIVE METRICS COMPARISON TABLE")
    print("="*100)
    
    # Create comparison table
    comparison_data = []
    
    for metric in metrics:
        precision_col = f'{metric}_precision'
        recall_col = f'{metric}_recall'
        f1_col = f'{metric}_fmeasure'
        
        if precision_col in results_df.columns:
            avg_precision = results_df[precision_col].mean()
            avg_recall = results_df[recall_col].mean()
            avg_f1 = results_df[f1_col].mean()
            
            std_precision = results_df[precision_col].std()
            std_recall = results_df[recall_col].std()
            std_f1 = results_df[f1_col].std()
            
            comparison_data.append({
                'Metric': metric.upper(),
                'Precision': f"{avg_precision:.4f} ± {std_precision:.4f}",
                'Recall': f"{avg_recall:.4f} ± {std_recall:.4f}",
                'F1-Score': f"{avg_f1:.4f} ± {std_f1:.4f}"
            })
    
    # Create and display comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    return comparison_df

if __name__ == "__main__":
    print("🎯 Analyzing Existing TextRank + GNN Results")
    print("=" * 50)
    
    # Analyze existing results
    detailed_df = analyze_existing_results()
    
    if detailed_df is not None:
        # Create comparison table
        comparison_df = create_comparison_table(detailed_df)
        
        print("\n✅ Analysis completed successfully!")
        print("📁 Generated files:")
        print("  - detailed_summarization_results.csv (Detailed metrics)")
        print("  - detailed_rouge_metrics.png (Visualization)")
        print("  - Comparison table displayed above")
    else:
        print("❌ Analysis failed. Please check your data files.") 