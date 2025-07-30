#!/usr/bin/env python3
"""
Comparison script between TextRank-only and TextRank+GNN approaches
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results():
    """Load results from both approaches"""
    # Load TextRank-only results
    textrank_results = pd.read_csv('textrank_summarization_results.csv')
    
    # Load TextRank+GNN results (from parent directory)
    gnn_results = pd.read_csv('../summarization_results.csv')
    
    return textrank_results, gnn_results

def calculate_averages(results_df):
    """Calculate average ROUGE scores"""
    return {
        'rouge1': results_df['rouge1'].mean(),
        'rouge2': results_df['rouge2'].mean(),
        'rougeL': results_df['rougeL'].mean(),
        'rouge1_std': results_df['rouge1'].std(),
        'rouge2_std': results_df['rouge2'].std(),
        'rougeL_std': results_df['rougeL'].std()
    }

def create_comparison_chart(textrank_avg, gnn_avg):
    """Create comparison visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart comparison
    methods = ['TextRank Only', 'TextRank + GNN']
    rouge1_scores = [textrank_avg['rouge1'], gnn_avg['rouge1']]
    rouge2_scores = [textrank_avg['rouge2'], gnn_avg['rouge2']]
    rougeL_scores = [textrank_avg['rougeL'], gnn_avg['rougeL']]
    
    x = np.arange(len(methods))
    width = 0.25
    
    bars1 = ax1.bar(x - width, rouge1_scores, width, label='ROUGE-1', color='#FF6B6B')
    bars2 = ax1.bar(x, rouge2_scores, width, label='ROUGE-2', color='#4ECDC4')
    bars3 = ax1.bar(x + width, rougeL_scores, width, label='ROUGE-L', color='#45B7D1')
    
    ax1.set_xlabel('Method')
    ax1.set_ylabel('ROUGE Score')
    ax1.set_title('ROUGE Score Comparison: TextRank vs TextRank+GNN')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Performance improvement chart
    improvements = []
    for i in range(3):
        if i == 0:
            scores = [textrank_avg['rouge1'], gnn_avg['rouge1']]
        elif i == 1:
            scores = [textrank_avg['rouge2'], gnn_avg['rouge2']]
        else:
            scores = [textrank_avg['rougeL'], gnn_avg['rougeL']]
        
        improvement = ((scores[1] - scores[0]) / scores[0]) * 100
        improvements.append(improvement)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax2.bar(['ROUGE-1', 'ROUGE-2', 'ROUGE-L'], improvements, color=colors)
    ax2.set_xlabel('ROUGE Metric')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Performance Improvement: GNN over TextRank')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{improvement:+.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Comparison chart saved as method_comparison.png")

def print_detailed_comparison(textrank_avg, gnn_avg):
    """Print detailed comparison statistics"""
    print("\n" + "="*60)
    print("📊 DETAILED COMPARISON: TextRank vs TextRank+GNN")
    print("="*60)
    
    print(f"\n📈 ROUGE-1 Scores:")
    print(f"   TextRank Only:  {textrank_avg['rouge1']:.4f} ± {textrank_avg['rouge1_std']:.4f}")
    print(f"   TextRank + GNN: {gnn_avg['rouge1']:.4f} ± {gnn_avg['rouge1_std']:.4f}")
    improvement1 = ((gnn_avg['rouge1'] - textrank_avg['rouge1']) / textrank_avg['rouge1']) * 100
    print(f"   Improvement:    {improvement1:+.1f}%")
    
    print(f"\n📈 ROUGE-2 Scores:")
    print(f"   TextRank Only:  {textrank_avg['rouge2']:.4f} ± {textrank_avg['rouge2_std']:.4f}")
    print(f"   TextRank + GNN: {gnn_avg['rouge2']:.4f} ± {gnn_avg['rouge2_std']:.4f}")
    improvement2 = ((gnn_avg['rouge2'] - textrank_avg['rouge2']) / textrank_avg['rouge2']) * 100
    print(f"   Improvement:    {improvement2:+.1f}%")
    
    print(f"\n📈 ROUGE-L Scores:")
    print(f"   TextRank Only:  {textrank_avg['rougeL']:.4f} ± {textrank_avg['rougeL_std']:.4f}")
    print(f"   TextRank + GNN: {gnn_avg['rougeL']:.4f} ± {gnn_avg['rougeL_std']:.4f}")
    improvementL = ((gnn_avg['rougeL'] - textrank_avg['rougeL']) / textrank_avg['rougeL']) * 100
    print(f"   Improvement:    {improvementL:+.1f}%")
    
    print(f"\n🎯 Overall Performance:")
    avg_improvement = (improvement1 + improvement2 + improvementL) / 3
    print(f"   Average Improvement: {avg_improvement:+.1f}%")
    
    print(f"\n⚡ Key Insights:")
    if avg_improvement > 0:
        print(f"   ✅ GNN enhancement provides {avg_improvement:.1f}% average improvement")
        print(f"   🧠 Neural network learning captures additional patterns")
        print(f"   📊 Most significant improvement in ROUGE-2 ({improvement2:+.1f}%)")
    else:
        print(f"   ⚠️  GNN enhancement shows {abs(avg_improvement):.1f}% average decrease")
        print(f"   🤔 Consider hyperparameter tuning for GNN")
        print(f"   📝 TextRank baseline performs competitively")

def main():
    """Main comparison function"""
    print("🔍 Comparing TextRank vs TextRank+GNN Approaches")
    print("=" * 50)
    
    try:
        # Load results
        textrank_results, gnn_results = load_results()
        
        # Calculate averages
        textrank_avg = calculate_averages(textrank_results)
        gnn_avg = calculate_averages(gnn_results)
        
        # Create comparison chart
        create_comparison_chart(textrank_avg, gnn_avg)
        
        # Print detailed comparison
        print_detailed_comparison(textrank_avg, gnn_avg)
        
        print(f"\n✅ Comparison completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure both textrank_summarization_results.csv and ../summarization_results.csv exist")
    except Exception as e:
        print(f"❌ Error during comparison: {e}")

if __name__ == "__main__":
    main() 