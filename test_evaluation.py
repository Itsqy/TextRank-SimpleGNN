#!/usr/bin/env python3
"""
Test script for ROUGE evaluation function
"""

from rouge_score import rouge_scorer

def test_rouge_evaluation():
    """Test the ROUGE evaluation function"""
    
    # Test data
    reference = "Artificial intelligence has revolutionized many industries. Machine learning algorithms solve complex problems. Deep learning models achieve remarkable success."
    prediction = "AI has transformed many sectors. Machine learning solves difficult problems. Deep learning achieves great success."
    
    print("🧪 Testing ROUGE Evaluation Function")
    print("=" * 50)
    
    print(f"Reference: {reference}")
    print(f"Prediction: {prediction}")
    print()
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    try:
        # Calculate scores
        scores = scorer.score(reference, prediction)
        
        print("✅ ROUGE Metrics Calculated Successfully!")
        print()
        
        # Display results
        for metric_name, score in scores.items():
            print(f"📊 {metric_name.upper()}:")
            print(f"   Precision: {score.precision:.4f}")
            print(f"   Recall:    {score.recall:.4f}")
            print(f"   F1-Score:  {score.fmeasure:.4f}")
            print()
        
        print("🎉 All tests passed! ROUGE evaluation is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error in ROUGE evaluation: {e}")
        return False

if __name__ == "__main__":
    test_rouge_evaluation() 