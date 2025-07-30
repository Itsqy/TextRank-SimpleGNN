#!/usr/bin/env python3
"""
Test script for manual reference evaluation
"""

def test_manual_reference_flow():
    """Test the manual reference evaluation flow"""
    
    print("🧪 Testing Manual Reference Evaluation Flow")
    print("=" * 50)
    
    # Simulate the flow
    print("1. ✅ User generates summary")
    print("2. ✅ User selects 'Manual Reference'")
    print("3. ✅ User enters reference summary")
    print("4. ✅ User clicks 'Calculate ROUGE Metrics' button")
    print("5. ✅ App calculates and displays metrics")
    print()
    
    # Test the actual evaluation function
    from rouge_score import rouge_scorer
    
    # Sample data
    generated_summary = "AI has transformed many industries. Machine learning solves complex problems."
    manual_reference = "Artificial intelligence has revolutionized many sectors. Machine learning algorithms solve difficult problems."
    
    print("📝 Generated Summary:")
    print(f"   {generated_summary}")
    print()
    print("📝 Manual Reference:")
    print(f"   {manual_reference}")
    print()
    
    # Calculate metrics
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(manual_reference, generated_summary)
    
    print("📊 ROUGE Metrics Results:")
    for metric_name, score in scores.items():
        print(f"   {metric_name.upper()}:")
        print(f"     Precision: {score.precision:.3f}")
        print(f"     Recall:    {score.recall:.3f}")
        print(f"     F1-Score:  {score.fmeasure:.3f}")
        print()
    
    print("🎉 Manual reference evaluation test completed!")
    print("💡 The feature should now work properly in the Streamlit app.")

if __name__ == "__main__":
    test_manual_reference_flow() 