"""
Complete Evaluation Section for Assignment

Generate confusion matrix, misclassification analysis, and final evaluation
using the best performing model (SVM with RBF kernel).
"""

import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')

from src.embeddings import GloVeEmbeddings
from src.data_processing import load_tweet_data, TweetPreprocessor, TweetVectorizer
from src.evaluation import SentimentEvaluator

def complete_evaluation():
    """Run complete evaluation with best model."""
    print("🎯 FINAL MODEL EVALUATION")
    print("="*50)
    print("Using: SVM with RBF kernel (Best performing model)")
    print("Expected F1: ~74.38%")
    print("="*50)
    
    # Load data
    print("\n📊 Loading data...")
    train_texts, train_labels = load_tweet_data("data/tweet_sentiment.train.jsonl")
    test_texts, test_labels = load_tweet_data("data/tweet_sentiment.test.jsonl")
    
    # Load GloVe embeddings
    print("🔤 Loading GloVe embeddings...")
    glove = GloVeEmbeddings("embeddings/glove.6B.100d.txt")
    glove.load_embeddings()
    
    # Prepare data (using best configuration: mean aggregation for SVM)
    print("🔄 Preparing data...")
    preprocessor = TweetPreprocessor()
    vectorizer = TweetVectorizer(glove, preprocessor=preprocessor, aggregation_method='mean')
    
    X_train = vectorizer.tweets_to_vectors(train_texts)
    X_test = vectorizer.tweets_to_vectors(test_texts)
    
    # Apply feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Train best model (SVM with RBF)
    print("\n🤖 Training SVM with RBF kernel...")
    model = SVC(
        kernel='rbf', 
        C=1.0,  # Default that worked well
        class_weight='balanced', 
        probability=True, 
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("✅ Training completed!")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Comprehensive evaluation using our evaluation module
    print("\n📊 COMPREHENSIVE EVALUATION")
    print("="*40)
    
    evaluator = SentimentEvaluator()
    
    # Generate complete evaluation report
    evaluation_report = evaluator.generate_evaluation_report(
        y_true=y_test,
        y_pred=y_pred,
        texts=test_texts,
        save_dir="final_evaluation"
    )
    
    # Print detailed summary
    evaluator.print_evaluation_summary()
    
    # Detailed misclassification analysis
    print("\n🔍 MISCLASSIFICATION ANALYSIS")
    print("="*40)
    
    error_analysis = evaluator.analyze_misclassifications(y_test, y_pred, test_texts, n_examples=3)
    
    print(f"\nError Analysis Summary:")
    analysis = error_analysis['error_analysis']
    print(f"  Total misclassified: {analysis['total_misclassified']} ({analysis['misclassification_rate']:.1%})")
    
    print(f"\n  Error rates by class:")
    for class_name, error_rate in analysis['error_rates_by_class'].items():
        print(f"    {class_name}: {error_rate:.1%}")
    
    print(f"\n  Most common error patterns:")
    for pattern in analysis['most_common_error_patterns'][:3]:
        print(f"    {pattern['pattern']}: {pattern['count']} cases ({pattern['percentage']:.1%})")
    
    # Show interesting misclassification examples
    print(f"\n📝 INTERESTING MISCLASSIFICATION EXAMPLES")
    print("="*50)
    
    examples = error_analysis['misclassified_examples']
    example_count = 0
    
    for error_type, data in examples.items():
        if example_count >= 3:  # Assignment asks for 2-3 examples
            break
            
        true_label, pred_label = error_type.split('_predicted_as_')
        print(f"\n{example_count + 1}. {true_label.upper()} predicted as {pred_label.upper()}:")
        print(f"   Count: {data['count']} cases")
        print(f"   Example: \"{data['examples'][0]}\"")
        
        # Brief analysis of why this might happen
        if true_label == 'neutral' and pred_label == 'negative':
            print("   → Analysis: Neutral tweets may contain complaint words that seem negative")
        elif true_label == 'positive' and pred_label == 'negative':
            print("   → Analysis: Positive tweets may mention problems they're happy were resolved")
        elif true_label == 'negative' and pred_label == 'neutral':
            print("   → Analysis: Some negative tweets may be stated factually without emotion")
        
        example_count += 1
    
    # Model insights
    print(f"\n🧠 MODEL INSIGHTS")
    print("="*30)
    
    results = evaluation_report['evaluation_metrics']
    
    print(f"Strengths:")
    print(f"  ✅ Strong negative sentiment detection (F1: {results['per_class']['negative']['f1-score']:.3f})")
    print(f"  ✅ Balanced approach with class weighting")
    print(f"  ✅ Good overall accuracy ({results['accuracy']:.3f})")
    
    print(f"\nChallenges:")
    print(f"  ⚠️  Neutral class hardest to predict (F1: {results['per_class']['neutral']['f1-score']:.3f})")
    print(f"  ⚠️  Class imbalance affects performance")
    
    print(f"\n💾 Results saved to:")
    print(f"  📁 final_evaluation/")
    print(f"    - confusion_matrix.png")
    print(f"    - confusion_matrix_normalized.png") 
    print(f"    - class_performance.png")
    print(f"    - evaluation_report.json")
    print(f"    - classification_report.txt")
    
    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_proba,
        'evaluation_report': evaluation_report,
        'evaluator': evaluator
    }

if __name__ == "__main__":
    results = complete_evaluation()
    
    print(f"\n🎯 EVALUATION SECTION COMPLETE!")
    print(f"Ready to proceed with reflection section and final deliverables.") 