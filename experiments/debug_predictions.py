"""
Debug Predictions Script

This script helps investigate the "all negative" predictions issue
by examining prediction probabilities and model behavior.
"""

import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add src to path
sys.path.append('../src')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_predictions_and_probabilities():
    """
    Analyze model predictions and probabilities to debug the issue.
    """
    print("🔍 DEBUGGING PREDICTIONS ISSUE")
    print("="*50)
    
    # Import required modules
    from src.embeddings import GloVeEmbeddings
    from src.data_processing import load_tweet_data, TweetPreprocessor, TweetVectorizer
    from src.models import SentimentModelTrainer
    from src.evaluation import SentimentEvaluator
    
    # Load data
    print("📊 Loading data...")
    train_texts, train_labels = load_tweet_data("data/tweet_sentiment.train.jsonl")
    test_texts, test_labels = load_tweet_data("data/tweet_sentiment.test.jsonl")
    
    # Load GloVe embeddings
    print("🔤 Loading GloVe embeddings...")
    glove = GloVeEmbeddings("embeddings/glove.6B.100d.txt")
    glove.load_embeddings()
    
    # Prepare data
    print("🔄 Preparing data...")
    preprocessor = TweetPreprocessor()
    vectorizer = TweetVectorizer(glove, preprocessor=preprocessor)
    
    X_train = vectorizer.tweets_to_vectors(train_texts)
    X_test = vectorizer.tweets_to_vectors(test_texts)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Check class distribution
    print(f"\n📈 Class distribution in training data:")
    train_dist = Counter(y_train)
    for label, count in train_dist.items():
        pct = count / len(y_train) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print(f"\n📈 Class distribution in test data:")
    test_dist = Counter(y_test)
    for label, count in test_dist.items():
        pct = count / len(y_test) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Train model
    print(f"\n🤖 Training logistic regression model...")
    trainer = SentimentModelTrainer('logistic_regression')
    training_results = trainer.train(X_train, y_train, tune_hyperparameters=True)
    
    print(f"Training completed. CV F1: {training_results['cv_mean']:.4f}")
    
    # Make predictions
    print(f"\n🔮 Making predictions...")
    y_pred = trainer.predict(X_test)
    y_proba = trainer.predict_proba(X_test)
    
    # Analyze predictions
    print(f"\n📊 PREDICTION ANALYSIS")
    print("="*30)
    
    pred_dist = Counter(y_pred)
    print(f"Prediction distribution:")
    for label, count in pred_dist.items():
        pct = count / len(y_pred) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Show unique predictions
    unique_preds = set(y_pred)
    print(f"\nUnique predictions: {unique_preds}")
    
    # Analyze probabilities
    print(f"\n🎯 PROBABILITY ANALYSIS")
    print("="*30)
    
    # Get class names
    class_names = trainer.classes_
    print(f"Class names: {class_names}")
    
    # Probability statistics
    print(f"\nProbability statistics:")
    for i, class_name in enumerate(class_names):
        probs = y_proba[:, i]
        print(f"  {class_name}:")
        print(f"    Mean: {probs.mean():.4f}")
        print(f"    Std:  {probs.std():.4f}")
        print(f"    Min:  {probs.min():.4f}")
        print(f"    Max:  {probs.max():.4f}")
    
    # Find most confident predictions for each class
    print(f"\n🎯 Most confident predictions:")
    for i, class_name in enumerate(class_names):
        max_prob_idx = np.argmax(y_proba[:, i])
        max_prob = y_proba[max_prob_idx, i]
        predicted_class = y_pred[max_prob_idx]
        true_class = y_test[max_prob_idx]
        
        print(f"  {class_name}:")
        print(f"    Max probability: {max_prob:.4f}")
        print(f"    Predicted as: {predicted_class}")
        print(f"    True label: {true_class}")
        print(f"    Text sample: {test_texts[max_prob_idx][:100]}...")
    
    # Check if probabilities are reasonable
    print(f"\n🔍 PROBABILITY DISTRIBUTION ANALYSIS")
    print("="*40)
    
    # Check for extreme probabilities
    max_probs = np.max(y_proba, axis=1)
    print(f"Maximum probability per sample:")
    print(f"  Mean: {max_probs.mean():.4f}")
    print(f"  Std:  {max_probs.std():.4f}")
    print(f"  Min:  {max_probs.min():.4f}")
    print(f"  Max:  {max_probs.max():.4f}")
    
    # Check for samples where model is very confident but wrong
    wrong_mask = y_pred != y_test
    confident_wrong = (max_probs > 0.8) & wrong_mask
    
    print(f"\nConfident but wrong predictions: {confident_wrong.sum()}")
    if confident_wrong.sum() > 0:
        print("Examples:")
        confident_wrong_indices = np.where(confident_wrong)[0][:5]
        for idx in confident_wrong_indices:
            print(f"  True: {y_test[idx]}, Pred: {y_pred[idx]}, Prob: {max_probs[idx]:.4f}")
            print(f"  Text: {test_texts[idx][:100]}...")
    
    # Analyze feature statistics
    print(f"\n📊 FEATURE ANALYSIS")
    print("="*25)
    
    print(f"Training features:")
    print(f"  Mean: {X_train.mean():.6f}")
    print(f"  Std:  {X_train.std():.6f}")
    print(f"  Min:  {X_train.min():.6f}")
    print(f"  Max:  {X_train.max():.6f}")
    
    print(f"Test features:")
    print(f"  Mean: {X_test.mean():.6f}")
    print(f"  Std:  {X_test.std():.6f}")
    print(f"  Min:  {X_test.min():.6f}")
    print(f"  Max:  {X_test.max():.6f}")
    
    # Check for zero features
    zero_features_train = (X_train == 0).all(axis=0).sum()
    zero_features_test = (X_test == 0).all(axis=0).sum()
    
    print(f"\nZero features:")
    print(f"  Training: {zero_features_train}/{X_train.shape[1]}")
    print(f"  Test: {zero_features_test}/{X_test.shape[1]}")
    
    # Look at per-class feature means
    print(f"\n📊 PER-CLASS FEATURE ANALYSIS")
    print("="*35)
    
    for label in np.unique(y_train):
        mask = y_train == label
        class_features = X_train[mask]
        print(f"{label}:")
        print(f"  Samples: {class_features.shape[0]}")
        print(f"  Mean feature value: {class_features.mean():.6f}")
        print(f"  Std feature value:  {class_features.std():.6f}")
    
    # Create visualizations
    print(f"\n📈 Creating visualizations...")
    
    # Probability distribution plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Probability distributions by class
    for i, class_name in enumerate(class_names):
        axes[0, 0].hist(y_proba[:, i], alpha=0.7, label=class_name, bins=50)
    axes[0, 0].set_xlabel('Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Probability Distributions by Class')
    axes[0, 0].legend()
    
    # Plot 2: Max probability distribution
    axes[0, 1].hist(max_probs, bins=50, alpha=0.7)
    axes[0, 1].set_xlabel('Max Probability')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Maximum Probability Distribution')
    
    # Plot 3: Feature distribution comparison
    feature_means_train = X_train.mean(axis=0)
    feature_means_test = X_test.mean(axis=0)
    
    axes[1, 0].scatter(feature_means_train, feature_means_test, alpha=0.5)
    axes[1, 0].plot([feature_means_train.min(), feature_means_train.max()], 
                    [feature_means_train.min(), feature_means_train.max()], 'r--')
    axes[1, 0].set_xlabel('Training Feature Means')
    axes[1, 0].set_ylabel('Test Feature Means')
    axes[1, 0].set_title('Feature Distribution: Train vs Test')
    
    # Plot 4: Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('debug_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Final evaluation
    print(f"\n📈 FINAL EVALUATION")
    print("="*25)
    
    evaluator = SentimentEvaluator()
    results = evaluator.evaluate(y_test, y_pred, class_names)
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Weighted F1: {results['weighted_avg']['f1-score']:.4f}")
    
    print(f"\nPer-class results:")
    for class_name, metrics in results['per_class'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-score: {metrics['f1-score']:.4f}")
        print(f"    Support: {metrics['support']}")
    
    return {
        'predictions': y_pred,
        'probabilities': y_proba,
        'true_labels': y_test,
        'class_names': class_names,
        'evaluation_results': results
    }


if __name__ == "__main__":
    results = analyze_predictions_and_probabilities() 