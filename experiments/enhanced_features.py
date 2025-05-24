"""
Enhanced Features Experiment

Add simple feature engineering to improve performance.
"""

import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.embeddings import GloVeEmbeddings
from src.data_processing import load_tweet_data, TweetPreprocessor, TweetVectorizer
from src.models import SentimentModelTrainer
from src.evaluation import SentimentEvaluator

def create_enhanced_features(texts, vectorizer):
    """Create enhanced features combining GloVe + simple text features."""
    
    # Get GloVe vectors
    glove_vectors = vectorizer.tweets_to_vectors(texts)
    
    # Simple sentiment word lists
    positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'best', 'awesome', 'fantastic'}
    negative_words = {'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'delayed', 'cancel'}
    
    enhanced_features = []
    
    for i, text in enumerate(texts):
        # Get GloVe vector
        glove_vec = glove_vectors[i]
        
        # Text statistics
        text_length = len(text)
        word_count = len(text.split())
        
        # Sentiment word counts
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        # Combine features
        enhanced_vec = np.concatenate([
            glove_vec,
            [text_length / 100,  # Normalize
             word_count / 20,
             pos_count,
             neg_count,
             pos_count - neg_count]  # Sentiment balance
        ])
        
        enhanced_features.append(enhanced_vec)
    
    return np.array(enhanced_features)

def run_enhanced_experiment():
    """Run experiment with enhanced features."""
    print("🚀 Enhanced Features Experiment")
    print("="*40)
    
    # Load data
    print("📊 Loading data...")
    train_texts, train_labels = load_tweet_data("../data/tweet_sentiment.train.jsonl")
    test_texts, test_labels = load_tweet_data("../data/tweet_sentiment.test.jsonl")
    
    # Load GloVe embeddings
    print("🔤 Loading GloVe embeddings...")
    glove = GloVeEmbeddings("../embeddings/glove.6B.100d.txt")
    glove.load_embeddings()
    
    # Create vectorizer
    preprocessor = TweetPreprocessor()
    vectorizer = TweetVectorizer(glove, preprocessor=preprocessor, aggregation_method='sum')
    
    # Create enhanced features
    print("⚡ Creating enhanced features...")
    X_train = create_enhanced_features(train_texts, vectorizer)
    X_test = create_enhanced_features(test_texts, vectorizer)
    
    print(f"Enhanced feature shape: {X_train.shape}")
    print(f"Original GloVe: 100, Enhanced: {X_train.shape[1]} (+{X_train.shape[1] - 100} features)")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    
    # Test best models with enhanced features
    models = {
        'Enhanced SVM': SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
        'Enhanced LR': LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🤖 Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        evaluator = SentimentEvaluator()
        eval_results = evaluator.evaluate(y_test, y_pred)
        
        results[name] = eval_results
        
        print(f"{name} Results:")
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  Weighted F1: {eval_results['weighted_avg']['f1-score']:.4f}")
        print(f"  Per-class F1:")
        for class_name, metrics in eval_results['per_class'].items():
            print(f"    {class_name}: {metrics['f1-score']:.4f}")
    
    return results

if __name__ == "__main__":
    results = run_enhanced_experiment() 