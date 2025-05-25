"""
Data Augmentation Experiment - Optional Bonus Task

This module implements synonym replacement data augmentation
to test if it improves sentiment classification performance.
"""

import sys
import os
import random
import numpy as np
from typing import List, Dict, Tuple
import nltk
from nltk.corpus import wordnet
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.embeddings import GloVeEmbeddings
from src.data_processing import load_tweet_data, TweetPreprocessor, TweetVectorizer
from src.models import SentimentModelTrainer
from src.evaluation import SentimentEvaluator

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class SynonymReplacer:
    """Simple synonym replacement for data augmentation."""
    
    def __init__(self, replacement_rate: float = 0.1):
        """
        Initialize synonym replacer.
        
        Args:
            replacement_rate (float): Fraction of words to replace with synonyms
        """
        self.replacement_rate = replacement_rate
        
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym.lower())
        
        return list(synonyms)
    
    def augment_text(self, text: str) -> str:
        """
        Augment text by replacing some words with synonyms.
        
        NOTE: This is a simple implementation that has limitations:
        - No part-of-speech filtering (can replace "do" auxiliary verb with "bash")
        - No context awareness (doesn't preserve grammatical structure)
        - No sentiment preservation (synonyms may change sentiment)
        
        Better approaches would include:
        - POS tagging to only replace appropriate word types
        - Context-aware embeddings (BERT-based)
        - Sentiment-preserving synonym selection
        
        Args:
            text (str): Original text
            
        Returns:
            str: Augmented text
        """
        words = text.split()
        augmented_words = []
        
        for word in words:
            # Skip if word contains special characters (URLs, mentions, etc.)
            if any(char in word for char in ['@', '#', 'http', '.']):
                augmented_words.append(word)
                continue
                
            # Replace with probability
            if random.random() < self.replacement_rate:
                synonyms = self.get_synonyms(word.lower())
                if synonyms:
                    # Choose random synonym
                    # TODO: Filter synonyms by POS tag and sentiment
                    synonym = random.choice(synonyms)
                    augmented_words.append(synonym)
                else:
                    augmented_words.append(word)
            else:
                augmented_words.append(word)
        
        return ' '.join(augmented_words)
    
    def augment_dataset(self, texts: List[str], labels: List[str], 
                       augment_ratio: float = 0.5) -> Tuple[List[str], List[str]]:
        """
        Augment dataset by creating synonym-replaced versions.
        
        Args:
            texts (List[str]): Original texts
            labels (List[str]): Original labels
            augment_ratio (float): Ratio of original data to augment
            
        Returns:
            Tuple: (augmented_texts, augmented_labels)
        """
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        # Determine how many samples to augment
        n_augment = int(len(texts) * augment_ratio)
        indices_to_augment = random.sample(range(len(texts)), n_augment)
        
        for idx in indices_to_augment:
            original_text = texts[idx]
            augmented_text = self.augment_text(original_text)
            
            # Only add if augmentation actually changed the text
            if augmented_text != original_text:
                augmented_texts.append(augmented_text)
                augmented_labels.append(labels[idx])
        
        return augmented_texts, augmented_labels


def run_data_augmentation_experiment():
    """Run the data augmentation experiment."""
    print("🚀 DATA AUGMENTATION EXPERIMENT - OPTIONAL BONUS")
    print("="*55)
    print("Testing synonym replacement data augmentation")
    print("="*55)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load data
    print("📊 Loading original data...")
    train_texts, train_labels = load_tweet_data("../data/tweet_sentiment.train.jsonl")
    test_texts, test_labels = load_tweet_data("../data/tweet_sentiment.test.jsonl")
    
    print(f"Original training data: {len(train_texts)} samples")
    
    # Load GloVe embeddings
    print("🔤 Loading GloVe embeddings...")
    glove = GloVeEmbeddings("../embeddings/glove.6B.100d.txt")
    glove.load_embeddings()
    
    # Baseline experiment (no augmentation)
    print("\n📈 BASELINE EXPERIMENT (No Augmentation)")
    print("-" * 45)
    
    baseline_results = train_and_evaluate(
        train_texts, train_labels, test_texts, test_labels, 
        glove, "Baseline"
    )
    
    # Data augmentation experiment
    print("\n🔄 DATA AUGMENTATION EXPERIMENT")
    print("-" * 40)
    
    # Create augmented training data
    print("Creating augmented training data...")
    replacer = SynonymReplacer(replacement_rate=0.15)  # Replace 15% of words
    
    augmented_texts, augmented_labels = replacer.augment_dataset(
        train_texts, train_labels, augment_ratio=0.3  # Augment 30% of data
    )
    
    print(f"Augmented training data: {len(augmented_texts)} samples")
    print(f"Added {len(augmented_texts) - len(train_texts)} augmented samples")
    
    # Show some examples
    print("\n📝 Augmentation Examples:")
    augmentation_examples = []
    for i in range(3):
        if i < len(train_texts):
            original = train_texts[i]
            augmented = replacer.augment_text(original)
            if augmented != original:
                print(f"  Original:  {original}")
                print(f"  Augmented: {augmented}")
                print()
                augmentation_examples.append({
                    "original": original,
                    "augmented": augmented
                })
    
    # Train with augmented data
    augmented_results = train_and_evaluate(
        augmented_texts, augmented_labels, test_texts, test_labels,
        glove, "Augmented"
    )
    
    # Compare results
    print("\n📊 RESULTS COMPARISON")
    print("="*30)
    
    print(f"Baseline Results:")
    print(f"  Accuracy: {baseline_results['accuracy']:.4f}")
    print(f"  Weighted F1: {baseline_results['weighted_avg']['f1-score']:.4f}")
    
    print(f"\nAugmented Results:")
    print(f"  Accuracy: {augmented_results['accuracy']:.4f}")
    print(f"  Weighted F1: {augmented_results['weighted_avg']['f1-score']:.4f}")
    
    # Calculate improvement
    acc_improvement = augmented_results['accuracy'] - baseline_results['accuracy']
    f1_improvement = augmented_results['weighted_avg']['f1-score'] - baseline_results['weighted_avg']['f1-score']
    
    print(f"\nImprovement:")
    print(f"  Accuracy: {acc_improvement:+.4f}")
    print(f"  Weighted F1: {f1_improvement:+.4f}")
    
    if f1_improvement > 0:
        print("  ✅ Data augmentation improved performance!")
    else:
        print("  ⚠️ Data augmentation did not improve performance")
    
    print(f"\n💡 Analysis:")
    if f1_improvement > 0.01:
        analysis = "Synonym replacement helped the model generalize better"
        print("  - Synonym replacement helped the model generalize better")
        print("  - Additional training data improved minority class performance")
    elif abs(f1_improvement) < 0.01:
        analysis = "Minimal impact from data augmentation - model already well-generalized"
        print("  - Minimal impact from data augmentation")
        print("  - Model may already be well-generalized with original data")
    else:
        analysis = "Data augmentation may have introduced noise"
        print("  - Data augmentation may have introduced noise")
        print("  - Synonym quality or replacement rate could be optimized")
    
    # Save results to file
    results_summary = {
        'experiment_info': {
            'experiment_type': 'data_augmentation',
            'augmentation_method': 'synonym_replacement',
            'replacement_rate': 0.15,
            'augment_ratio': 0.3,
            'original_training_samples': len(train_texts),
            'augmented_training_samples': len(augmented_texts),
            'added_samples': len(augmented_texts) - len(train_texts),
            'test_samples': len(test_texts)
        },
        'baseline_results': baseline_results,
        'augmented_results': augmented_results,
        'improvement': {
            'accuracy': acc_improvement,
            'weighted_f1': f1_improvement,
            'macro_f1': augmented_results['macro_avg']['f1-score'] - baseline_results['macro_avg']['f1-score']
        },
        'analysis': analysis,
        'augmentation_examples': augmentation_examples,
        'conclusion': 'Data augmentation improved performance' if f1_improvement > 0 else 'Data augmentation did not improve performance',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save to results directory
    import json
    os.makedirs('results', exist_ok=True)
    
    with open('results/data_augmentation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: experiments/results/data_augmentation_results.json")
    
    return {
        'baseline': baseline_results,
        'augmented': augmented_results,
        'improvement': {
            'accuracy': acc_improvement,
            'f1': f1_improvement
        },
        'full_results': results_summary
    }


def train_and_evaluate(train_texts: List[str], train_labels: List[str],
                      test_texts: List[str], test_labels: List[str],
                      glove: GloVeEmbeddings, experiment_name: str) -> Dict:
    """Train model and evaluate performance."""
    
    # Prepare data
    preprocessor = TweetPreprocessor()
    vectorizer = TweetVectorizer(glove, preprocessor=preprocessor, aggregation_method='mean')
    
    X_train = vectorizer.tweets_to_vectors(train_texts)
    X_test = vectorizer.tweets_to_vectors(test_texts)
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    
    # Train best model (SVM with RBF)
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    evaluator = SentimentEvaluator()
    results = evaluator.evaluate(y_test, y_pred)
    
    print(f"{experiment_name} - Accuracy: {results['accuracy']:.4f}, F1: {results['weighted_avg']['f1-score']:.4f}")
    
    return results


if __name__ == "__main__":
    results = run_data_augmentation_experiment()
    
    print(f"\n🎯 DATA AUGMENTATION EXPERIMENT COMPLETE!")
    print(f"This completes the optional bonus task from the assignment.") 