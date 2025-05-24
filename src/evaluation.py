"""
Model Evaluation and Error Analysis Module

This module provides comprehensive evaluation metrics, confusion matrix analysis,
and error analysis for sentiment classification models.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentEvaluator:
    """
    A class to handle comprehensive evaluation of sentiment classification models.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.evaluation_results = {}
        self.confusion_matrix = None
        self.classification_report = None
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 class_names: Optional[List[str]] = None) -> Dict:
        """
        Perform comprehensive evaluation of model predictions.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (List[str]): Names of the classes
            
        Returns:
            Dict: Comprehensive evaluation results
        """
        logger.info("Evaluating model performance...")
        
        # Ensure arrays are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Get unique labels
        if class_names is None:
            class_names = sorted(list(set(y_true) | set(y_pred)))
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=class_names, average=None, zero_division=0
        )
        
        # Calculate weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=class_names, average='weighted', zero_division=0
        )
        
        # Calculate macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=class_names, average='macro', zero_division=0
        )
        
        # Create per-class results
        per_class_results = {}
        for i, class_name in enumerate(class_names):
            per_class_results[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1-score': float(f1[i]),
                'support': int(support[i])
            }
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        
        # Compile results
        evaluation_results = {
            'accuracy': float(accuracy),
            'macro_avg': {
                'precision': float(precision_macro),
                'recall': float(recall_macro),
                'f1-score': float(f1_macro),
                'support': int(len(y_true))
            },
            'weighted_avg': {
                'precision': float(precision_weighted),
                'recall': float(recall_weighted),
                'f1-score': float(f1_weighted),
                'support': int(len(y_true))
            },
            'per_class': per_class_results,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'total_samples': int(len(y_true))
        }
        
        # Store results
        self.evaluation_results = evaluation_results
        self.confusion_matrix = cm
        self.classification_report = classification_report(
            y_true, y_pred, labels=class_names, target_names=class_names
        )
        
        # Log key metrics
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Weighted F1-score: {f1_weighted:.4f}")
        logger.info(f"Macro F1-score: {f1_macro:.4f}")
        
        return evaluation_results
    
    def plot_confusion_matrix(self, normalize: bool = False, save_path: str = None):
        """
        Plot confusion matrix with optional normalization.
        
        Args:
            normalize (bool): Whether to normalize the confusion matrix
            save_path (str): Path to save the plot
        """
        if self.confusion_matrix is None:
            raise ValueError("No confusion matrix available. Run evaluate() first.")
        
        cm = self.confusion_matrix.copy()
        class_names = self.evaluation_results['class_names']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_misclassifications(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 texts: List[str], n_examples: int = 5) -> Dict:
        """
        Analyze misclassified examples to understand model errors.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            texts (List[str]): Original text samples
            n_examples (int): Number of examples to show per error type
            
        Returns:
            Dict: Analysis of misclassifications
        """
        logger.info("Analyzing misclassifications...")
        
        # Create dataframe for analysis
        df = pd.DataFrame({
            'text': texts,
            'true_label': y_true,
            'pred_label': y_pred,
            'correct': y_true == y_pred
        })
        
        # Find misclassified examples
        misclassified = df[~df['correct']].copy()
        
        if len(misclassified) == 0:
            logger.info("No misclassifications found!")
            return {'misclassified_examples': {}, 'error_analysis': {}}
        
        # Group by error type
        error_types = misclassified.groupby(['true_label', 'pred_label']).size().to_dict()
        
        # Get examples for each error type
        misclassified_examples = {}
        
        for (true_label, pred_label), count in error_types.items():
            if count > 0:
                error_key = f"{true_label}_predicted_as_{pred_label}"
                examples = misclassified[
                    (misclassified['true_label'] == true_label) & 
                    (misclassified['pred_label'] == pred_label)
                ]['text'].head(n_examples).tolist()
                
                misclassified_examples[error_key] = {
                    'count': count,
                    'examples': examples
                }
        
        # Calculate error rates
        total_per_class = df['true_label'].value_counts().to_dict()
        error_rates = {}
        
        for true_label in df['true_label'].unique():
            class_errors = len(misclassified[misclassified['true_label'] == true_label])
            class_total = total_per_class[true_label]
            error_rates[true_label] = class_errors / class_total if class_total > 0 else 0
        
        # Most common error patterns
        most_common_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        error_analysis = {
            'total_misclassified': len(misclassified),
            'misclassification_rate': len(misclassified) / len(df),
            'error_rates_by_class': error_rates,
            'most_common_error_patterns': [
                {
                    'pattern': f"{true_label} → {pred_label}",
                    'count': count,
                    'percentage': count / len(misclassified)
                }
                for (true_label, pred_label), count in most_common_errors
            ]
        }
        
        # Log key findings
        logger.info(f"Total misclassifications: {len(misclassified)} ({len(misclassified)/len(df):.1%})")
        logger.info("Most common error patterns:")
        for pattern in error_analysis['most_common_error_patterns']:
            logger.info(f"  {pattern['pattern']}: {pattern['count']} ({pattern['percentage']:.1%})")
        
        return {
            'misclassified_examples': misclassified_examples,
            'error_analysis': error_analysis
        }
    
    def plot_class_performance(self, save_path: str = None):
        """
        Plot per-class performance metrics.
        
        Args:
            save_path (str): Path to save the plot
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        
        per_class = self.evaluation_results['per_class']
        classes = list(per_class.keys())
        
        # Extract metrics
        precision = [per_class[c]['precision'] for c in classes]
        recall = [per_class[c]['recall'] for c in classes]
        f1 = [per_class[c]['f1-score'] for c in classes]
        
        # Create plot
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Sentiment Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            ax.text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width, f + 0.01, f'{f:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 texts: List[str], save_dir: str = "results") -> Dict:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            texts (List[str]): Original text samples
            save_dir (str): Directory to save results
            
        Returns:
            Dict: Complete evaluation report
        """
        logger.info("Generating comprehensive evaluation report...")
        
        # Create results directory
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Perform evaluation
        evaluation_results = self.evaluate(y_true, y_pred)
        
        # Analyze misclassifications
        error_analysis = self.analyze_misclassifications(y_true, y_pred, texts)
        
        # Create visualizations
        self.plot_confusion_matrix(save_path=f"{save_dir}/confusion_matrix.png")
        self.plot_confusion_matrix(normalize=True, save_path=f"{save_dir}/confusion_matrix_normalized.png")
        self.plot_class_performance(save_path=f"{save_dir}/class_performance.png")
        
        # Compile complete report
        complete_report = {
            'evaluation_metrics': evaluation_results,
            'error_analysis': error_analysis,
            'classification_report_text': self.classification_report
        }
        
        # Save report
        import json
        
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {key: recursive_convert(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        with open(f"{save_dir}/evaluation_report.json", 'w') as f:
            json.dump(recursive_convert(complete_report), f, indent=2)
        
        # Save classification report as text
        with open(f"{save_dir}/classification_report.txt", 'w') as f:
            f.write(self.classification_report)
        
        logger.info(f"Evaluation report saved to {save_dir}/")
        return complete_report
    
    def print_evaluation_summary(self):
        """Print a concise evaluation summary."""
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate() first.")
            return
        
        results = self.evaluation_results
        
        print("\n" + "="*50)
        print("SENTIMENT ANALYSIS EVALUATION SUMMARY")
        print("="*50)
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Weighted F1-Score: {results['weighted_avg']['f1-score']:.4f}")
        print(f"  Macro F1-Score: {results['macro_avg']['f1-score']:.4f}")
        
        print(f"\nPer-Class Performance:")
        for class_name, metrics in results['per_class'].items():
            print(f"  {class_name.capitalize()}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1-score']:.4f}")
            print(f"    Support: {metrics['support']}")
        
        print(f"\nDataset Info:")
        print(f"  Total Samples: {results['total_samples']}")
        print(f"  Classes: {', '.join(results['class_names'])}")
        
        print("\n" + "="*50)


if __name__ == "__main__":
    # Example usage
    from data_processing import load_tweet_data
    from models import SentimentModelTrainer
    from embeddings import GloVeEmbeddings
    
    # Load data
    train_texts, train_labels = load_tweet_data("data/tweet_sentiment.train.jsonl")
    test_texts, test_labels = load_tweet_data("data/tweet_sentiment.test.jsonl")
    
    # For demonstration, create mock predictions (replace with actual model predictions)
    np.random.seed(42)
    mock_predictions = np.random.choice(test_labels, size=len(test_labels))
    
    # Create evaluator and analyze
    evaluator = SentimentEvaluator()
    
    # Generate comprehensive report
    report = evaluator.generate_evaluation_report(
        y_true=test_labels,
        y_pred=mock_predictions,
        texts=test_texts
    )
    
    # Print summary
    evaluator.print_evaluation_summary()
    
    print("\nEvaluation complete! Check the 'results/' directory for detailed reports and visualizations.") 