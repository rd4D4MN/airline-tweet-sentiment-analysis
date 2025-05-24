"""
Exploratory Data Analysis Module

This module provides functionality for analyzing tweet sentiment data,
including class distribution analysis, anomaly detection, and visualization.
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


class SentimentDataAnalyzer:
    """
    A class to perform exploratory data analysis on sentiment data.
    """
    
    def __init__(self, train_texts: List[str], train_labels: List[str],
                 test_texts: List[str], test_labels: List[str]):
        """
        Initialize the analyzer with training and test data.
        
        Args:
            train_texts (List[str]): Training tweet texts
            train_labels (List[str]): Training labels
            test_texts (List[str]): Test tweet texts
            test_labels (List[str]): Test labels
        """
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.test_texts = test_texts
        self.test_labels = test_labels
        
        # Create dataframes for easier analysis
        self.train_df = pd.DataFrame({
            'text': train_texts,
            'label': train_labels,
            'split': 'train'
        })
        
        self.test_df = pd.DataFrame({
            'text': test_texts,
            'label': test_labels,
            'split': 'test'
        })
        
        self.full_df = pd.concat([self.train_df, self.test_df], ignore_index=True)
        
        # Add text length features
        self.full_df['text_length'] = self.full_df['text'].str.len()
        self.full_df['word_count'] = self.full_df['text'].str.split().str.len()
    
    def analyze_class_distribution(self, plot: bool = True) -> Dict:
        """
        Analyze the distribution of sentiment classes.
        
        Args:
            plot (bool): Whether to create visualization plots
            
        Returns:
            Dict: Dictionary containing distribution statistics
        """
        logger.info("Analyzing class distribution...")
        
        # Overall distribution
        overall_dist = Counter(self.full_df['label'])
        train_dist = Counter(self.train_df['label'])
        test_dist = Counter(self.test_df['label'])
        
        # Calculate percentages
        total_samples = len(self.full_df)
        train_samples = len(self.train_df)
        test_samples = len(self.test_df)
        
        overall_pct = {label: count/total_samples for label, count in overall_dist.items()}
        train_pct = {label: count/train_samples for label, count in train_dist.items()}
        test_pct = {label: count/test_samples for label, count in test_dist.items()}
        
        # Create distribution summary
        distribution_stats = {
            'overall': {
                'counts': dict(overall_dist),
                'percentages': overall_pct,
                'total_samples': total_samples
            },
            'train': {
                'counts': dict(train_dist),
                'percentages': train_pct,
                'total_samples': train_samples
            },
            'test': {
                'counts': dict(test_dist),
                'percentages': test_pct,
                'total_samples': test_samples
            }
        }
        
        # Check for class imbalance
        max_pct = max(overall_pct.values())
        min_pct = min(overall_pct.values())
        imbalance_ratio = max_pct / min_pct
        
        distribution_stats['imbalance_analysis'] = {
            'max_class_percentage': max_pct,
            'min_class_percentage': min_pct,
            'imbalance_ratio': imbalance_ratio,
            'is_imbalanced': imbalance_ratio > 2.0  # Threshold for considering dataset imbalanced
        }
        
        # Log results
        logger.info(f"Class distribution:")
        for label, count in overall_dist.items():
            logger.info(f"  {label}: {count} ({overall_pct[label]:.2%})")
        
        if distribution_stats['imbalance_analysis']['is_imbalanced']:
            logger.warning(f"Dataset is imbalanced! Ratio: {imbalance_ratio:.2f}")
        
        # Create visualizations
        if plot:
            self._plot_class_distribution(distribution_stats)
        
        return distribution_stats
    
    def analyze_text_length_distribution(self, plot: bool = True) -> Dict:
        """
        Analyze the distribution of text lengths.
        
        Args:
            plot (bool): Whether to create visualization plots
            
        Returns:
            Dict: Dictionary containing text length statistics
        """
        logger.info("Analyzing text length distribution...")
        
        # Calculate statistics by label
        length_stats = {}
        for label in self.full_df['label'].unique():
            label_data = self.full_df[self.full_df['label'] == label]
            
            length_stats[label] = {
                'char_length': {
                    'mean': label_data['text_length'].mean(),
                    'median': label_data['text_length'].median(),
                    'std': label_data['text_length'].std(),
                    'min': label_data['text_length'].min(),
                    'max': label_data['text_length'].max()
                },
                'word_count': {
                    'mean': label_data['word_count'].mean(),
                    'median': label_data['word_count'].median(),
                    'std': label_data['word_count'].std(),
                    'min': label_data['word_count'].min(),
                    'max': label_data['word_count'].max()
                }
            }
        
        # Overall statistics
        length_stats['overall'] = {
            'char_length': {
                'mean': self.full_df['text_length'].mean(),
                'median': self.full_df['text_length'].median(),
                'std': self.full_df['text_length'].std(),
                'min': self.full_df['text_length'].min(),
                'max': self.full_df['text_length'].max()
            },
            'word_count': {
                'mean': self.full_df['word_count'].mean(),
                'median': self.full_df['word_count'].median(),
                'std': self.full_df['word_count'].std(),
                'min': self.full_df['word_count'].min(),
                'max': self.full_df['word_count'].max()
            }
        }
        
        # Log results
        logger.info(f"Text length statistics:")
        logger.info(f"  Average character length: {length_stats['overall']['char_length']['mean']:.1f}")
        logger.info(f"  Average word count: {length_stats['overall']['word_count']['mean']:.1f}")
        
        # Create visualizations
        if plot:
            self._plot_text_length_distribution(length_stats)
        
        return length_stats
    
    def detect_anomalies(self) -> Dict:
        """
        Detect potential anomalies in the dataset.
        
        Returns:
            Dict: Dictionary containing anomaly detection results
        """
        logger.info("Detecting potential anomalies...")
        
        anomalies = {
            'empty_texts': [],
            'very_short_texts': [],
            'very_long_texts': [],
            'duplicate_texts': [],
            'invalid_labels': []
        }
        
        # Check for empty texts
        empty_mask = self.full_df['text'].str.strip().str.len() == 0
        anomalies['empty_texts'] = self.full_df[empty_mask].index.tolist()
        
        # Check for very short texts (< 3 characters)
        short_mask = self.full_df['text_length'] < 3
        anomalies['very_short_texts'] = self.full_df[short_mask].index.tolist()
        
        # Check for very long texts (> 3 standard deviations from mean)
        mean_length = self.full_df['text_length'].mean()
        std_length = self.full_df['text_length'].std()
        long_threshold = mean_length + 3 * std_length
        long_mask = self.full_df['text_length'] > long_threshold
        anomalies['very_long_texts'] = self.full_df[long_mask].index.tolist()
        
        # Check for duplicate texts
        duplicated_mask = self.full_df.duplicated(subset=['text'], keep=False)
        anomalies['duplicate_texts'] = self.full_df[duplicated_mask].index.tolist()
        
        # Check for invalid labels
        expected_labels = {'positive', 'negative', 'neutral'}
        invalid_mask = ~self.full_df['label'].isin(expected_labels)
        anomalies['invalid_labels'] = self.full_df[invalid_mask].index.tolist()
        
        # Log anomalies
        for anomaly_type, indices in anomalies.items():
            if indices:
                logger.warning(f"Found {len(indices)} {anomaly_type}")
            else:
                logger.info(f"No {anomaly_type} found")
        
        return anomalies
    
    def get_sample_texts(self, n_samples: int = 5) -> Dict:
        """
        Get sample texts for each sentiment class.
        
        Args:
            n_samples (int): Number of samples per class
            
        Returns:
            Dict: Dictionary containing sample texts for each class
        """
        samples = {}
        
        for label in self.full_df['label'].unique():
            label_data = self.full_df[self.full_df['label'] == label]
            sample_texts = label_data['text'].sample(n=min(n_samples, len(label_data)), 
                                                   random_state=42).tolist()
            samples[label] = sample_texts
        
        return samples
    
    def _plot_class_distribution(self, distribution_stats: Dict):
        """Create class distribution plots."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall distribution
        labels = list(distribution_stats['overall']['counts'].keys())
        counts = list(distribution_stats['overall']['counts'].values())
        
        axes[0].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Overall Class Distribution')
        
        # Train vs Test distribution
        train_counts = [distribution_stats['train']['counts'].get(label, 0) for label in labels]
        test_counts = [distribution_stats['test']['counts'].get(label, 0) for label in labels]
        
        x = np.arange(len(labels))
        width = 0.35
        
        axes[1].bar(x - width/2, train_counts, width, label='Train', alpha=0.8)
        axes[1].bar(x + width/2, test_counts, width, label='Test', alpha=0.8)
        
        axes[1].set_xlabel('Sentiment Class')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Train vs Test Distribution')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('results/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_text_length_distribution(self, length_stats: Dict):
        """Create text length distribution plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Character length distribution by class
        for label in self.full_df['label'].unique():
            label_data = self.full_df[self.full_df['label'] == label]
            axes[0, 0].hist(label_data['text_length'], alpha=0.7, label=label, bins=50)
        
        axes[0, 0].set_xlabel('Character Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Character Length Distribution by Class')
        axes[0, 0].legend()
        
        # Word count distribution by class
        for label in self.full_df['label'].unique():
            label_data = self.full_df[self.full_df['label'] == label]
            axes[0, 1].hist(label_data['word_count'], alpha=0.7, label=label, bins=30)
        
        axes[0, 1].set_xlabel('Word Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Word Count Distribution by Class')
        axes[0, 1].legend()
        
        # Box plots for character length
        sns.boxplot(data=self.full_df, x='label', y='text_length', ax=axes[1, 0])
        axes[1, 0].set_title('Character Length by Sentiment Class')
        axes[1, 0].set_ylabel('Character Length')
        
        # Box plots for word count
        sns.boxplot(data=self.full_df, x='label', y='word_count', ax=axes[1, 1])
        axes[1, 1].set_title('Word Count by Sentiment Class')
        axes[1, 1].set_ylabel('Word Count')
        
        plt.tight_layout()
        plt.savefig('results/text_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_eda_report(self) -> Dict:
        """
        Generate a comprehensive EDA report.
        
        Returns:
            Dict: Complete EDA report
        """
        logger.info("Generating comprehensive EDA report...")
        
        # Create results directory
        import os
        os.makedirs('results', exist_ok=True)
        
        report = {
            'dataset_overview': {
                'total_samples': len(self.full_df),
                'train_samples': len(self.train_df),
                'test_samples': len(self.test_df),
                'unique_labels': self.full_df['label'].unique().tolist()
            },
            'class_distribution': self.analyze_class_distribution(plot=True),
            'text_length_analysis': self.analyze_text_length_distribution(plot=True),
            'anomalies': self.detect_anomalies(),
            'sample_texts': self.get_sample_texts(n_samples=3)
        }
        
        # Save report
        import json
        with open('results/eda_report.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Recursively convert numpy types
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {key: recursive_convert(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(item) for item in obj]
                else:
                    return convert_numpy(obj)
            
            json.dump(recursive_convert(report), f, indent=2)
        
        logger.info("EDA report saved to results/eda_report.json")
        return report


if __name__ == "__main__":
    # Example usage
    from data_processing import load_tweet_data
    
    # Load data
    train_texts, train_labels = load_tweet_data("data/tweet_sentiment.train.jsonl")
    test_texts, test_labels = load_tweet_data("data/tweet_sentiment.test.jsonl")
    
    # Create analyzer
    analyzer = SentimentDataAnalyzer(train_texts, train_labels, test_texts, test_labels)
    
    # Generate report
    report = analyzer.generate_eda_report()
    
    # Print summary
    print("\n=== EDA SUMMARY ===")
    print(f"Total samples: {report['dataset_overview']['total_samples']}")
    print(f"Classes: {report['dataset_overview']['unique_labels']}")
    
    print(f"\nClass distribution:")
    for label, count in report['class_distribution']['overall']['counts'].items():
        pct = report['class_distribution']['overall']['percentages'][label]
        print(f"  {label}: {count} ({pct:.1%})")
    
    if report['class_distribution']['imbalance_analysis']['is_imbalanced']:
        ratio = report['class_distribution']['imbalance_analysis']['imbalance_ratio']
        print(f"\n⚠️  Dataset is imbalanced (ratio: {ratio:.2f})")
    
    print(f"\nAnomalies detected:")
    for anomaly_type, indices in report['anomalies'].items():
        if indices:
            print(f"  {anomaly_type}: {len(indices)} cases") 