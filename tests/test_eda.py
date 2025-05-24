"""
Unit tests for the eda module (Phase 3).
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import json
from unittest.mock import Mock, patch

import sys
sys.path.append('../src')

from src.eda import SentimentDataAnalyzer


class TestSentimentDataAnalyzer:
    """Test cases for SentimentDataAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data
        self.train_texts = [
            "I love this airline! Great service.",
            "Terrible flight delay, very disappointed.",
            "Flight was okay, nothing special.",
            "Amazing customer service, highly recommend!",
            "Bad experience, will not fly again."
        ]
        
        self.train_labels = [
            "positive",
            "negative", 
            "neutral",
            "positive",
            "negative"
        ]
        
        self.test_texts = [
            "Excellent flight experience.",
            "Poor service, very upset.",
            "Average flight, nothing to complain about."
        ]
        
        self.test_labels = [
            "positive",
            "negative",
            "neutral"
        ]
        
        # Initialize analyzer
        self.analyzer = SentimentDataAnalyzer(
            self.train_texts, self.train_labels,
            self.test_texts, self.test_labels
        )
    
    def test_initialization(self):
        """Test SentimentDataAnalyzer initialization."""
        assert len(self.analyzer.train_texts) == 5
        assert len(self.analyzer.train_labels) == 5
        assert len(self.analyzer.test_texts) == 3
        assert len(self.analyzer.test_labels) == 3
        
        # Check dataframes
        assert len(self.analyzer.train_df) == 5
        assert len(self.analyzer.test_df) == 3
        assert len(self.analyzer.full_df) == 8
        
        # Check columns
        expected_columns = ['text', 'label', 'split', 'text_length', 'word_count']
        assert all(col in self.analyzer.full_df.columns for col in expected_columns)
    
    def test_analyze_class_distribution(self):
        """Test class distribution analysis."""
        # Test without plotting to avoid display issues in tests
        distribution_stats = self.analyzer.analyze_class_distribution(plot=False)
        
        # Check structure
        assert 'overall' in distribution_stats
        assert 'train' in distribution_stats
        assert 'test' in distribution_stats
        assert 'imbalance_analysis' in distribution_stats
        
        # Check overall distribution
        overall = distribution_stats['overall']
        assert 'counts' in overall
        assert 'percentages' in overall
        assert 'total_samples' in overall
        
        assert overall['total_samples'] == 8
        assert overall['counts']['positive'] == 3  # 2 train + 1 test
        assert overall['counts']['negative'] == 3  # 2 train + 1 test
        assert overall['counts']['neutral'] == 2   # 1 train + 1 test
        
        # Check percentages sum to 1
        percentages = list(overall['percentages'].values())
        assert abs(sum(percentages) - 1.0) < 1e-6
        
        # Check imbalance analysis
        imbalance = distribution_stats['imbalance_analysis']
        assert 'is_imbalanced' in imbalance
        assert 'imbalance_ratio' in imbalance
        assert 'max_class_percentage' in imbalance
        assert 'min_class_percentage' in imbalance
    
    def test_analyze_text_length_distribution(self):
        """Test text length distribution analysis."""
        length_stats = self.analyzer.analyze_text_length_distribution(plot=False)
        
        # Check structure
        assert 'overall' in length_stats
        for label in ['positive', 'negative', 'neutral']:
            assert label in length_stats
        
        # Check overall statistics
        overall = length_stats['overall']
        assert 'char_length' in overall
        assert 'word_count' in overall
        
        # Check that statistics are reasonable
        assert overall['char_length']['mean'] > 0
        assert overall['word_count']['mean'] > 0
        assert overall['char_length']['min'] >= 0
        assert overall['word_count']['min'] >= 0
        
        # Check per-class statistics
        for label in ['positive', 'negative', 'neutral']:
            label_stats = length_stats[label]
            assert 'char_length' in label_stats
            assert 'word_count' in label_stats
            
            # Check that required statistics are present
            for metric in ['mean', 'median', 'std', 'min', 'max']:
                assert metric in label_stats['char_length']
                assert metric in label_stats['word_count']
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        anomalies = self.analyzer.detect_anomalies()
        
        # Check structure
        expected_anomaly_types = [
            'empty_texts', 'very_short_texts', 'very_long_texts',
            'duplicate_texts', 'invalid_labels'
        ]
        
        for anomaly_type in expected_anomaly_types:
            assert anomaly_type in anomalies
            assert isinstance(anomalies[anomaly_type], list)
        
        # With our test data, we shouldn't have these anomalies
        assert len(anomalies['empty_texts']) == 0
        assert len(anomalies['invalid_labels']) == 0
    
    def test_detect_anomalies_with_actual_anomalies(self):
        """Test anomaly detection with data that has anomalies."""
        # Create data with anomalies
        problematic_texts = [
            "",  # empty text
            "a",  # very short text  
            "This is a very long tweet that goes on and on and on with many many words to make it extremely long beyond normal tweet length which should be detected as an anomaly because it's much longer than typical tweets",  # very long
            "Normal tweet here",
            "Normal tweet here"  # duplicate
        ]
        
        problematic_labels = [
            "positive",
            "negative", 
            "neutral",
            "positive",
            "invalid_label"  # invalid label
        ]
        
        # Create analyzer with problematic data
        problematic_analyzer = SentimentDataAnalyzer(
            problematic_texts, problematic_labels, [], []
        )
        
        anomalies = problematic_analyzer.detect_anomalies()
        
        # Should detect anomalies
        assert len(anomalies['empty_texts']) > 0
        assert len(anomalies['very_short_texts']) > 0
        assert len(anomalies['duplicate_texts']) > 0
        assert len(anomalies['invalid_labels']) > 0
    
    def test_get_sample_texts(self):
        """Test sample text extraction."""
        samples = self.analyzer.get_sample_texts(n_samples=2)
        
        # Check structure
        for label in ['positive', 'negative', 'neutral']:
            assert label in samples
            assert isinstance(samples[label], list)
            # Should have at most 2 samples (or less if class has fewer samples)
            assert len(samples[label]) <= 2
            assert len(samples[label]) <= len([l for l in self.analyzer.full_df['label'] if l == label])
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_methods(self, mock_savefig, mock_show):
        """Test plotting methods don't crash."""
        # Test class distribution plot
        distribution_stats = self.analyzer.analyze_class_distribution(plot=True)
        
        # Test text length distribution plot
        length_stats = self.analyzer.analyze_text_length_distribution(plot=True)
        
        # Check that plotting functions were called
        assert mock_show.call_count >= 2
        assert mock_savefig.call_count >= 2

    def test_dataframe_features(self):
        """Test that dataframe features are correctly calculated."""
        df = self.analyzer.full_df
        
        # Check text length calculation
        for idx, row in df.iterrows():
            expected_length = len(row['text'])
            assert row['text_length'] == expected_length
        
        # Check word count calculation
        for idx, row in df.iterrows():
            expected_words = len(row['text'].split())
            assert row['word_count'] == expected_words
        
        # Check split column
        train_mask = df['split'] == 'train'
        test_mask = df['split'] == 'test'
        
        assert sum(train_mask) == 5
        assert sum(test_mask) == 3
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with minimal data
        minimal_analyzer = SentimentDataAnalyzer(
            ["test"], ["positive"], 
            ["test2"], ["negative"]
        )
        
        # Should not crash
        distribution = minimal_analyzer.analyze_class_distribution(plot=False)
        assert distribution is not None
        
        length_analysis = minimal_analyzer.analyze_text_length_distribution(plot=False)
        assert length_analysis is not None
        
        anomalies = minimal_analyzer.detect_anomalies()
        assert anomalies is not None
        
        samples = minimal_analyzer.get_sample_texts(n_samples=1)
        assert samples is not None


if __name__ == "__main__":
    pytest.main([__file__]) 