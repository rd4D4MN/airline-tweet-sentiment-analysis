"""
Unit tests for the data_processing module (Phase 2).
"""

import pytest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch

import sys
sys.path.append('../src')

from src.data_processing import TweetPreprocessor, TweetVectorizer, load_tweet_data, load_and_prepare_data
from src.embeddings import GloVeEmbeddings


class TestTweetPreprocessor:
    """Test cases for TweetPreprocessor class."""
    
    def test_default_initialization(self):
        """Test default initialization of TweetPreprocessor."""
        preprocessor = TweetPreprocessor()
        
        assert preprocessor.remove_stopwords == False
        assert preprocessor.use_stemming == False
        assert preprocessor.remove_urls == True
        assert preprocessor.remove_mentions == True
        assert preprocessor.remove_hashtags == False
        assert preprocessor.lowercase == True
    
    def test_custom_initialization(self):
        """Test custom initialization of TweetPreprocessor."""
        preprocessor = TweetPreprocessor(
            remove_stopwords=True,
            use_stemming=True,
            remove_urls=False,
            remove_mentions=False,
            remove_hashtags=True,
            lowercase=False
        )
        
        assert preprocessor.remove_stopwords == True
        assert preprocessor.use_stemming == True
        assert preprocessor.remove_urls == False
        assert preprocessor.remove_mentions == False
        assert preprocessor.remove_hashtags == True
        assert preprocessor.lowercase == False
    
    def test_clean_text_remove_urls(self):
        """Test URL removal from tweets."""
        preprocessor = TweetPreprocessor(remove_urls=True)
        
        text = "Check this out http://example.com and https://test.org"
        cleaned = preprocessor.clean_text(text)
        
        assert "http://example.com" not in cleaned
        assert "https://test.org" not in cleaned
        assert "Check this out" in cleaned
    
    def test_clean_text_keep_urls(self):
        """Test keeping URLs in tweets."""
        preprocessor = TweetPreprocessor(remove_urls=False)
        
        text = "Check this out http://example.com"
        cleaned = preprocessor.clean_text(text)
        
        assert "http://example.com" in cleaned
    
    def test_clean_text_remove_mentions(self):
        """Test mention removal from tweets."""
        preprocessor = TweetPreprocessor(remove_mentions=True)
        
        text = "Hello @user1 and @user2 how are you?"
        
        # Test the full preprocessing pipeline since mentions are handled by tokenizer
        tokens = preprocessor.tokenize_and_preprocess(text)
        
        # Mentions should be removed by the tokenizer when strip_handles=True
        assert "@user1" not in tokens
        assert "@user2" not in tokens
    
    def test_clean_text_remove_hashtags(self):
        """Test hashtag symbol removal."""
        preprocessor = TweetPreprocessor(remove_hashtags=True)
        
        text = "Great day #sunshine #happy"
        cleaned = preprocessor.clean_text(text)
        
        assert "#" not in cleaned
        assert "sunshine" in cleaned
        assert "happy" in cleaned
    
    def test_tokenize_and_preprocess_basic(self):
        """Test basic tokenization and preprocessing."""
        preprocessor = TweetPreprocessor()
        
        text = "Hello world! This is a test."
        tokens = preprocessor.tokenize_and_preprocess(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
    
    def test_tokenize_and_preprocess_empty(self):
        """Test preprocessing empty or None text."""
        preprocessor = TweetPreprocessor()
        
        # Empty string
        tokens1 = preprocessor.tokenize_and_preprocess("")
        assert tokens1 == []
        
        # None input
        tokens2 = preprocessor.tokenize_and_preprocess(None)
        assert tokens2 == []
    
    def test_tokenize_and_preprocess_case_handling(self):
        """Test case handling in tokenization."""
        preprocessor_lower = TweetPreprocessor(lowercase=True)
        preprocessor_preserve = TweetPreprocessor(lowercase=False)
        
        text = "Hello WORLD Test"
        
        tokens_lower = preprocessor_lower.tokenize_and_preprocess(text)
        tokens_preserve = preprocessor_preserve.tokenize_and_preprocess(text)
        
        # Check that lowercase version has lowercase tokens
        assert any(token.islower() for token in tokens_lower if token.isalpha())


class TestTweetVectorizer:
    """Test cases for TweetVectorizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock GloVe embeddings
        self.mock_glove = Mock(spec=GloVeEmbeddings)
        self.mock_glove.embedding_dim = 3
        
        # Define test vocabulary
        self.test_vocab = {
            'hello': np.array([0.1, 0.2, 0.3]),
            'world': np.array([0.4, 0.5, 0.6]),
            'good': np.array([0.7, 0.8, 0.9]),
            'bad': np.array([-0.1, -0.2, -0.3])
        }
        
        # Mock get_vector method
        def mock_get_vector(word):
            return self.test_vocab.get(word.lower())
        
        self.mock_glove.get_vector = mock_get_vector
    
    def test_initialization(self):
        """Test TweetVectorizer initialization."""
        vectorizer = TweetVectorizer(self.mock_glove)
        
        assert vectorizer.glove_embeddings == self.mock_glove
        assert vectorizer.preprocessor is not None
        assert vectorizer.aggregation_method == 'mean'
        assert vectorizer.total_tweets == 0
        assert vectorizer.empty_tweets == 0
        assert vectorizer.oov_words == 0
        assert vectorizer.total_words == 0
    
    def test_initialization_custom(self):
        """Test TweetVectorizer with custom parameters."""
        custom_preprocessor = TweetPreprocessor(lowercase=False)
        vectorizer = TweetVectorizer(
            self.mock_glove, 
            preprocessor=custom_preprocessor,
            aggregation_method='sum'
        )
        
        assert vectorizer.preprocessor == custom_preprocessor
        assert vectorizer.aggregation_method == 'sum'
    
    def test_tweet_to_vector_mean_aggregation(self):
        """Test tweet vectorization with mean aggregation."""
        vectorizer = TweetVectorizer(self.mock_glove, aggregation_method='mean')
        
        # Test tweet with known words
        tweet = "hello world"
        vector = vectorizer.tweet_to_vector(tweet)
        
        expected = np.mean([
            self.test_vocab['hello'],
            self.test_vocab['world']
        ], axis=0).astype(np.float32)
        
        np.testing.assert_array_almost_equal(vector, expected)
        assert vectorizer.total_tweets == 1
    
    def test_tweet_to_vector_sum_aggregation(self):
        """Test tweet vectorization with sum aggregation."""
        vectorizer = TweetVectorizer(self.mock_glove, aggregation_method='sum')
        
        tweet = "hello world"
        vector = vectorizer.tweet_to_vector(tweet)
        
        expected = np.sum([
            self.test_vocab['hello'],
            self.test_vocab['world']
        ], axis=0).astype(np.float32)
        
        np.testing.assert_array_almost_equal(vector, expected)
    
    def test_tweet_to_vector_max_aggregation(self):
        """Test tweet vectorization with max aggregation."""
        vectorizer = TweetVectorizer(self.mock_glove, aggregation_method='max')
        
        tweet = "hello world"
        vector = vectorizer.tweet_to_vector(tweet)
        
        expected = np.max([
            self.test_vocab['hello'],
            self.test_vocab['world']
        ], axis=0).astype(np.float32)
        
        np.testing.assert_array_almost_equal(vector, expected)
    
    def test_tweet_to_vector_empty_tweet(self):
        """Test handling of empty tweets."""
        vectorizer = TweetVectorizer(self.mock_glove)
        
        vector = vectorizer.tweet_to_vector("")
        expected = np.zeros(3, dtype=np.float32)
        
        np.testing.assert_array_equal(vector, expected)
        assert vectorizer.empty_tweets == 1
    
    def test_tweet_to_vector_oov_words(self):
        """Test handling of out-of-vocabulary words."""
        vectorizer = TweetVectorizer(self.mock_glove)
        
        # Tweet with mix of known and unknown words
        tweet = "hello unknown_word world"
        vector = vectorizer.tweet_to_vector(tweet)
        
        # Should average only the known words
        expected = np.mean([
            self.test_vocab['hello'],
            self.test_vocab['world']
        ], axis=0).astype(np.float32)
        
        np.testing.assert_array_almost_equal(vector, expected)
        assert vectorizer.oov_words == 1
    
    def test_tweet_to_vector_all_oov(self):
        """Test handling of tweets with all OOV words."""
        vectorizer = TweetVectorizer(self.mock_glove)
        
        tweet = "unknown_word1 unknown_word2"
        vector = vectorizer.tweet_to_vector(tweet)
        
        expected = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_equal(vector, expected)
        assert vectorizer.empty_tweets == 1
    
    def test_tweets_to_vectors(self):
        """Test vectorizing multiple tweets."""
        vectorizer = TweetVectorizer(self.mock_glove)
        
        tweets = ["hello world", "good day", "bad news"]
        vectors = vectorizer.tweets_to_vectors(tweets)
        
        assert vectors.shape == (3, 3)  # 3 tweets, 3 dimensions
        assert vectorizer.total_tweets == 3
    
    def test_get_stats(self):
        """Test getting vectorization statistics."""
        vectorizer = TweetVectorizer(self.mock_glove)
        
        # Process some tweets
        vectorizer.tweet_to_vector("hello world")
        vectorizer.tweet_to_vector("unknown_word")
        vectorizer.tweet_to_vector("")
        
        stats = vectorizer.get_stats()
        
        assert stats['total_tweets'] == 3
        assert stats['empty_tweets'] == 2  # unknown_word and empty
        assert stats['empty_rate'] > 0
        assert stats['oov_rate'] > 0
        assert stats['aggregation_method'] == 'mean'
    
    def test_invalid_aggregation_method(self):
        """Test error handling for invalid aggregation method."""
        vectorizer = TweetVectorizer(self.mock_glove, aggregation_method='invalid')
        
        with pytest.raises(ValueError):
            vectorizer.tweet_to_vector("hello world")


class TestDataLoading:
    """Test cases for data loading functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_data = [
            {"text": "I love this airline!", "label": "positive"},
            {"text": "Terrible service", "label": "negative"},
            {"text": "Flight was okay", "label": "neutral"},
            {"text": "Great experience", "label": "positive"}
        ]
        
        self.test_file = os.path.join(self.temp_dir, "test_data.jsonl")
        with open(self.test_file, 'w') as f:
            for item in self.test_data:
                f.write(json.dumps(item) + '\n')
    
    def teardown_method(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_tweet_data_success(self):
        """Test successful loading of tweet data."""
        texts, labels = load_tweet_data(self.test_file)
        
        assert len(texts) == 4
        assert len(labels) == 4
        assert texts[0] == "I love this airline!"
        assert labels[0] == "positive"
        assert labels[1] == "negative"
        assert labels[2] == "neutral"
    
    def test_load_tweet_data_file_not_found(self):
        """Test handling of missing data file."""
        with pytest.raises(FileNotFoundError):
            load_tweet_data("nonexistent_file.jsonl")
    
    def test_load_tweet_data_malformed_json(self):
        """Test handling of malformed JSON lines."""
        malformed_file = os.path.join(self.temp_dir, "malformed.jsonl")
        with open(malformed_file, 'w') as f:
            f.write('{"text": "valid", "label": "positive"}\n')
            f.write('invalid json line\n')
            f.write('{"text": "another valid", "label": "negative"}\n')
        
        texts, labels = load_tweet_data(malformed_file)
        
        # Should skip malformed line
        assert len(texts) == 2
        assert len(labels) == 2
        assert texts[0] == "valid"
        assert texts[1] == "another valid"
    
    @patch('src.data_processing.TweetVectorizer')
    @patch('src.data_processing.load_tweet_data')
    def test_load_and_prepare_data(self, mock_load_data, mock_vectorizer_class):
        """Test the complete data preparation pipeline."""
        # Mock the data loading
        mock_load_data.side_effect = [
            (["train tweet 1", "train tweet 2"], ["positive", "negative"]),
            (["test tweet 1"], ["neutral"])
        ]
        
        # Mock the vectorizer
        mock_vectorizer = Mock()
        mock_vectorizer.tweets_to_vectors.side_effect = [
            np.array([[0.1, 0.2], [0.3, 0.4]]),  # Training data
            np.array([[0.5, 0.6]])  # Test data
        ]
        mock_vectorizer.get_stats.return_value = {"oov_rate": 0.1}
        mock_vectorizer_class.return_value = mock_vectorizer
        
        # Mock GloVe embeddings
        mock_glove = Mock()
        
        # Call the function
        X_train, X_test, y_train, y_test, train_texts, test_texts = load_and_prepare_data(
            "train.jsonl", "test.jsonl", mock_glove
        )
        
        # Verify results
        assert X_train.shape == (2, 2)
        assert X_test.shape == (1, 2)
        assert len(y_train) == 2
        assert len(y_test) == 1
        assert len(train_texts) == 2
        assert len(test_texts) == 1
        
        # Verify function calls
        assert mock_load_data.call_count == 2
        assert mock_vectorizer.tweets_to_vectors.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__]) 