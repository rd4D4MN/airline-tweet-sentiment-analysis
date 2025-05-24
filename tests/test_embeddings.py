"""
Unit tests for the embeddings module.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, mock_open

import sys
sys.path.append('../src')

from src.embeddings import GloVeEmbeddings, load_glove_embeddings


class TestGloVeEmbeddings:
    """Test cases for GloVeEmbeddings class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_embeddings_path = os.path.join(self.temp_dir, "test_glove.txt")
        
        # Create a mock GloVe file
        test_content = """hello 0.1 0.2 0.3
world 0.4 0.5 0.6
test -0.1 0.7 -0.3
python 1.0 -1.0 0.0"""
        
        with open(self.test_embeddings_path, 'w') as f:
            f.write(test_content)
    
    def teardown_method(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test GloVeEmbeddings initialization."""
        glove = GloVeEmbeddings(self.test_embeddings_path, self.temp_dir)
        
        assert glove.embeddings_path == self.test_embeddings_path
        assert glove.cache_dir == self.temp_dir
        assert glove.word_vectors == {}
        assert glove.embedding_dim is None
        assert glove.vocab_size == 0
    
    def test_load_embeddings_success(self):
        """Test successful loading of embeddings."""
        glove = GloVeEmbeddings(self.test_embeddings_path, self.temp_dir)
        word_vectors = glove.load_embeddings()
        
        # Check that embeddings were loaded
        assert len(word_vectors) == 4
        assert 'hello' in word_vectors
        assert 'world' in word_vectors
        assert 'test' in word_vectors
        assert 'python' in word_vectors
        
        # Check embedding dimensions
        assert glove.embedding_dim == 3
        assert glove.vocab_size == 4
        
        # Check specific vectors
        np.testing.assert_array_almost_equal(
            word_vectors['hello'], np.array([0.1, 0.2, 0.3])
        )
        np.testing.assert_array_almost_equal(
            word_vectors['python'], np.array([1.0, -1.0, 0.0])
        )
    
    def test_load_embeddings_file_not_found(self):
        """Test handling of missing embeddings file."""
        glove = GloVeEmbeddings("nonexistent_file.txt", self.temp_dir)
        
        with pytest.raises(FileNotFoundError):
            glove.load_embeddings()
    
    def test_get_vector_existing_word(self):
        """Test getting vector for existing word."""
        glove = GloVeEmbeddings(self.test_embeddings_path, self.temp_dir)
        glove.load_embeddings()
        
        vector = glove.get_vector('hello')
        assert vector is not None
        np.testing.assert_array_almost_equal(
            vector, np.array([0.1, 0.2, 0.3])
        )
    
    def test_get_vector_nonexistent_word(self):
        """Test getting vector for non-existent word."""
        glove = GloVeEmbeddings(self.test_embeddings_path, self.temp_dir)
        glove.load_embeddings()
        
        vector = glove.get_vector('nonexistent')
        assert vector is None
    
    def test_get_vector_case_insensitive(self):
        """Test that get_vector is case insensitive."""
        glove = GloVeEmbeddings(self.test_embeddings_path, self.temp_dir)
        glove.load_embeddings()
        
        vector_lower = glove.get_vector('hello')
        vector_upper = glove.get_vector('HELLO')
        vector_mixed = glove.get_vector('Hello')
        
        np.testing.assert_array_equal(vector_lower, vector_upper)
        np.testing.assert_array_equal(vector_lower, vector_mixed)
    
    def test_get_vectors_multiple_words(self):
        """Test getting vectors for multiple words."""
        glove = GloVeEmbeddings(self.test_embeddings_path, self.temp_dir)
        glove.load_embeddings()
        
        words = ['hello', 'world', 'nonexistent']
        vectors = glove.get_vectors(words)
        
        assert len(vectors) == 3
        assert vectors[0] is not None  # hello
        assert vectors[1] is not None  # world
        assert vectors[2] is None      # nonexistent
    
    def test_get_oov_rate(self):
        """Test calculation of out-of-vocabulary rate."""
        glove = GloVeEmbeddings(self.test_embeddings_path, self.temp_dir)
        glove.load_embeddings()
        
        # All words in vocabulary
        words1 = ['hello', 'world', 'test']
        oov_rate1 = glove.get_oov_rate(words1)
        assert oov_rate1 == 0.0
        
        # Half words out of vocabulary
        words2 = ['hello', 'unknown1', 'world', 'unknown2']
        oov_rate2 = glove.get_oov_rate(words2)
        assert oov_rate2 == 0.5
        
        # All words out of vocabulary
        words3 = ['unknown1', 'unknown2', 'unknown3']
        oov_rate3 = glove.get_oov_rate(words3)
        assert oov_rate3 == 1.0
        
        # Empty list
        oov_rate4 = glove.get_oov_rate([])
        assert oov_rate4 == 0.0
    
    def test_get_embedding_stats(self):
        """Test getting embedding statistics."""
        glove = GloVeEmbeddings(self.test_embeddings_path, self.temp_dir)
        glove.load_embeddings()
        
        stats = glove.get_embedding_stats()
        
        assert stats['vocab_size'] == 4
        assert stats['embedding_dim'] == 3
        assert stats['embeddings_path'] == self.test_embeddings_path
        assert 'cache_path' in stats
    
    def test_cache_functionality(self):
        """Test that caching works correctly."""
        glove = GloVeEmbeddings(self.test_embeddings_path, self.temp_dir)
        
        # First load should create cache
        word_vectors1 = glove.load_embeddings()
        cache_path = glove._get_cache_path()
        assert os.path.exists(cache_path)
        
        # Second load should use cache
        glove2 = GloVeEmbeddings(self.test_embeddings_path, self.temp_dir)
        word_vectors2 = glove2.load_embeddings()
        
        # Results should be identical
        assert len(word_vectors1) == len(word_vectors2)
        assert glove2.vocab_size == glove.vocab_size
        assert glove2.embedding_dim == glove.embedding_dim
        
        for word in word_vectors1:
            np.testing.assert_array_equal(word_vectors1[word], word_vectors2[word])


class TestLoadGloVeEmbeddings:
    """Test cases for the convenience function."""
    
    def test_load_glove_embeddings_function(self):
        """Test the convenience function works correctly."""
        temp_dir = tempfile.mkdtemp()
        test_embeddings_path = os.path.join(temp_dir, "test_glove.txt")
        
        # Create test file
        test_content = "hello 0.1 0.2\nworld 0.3 0.4"
        with open(test_embeddings_path, 'w') as f:
            f.write(test_content)
        
        try:
            word_vectors = load_glove_embeddings(test_embeddings_path, temp_dir)
            
            assert len(word_vectors) == 2
            assert 'hello' in word_vectors
            assert 'world' in word_vectors
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__]) 