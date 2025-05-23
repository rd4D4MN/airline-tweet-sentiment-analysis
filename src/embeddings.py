"""
GloVe Embeddings Loading and Management Module

This module provides functionality to load, cache, and manage GloVe word embeddings
for efficient use in the sentiment analysis pipeline.
"""

import os
import pickle
import logging
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GloVeEmbeddings:
    """
    A class to handle GloVe embeddings loading, caching, and retrieval.
    
    This class optimizes memory usage by providing caching functionality
    and efficient lookup methods for word vectors.
    """
    
    def __init__(self, embeddings_path: str, cache_dir: str = "cache"):
        """
        Initialize the GloVe embeddings handler.
        
        Args:
            embeddings_path (str): Path to the GloVe embeddings file
            cache_dir (str): Directory to store cached embeddings
        """
        self.embeddings_path = embeddings_path
        self.cache_dir = cache_dir
        self.word_vectors: Dict[str, np.ndarray] = {}
        self.embedding_dim: Optional[int] = None
        self.vocab_size: int = 0
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_path(self) -> str:
        """Get the cache file path for the embeddings."""
        filename = os.path.basename(self.embeddings_path).replace('.txt', '.pkl')
        return os.path.join(self.cache_dir, f"glove_cache_{filename}")
    
    def load_embeddings(self, force_reload: bool = False) -> Dict[str, np.ndarray]:
        """
        Load GloVe embeddings from file or cache.
        
        Args:
            force_reload (bool): If True, reload from original file even if cache exists
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping words to their embedding vectors
        """
        cache_path = self._get_cache_path()
        
        # Try to load from cache first
        if not force_reload and os.path.exists(cache_path):
            logger.info(f"Loading embeddings from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.word_vectors = cached_data['word_vectors']
                    self.embedding_dim = cached_data['embedding_dim']
                    self.vocab_size = cached_data['vocab_size']
                    logger.info(f"Loaded {self.vocab_size} embeddings with dimension {self.embedding_dim}")
                    return self.word_vectors
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Loading from original file.")
        
        # Load from original file
        logger.info(f"Loading embeddings from: {self.embeddings_path}")
        
        if not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")
        
        # Get file size for progress bar
        file_size = os.path.getsize(self.embeddings_path)
        
        with open(self.embeddings_path, 'r', encoding='utf-8') as f:
            # Use tqdm for progress tracking
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Loading GloVe") as pbar:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    word = parts[0]
                    vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    
                    # Set embedding dimension from first vector
                    if self.embedding_dim is None:
                        self.embedding_dim = len(vector)
                        logger.info(f"Detected embedding dimension: {self.embedding_dim}")
                    
                    self.word_vectors[word] = vector
                    pbar.update(len(line.encode('utf-8')))
        
        self.vocab_size = len(self.word_vectors)
        logger.info(f"Loaded {self.vocab_size} word embeddings")
        
        # Cache the embeddings
        try:
            cache_data = {
                'word_vectors': self.word_vectors,
                'embedding_dim': self.embedding_dim,
                'vocab_size': self.vocab_size
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Cached embeddings to: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")
        
        return self.word_vectors
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get the embedding vector for a word.
        
        Args:
            word (str): The word to get the embedding for
            
        Returns:
            Optional[np.ndarray]: The embedding vector or None if word not found
        """
        return self.word_vectors.get(word.lower())
    
    def get_vectors(self, words: List[str]) -> List[Optional[np.ndarray]]:
        """
        Get embedding vectors for a list of words.
        
        Args:
            words (List[str]): List of words to get embeddings for
            
        Returns:
            List[Optional[np.ndarray]]: List of embedding vectors (None for OOV words)
        """
        return [self.get_vector(word) for word in words]
    
    def get_oov_rate(self, words: List[str]) -> float:
        """
        Calculate the out-of-vocabulary rate for a list of words.
        
        Args:
            words (List[str]): List of words to check
            
        Returns:
            float: OOV rate (0.0 to 1.0)
        """
        if not words:
            return 0.0
        
        oov_count = sum(1 for word in words if self.get_vector(word) is None)
        return oov_count / len(words)
    
    def get_embedding_stats(self) -> Dict:
        """
        Get statistics about the loaded embeddings.
        
        Returns:
            Dict: Dictionary containing embedding statistics
        """
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'cache_path': self._get_cache_path(),
            'embeddings_path': self.embeddings_path
        }


def load_glove_embeddings(embeddings_path: str, cache_dir: str = "cache") -> Dict[str, np.ndarray]:
    """
    Convenience function to load GloVe embeddings.
    
    This function provides a simple interface to load GloVe embeddings
    with automatic caching for improved performance.
    
    Args:
        embeddings_path (str): Path to the GloVe embeddings file
        cache_dir (str): Directory to store cached embeddings
        
    Returns:
        Dict[str, np.ndarray]: Dictionary mapping words to their embedding vectors
        
    Example:
        >>> embeddings = load_glove_embeddings('embeddings/glove.6B.100d.txt')
        >>> vector = embeddings.get('happy')
    """
    glove = GloVeEmbeddings(embeddings_path, cache_dir)
    return glove.load_embeddings()


if __name__ == "__main__":
    # Example usage
    embeddings_path = "embeddings/glove.6B.100d.txt"
    glove = GloVeEmbeddings(embeddings_path)
    word_vectors = glove.load_embeddings()
    
    # Test some words
    test_words = ["happy", "sad", "airplane", "flight", "customer"]
    for word in test_words:
        vector = glove.get_vector(word)
        if vector is not None:
            print(f"{word}: {vector.shape}")
        else:
            print(f"{word}: Not found in vocabulary")
    
    # Print statistics
    stats = glove.get_embedding_stats()
    print(f"Embedding Statistics: {stats}") 