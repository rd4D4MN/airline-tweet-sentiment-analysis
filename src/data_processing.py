"""
Data Loading and Preprocessing Module

This module handles loading tweet data, preprocessing text, and converting tweets
to vector representations using GloVe embeddings.
"""

import json
import re
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

from .embeddings import GloVeEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TweetPreprocessor:
    """
    A class to handle tweet text preprocessing with various cleaning and normalization options.
    """
    
    def __init__(self, 
                 remove_stopwords: bool = False,
                 use_stemming: bool = False,
                 remove_urls: bool = True,
                 remove_mentions: bool = True,
                 remove_hashtags: bool = False,
                 lowercase: bool = True):
        """
        Initialize the tweet preprocessor.
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            use_stemming (bool): Whether to apply stemming
            remove_urls (bool): Whether to remove URLs
            remove_mentions (bool): Whether to remove mentions (@username)
            remove_hashtags (bool): Whether to remove hashtags
            lowercase (bool): Whether to convert to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.lowercase = lowercase
        
        # Initialize NLTK components
        self.tokenizer = TweetTokenizer(preserve_case=not lowercase, 
                                      reduce_len=True, 
                                      strip_handles=remove_mentions)
        
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if use_stemming:
            self.stemmer = PorterStemmer()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize tweet text.
        
        Args:
            text (str): Raw tweet text
            
        Returns:
            str: Cleaned tweet text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions (if not handled by tokenizer)
        if self.remove_mentions and not self.tokenizer.strip_handles:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the word, remove the #)
        if self.remove_hashtags:
            text = re.sub(r'#', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_preprocess(self, text: str) -> List[str]:
        """
        Tokenize and preprocess tweet text.
        
        Args:
            text (str): Tweet text to process
            
        Returns:
            List[str]: List of preprocessed tokens
        """
        # Clean the text first
        text = self.clean_text(text)
        
        if not text:
            return []
        
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # Convert to lowercase if not done by tokenizer
        if self.lowercase and self.tokenizer.preserve_case:
            tokens = [token.lower() for token in tokens]
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Remove empty tokens and non-alphabetic tokens (optional)
        tokens = [token for token in tokens if token and len(token) > 1]
        
        return tokens


class TweetVectorizer:
    """
    A class to convert tweets to vector representations using GloVe embeddings.
    """
    
    def __init__(self, 
                 glove_embeddings: GloVeEmbeddings,
                 preprocessor: Optional[TweetPreprocessor] = None,
                 aggregation_method: str = 'mean'):
        """
        Initialize the tweet vectorizer.
        
        Args:
            glove_embeddings (GloVeEmbeddings): Loaded GloVe embeddings
            preprocessor (TweetPreprocessor): Text preprocessor
            aggregation_method (str): Method to aggregate word vectors ('mean', 'sum', 'max')
        """
        self.glove_embeddings = glove_embeddings
        self.preprocessor = preprocessor or TweetPreprocessor()
        self.aggregation_method = aggregation_method
        
        # Statistics
        self.total_tweets = 0
        self.empty_tweets = 0
        self.oov_words = 0
        self.total_words = 0
    
    def tweet_to_vector(self, tweet: str) -> np.ndarray:
        """
        Convert a tweet to a vector representation.
        
        Args:
            tweet (str): Tweet text
            
        Returns:
            np.ndarray: Vector representation of the tweet
        """
        # Preprocess the tweet
        tokens = self.preprocessor.tokenize_and_preprocess(tweet)
        
        self.total_tweets += 1
        
        # Handle empty tweets
        if not tokens:
            self.empty_tweets += 1
            return np.zeros(self.glove_embeddings.embedding_dim, dtype=np.float32)
        
        # Get vectors for tokens
        vectors = []
        for token in tokens:
            vector = self.glove_embeddings.get_vector(token)
            if vector is not None:
                vectors.append(vector)
            else:
                self.oov_words += 1
            
            self.total_words += 1
        
        # Handle case where no tokens have embeddings
        if not vectors:
            self.empty_tweets += 1
            return np.zeros(self.glove_embeddings.embedding_dim, dtype=np.float32)
        
        # Aggregate vectors
        vectors = np.array(vectors)
        
        if self.aggregation_method == 'mean':
            return np.mean(vectors, axis=0).astype(np.float32)
        elif self.aggregation_method == 'sum':
            return np.sum(vectors, axis=0).astype(np.float32)
        elif self.aggregation_method == 'max':
            return np.max(vectors, axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def tweets_to_vectors(self, tweets: List[str]) -> np.ndarray:
        """
        Convert a list of tweets to vector representations.
        
        Args:
            tweets (List[str]): List of tweet texts
            
        Returns:
            np.ndarray: Matrix of tweet vectors (n_tweets, embedding_dim)
        """
        vectors = []
        for tweet in tweets:
            vector = self.tweet_to_vector(tweet)
            vectors.append(vector)
        
        return np.array(vectors)
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vectorization process.
        
        Returns:
            Dict: Dictionary containing vectorization statistics
        """
        oov_rate = self.oov_words / max(self.total_words, 1)
        empty_rate = self.empty_tweets / max(self.total_tweets, 1)
        
        return {
            'total_tweets': self.total_tweets,
            'empty_tweets': self.empty_tweets,
            'empty_rate': empty_rate,
            'total_words': self.total_words,
            'oov_words': self.oov_words,
            'oov_rate': oov_rate,
            'aggregation_method': self.aggregation_method
        }


def load_tweet_data(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Load tweet data from JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        Tuple[List[str], List[str]]: Tuple of (texts, labels)
    """
    texts = []
    labels = []
    
    logger.info(f"Loading data from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                texts.append(data['text'])
                labels.append(data['label'])
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error parsing line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(texts)} tweets with {len(set(labels))} unique labels")
    return texts, labels


def load_and_prepare_data(train_path: str, 
                         test_path: str,
                         glove_embeddings: GloVeEmbeddings,
                         preprocessor_config: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load and prepare training and test data.
    
    Args:
        train_path (str): Path to training data
        test_path (str): Path to test data
        glove_embeddings (GloVeEmbeddings): Loaded GloVe embeddings
        preprocessor_config (Dict): Configuration for preprocessor
        
    Returns:
        Tuple containing X_train, X_test, y_train, y_test, train_texts, test_texts
    """
    # Load data
    train_texts, train_labels = load_tweet_data(train_path)
    test_texts, test_labels = load_tweet_data(test_path)
    
    # Initialize preprocessor and vectorizer
    preprocessor_config = preprocessor_config or {}
    preprocessor = TweetPreprocessor(**preprocessor_config)
    vectorizer = TweetVectorizer(glove_embeddings, preprocessor)
    
    # Convert tweets to vectors
    logger.info("Converting training tweets to vectors...")
    X_train = vectorizer.tweets_to_vectors(train_texts)
    
    logger.info("Converting test tweets to vectors...")
    X_test = vectorizer.tweets_to_vectors(test_texts)
    
    # Convert labels to numpy arrays
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    
    # Log statistics
    stats = vectorizer.get_stats()
    logger.info(f"Vectorization statistics: {stats}")
    
    return X_train, X_test, y_train, y_test, train_texts, test_texts


if __name__ == "__main__":
    # Example usage
    from embeddings import GloVeEmbeddings
    
    # Load embeddings
    glove = GloVeEmbeddings("embeddings/glove.6B.100d.txt")
    word_vectors = glove.load_embeddings()
    
    # Test preprocessing
    preprocessor = TweetPreprocessor()
    test_tweet = "@airline your flight was delayed again! #frustrated http://example.com"
    tokens = preprocessor.tokenize_and_preprocess(test_tweet)
    print(f"Original: {test_tweet}")
    print(f"Tokens: {tokens}")
    
    # Test vectorization
    vectorizer = TweetVectorizer(glove, preprocessor)
    vector = vectorizer.tweet_to_vector(test_tweet)
    print(f"Vector shape: {vector.shape}")
    print(f"Vector (first 10 dims): {vector[:10]}")
    
    # Print stats
    stats = vectorizer.get_stats()
    print(f"Vectorization stats: {stats}") 