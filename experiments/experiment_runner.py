"""
Systematic Experiment Runner for Sentiment Analysis Model Comparison

This module provides a framework for running systematic experiments
to compare different approaches and find the best performing model.
"""

import logging
import json
import os
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from datetime import datetime
import sys
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path properly
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import required modules
from src.embeddings import GloVeEmbeddings
from src.data_processing import load_tweet_data


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    experiment_id: str
    description: str
    model_type: str
    aggregation_method: str = 'mean'
    preprocessor_config: Dict = None
    hyperparameter_tuning: bool = True
    custom_params: Dict = None
    
    def __post_init__(self):
        if self.preprocessor_config is None:
            self.preprocessor_config = {}
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    experiment_id: str
    config: ExperimentConfig
    training_time: float
    cv_f1_mean: float
    cv_f1_std: float
    test_f1: float
    test_accuracy: float
    per_class_f1: Dict[str, float]
    confusion_matrix: List[List[int]]
    detailed_results: Dict
    timestamp: str
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ExperimentRunner:
    """
    Main class for running systematic experiments.
    """
    
    def __init__(self, results_dir: str = "experiments/results"):
        """
        Initialize the experiment runner.
        
        Args:
            results_dir (str): Directory to store experiment results
        """
        self.results_dir = results_dir
        self.experiments = []
        self.results = []
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Load data once for all experiments
        self.data = None
        self.glove_embeddings = None
        
    def setup_data(self, train_path: str, test_path: str, glove_path: str):
        """
        Set up data for experiments.
        
        Args:
            train_path (str): Path to training data
            test_path (str): Path to test data  
            glove_path (str): Path to GloVe embeddings
        """
        logger.info("Setting up data for experiments...")
        
        # Load GloVe embeddings
        self.glove_embeddings = GloVeEmbeddings(glove_path)
        self.glove_embeddings.load_embeddings()
        
        # Load raw data
        train_texts, train_labels = load_tweet_data(train_path)
        test_texts, test_labels = load_tweet_data(test_path)
        
        self.data = {
            'train_texts': train_texts,
            'train_labels': train_labels,
            'test_texts': test_texts,
            'test_labels': test_labels
        }
        
        logger.info(f"Data loaded: {len(train_texts)} train, {len(test_texts)} test")
    
    def add_experiment(self, config: ExperimentConfig):
        """Add an experiment configuration."""
        self.experiments.append(config)
        logger.info(f"Added experiment: {config.experiment_id}")
    
    def prepare_data_for_experiment(self, config: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for a specific experiment configuration.
        
        Args:
            config (ExperimentConfig): Experiment configuration
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        from src.data_processing import TweetPreprocessor, TweetVectorizer
        
        # Create preprocessor with experiment-specific config
        preprocessor = TweetPreprocessor(**config.preprocessor_config)
        
        # Create vectorizer with experiment-specific aggregation
        vectorizer = TweetVectorizer(
            self.glove_embeddings,
            preprocessor=preprocessor,
            aggregation_method=config.aggregation_method
        )
        
        # Convert to vectors
        X_train = vectorizer.tweets_to_vectors(self.data['train_texts'])
        X_test = vectorizer.tweets_to_vectors(self.data['test_texts'])
        
        # Apply feature scaling to handle different aggregation methods
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Get labels
        y_train = np.array(self.data['train_labels'])
        y_test = np.array(self.data['test_labels'])
        
        # Log vectorization stats
        stats = vectorizer.get_stats()
        logger.info(f"Vectorization stats: OOV rate: {stats['oov_rate']:.3f}, Empty rate: {stats['empty_rate']:.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            config (ExperimentConfig): Experiment configuration
            
        Returns:
            ExperimentResult: Results from the experiment
        """
        logger.info(f"Running experiment: {config.experiment_id}")
        start_time = time.time()
        
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data_for_experiment(config)
            
            # Import models
            from src.models import SentimentModelTrainer
            from src.evaluation import SentimentEvaluator
            
            # Create and configure model
            trainer = SentimentModelTrainer(config.model_type)
            
            # Apply custom parameters if any
            if config.custom_params:
                for param, value in config.custom_params.items():
                    if hasattr(trainer.model, param):
                        setattr(trainer.model, param, value)
            
            # Train model
            training_results = trainer.train(
                X_train, y_train, 
                tune_hyperparameters=config.hyperparameter_tuning
            )
            
            # Make predictions
            y_pred = trainer.predict(X_test)
            
            # Evaluate
            evaluator = SentimentEvaluator()
            evaluation_results = evaluator.evaluate(y_test, y_pred)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Extract key metrics
            cv_f1_mean = training_results['cv_mean']
            cv_f1_std = training_results['cv_std']
            test_f1 = evaluation_results['weighted_avg']['f1-score']
            test_accuracy = evaluation_results['accuracy']
            
            # Per-class F1 scores
            per_class_f1 = {
                class_name: metrics['f1-score'] 
                for class_name, metrics in evaluation_results['per_class'].items()
            }
            
            # Create result
            result = ExperimentResult(
                experiment_id=config.experiment_id,
                config=config,
                training_time=training_time,
                cv_f1_mean=cv_f1_mean,
                cv_f1_std=cv_f1_std,
                test_f1=test_f1,
                test_accuracy=test_accuracy,
                per_class_f1=per_class_f1,
                confusion_matrix=evaluation_results['confusion_matrix'],
                detailed_results={
                    'training_results': training_results,
                    'evaluation_results': evaluation_results
                },
                timestamp=datetime.now().isoformat()
            )
            
            # Save individual result
            self.save_experiment_result(result)
            
            logger.info(f"Experiment {config.experiment_id} completed: CV F1={cv_f1_mean:.4f}, Test F1={test_f1:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Experiment {config.experiment_id} failed: {e}")
            raise
    
    def run_all_experiments(self) -> List[ExperimentResult]:
        """
        Run all configured experiments.
        
        Returns:
            List[ExperimentResult]: Results from all experiments
        """
        logger.info(f"Running {len(self.experiments)} experiments...")
        
        self.results = []
        for config in self.experiments:
            try:
                result = self.run_single_experiment(config)
                self.results.append(result)
            except Exception as e:
                logger.error(f"Skipping failed experiment {config.experiment_id}: {e}")
                continue
        
        # Save comparison results
        self.save_comparison_results()
        
        return self.results
    
    def save_experiment_result(self, result: ExperimentResult):
        """Save individual experiment result."""
        filename = f"{result.experiment_id}_result.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert to serializable format
        result_dict = asdict(result)
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
    
    def save_comparison_results(self):
        """Save comparison of all experiments."""
        if not self.results:
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for result in self.results:
            row = {
                'experiment_id': result.experiment_id,
                'description': result.config.description,
                'model_type': result.config.model_type,
                'aggregation_method': result.config.aggregation_method,
                'cv_f1_mean': result.cv_f1_mean,
                'cv_f1_std': result.cv_f1_std,
                'test_f1': result.test_f1,
                'test_accuracy': result.test_accuracy,
                'training_time': result.training_time,
            }
            
            # Add per-class F1 scores
            for class_name, f1_score in result.per_class_f1.items():
                row[f'{class_name}_f1'] = f1_score
            
            comparison_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(comparison_data)
        
        # Sort by test F1 score (descending)
        df = df.sort_values('test_f1', ascending=False)
        
        # Save as CSV
        csv_path = os.path.join(self.results_dir, 'experiment_comparison.csv')
        df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = os.path.join(self.results_dir, 'experiment_comparison.json')
        comparison_dict = {
            'summary': df.to_dict('records'),
            'best_experiment': df.iloc[0].to_dict(),
            'total_experiments': len(self.results),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(json_path, 'w') as f:
            json.dump(comparison_dict, f, indent=2, default=str)
        
        logger.info(f"Comparison results saved to {self.results_dir}")
    
    def print_comparison_summary(self):
        """Print a summary of experiment results."""
        if not self.results:
            print("No experiment results available.")
            return
        
        # Sort by test F1 score
        sorted_results = sorted(self.results, key=lambda x: x.test_f1, reverse=True)
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPARISON SUMMARY")
        print("="*80)
        
        print(f"{'Rank':<4} {'Experiment ID':<20} {'Model':<15} {'Agg':<6} {'CV F1':<8} {'Test F1':<8} {'Accuracy':<8}")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results, 1):
            print(f"{i:<4} {result.experiment_id:<20} {result.config.model_type:<15} "
                  f"{result.config.aggregation_method:<6} {result.cv_f1_mean:<8.4f} "
                  f"{result.test_f1:<8.4f} {result.test_accuracy:<8.4f}")
        
        # Best experiment details
        best = sorted_results[0]
        print(f"\n🏆 BEST EXPERIMENT: {best.experiment_id}")
        print(f"   Description: {best.config.description}")
        print(f"   Test F1: {best.test_f1:.4f}")
        print(f"   Per-class F1 scores:")
        for class_name, f1_score in best.per_class_f1.items():
            print(f"     {class_name}: {f1_score:.4f}")
    
    def get_best_experiment(self) -> Optional[ExperimentResult]:
        """Get the best performing experiment."""
        if not self.results:
            return None
        
        return max(self.results, key=lambda x: x.test_f1)


def create_experiment_configs() -> List[ExperimentConfig]:
    """
    Create a comprehensive set of experiment configurations.
    
    Returns:
        List[ExperimentConfig]: List of experiment configurations
    """
    configs = []
    
    # Baseline: Current approach
    configs.append(ExperimentConfig(
        experiment_id="baseline",
        description="Baseline: Logistic Regression with mean aggregation",
        model_type="logistic_regression",
        aggregation_method="mean"
    ))
    
    # Aggregation method experiments
    configs.append(ExperimentConfig(
        experiment_id="agg_sum",
        description="Logistic Regression with sum aggregation",
        model_type="logistic_regression",
        aggregation_method="sum"
    ))
    
    configs.append(ExperimentConfig(
        experiment_id="agg_max",
        description="Logistic Regression with max aggregation", 
        model_type="logistic_regression",
        aggregation_method="max"
    ))
    
    # Different models
    configs.append(ExperimentConfig(
        experiment_id="random_forest",
        description="Random Forest with mean aggregation",
        model_type="random_forest",
        aggregation_method="mean"
    ))
    
    configs.append(ExperimentConfig(
        experiment_id="svm_rbf",
        description="SVM with RBF kernel",
        model_type="svm",
        aggregation_method="mean"
    ))
    
    configs.append(ExperimentConfig(
        experiment_id="naive_bayes",
        description="Naive Bayes with mean aggregation",
        model_type="naive_bayes",
        aggregation_method="mean"
    ))
    
    # Preprocessing variations
    configs.append(ExperimentConfig(
        experiment_id="no_stopwords",
        description="Logistic Regression without stopword removal",
        model_type="logistic_regression",
        aggregation_method="mean",
        preprocessor_config={"remove_stopwords": False}
    ))
    
    configs.append(ExperimentConfig(
        experiment_id="no_stemming",
        description="Logistic Regression without stemming",
        model_type="logistic_regression", 
        aggregation_method="mean",
        preprocessor_config={"use_stemming": False}
    ))
    
    # Combined best approaches
    configs.append(ExperimentConfig(
        experiment_id="rf_sum",
        description="Random Forest with sum aggregation",
        model_type="random_forest",
        aggregation_method="sum"
    ))
    
    configs.append(ExperimentConfig(
        experiment_id="svm_max",
        description="SVM with max aggregation",
        model_type="svm",
        aggregation_method="max"
    ))
    
    return configs


if __name__ == "__main__":
    # Example usage
    runner = ExperimentRunner()
    
    # Setup data (paths relative to project root)
    runner.setup_data(
        "../data/tweet_sentiment.train.jsonl",
        "../data/tweet_sentiment.test.jsonl", 
        "../embeddings/glove.6B.100d.txt"
    )
    
    # Add experiments
    configs = create_experiment_configs()
    for config in configs:
        runner.add_experiment(config)
    
    # Run all experiments
    results = runner.run_all_experiments()
    
    # Print summary
    runner.print_comparison_summary() 