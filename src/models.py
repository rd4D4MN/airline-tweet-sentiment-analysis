"""
Model Training Module

This module provides functionality for training sentiment classification models
using various CPU-friendly algorithms with optional hyperparameter tuning.
"""

import logging
import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentModelTrainer:
    """
    A class to handle training of sentiment classification models.
    """
    
    def __init__(self, model_type: str = 'logistic_regression', random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            model_type (str): Type of model to use ('logistic_regression', 'svm', 'random_forest', 'naive_bayes')
            random_state (int): Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.label_encoder = LabelEncoder()
        self.training_history = {}
        
        # Initialize the model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on the specified type."""
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=2000,
                class_weight='balanced'  # Handle class imbalance
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                random_state=self.random_state,
                class_weight='balanced',
                probability=True  # Enable probability estimates
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_estimators=100
            )
        elif self.model_type == 'naive_bayes':
            self.model = GaussianNB()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def get_hyperparameter_grid(self) -> Dict:
        """
        Get hyperparameter grid for the current model type.
        
        Returns:
            Dict: Hyperparameter grid for GridSearchCV
        """
        if self.model_type == 'logistic_regression':
            return {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif self.model_type == 'svm':
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'naive_bayes':
            return {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
        else:
            return {}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              tune_hyperparameters: bool = False, cv_folds: int = 5) -> Dict:
        """
        Train the sentiment classification model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            tune_hyperparameters (bool): Whether to perform hyperparameter tuning
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict: Training results and statistics
        """
        logger.info(f"Training {self.model_type} model...")
        start_time = time.time()
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Store original label classes
        self.classes_ = self.label_encoder.classes_
        
        training_results = {
            'model_type': self.model_type,
            'training_samples': len(X_train),
            'feature_dim': X_train.shape[1],
            'classes': self.classes_.tolist(),
            'hyperparameter_tuning': tune_hyperparameters
        }
        
        if tune_hyperparameters:
            logger.info("Performing hyperparameter tuning...")
            param_grid = self.get_hyperparameter_grid()
            
            if param_grid:
                # Perform grid search
                grid_search = GridSearchCV(
                    self.model, 
                    param_grid, 
                    cv=cv_folds, 
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train_encoded)
                
                # Update model with best parameters
                self.model = grid_search.best_estimator_
                
                training_results.update({
                    'best_params': grid_search.best_params_,
                    'best_cv_score': grid_search.best_score_,
                    'cv_results': {
                        'mean_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                        'std_scores': grid_search.cv_results_['std_test_score'].tolist()
                    }
                })
                
                logger.info(f"Best parameters: {grid_search.best_params_}")
                logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            else:
                logger.warning("No hyperparameter grid defined for this model type")
                self.model.fit(X_train, y_train_encoded)
        else:
            # Train with default parameters
            self.model.fit(X_train, y_train_encoded)
        
        # Perform cross-validation with final model
        cv_scores = cross_val_score(self.model, X_train, y_train_encoded, 
                                  cv=cv_folds, scoring='f1_weighted')
        
        training_time = time.time() - start_time
        
        training_results.update({
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time_seconds': training_time
        })
        
        # Store training history
        self.training_history = training_results
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Cross-validation F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return training_results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        predictions_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability predictions")
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'classes': self.classes_,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.classes_ = model_data['classes']
        self.training_history = model_data.get('training_history', {})
        
        logger.info(f"Model loaded from: {filepath}")
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if available.
        
        Returns:
            Optional[np.ndarray]: Feature importance scores
        """
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute values of coefficients
            return np.abs(self.model.coef_).mean(axis=0)
        else:
            return None


class ModelComparison:
    """
    A class to compare multiple sentiment classification models.
    """
    
    def __init__(self, model_types: List[str] = None, random_state: int = 42):
        """
        Initialize the model comparison.
        
        Args:
            model_types (List[str]): List of model types to compare
            random_state (int): Random state for reproducibility
        """
        self.model_types = model_types or ['logistic_regression', 'svm', 'random_forest', 'naive_bayes']
        self.random_state = random_state
        self.trained_models = {}
        self.comparison_results = {}
    
    def compare_models(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      tune_hyperparameters: bool = False) -> Dict:
        """
        Compare multiple models on the given dataset.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            tune_hyperparameters (bool): Whether to tune hyperparameters
            
        Returns:
            Dict: Comparison results for all models
        """
        logger.info(f"Comparing {len(self.model_types)} models...")
        
        for model_type in self.model_types:
            logger.info(f"Training {model_type}...")
            
            try:
                # Initialize and train model
                trainer = SentimentModelTrainer(model_type, self.random_state)
                training_results = trainer.train(X_train, y_train, tune_hyperparameters)
                
                # Make predictions
                y_pred = trainer.predict(X_test)
                
                # Calculate metrics (will be implemented in evaluation module)
                try:
                    from .evaluation import SentimentEvaluator
                except ImportError:
                    from evaluation import SentimentEvaluator
                evaluator = SentimentEvaluator()
                evaluation_results = evaluator.evaluate(y_test, y_pred, trainer.classes_)
                
                # Store results
                self.trained_models[model_type] = trainer
                self.comparison_results[model_type] = {
                    'training_results': training_results,
                    'evaluation_results': evaluation_results
                }
                
                logger.info(f"{model_type} - F1 Score: {evaluation_results['weighted_avg']['f1-score']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                continue
        
        # Determine best model
        if self.comparison_results:
            best_model = max(self.comparison_results.keys(), 
                           key=lambda k: self.comparison_results[k]['evaluation_results']['weighted_avg']['f1-score'])
            
            self.comparison_results['best_model'] = {
                'model_type': best_model,
                'f1_score': self.comparison_results[best_model]['evaluation_results']['weighted_avg']['f1-score']
            }
            
            logger.info(f"Best model: {best_model} (F1: {self.comparison_results['best_model']['f1_score']:.4f})")
        
        return self.comparison_results
    
    def get_best_model(self) -> Optional[SentimentModelTrainer]:
        """
        Get the best performing model.
        
        Returns:
            Optional[SentimentModelTrainer]: Best model trainer
        """
        if 'best_model' in self.comparison_results:
            best_model_type = self.comparison_results['best_model']['model_type']
            return self.trained_models.get(best_model_type)
        return None
    
    def save_comparison_results(self, filepath: str):
        """
        Save comparison results to disk.
        
        Args:
            filepath (str): Path to save results
        """
        import json
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy types for JSON serialization
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
        
        with open(filepath, 'w') as f:
            json.dump(recursive_convert(self.comparison_results), f, indent=2)
        
        logger.info(f"Comparison results saved to: {filepath}")


if __name__ == "__main__":
    # Example usage
    try:
        from .data_processing import load_and_prepare_data
        from .embeddings import GloVeEmbeddings
    except ImportError:
        from data_processing import load_and_prepare_data
        from embeddings import GloVeEmbeddings
    
    # Load embeddings and data
    glove = GloVeEmbeddings("embeddings/glove.6B.100d.txt")
    glove.load_embeddings()
    
    X_train, X_test, y_train, y_test, _, _ = load_and_prepare_data(
        "data/tweet_sentiment.train.jsonl",
        "data/tweet_sentiment.test.jsonl",
        glove
    )
    
    # Train a single model
    trainer = SentimentModelTrainer('logistic_regression')
    results = trainer.train(X_train, y_train, tune_hyperparameters=True)
    
    # Make predictions
    predictions = trainer.predict(X_test)
    
    # Save model
    trainer.save_model("models/logistic_regression_model.joblib")
    
    print(f"Training completed. CV F1 score: {results['cv_mean']:.4f}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}") 