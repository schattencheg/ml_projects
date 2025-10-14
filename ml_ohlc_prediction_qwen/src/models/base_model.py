from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict


class BaseModel(ABC):
    """
    Abstract base class for all models
    """
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'BaseModel':
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments for training
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> 'BaseModel':
        """
        Load a trained model from disk
        
        Args:
            path: Path to load the model from
            
        Returns:
            The loaded model instance
        """
        pass