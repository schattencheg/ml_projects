"""
Sklearn model wrappers - Linear/Logistic Regression and Random Forest.
"""

import numpy as np
from typing import Optional
from src.models_lib.base_model import BaseModel
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier


class LinearRegressionModel(BaseModel):
    """Linear regression for classification (rounds predictions to integers)."""
    
    def __init__(self, name: str = "LinearRegression", **params):
        super().__init__(name)
        self.params = {'n_jobs': -1, **params}
    
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        self.model = LinearRegression(**self.params)
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return np.round(self.model.predict(X)).astype(int)
    
    def get_coefficients(self) -> Optional[np.ndarray]:
        return self.model.coef_ if self.model else None


class LogisticRegressionModel(BaseModel):
    """Logistic regression classifier."""
    
    DEFAULT_PARAMS = {'max_iter': 1000, 'random_state': 42, 'n_jobs': -1}
    
    def __init__(self, name: str = "LogisticRegression", **params):
        super().__init__(name)
        self.params = {**self.DEFAULT_PARAMS, **params}
    
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        self.model = LogisticRegression(**self.params)
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.model.predict(X)
    
    def _predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_coefficients(self) -> Optional[np.ndarray]:
        return self.model.coef_ if self.model else None


class RandomForestModel(BaseModel):
    """Random Forest classifier."""
    
    DEFAULT_PARAMS = {'n_estimators': 100, 'max_depth': None, 'random_state': 42, 'n_jobs': -1}
    
    def __init__(self, name: str = "RandomForest", **params):
        super().__init__(name)
        self.params = {**self.DEFAULT_PARAMS, **params}
    
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.model.predict(X)
    
    def _predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        return self.model.feature_importances_ if self.model else None
