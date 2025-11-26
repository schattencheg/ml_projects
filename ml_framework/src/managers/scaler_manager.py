"""
ScalerManager - Feature scaling with save/load support.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Optional, List, Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

SCALERS = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}


class ScalerManager:
    """Manages feature scaling with StandardScaler, MinMaxScaler, or RobustScaler."""
    
    def __init__(self, scaler_type: str = 'standard'):
        if scaler_type not in SCALERS:
            raise ValueError(f"Unknown scaler: {scaler_type}. Use: {list(SCALERS.keys())}")
        self.scaler_type = scaler_type
        self.scaler = SCALERS[scaler_type]()
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame],
            feature_cols: Optional[List[str]] = None) -> 'ScalerManager':
        """Fit scaler on data. For DataFrame, scales specified or all numeric columns."""
        if isinstance(data, pd.DataFrame):
            self.feature_names = feature_cols or data.select_dtypes(include=[np.number]).columns.tolist()
            X = data[self.feature_names].values
        else:
            X = data
            self.feature_names = None
        
        self.scaler.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Transform data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted")
        
        if isinstance(data, pd.DataFrame):
            result = data.copy()
            result[self.feature_names] = self.scaler.transform(data[self.feature_names])
            return result
        return self.scaler.transform(data)
    
    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame],
                      feature_cols: Optional[List[str]] = None) -> Union[np.ndarray, pd.DataFrame]:
        """Fit and transform in one step."""
        return self.fit(data, feature_cols).transform(data)
    
    def inverse_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Inverse transform to original scale."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted")
        
        if isinstance(data, pd.DataFrame):
            result = data.copy()
            result[self.feature_names] = self.scaler.inverse_transform(data[self.feature_names])
            return result
        return self.scaler.inverse_transform(data)
    
    def save(self, save_dir: Union[str, Path]):
        """Save scaler to directory."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted scaler")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'scaler': self.scaler, 'scaler_type': self.scaler_type,
            'feature_names': self.feature_names, 'is_fitted': True
        }, save_dir / "scaler.joblib")
    
    @classmethod
    def load(cls, load_dir: Union[str, Path]) -> 'ScalerManager':
        """Load scaler from directory."""
        data = joblib.load(Path(load_dir) / "scaler.joblib")
        instance = cls(scaler_type=data['scaler_type'])
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.is_fitted = data['is_fitted']
        return instance
    
    def __repr__(self) -> str:
        n = len(self.feature_names) if self.feature_names else 0
        return f"ScalerManager({self.scaler_type}, fitted={self.is_fitted}, features={n})"
