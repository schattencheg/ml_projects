"""
ScalerManager - Manages data scaling with save/load capabilities.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Optional, List, Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class ScalerManager:
    """
    Manages data scaling operations.
    
    Features:
    - Multiple scaler types (Standard, MinMax, Robust)
    - Scale only float fields or selected fields
    - Save/load scaler state
    - Fit on train, transform on train/test/val
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize ScalerManager.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
        
        # Create scaler instance
        self._create_scaler()
    
    def _create_scaler(self):
        """Create scaler instance based on type."""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}. "
                           f"Use 'standard', 'minmax', or 'robust'")
    
    def fit(self, 
            data: Union[np.ndarray, pd.DataFrame],
            feature_cols: Optional[List[str]] = None,
            only_float: bool = True) -> 'ScalerManager':
        """
        Fit the scaler on training data.
        
        Args:
            data: Training data (DataFrame or array)
            feature_cols: List of columns to scale (None = all numeric)
            only_float: If True, scale only float columns (ignored if feature_cols provided)
            
        Returns:
            Self
        """
        if isinstance(data, pd.DataFrame):
            # Select columns to scale
            if feature_cols is not None:
                self.feature_names = feature_cols
                X = data[feature_cols].values
            elif only_float:
                # Select only float columns
                float_cols = data.select_dtypes(include=['float64', 'float32']).columns.tolist()
                self.feature_names = float_cols
                X = data[float_cols].values
            else:
                # Select all numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                self.feature_names = numeric_cols
                X = data[numeric_cols].values
        else:
            X = data
            self.feature_names = None
        
        # Fit scaler
        self.scaler.fit(X)
        self.is_fitted = True
        
        print(f"✓ Scaler fitted on {X.shape[1]} features")
        if self.feature_names:
            print(f"  Features: {', '.join(self.feature_names[:5])}" + 
                  (f" ... (+{len(self.feature_names)-5} more)" if len(self.feature_names) > 5 else ""))
        
        return self
    
    def transform(self, 
                  data: Union[np.ndarray, pd.DataFrame],
                  inplace: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform data using fitted scaler.
        
        Args:
            data: Data to transform
            inplace: If True and data is DataFrame, modify in place
            
        Returns:
            Scaled data (same type as input)
        """
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted. Call fit() first.")
        
        if isinstance(data, pd.DataFrame):
            if self.feature_names is None:
                raise ValueError("Cannot transform DataFrame without feature names. "
                               "Fit with DataFrame first.")
            
            if inplace:
                data[self.feature_names] = self.scaler.transform(data[self.feature_names])
                return data
            else:
                data_copy = data.copy()
                data_copy[self.feature_names] = self.scaler.transform(data[self.feature_names])
                return data_copy
        else:
            return self.scaler.transform(data)
    
    def fit_transform(self,
                     data: Union[np.ndarray, pd.DataFrame],
                     feature_cols: Optional[List[str]] = None,
                     only_float: bool = True,
                     inplace: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit scaler and transform data in one step.
        
        Args:
            data: Training data
            feature_cols: List of columns to scale
            only_float: If True, scale only float columns
            inplace: If True and data is DataFrame, modify in place
            
        Returns:
            Scaled data
        """
        self.fit(data, feature_cols, only_float)
        return self.transform(data, inplace)
    
    def inverse_transform(self,
                         data: Union[np.ndarray, pd.DataFrame],
                         inplace: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            data: Scaled data
            inplace: If True and data is DataFrame, modify in place
            
        Returns:
            Data in original scale
        """
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted. Call fit() first.")
        
        if isinstance(data, pd.DataFrame):
            if self.feature_names is None:
                raise ValueError("Cannot inverse transform DataFrame without feature names.")
            
            if inplace:
                data[self.feature_names] = self.scaler.inverse_transform(data[self.feature_names])
                return data
            else:
                data_copy = data.copy()
                data_copy[self.feature_names] = self.scaler.inverse_transform(data[self.feature_names])
                return data_copy
        else:
            return self.scaler.inverse_transform(data)
    
    def save(self, save_dir: Union[str, Path]):
        """
        Save scaler to file.
        
        Args:
            save_dir: Directory to save scaler (will save as scaler.joblib)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted scaler")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        scaler_path = save_dir / "scaler.joblib"
        
        # Save scaler and metadata
        scaler_data = {
            'scaler': self.scaler,
            'scaler_type': self.scaler_type,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(scaler_data, scaler_path)
        print(f"✓ Saved scaler to {scaler_path}")
    
    @classmethod
    def load(cls, load_dir: Union[str, Path]) -> 'ScalerManager':
        """
        Load scaler from file.
        
        Args:
            load_dir: Directory containing scaler.joblib
            
        Returns:
            ScalerManager instance
        """
        load_dir = Path(load_dir)
        scaler_path = load_dir / "scaler.joblib"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        # Load scaler data
        scaler_data = joblib.load(scaler_path)
        
        # Create instance
        instance = cls(scaler_type=scaler_data['scaler_type'])
        instance.scaler = scaler_data['scaler']
        instance.feature_names = scaler_data['feature_names']
        instance.is_fitted = scaler_data['is_fitted']
        
        print(f"✓ Loaded scaler from {scaler_path}")
        
        return instance
    
    def get_feature_names(self) -> Optional[List[str]]:
        """Get list of feature names being scaled."""
        return self.feature_names
    
    def get_params(self) -> dict:
        """Get scaler parameters."""
        if self.scaler is None:
            return {}
        return {
            'scaler_type': self.scaler_type,
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names) if self.feature_names else None
        }
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        n_features = len(self.feature_names) if self.feature_names else "unknown"
        return f"ScalerManager(type='{self.scaler_type}', {status}, features={n_features})"
