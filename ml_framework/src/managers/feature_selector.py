"""
FeatureSelector - Track feature importance and select optimal features.
Performs feature selection BEFORE training models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Literal
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
import warnings


class FeatureSelector:
    """
    Manages feature importance tracking and feature selection.
    
    Features:
    - Multiple selection methods (correlation, mutual info, tree-based, RFE)
    - Feature importance visualization
    - Automatic feature dropping based on importance
    - Save/load selected features
    - Integration with training pipeline
    """
    
    def __init__(self, method: Literal['correlation', 'mutual_info', 'tree', 'rfe', 'lasso'] = 'tree'):
        """
        Initialize FeatureSelector.
        
        Args:
            method: Feature selection method
                - 'correlation': Remove highly correlated features
                - 'mutual_info': Mutual information with target
                - 'tree': Tree-based feature importance
                - 'rfe': Recursive Feature Elimination
                - 'lasso': L1-based feature selection
        """
        self.method = method
        self.selected_features = None
        self.feature_importance = None
        self.dropped_features = None
        self.selector = None
        self.is_fitted = False
        
    def fit(self,
           X: pd.DataFrame,
           y: pd.Series,
           n_features: Optional[int] = None,
           threshold: Optional[float] = None,
           correlation_threshold: float = 0.95) -> 'FeatureSelector':
        """
        Fit feature selector and identify important features.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select (None = auto)
            threshold: Importance threshold for selection (None = auto)
            correlation_threshold: Threshold for correlation-based removal
            
        Returns:
            Self
        """
        print("\n" + "="*70)
        print(f"FEATURE SELECTION - Method: {self.method.upper()}")
        print("="*70)
        
        print(f"\nInitial features: {X.shape[1]}")
        
        if self.method == 'correlation':
            self._fit_correlation(X, y, correlation_threshold)
        elif self.method == 'mutual_info':
            self._fit_mutual_info(X, y, n_features, threshold)
        elif self.method == 'tree':
            self._fit_tree_based(X, y, n_features, threshold)
        elif self.method == 'rfe':
            self._fit_rfe(X, y, n_features)
        elif self.method == 'lasso':
            self._fit_lasso(X, y, threshold)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted = True
        
        print(f"\nSelected features: {len(self.selected_features)}")
        print(f"Dropped features: {len(self.dropped_features)}")
        print(f"Reduction: {len(self.dropped_features) / X.shape[1] * 100:.1f}%")
        print("="*70 + "\n")
        
        return self
    
    def _fit_correlation(self, X: pd.DataFrame, y: pd.Series, threshold: float):
        """Remove highly correlated features."""
        print(f"\nRemoving features with correlation > {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        self.selected_features = [col for col in X.columns if col not in to_drop]
        self.dropped_features = to_drop
        
        # Store correlation as importance
        self.feature_importance = pd.Series(
            [corr_matrix[col].mean() for col in self.selected_features],
            index=self.selected_features
        ).sort_values(ascending=False)
        
        print(f"  Dropped {len(to_drop)} highly correlated features")
    
    def _fit_mutual_info(self, X: pd.DataFrame, y: pd.Series, n_features: Optional[int], threshold: Optional[float]):
        """Select features based on mutual information."""
        print(f"\nCalculating mutual information scores...")
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        self.feature_importance = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        # Select features
        if n_features is not None:
            self.selected_features = self.feature_importance.nlargest(n_features).index.tolist()
        elif threshold is not None:
            self.selected_features = self.feature_importance[self.feature_importance > threshold].index.tolist()
        else:
            # Auto: select features with MI > median
            median_mi = self.feature_importance.median()
            self.selected_features = self.feature_importance[self.feature_importance > median_mi].index.tolist()
        
        self.dropped_features = [col for col in X.columns if col not in self.selected_features]
        
        print(f"  Mutual information scores calculated")
    
    def _fit_tree_based(self, X: pd.DataFrame, y: pd.Series, n_features: Optional[int], threshold: Optional[float]):
        """Select features based on tree importance."""
        print(f"\nTraining Random Forest for feature importance...")
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importance
        self.feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        # Select features
        if n_features is not None:
            self.selected_features = self.feature_importance.nlargest(n_features).index.tolist()
        elif threshold is not None:
            self.selected_features = self.feature_importance[self.feature_importance > threshold].index.tolist()
        else:
            # Auto: select features with importance > mean
            mean_importance = self.feature_importance.mean()
            self.selected_features = self.feature_importance[self.feature_importance > mean_importance].index.tolist()
        
        self.dropped_features = [col for col in X.columns if col not in self.selected_features]
        self.selector = rf
        
        print(f"  Feature importance calculated")
    
    def _fit_rfe(self, X: pd.DataFrame, y: pd.Series, n_features: Optional[int]):
        """Select features using Recursive Feature Elimination."""
        print(f"\nPerforming Recursive Feature Elimination...")
        
        if n_features is None:
            n_features = max(5, X.shape[1] // 2)  # Select half by default
        
        # Use Random Forest as estimator
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        # Perform RFE
        self.selector = RFE(estimator, n_features_to_select=n_features, step=1)
        self.selector.fit(X, y)
        
        # Get selected features
        self.selected_features = X.columns[self.selector.support_].tolist()
        self.dropped_features = X.columns[~self.selector.support_].tolist()
        
        # Get ranking as importance (lower is better, so invert)
        ranking = self.selector.ranking_
        self.feature_importance = pd.Series(
            1.0 / ranking,
            index=X.columns
        ).sort_values(ascending=False)
        
        print(f"  RFE completed")
    
    def _fit_lasso(self, X: pd.DataFrame, y: pd.Series, threshold: Optional[float]):
        """Select features using L1-based selection."""
        print(f"\nPerforming L1-based feature selection...")
        
        from sklearn.linear_model import LogisticRegression
        
        # Train L1-regularized model
        lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
        
        if threshold is None:
            threshold = 'mean'
        
        self.selector = SelectFromModel(lasso, threshold=threshold)
        self.selector.fit(X, y)
        
        # Get selected features
        self.selected_features = X.columns[self.selector.get_support()].tolist()
        self.dropped_features = X.columns[~self.selector.get_support()].tolist()
        
        # Get coefficients as importance
        lasso.fit(X, y)
        self.feature_importance = pd.Series(
            np.abs(lasso.coef_).mean(axis=0) if len(lasso.coef_.shape) > 1 else np.abs(lasso.coef_),
            index=X.columns
        ).sort_values(ascending=False)
        
        print(f"  L1-based selection completed")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting only important features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with selected features only
        """
        if not self.is_fitted:
            raise ValueError("FeatureSelector is not fitted. Call fit() first.")
        
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            **kwargs: Arguments for fit()
            
        Returns:
            Transformed DataFrame
        """
        self.fit(X, y, **kwargs)
        return self.transform(X)
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.Series:
        """
        Get feature importance scores.
        
        Args:
            top_n: Return only top N features (None = all)
            
        Returns:
            Series with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available")
        
        if top_n is not None:
            return self.feature_importance.head(top_n)
        return self.feature_importance
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        if not self.is_fitted:
            raise ValueError("FeatureSelector is not fitted")
        return self.selected_features
    
    def get_dropped_features(self) -> List[str]:
        """Get list of dropped feature names."""
        if not self.is_fitted:
            raise ValueError("FeatureSelector is not fitted")
        return self.dropped_features
    
    def plot_importance(self, top_n: int = 20, save_path: Optional[Path] = None):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            save_path: Path to save plot (None = display only)
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available")
        
        import matplotlib.pyplot as plt
        
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 6))
        top_features.plot(kind='barh')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance ({self.method})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"✓ Saved feature importance plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save(self, save_dir: Path):
        """
        Save feature selector state.
        
        Args:
            save_dir: Directory to save selector
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted FeatureSelector")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        selector_path = save_dir / "feature_selector.joblib"
        
        # Save selector data
        selector_data = {
            'method': self.method,
            'selected_features': self.selected_features,
            'dropped_features': self.dropped_features,
            'feature_importance': self.feature_importance,
            'selector': self.selector,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(selector_data, selector_path)
        print(f"✓ Saved feature selector to {selector_path}")
    
    @classmethod
    def load(cls, load_dir: Path) -> 'FeatureSelector':
        """
        Load feature selector from file.
        
        Args:
            load_dir: Directory containing feature_selector.joblib
            
        Returns:
            FeatureSelector instance
        """
        load_dir = Path(load_dir)
        selector_path = load_dir / "feature_selector.joblib"
        
        if not selector_path.exists():
            raise FileNotFoundError(f"Feature selector file not found: {selector_path}")
        
        # Load selector data
        selector_data = joblib.load(selector_path)
        
        # Create instance
        instance = cls(method=selector_data['method'])
        instance.selected_features = selector_data['selected_features']
        instance.dropped_features = selector_data['dropped_features']
        instance.feature_importance = selector_data['feature_importance']
        instance.selector = selector_data['selector']
        instance.is_fitted = selector_data['is_fitted']
        
        print(f"✓ Loaded feature selector from {selector_path}")
        
        return instance
    
    def print_summary(self):
        """Print feature selection summary."""
        if not self.is_fitted:
            print("FeatureSelector is not fitted")
            return
        
        print("\n" + "="*70)
        print("FEATURE SELECTION SUMMARY")
        print("="*70)
        
        print(f"\nMethod: {self.method}")
        print(f"Selected features: {len(self.selected_features)}")
        print(f"Dropped features: {len(self.dropped_features)}")
        
        if self.feature_importance is not None:
            print(f"\nTop 10 Most Important Features:")
            for i, (feat, imp) in enumerate(self.feature_importance.head(10).items(), 1):
                print(f"  {i:2d}. {feat:<30s} {imp:.6f}")
        
        if len(self.dropped_features) > 0:
            print(f"\nDropped features ({len(self.dropped_features)}):")
            for feat in self.dropped_features[:10]:
                print(f"  - {feat}")
            if len(self.dropped_features) > 10:
                print(f"  ... and {len(self.dropped_features) - 10} more")
        
        print("="*70 + "\n")
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        n_features = len(self.selected_features) if self.selected_features else "unknown"
        return f"FeatureSelector(method='{self.method}', {status}, features={n_features})"
    
    # =========================================================================
    # JUPYTER-FRIENDLY METHODS
    # =========================================================================
    
    def display_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Display feature importance as a styled DataFrame (Jupyter-friendly).
        
        Args:
            top_n: Number of top features to display
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            print("No feature importance available. Call fit() first.")
            return pd.DataFrame()
        
        df = self.feature_importance.head(top_n).reset_index()
        df.columns = ['Feature', 'Importance']
        
        # Check if in Jupyter
        try:
            from IPython import get_ipython
            shell = get_ipython()
            if shell and 'ZMQInteractiveShell' in shell.__class__.__name__:
                from IPython.display import display
                styled = df.style.bar(subset=['Importance'], color='steelblue')
                display(styled)
        except (ImportError, NameError):
            pass
        
        return df
    
    def display_summary(self) -> pd.DataFrame:
        """
        Display feature selection summary as DataFrame (Jupyter-friendly).
        
        Returns:
            DataFrame with selection summary
        """
        if not self.is_fitted:
            print("FeatureSelector is not fitted")
            return pd.DataFrame()
        
        summary_data = {
            'Metric': ['Method', 'Total Features', 'Selected', 'Dropped'],
            'Value': [
                self.method,
                len(self.selected_features) + len(self.dropped_features),
                len(self.selected_features),
                len(self.dropped_features)
            ]
        }
        df = pd.DataFrame(summary_data)
        
        # Check if in Jupyter
        try:
            from IPython import get_ipython
            shell = get_ipython()
            if shell and 'ZMQInteractiveShell' in shell.__class__.__name__:
                from IPython.display import display
                display(df)
        except (ImportError, NameError):
            pass
        
        return df
