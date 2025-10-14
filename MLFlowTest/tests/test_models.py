"""Tests for ML models."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import LSTMModel, RandomForestModel, XGBoostModel, EnsembleModel

class TestModels:
    """Test cases for ML models."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples, 1)
        
        return X, y
    
    @pytest.fixture
    def sample_sequence_data(self):
        """Create sample sequence data for LSTM testing."""
        np.random.seed(42)
        n_samples = 50
        sequence_length = 20
        n_features = 5
        
        X = np.random.randn(n_samples, sequence_length, n_features)
        y = np.random.randn(n_samples, 1)
        
        return X, y
    
    def test_random_forest_model(self, sample_data):
        """Test Random Forest model."""
        X, y = sample_data
        
        config = {
            'n_estimators': 10,
            'max_depth': 5,
            'random_state': 42
        }
        
        model = RandomForestModel(config)
        
        # Test training
        results = model.train(X, y)
        assert model.is_trained
        assert 'train_score' in results
        
        # Test prediction
        predictions = model.predict(X[:10])
        assert predictions.shape[0] == 10
        
        # Test evaluation
        metrics = model.evaluate(X[:20], y[:20])
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'R2' in metrics
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0
    
    def test_xgboost_model(self, sample_data):
        """Test XGBoost model."""
        X, y = sample_data
        
        config = {
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        model = XGBoostModel(config)
        
        # Test training
        results = model.train(X, y)
        assert model.is_trained
        assert 'train_score' in results
        
        # Test prediction
        predictions = model.predict(X[:10])
        assert predictions.shape[0] == 10
        
        # Test evaluation
        metrics = model.evaluate(X[:20], y[:20])
        assert 'MAE' in metrics
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
    
    def test_lstm_model(self, sample_sequence_data):
        """Test LSTM model."""
        X, y = sample_sequence_data
        
        config = {
            'sequence_length': 20,
            'hidden_units': [32, 16],
            'dropout': 0.2,
            'epochs': 2,  # Small for testing
            'batch_size': 16
        }
        
        model = LSTMModel(config)
        
        # Test model building
        input_shape = (X.shape[1], X.shape[2])
        output_shape = y.shape[1]
        model.build_model(input_shape, output_shape)
        
        # Test training
        results = model.train(X, y)
        assert model.is_trained
        assert 'history' in results
        
        # Test prediction
        predictions = model.predict(X[:5])
        assert predictions.shape[0] == 5
        
        # Test evaluation
        metrics = model.evaluate(X[:10], y[:10])
        assert 'MAE' in metrics
    
    def test_ensemble_model(self, sample_data):
        """Test Ensemble model."""
        X, y = sample_data
        
        config = {'method': 'weighted_average'}
        ensemble = EnsembleModel(config)
        
        # Add base models
        rf_config = {'n_estimators': 5, 'random_state': 42}
        xgb_config = {'n_estimators': 5, 'random_state': 42}
        
        ensemble.add_model('rf', RandomForestModel(rf_config))
        ensemble.add_model('xgb', XGBoostModel(xgb_config))
        
        # Test training
        results = ensemble.train(X, y)
        assert ensemble.is_trained
        assert 'base_models' in results
        
        # Test prediction
        predictions = ensemble.predict(X[:10])
        assert predictions.shape[0] == 10
        
        # Test individual predictions
        individual_preds = ensemble.predict_with_individual_models(X[:5])
        assert 'ensemble' in individual_preds
        assert 'rf' in individual_preds
        assert 'xgb' in individual_preds
        
        # Test model weights
        weights = ensemble.get_model_weights()
        assert isinstance(weights, dict)
        assert len(weights) == 2
    
    def test_model_save_load(self, sample_data, tmp_path):
        """Test model save and load functionality."""
        X, y = sample_data
        
        config = {
            'n_estimators': 5,
            'random_state': 42
        }
        
        # Train model
        model = RandomForestModel(config)
        model.train(X, y)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        model.save_model(str(model_path))
        assert model_path.exists()
        
        # Load model
        new_model = RandomForestModel({})
        new_model.load_model(str(model_path))
        
        assert new_model.is_trained
        
        # Test predictions are the same
        pred1 = model.predict(X[:5])
        pred2 = new_model.predict(X[:5])
        
        np.testing.assert_array_almost_equal(pred1, pred2)

if __name__ == "__main__":
    pytest.main([__file__])
