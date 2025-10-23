"""
Model Training Module

Functions for training and evaluating ML models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from src.data_preparation import prepare_data, fit_scaler
from src.neural_models import (
    create_lstm_model, create_cnn_model, create_hybrid_lstm_cnn_model,
    KerasClassifierWrapper, TENSORFLOW_AVAILABLE
)
from src.model_loader import save_all_models
from src.utils import Utils

# Check for additional ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def get_model_configs():
    """
    Get configurations for all available models.
    
    Returns:
    --------
    models : dict
        Dictionary of model_name -> (model_instance, params_dict)
    """
    models = {
        # Traditional ML models
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"max_iter": 1000}
        )
    }
    '''
        "ridge_classifier": (
            RidgeClassifier(random_state=42),
            {}
        ),
        "naive_bayes": (
            GaussianNB(),
            {}
        ),
        "knn": (
            KNeighborsClassifier(n_neighbors=5),
            {"n_neighbors": 5}
        ),
        "decision_tree": (
            DecisionTreeClassifier(max_depth=10, random_state=42),
            {"max_depth": 10}
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            {"n_estimators": 100, "max_depth": 10}
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            {"n_estimators": 100, "max_depth": 5}
        ),
        "svm": (
            SVC(kernel='rbf', probability=True, random_state=42),
            {"kernel": "rbf"}
        ),
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models["xgboost"] = (
            xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
        )
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models["lightgbm"] = (
            lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
        )
    '''
    
    return models


def add_neural_network_models(models, input_shape, sequence_length=60):
    """
    Add neural network models to the models dictionary.
    
    Parameters:
    -----------
    models : dict
        Existing models dictionary
    input_shape : tuple
        Input shape for neural networks (sequence_length, features)
    sequence_length : int
        Sequence length for time series models
    
    Returns:
    --------
    models : dict
        Updated models dictionary with neural networks
    """
    if not TENSORFLOW_AVAILABLE:
        return models
    
    # Add LSTM
    models["lstm"] = (
        KerasClassifierWrapper(
            create_lstm_model,
            input_shape,
            sequence_length=sequence_length,
            dropout_rate=0.2
        ),
        {"sequence_length": sequence_length}
    )
    
    # Add CNN
    models["cnn"] = (
        KerasClassifierWrapper(
            create_cnn_model,
            input_shape,
            sequence_length=sequence_length,
            dropout_rate=0.2
        ),
        {"sequence_length": sequence_length}
    )
    
    # Add Hybrid LSTM-CNN
    models["lstm_cnn_hybrid"] = (
        KerasClassifierWrapper(
            create_hybrid_lstm_cnn_model,
            input_shape,
            sequence_length=sequence_length,
            dropout_rate=0.3
        ),
        {"sequence_length": sequence_length}
    )
    
    return models


def train_and_evaluate_model(model, model_name, X_train_scaled, y_train, X_val_scaled, y_val):
    """
    Train and evaluate a single model.
    
    Parameters:
    -----------
    model : sklearn-compatible model
        Model to train
    model_name : str
        Name of the model
    X_train_scaled : np.ndarray
        Scaled training features
    y_train : pd.Series
        Training labels
    X_val_scaled : np.ndarray
        Scaled validation features
    y_val : pd.Series
        Validation labels
    
    Returns:
    --------
    results : dict
        Dictionary with model performance metrics
    """
    Utils.print_color(f"\n{'='*80}", 'magenta')
    Utils.print_color(f"Model: {model_name.upper()}", 'magenta')
    Utils.print_color(f"{'='*80}", 'magenta')
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on validation set
    y_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    
    # Calculate ROC AUC
    try:
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_val_scaled)
        else:
            y_pred_proba = y_pred.astype(float)
        
        roc_auc = roc_auc_score(y_val, y_pred_proba)
    except Exception as e:
        print(f"  Warning: Could not calculate ROC AUC: {str(e)}")
        roc_auc = 0.0
    
    # Print results
    Utils.print_color(f"\nValidation Set Performance:", 'green')
    Utils.print_color(f"  Accuracy:  {accuracy:.4f}", 'green')
    Utils.print_color(f"  F1 Score:  {f1:.4f}", 'green')
    Utils.print_color(f"  Precision: {precision:.4f}", 'green')
    Utils.print_color(f"  Recall:    {recall:.4f}", 'green')
    Utils.print_color(f"  ROC AUC:   {roc_auc:.4f}", 'green')
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['No Increase', 'Increase'], zero_division=np.nan))
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'model': model
    }


def train(df_train: pd.DataFrame, target_bars: int = 45, target_pct: float = 3.0):
    """
    Train models on the provided training dataframe.
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Training dataframe with OHLCV data
    target_bars : int
        Number of bars to look ahead for target
    target_pct : float
        Percentage increase threshold for target
    
    Returns:
    --------
    models : dict
        Dictionary of trained models
    scaler : StandardScaler
        Fitted scaler for feature normalization
    results : dict
        Training results with metrics
    best_model_name : str
        Name of the best performing model
    """
    # Prepare data
    X, y = prepare_data(df_train, target_bars, target_pct)

    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Target balance: {y.value_counts(normalize=True)}")
    print()

    # Split data into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Get model configurations
    models = get_model_configs()
    
    # Add neural network models if TensorFlow is available
    if False and TENSORFLOW_AVAILABLE:
        sequence_length = 60
        input_shape = (sequence_length, X_train.shape[1])
        models = add_neural_network_models(models, input_shape, sequence_length)

    # Fit scaler on training data
    scaler, X_train_scaled = fit_scaler(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train and evaluate models
    print("="*80)
    print("TRAINING AND EVALUATING MODELS")
    print("="*80)

    best_model = None
    best_score = 0
    best_model_name = ""
    results = {}

    for model_name, model_data in models.items():
        model = model_data[0]
        
        # Train and evaluate
        result = train_and_evaluate_model(
            model, model_name, X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        results[model_name] = result
        
        # Track best model
        if result['accuracy'] > best_score:
            best_score = result['accuracy']
            best_model = model
            best_model_name = model_name

    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model_name.upper()} with accuracy: {best_score:.4f}")
    print(f"{'='*80}")

    # Save all models and scaler
    save_all_models(models, scaler, models_dir='models')
    
    # Also save the best model separately
    import os
    import joblib
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{best_model_name}_best.joblib'
    joblib.dump(best_model, model_path)
    print(f"\nBest model also saved separately to: {model_path}")

    return models, scaler, results, best_model_name


def test(models: dict, scaler, df_test: pd.DataFrame, target_bars: int = 45, target_pct: float = 3.0):
    """
    Test trained models on new data.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    scaler : StandardScaler
        Fitted scaler from training
    df_test : pd.DataFrame
        Test dataframe with OHLCV data
    target_bars : int
        Number of bars to look ahead for target (should match training)
    target_pct : float
        Percentage increase threshold for target (should match training)
    
    Returns:
    --------
    results_test : dict
        Dictionary with test metrics for each model
    """
    # Prepare test data using the same pipeline as training
    X_test, y_test = prepare_data(df_test, target_bars, target_pct)
    
    # Scale using the fitted scaler from training (DO NOT refit!)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n{'='*80}")
    print(f"TESTING ON HELD-OUT TEST DATA")
    print(f"{'='*80}")
    print(f"Test dataset shape: {X_test.shape}")
    print(f"Test target distribution:\n{y_test.value_counts()}")
    print()

    results_test = {}

    for model_name, model_data in models.items():
        print(f"\n{'='*80}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*80}")
        
        model = model_data[0]
        
        # Make predictions on test set
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, zero_division=np.nan)
        recall = recall_score(y_test, y_pred_test)
        
        # Calculate ROC AUC
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_pred_proba_test = model.decision_function(X_test_scaled)
            else:
                y_pred_proba_test = y_pred_test.astype(float)
            
            roc_auc = roc_auc_score(y_test, y_pred_proba_test)
        except Exception as e:
            print(f"  Warning: Could not calculate ROC AUC for {model_name}: {str(e)}")
            roc_auc = 0.0
        
        Utils.print_color(f"\nTest Set Performance:", 'green')
        Utils.print_color(f"  Accuracy:  {accuracy:.4f}", 'green')
        Utils.print_color(f"  F1 Score:  {f1:.4f}", 'green')
        Utils.print_color(f"  Precision: {precision:.4f}", 'green')
        Utils.print_color(f"  Recall:    {recall:.4f}", 'green')
        Utils.print_color(f"  ROC AUC:   {roc_auc:.4f}", 'green')
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_test, target_names=['No Increase', 'Increase'], zero_division=np.nan))
        
        # Store results
        results_test[model_name] = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'y_pred': y_pred_test,
        }

    return results_test
