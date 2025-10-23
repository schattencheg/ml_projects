"""
Model Loader Utility

Utility functions to load trained models and scalers from the models directory.
"""

import os
import joblib
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler


def load_model(model_name: str, models_dir: str = 'models') -> object:
    """
    Load a single trained model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model (without extension)
    models_dir : str
        Directory containing saved models
        
    Returns:
    --------
    model : object
        Loaded model
    """
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Loaded model: {model_name}")
    
    return model


def load_scaler(scaler_name: str = 'scaler', models_dir: str = 'models') -> StandardScaler:
    """
    Load a fitted scaler.
    
    Parameters:
    -----------
    scaler_name : str
        Name of the scaler file (without extension)
    models_dir : str
        Directory containing saved scaler
        
    Returns:
    --------
    scaler : StandardScaler
        Loaded scaler
    """
    scaler_path = os.path.join(models_dir, f"{scaler_name}.joblib")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    scaler = joblib.load(scaler_path)
    print(f"Loaded scaler: {scaler_name}")
    
    return scaler


def load_all_models(models_dir: str = 'models') -> Dict[str, object]:
    """
    Load all available models from the models directory.
    
    Parameters:
    -----------
    models_dir : str
        Directory containing saved models
        
    Returns:
    --------
    models : Dict[str, object]
        Dictionary of model_name -> model object
    """
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    models = {}
    
    # Get all .joblib files in the directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    # Filter out scaler files
    model_files = [f for f in model_files if 'scaler' not in f.lower()]
    
    if len(model_files) == 0:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    
    # Load each model
    for model_file in model_files:
        model_name = model_file.replace('.joblib', '')
        # Remove '_best' suffix if present
        display_name = model_name.replace('_best', '')
        
        model_path = os.path.join(models_dir, model_file)
        model = joblib.load(model_path)
        models[display_name] = model
        print(f"  ✓ Loaded: {display_name}")
    
    return models


def list_available_models(models_dir: str = 'models') -> List[str]:
    """
    List all available models in the models directory.
    
    Parameters:
    -----------
    models_dir : str
        Directory containing saved models
        
    Returns:
    --------
    model_names : List[str]
        List of available model names
    """
    if not os.path.exists(models_dir):
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    model_files = [f for f in model_files if 'scaler' not in f.lower()]
    model_best = [f for f in model_files if 'best' in f.lower()]
    model_not_best = [f for f in model_files if 'best' not in f.lower()]
    model_names = model_best + model_not_best
    model_names = [f.replace('.joblib', '').replace('_best', '') for f in model_names]
    
    return model_names


def save_model(model, model_name: str, models_dir: str = 'models'):
    """
    Save a trained model.
    
    Parameters:
    -----------
    model : object
        Trained model to save
    model_name : str
        Name for the saved model
    models_dir : str
        Directory to save the model
    """
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


def save_scaler(scaler: StandardScaler, scaler_name: str = 'scaler', models_dir: str = 'models'):
    """
    Save a fitted scaler.
    
    Parameters:
    -----------
    scaler : StandardScaler
        Fitted scaler to save
    scaler_name : str
        Name for the saved scaler
    models_dir : str
        Directory to save the scaler
    """
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, f"{scaler_name}.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")


def save_all_models(models: Dict[str, object], scaler: StandardScaler, models_dir: str = 'models'):
    """
    Save all models and scaler.
    
    Parameters:
    -----------
    models : Dict[str, object]
        Dictionary of model_name -> model object
    scaler : StandardScaler
        Fitted scaler
    models_dir : str
        Directory to save models
    """
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\nSaving models to {models_dir}/...")
    
    # Save scaler
    save_scaler(scaler, models_dir=models_dir)
    
    # Save each model
    for model_name, model_data in models.items():
        # Extract model from tuple if necessary
        if isinstance(model_data, tuple):
            model = model_data[0]
        else:
            model = model_data
        
        save_model(model, model_name, models_dir=models_dir)
    
    print(f"\n✓ All models and scaler saved successfully!")
