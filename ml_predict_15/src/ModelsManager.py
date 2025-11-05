"""
Models Manager Module

Handles model creation, loading, and saving of pretrained models.
"""

import os
import joblib
from datetime import datetime
from src.ModelConfig import get_model_config
import multiprocessing
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

# Import centralized model configuration
try:
    from src.ModelConfig import get_model_config
    MODEL_CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ModelConfig not available: {e}")
    MODEL_CONFIG_AVAILABLE = False

# Import neural networks manager
try:
    from .NeuralNetworksManager import NeuralNetworksManager
    NEURAL_NETWORKS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Neural networks not available: {e}")
    NEURAL_NETWORKS_AVAILABLE = False


class ModelsManager:
    """
    Manages ML model creation, loading, and saving.
    """
    
    def __init__(self, models_dir='models', include_neural_networks=True, 
                 sequence_length=30, epochs=50, batch_size=32):
        """
        Initialize ModelsManager.
        
        Parameters:
        -----------
        models_dir : str
            Directory to save/load models
        include_neural_networks : bool
            Whether to include neural network models
        sequence_length : int
            Sequence length for neural networks
        epochs : int
            Training epochs for neural networks
        batch_size : int
            Batch size for neural networks
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Get centralized model configuration
        if MODEL_CONFIG_AVAILABLE:
            self.model_config_manager = get_model_config()
        else:
            raise ImportError("ModelConfig is required but not available")
        
        # Initialize neural networks manager if available and requested
        self.neural_networks_manager = None
        if include_neural_networks and NEURAL_NETWORKS_AVAILABLE:
            self.neural_networks_manager = NeuralNetworksManager(
                sequence_length=sequence_length,
                epochs=epochs,
                batch_size=batch_size
            )
    
    def create_models(self, enabled_only=True, include_neural_networks=True):
        """
        Create fresh model instances.
        
        Parameters:
        -----------
        enabled_only : bool
            If True, only create enabled models
        include_neural_networks : bool
            If True, include neural network models
            
        Returns:
        --------
        dict : Dictionary of model_name -> model_instance
        """
        models = {}
        
        # Create traditional ML models from centralized config
        for name, config in self.model_config_manager.traditional_models.items():
            if enabled_only and not config['enabled']:
                continue
            
            try:
                model = config['class'](**config['params'])
                models[name] = model
                print(f"✓ Created model: {name}")
            except Exception as e:
                print(f"✗ Failed to create model {name}: {e}")
        
        # Create neural network models
        if include_neural_networks and self.neural_networks_manager is not None:
            print(f"\nCreating neural network models...")
            neural_models = self.neural_networks_manager.create_models(enabled_only=enabled_only)
            models.update(neural_models)
        
        print(f"\nTotal models created: {len(models)}")
        return models
    
    def save_models(self, models, scaler=None, suffix=''):
        """
        Save trained models and scaler to disk in a timestamped subdirectory.
        
        Parameters:
        -----------
        models : dict
            Dictionary of model_name -> trained_model
        scaler : sklearn scaler
            Fitted scaler (optional)
        suffix : str
            Optional timestamp suffix (YYYY-MM-DD_HH-MM-SS format)
            If empty, current timestamp will be used
            
        Returns:
        --------
        dict : Paths where models were saved
        """
        # Create timestamp in YYYY-MM-DD_HH-MM-SS format
        if not suffix:
            suffix = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Create timestamped subdirectory
        save_dir = os.path.join(self.models_dir, suffix)
        os.makedirs(save_dir, exist_ok=True)
        
        saved_paths = {}
        
        print(f"\n{'='*70}")
        print(f"SAVING MODELS TO: {save_dir}")
        print(f"{'='*70}")
        
        # Save each model
        for name, model in models.items():
            filename = f"{name}.joblib"
            filepath = os.path.join(save_dir, filename)
            joblib.dump(model, filepath)
            saved_paths[name] = filepath
            print(f"✓ Saved {name}")
        
        # Save scaler
        if scaler is not None:
            scaler_path = os.path.join(save_dir, "scaler.joblib")
            joblib.dump(scaler, scaler_path)
            saved_paths['scaler'] = scaler_path
            print(f"✓ Saved scaler")
        
        # Save metadata
        metadata = {
            'timestamp': suffix,
            'models': list(models.keys()),
            'has_scaler': scaler is not None,
            'save_dir': save_dir
        }
        metadata_path = os.path.join(save_dir, "metadata.joblib")
        joblib.dump(metadata, metadata_path)
        saved_paths['metadata'] = metadata_path
        print(f"✓ Saved metadata")
        
        print(f"\n✓ Saved {len(models)} models successfully to {save_dir}")
        print(f"{'='*70}\n")
        return saved_paths
    
    def load_models(self, suffix='latest') :
        """
        Load trained models and scaler from disk from a timestamped subdirectory.
        
        Parameters:
        -----------
        suffix : str
            Timestamp subdirectory name ('latest' or specific timestamp like '2024-01-15_14-30-45')
            
        Returns:
        --------
        tuple : (models_dict, scaler, metadata)
        """
        # Find the latest models if suffix is 'latest'
        if suffix == 'latest':
            suffix = self._find_latest_suffix()
            if suffix is None:
                print("✗ No saved models found")
                return {}, None, None
        
        # Construct the load directory path
        load_dir = os.path.join(self.models_dir, suffix)
        
        if not os.path.exists(load_dir):
            print(f"✗ Model directory not found: {load_dir}")
            return {}, None, None
        
        models = {}
        scaler = None
        metadata = None
        
        print(f"\n{'='*70}")
        print(f"LOADING MODELS FROM: {load_dir}")
        print(f"{'='*70}")
        
        # Load metadata
        metadata_path = os.path.join(load_dir, "metadata.joblib")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            print(f"✓ Loaded metadata")
        
        # Load models
        for name in self.model_config_manager.model_names.keys():
            filepath = os.path.join(load_dir, f"{name}.joblib")
            if os.path.exists(filepath):
                models[name] = joblib.load(filepath)
                print(f"✓ Loaded {name}")
        
        # Load scaler
        scaler_path = os.path.join(load_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"✓ Loaded scaler")
        
        print(f"\n✓ Loaded {len(models)} models successfully from {load_dir}")
        print(f"{'='*70}\n")
        return models, scaler, metadata
    
    def _find_latest_suffix(self):
        """Find the latest timestamped subdirectory in the models directory."""
        if not os.path.exists(self.models_dir):
            return None
        
        # Get all subdirectories that look like timestamps (YYYY-MM-DD_HH-MM-SS)
        subdirs = []
        for item in os.listdir(self.models_dir):
            item_path = os.path.join(self.models_dir, item)
            if os.path.isdir(item_path):
                # Check if it contains metadata.joblib to confirm it's a valid model directory
                metadata_path = os.path.join(item_path, 'metadata.joblib')
                if os.path.exists(metadata_path):
                    subdirs.append(item)
        
        if not subdirs:
            return None
        
        # Sort and return the latest (timestamps sort lexicographically)
        return max(subdirs)
    
    def list_saved_models(self):
        """
        List all saved model versions (timestamped subdirectories).
        
        Returns:
        --------
        list : List of (timestamp, metadata) tuples sorted by timestamp (newest first)
        """
        if not os.path.exists(self.models_dir):
            return []
        
        versions = []
        
        # Iterate through subdirectories
        for item in os.listdir(self.models_dir):
            item_path = os.path.join(self.models_dir, item)
            if os.path.isdir(item_path):
                # Check if it contains metadata.joblib
                metadata_path = os.path.join(item_path, 'metadata.joblib')
                if os.path.exists(metadata_path):
                    try:
                        metadata = joblib.load(metadata_path)
                        versions.append((item, metadata))
                    except Exception as e:
                        print(f"Warning: Could not load metadata from {item}: {e}")
        
        # Sort by timestamp (newest first)
        return sorted(versions, key=lambda x: x[0], reverse=True)
    
    def enable_model(self, model_name, enabled=True):
        """
        Enable or disable a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        enabled : bool
            Whether to enable or disable
        """
        # Delegate to centralized config
        self.model_config_manager.enable_model(model_name, enabled)
    
    def get_enabled_models(self, include_neural_networks=True):
        """
        Get list of enabled model names.
        
        Parameters:
        -----------
        include_neural_networks : bool
            Whether to include neural network models
        
        Returns:
        --------
        list : List of enabled model names
        """
        # Get from centralized config
        enabled_models = self.model_config_manager.get_enabled_traditional_models()
        
        if include_neural_networks:
            enabled_models.extend(self.model_config_manager.get_enabled_neural_network_models())
        
        return enabled_models
    
    def enable_neural_network(self, model_name, enabled=True):
        """
        Enable or disable a neural network model.
        
        Parameters:
        -----------
        model_name : str
            Name of the neural network model
        enabled : bool
            Whether to enable or disable
        """
        # Delegate to centralized config
        self.model_config_manager.enable_model(model_name, enabled)
    
    def configure_neural_networks(self, sequence_length=None, epochs=None, batch_size=None):
        """
        Update neural network configuration.
        
        Parameters:
        -----------
        sequence_length : int, optional
            New sequence length
        epochs : int, optional
            New number of epochs
        batch_size : int, optional
            New batch size
        """
        if self.neural_networks_manager is not None:
            if sequence_length is not None:
                self.neural_networks_manager.sequence_length = sequence_length
            if epochs is not None:
                self.neural_networks_manager.epochs = epochs
            if batch_size is not None:
                self.neural_networks_manager.batch_size = batch_size
            print(f"✓ Neural network configuration updated")
        else:
            print(f"✗ Neural networks not available")
    
    def print_config(self):
        """Print current model configuration."""
        # Delegate to centralized config
        self.model_config_manager.print_config()
