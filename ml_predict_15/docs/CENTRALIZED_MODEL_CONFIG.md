# Centralized Model Configuration Guide

## Overview

The `ModelConfig` class provides a **single source of truth** for all ML model configurations in the ml_predict_15 project. It centralizes the enable/disable settings and parameters for both traditional ML models and neural network models.

## Key Benefits

✅ **Single Configuration Point** - All model settings in one place  
✅ **Consistent Management** - Same interface for traditional and neural network models  
✅ **Easy Enable/Disable** - Simple methods to control which models to train  
✅ **Category-Based Control** - Enable/disable entire categories (CNN, LSTM, GRU)  
✅ **Better Organization** - Clear separation of concerns  
✅ **Backward Compatible** - Existing code continues to work  

## Architecture

### Before Refactoring
```
ModelsManager.py
├── model_config = {...}  # Traditional ML models
└── neural_networks_manager
    └── model_configs = {...}  # Neural network models
```

### After Refactoring
```
ModelConfig.py (Centralized)
├── traditional_models = {...}
└── neural_network_models = {...}
    ↓
ModelsManager.py → uses ModelConfig
NeuralNetworksManager.py → uses ModelConfig
```

## ModelConfig Class

### Structure

```python
class ModelConfig:
    def __init__(self):
        # Traditional ML Models
        self.traditional_models = {
            'logistic_regression': {
                'enabled': True,
                'class': LogisticRegression,
                'params': {...},
                'description': '...',
                'training_time': '~2-5 sec'
            },
            # ... more models
        }
        
        # Neural Network Models
        self.neural_network_models = {
            'cnn_simple': {
                'enabled': True,
                'build_fn': None,  # Set by NeuralNetworksManager
                'description': '...',
                'training_time': 'Variable',
                'category': 'CNN'
            },
            # ... more models
        }
```

### Key Methods

#### Enable/Disable Models

```python
# Enable or disable any model (traditional or neural network)
config.enable_model('random_forest', True)
config.enable_model('cnn_simple', False)

# Enable/disable all traditional models
config.enable_all_traditional(True)

# Enable/disable all neural networks
config.enable_all_neural_networks(False)

# Enable/disable by category (CNN, LSTM, GRU, Hybrid)
config.enable_by_category('CNN', True)
config.enable_by_category('LSTM', False)
```

#### Query Models

```python
# Get enabled models
enabled_trad = config.get_enabled_traditional_models()
enabled_nn = config.get_enabled_neural_network_models()
all_enabled = config.get_all_enabled_models()

# Get disabled models
disabled_trad = config.get_disabled_traditional_models()
disabled_nn = config.get_disabled_neural_network_models()

# Get model info
info = config.get_model_info('xgboost')
# Returns: {'type': 'traditional', 'config': {...}}
```

#### Display Configuration

```python
# Print full configuration
config.print_config()

# Print without showing disabled models
config.print_config(show_disabled=False)

# Export configuration as dictionary
config_dict = config.export_config()
```

## Usage Examples

### Basic Usage

```python
from src.ModelConfig import get_model_config

# Get singleton instance
config = get_model_config()

# View current configuration
config.print_config()

# Enable specific models
config.enable_model('random_forest', True)
config.enable_model('svm', False)

# Enable all CNNs
config.enable_by_category('CNN', True)
```

### Using with ModelsManager

```python
from src.ModelsManager import ModelsManager

# Create manager (automatically uses centralized config)
manager = ModelsManager(models_dir='models')

# View configuration
manager.print_config()

# Enable/disable models
manager.enable_model('xgboost', True)
manager.enable_neural_network('lstm_simple', False)

# Create only enabled models
models = manager.create_models(enabled_only=True)

# Get list of enabled models
enabled = manager.get_enabled_models(include_neural_networks=True)
```

### Configuration Presets

#### Fast Models Only (< 1 minute)
```python
config = get_model_config()

# Disable all first
config.enable_all_traditional(False)
config.enable_all_neural_networks(False)

# Enable only fast models
config.enable_model('logistic_regression', True)
config.enable_model('xgboost', True)
config.enable_model('lightgbm', True)
```

#### Tree-Based Models Only
```python
config = get_model_config()

# Disable all
config.enable_all_traditional(False)
config.enable_all_neural_networks(False)

# Enable tree-based
config.enable_model('decision_tree', True)
config.enable_model('random_forest', True)
config.enable_model('gradient_boosting', True)
config.enable_model('xgboost', True)
config.enable_model('lightgbm', True)
```

#### Neural Networks Only
```python
config = get_model_config()

# Disable traditional models
config.enable_all_traditional(False)

# Enable all neural networks
config.enable_all_neural_networks(True)

# Or enable specific categories
config.enable_by_category('CNN', True)
config.enable_by_category('LSTM', True)
config.enable_by_category('GRU', False)
```

#### Single Model for Testing
```python
config = get_model_config()

# Disable all
config.enable_all_traditional(False)
config.enable_all_neural_networks(False)

# Enable only one
config.enable_model('xgboost', True)
```

## Traditional ML Models

### Available Models

| Model Name | Default | Training Time | Description |
|------------|---------|---------------|-------------|
| logistic_regression | ✓ Enabled | ~2-5 sec | Logistic Regression with balanced class weights |
| ridge_classifier | ✗ Disabled | ~2-5 sec | Ridge Classifier with L2 regularization |
| naive_bayes | ✗ Disabled | ~3-5 sec | Gaussian Naive Bayes |
| decision_tree | ✗ Disabled | ~5-10 sec | Decision Tree with max depth 10 |
| random_forest | ✗ Disabled | ~15-30 sec | Random Forest with 100 trees |
| gradient_boosting | ✗ Disabled | ~60+ sec | Gradient Boosting (slower than XGBoost) |
| knn | ✗ Disabled | ~50-120 sec | K-Nearest Neighbors (slow on large datasets) |
| svm | ✗ Disabled | ~100+ sec | Support Vector Machine with RBF kernel |
| xgboost | ✓ Enabled | ~3-5 sec | XGBoost with histogram-based tree method |
| lightgbm | ✓ Enabled | ~3-5 sec | LightGBM with fast training |

## Neural Network Models

### Available Models

#### CNN Variants
| Model Name | Default | Category | Description |
|------------|---------|----------|-------------|
| cnn_simple | ✓ Enabled | CNN | Simple 1D-CNN with 2 conv layers |
| cnn_deep | ✓ Enabled | CNN | Deep 1D-CNN with 4 conv layers |
| cnn_residual | ✓ Enabled | CNN | 1D-CNN with residual connections |
| cnn_attention | ✓ Enabled | CNN | 1D-CNN with attention mechanism |
| cnn_dilated | ✓ Enabled | CNN | 1D-CNN with dilated convolutions |

#### LSTM Variants
| Model Name | Default | Category | Description |
|------------|---------|----------|-------------|
| lstm_simple | ✓ Enabled | LSTM | Simple LSTM with 2 layers |
| lstm_bidirectional | ✓ Enabled | LSTM | Bidirectional LSTM |
| lstm_stacked | ✓ Enabled | LSTM | Stacked LSTM with 3 layers |
| lstm_attention | ✓ Enabled | LSTM | LSTM with attention mechanism |

#### Hybrid Models
| Model Name | Default | Category | Description |
|------------|---------|----------|-------------|
| lstm_cnn_hybrid | ✓ Enabled | Hybrid | Hybrid CNN-LSTM model |

#### GRU Variants
| Model Name | Default | Category | Description |
|------------|---------|----------|-------------|
| gru_simple | ✓ Enabled | GRU | Simple GRU model |
| gru_bidirectional | ✓ Enabled | GRU | Bidirectional GRU |

## Output Examples

### Configuration Display

```
================================================================================
CENTRALIZED MODEL CONFIGURATION
================================================================================

TRADITIONAL ML MODELS:

Enabled models (3):
  ✓ logistic_regression      - Logistic Regression with balanced class weights
    Training time: ~2-5 sec
  ✓ xgboost                  - XGBoost with histogram-based tree method
    Training time: ~3-5 sec
  ✓ lightgbm                 - LightGBM with fast training
    Training time: ~3-5 sec

Disabled models (7):
  ✗ ridge_classifier         - Ridge Classifier with L2 regularization
  ✗ naive_bayes              - Gaussian Naive Bayes
  ...

--------------------------------------------------------------------------------
NEURAL NETWORK MODELS:

Enabled models (12):

  CNN Models:
    ✓ cnn_simple               - Simple 1D-CNN with 2 conv layers
    ✓ cnn_deep                 - Deep 1D-CNN with 4 conv layers
    ...

  LSTM Models:
    ✓ lstm_simple              - Simple LSTM with 2 layers
    ✓ lstm_bidirectional       - Bidirectional LSTM
    ...

--------------------------------------------------------------------------------
SUMMARY:
  Total models: 22
  Traditional ML: 3 enabled, 7 disabled
  Neural Networks: 12 enabled, 0 disabled
  Total enabled: 15
================================================================================
```

## Integration with Existing Code

### ModelsManager Integration

The `ModelsManager` class now uses `ModelConfig` internally:

```python
class ModelsManager:
    def __init__(self, ...):
        # Get centralized configuration
        self.model_config_manager = get_model_config()
        
    def create_models(self, enabled_only=True, ...):
        # Create from centralized config
        for name, config in self.model_config_manager.traditional_models.items():
            if enabled_only and not config['enabled']:
                continue
            # Create model...
```

### NeuralNetworksManager Integration

The `NeuralNetworksManager` class also uses `ModelConfig`:

```python
class NeuralNetworksManager:
    def __init__(self, ...):
        # Get centralized configuration
        self.model_config_manager = get_model_config()
        # Set build_fn references
        self._setup_build_functions()
        
    def create_models(self, enabled_only=True):
        # Create from centralized config
        for name, config in self.model_config_manager.neural_network_models.items():
            if enabled_only and not config['enabled']:
                continue
            # Create model...
```

## Backward Compatibility

✅ **Fully backward compatible** - All existing code continues to work  
✅ **No breaking changes** - Public APIs remain unchanged  
✅ **Legacy support** - Old `model_config` dictionaries kept for compatibility  

### Migration Path

Old code:
```python
manager = ModelsManager()
manager.model_config['xgboost']['enabled'] = True  # Still works (deprecated)
```

New code (recommended):
```python
manager = ModelsManager()
manager.enable_model('xgboost', True)  # Uses centralized config
```

## Best Practices

### 1. Use Centralized Config for All Changes
```python
# Good: Use centralized config
config = get_model_config()
config.enable_model('xgboost', True)

# Avoid: Direct dictionary manipulation (deprecated)
manager.model_config['xgboost']['enabled'] = True
```

### 2. Use Category-Based Control for Neural Networks
```python
# Enable all CNNs at once
config.enable_by_category('CNN', True)

# Better than enabling individually
config.enable_model('cnn_simple', True)
config.enable_model('cnn_deep', True)
# ...
```

### 3. Create Configuration Presets
```python
def configure_fast_training():
    """Configure for fast training (< 1 minute)."""
    config = get_model_config()
    config.enable_all_traditional(False)
    config.enable_all_neural_networks(False)
    config.enable_model('logistic_regression', True)
    config.enable_model('xgboost', True)
    config.enable_model('lightgbm', True)
    return config

# Use preset
config = configure_fast_training()
```

### 4. Document Your Configuration
```python
# Export configuration for documentation
config = get_model_config()
config_dict = config.export_config()

# Save to file
import json
with open('model_config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)
```

## Troubleshooting

### Issue: Changes Not Taking Effect

**Problem:** Model enable/disable changes don't affect training.

**Solution:** Make sure you're using the singleton instance:
```python
from src.ModelConfig import get_model_config

# Always use get_model_config() to get the singleton
config = get_model_config()
config.enable_model('xgboost', True)
```

### Issue: Neural Network build_fn is None

**Problem:** Neural network models have `build_fn: None`.

**Solution:** The `build_fn` is set by `NeuralNetworksManager` during initialization:
```python
# This happens automatically
nn_manager = NeuralNetworksManager()
# build_fn references are set via _setup_build_functions()
```

### Issue: Model Not Found

**Problem:** `enable_model()` says model not found.

**Solution:** Check the exact model name:
```python
# List all available models
config = get_model_config()
print("Traditional:", list(config.traditional_models.keys()))
print("Neural Networks:", list(config.neural_network_models.keys()))
```

## Summary

The centralized `ModelConfig` class provides:

1. **Single Source of Truth** - All model configurations in one place
2. **Unified Interface** - Same methods for traditional and neural network models
3. **Easy Management** - Simple enable/disable with category support
4. **Better Organization** - Clear separation and structure
5. **Backward Compatible** - Existing code continues to work
6. **Extensible** - Easy to add new models or features

This refactoring improves code organization and maintainability while preserving all existing functionality.
