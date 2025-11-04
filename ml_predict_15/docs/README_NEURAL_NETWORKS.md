# Neural Networks for Cryptocurrency Prediction

This document describes the comprehensive neural network implementation added to the ML prediction system.

## Overview

The system now includes multiple neural network architectures specifically designed for cryptocurrency price prediction:

- **Multiple 1D-CNN variants** for pattern detection in time series
- **Multiple LSTM variants** for sequential pattern recognition  
- **GRU variants** for efficient sequence modeling
- **GAN implementation** for data augmentation and feature extraction
- **Hybrid models** combining CNN and LSTM approaches

## Installation

Install the required dependencies:

```bash
pip install tensorflow keras mlflow
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## MLflow Integration

All neural network models are automatically tracked with MLflow when available:

### Start MLflow Server

```bash
# Windows
start_mlflow.bat

# Or manually
mlflow server --host 127.0.0.1 --port 5000
```

### Automatic Tracking

Neural networks are automatically tracked with:
- **Model Architecture**: Layer details, parameter counts
- **Training Parameters**: Sequence length, epochs, batch size
- **Performance Metrics**: Accuracy, F1, precision, recall
- **Model Artifacts**: Trained models, summaries, configs
- **Comparison**: Side-by-side experiment comparison

## Neural Network Models

### 1D-CNN Variants

#### 1. Simple CNN (`cnn_simple`)
- 2 convolutional layers with batch normalization
- MaxPooling and dropout for regularization
- Global max pooling for feature extraction
- Best for: Basic pattern detection

#### 2. Deep CNN (`cnn_deep`) 
- 6 convolutional layers in 3 blocks
- Progressive filter size increase (32→64→128)
- Multiple dropout layers
- Best for: Complex pattern recognition

#### 3. Residual CNN (`cnn_residual`)
- Residual connections to prevent vanishing gradients
- Skip connections between layers
- Batch normalization throughout
- Best for: Deep networks, gradient flow

#### 4. Attention CNN (`cnn_attention`)
- Attention mechanism for feature weighting
- Learns which parts of sequence are important
- Softmax attention weights
- Best for: Focus on relevant time periods

#### 5. Dilated CNN (`cnn_dilated`)
- Dilated convolutions with increasing rates (1,2,4,8)
- Captures multi-scale temporal patterns
- Larger receptive field without pooling
- Best for: Multi-timeframe analysis

### LSTM Variants

#### 1. Simple LSTM (`lstm_simple`)
- 2-layer LSTM with 50 units each
- Dropout between layers
- Standard sequential processing
- Best for: Basic sequence modeling

#### 2. Bidirectional LSTM (`lstm_bidirectional`)
- Processes sequences in both directions
- Captures past and future context
- Double the parameters of simple LSTM
- Best for: Context-aware predictions

#### 3. Stacked LSTM (`lstm_stacked`)
- 3-layer deep LSTM (64→64→32 units)
- Progressive size reduction
- Multiple levels of abstraction
- Best for: Complex temporal patterns

#### 4. Attention LSTM (`lstm_attention`)
- LSTM with attention mechanism
- Learns to focus on important timesteps
- Attention weights visualization possible
- Best for: Long sequences, interpretability

#### 5. Hybrid CNN-LSTM (`lstm_cnn_hybrid`)
- CNN layers for local pattern detection
- LSTM layers for sequence modeling
- Combines spatial and temporal learning
- Best for: Multi-scale pattern recognition

### GRU Variants

#### 1. Simple GRU (`gru_simple`)
- 2-layer GRU with 50 units each
- Faster training than LSTM
- Fewer parameters than LSTM
- Best for: Efficient sequence modeling

#### 2. Bidirectional GRU (`gru_bidirectional`)
- Bidirectional processing with GRU
- Balance between speed and performance
- Good alternative to bidirectional LSTM
- Best for: Fast bidirectional modeling

## GAN Implementation

### CryptocurrencyGAN

A Generative Adversarial Network specifically designed for cryptocurrency data:

#### Features:
- **Generator**: Creates synthetic price sequences
- **Discriminator**: Distinguishes real from fake sequences
- **Data Augmentation**: Generate additional training data
- **Feature Extraction**: Use discriminator as feature extractor

#### Usage:
```python
from src.NeuralNetworksManager import CryptocurrencyGAN

# Create and train GAN
gan = CryptocurrencyGAN(sequence_length=30, latent_dim=100)
gan.train(X_train, epochs=1000, batch_size=32)

# Generate synthetic data
synthetic_sequences = gan.generate_sequences(n_samples=1000)

# Extract features
features = gan.get_discriminator_features(X_test)
```

## Usage Examples

### Basic Usage

```python
from src.ModelsManager import ModelsManager
from src.Trainer import Trainer

# Create models manager with neural networks
models_manager = ModelsManager(
    include_neural_networks=True,
    sequence_length=30,  # Length of input sequences
    epochs=50,          # Training epochs
    batch_size=32       # Batch size
)

# Create all models (traditional + neural networks)
models = models_manager.create_models()

# Train with MLflow tracking (automatic)
trainer = Trainer(
    use_mlflow=True,  # Enable MLflow tracking
    mlflow_experiment="ml_predict_15/crypto_prediction",
    mlflow_tracking_uri="http://localhost:5000"
)

# Train models (handles sequences automatically + MLflow logging)
trained_models, scaler, results, best_model = trainer.train(models, X_train, y_train)
```

### MLflow Usage

```python
# Start MLflow server first: start_mlflow.bat

# Training automatically logs to MLflow
trainer = Trainer(use_mlflow=True)
trained_models, scaler, results, best_model = trainer.train(models, X_train, y_train)

# View results at: http://localhost:5000

# Load model from MLflow registry
from src.MLflowTracker import MLflowTracker
model = MLflowTracker.load_model_from_mlflow(
    model_name="ml_predict_15_cnn_simple",
    version="latest"
)
```

### Advanced Configuration

```python
from src.NeuralNetworksManager import NeuralNetworksManager

# Create neural networks manager
nn_manager = NeuralNetworksManager(
    sequence_length=50,  # Longer sequences
    epochs=100,         # More training
    batch_size=16       # Smaller batches
)

# Enable only specific models
nn_manager.enable_model('cnn_attention', True)
nn_manager.enable_model('lstm_bidirectional', True)
nn_manager.enable_model('lstm_cnn_hybrid', True)

# Create selected models
models = nn_manager.create_models(enabled_only=True)
```

### Integration with Existing Pipeline

```python
# In your main training script
models_manager = ModelsManager(
    models_dir='models',
    include_neural_networks=True,
    sequence_length=30,
    epochs=50,
    batch_size=32
)

# Configure neural networks
models_manager.configure_neural_networks(
    sequence_length=40,  # Adjust sequence length
    epochs=75,          # More epochs
    batch_size=16       # Smaller batches
)

# Enable/disable specific neural networks
models_manager.enable_neural_network('cnn_deep', False)
models_manager.enable_neural_network('lstm_attention', True)

# Print full configuration
models_manager.print_config()
```

## Model Selection Guidelines

### For Different Use Cases:

**High Frequency Trading (Short-term patterns):**
- `cnn_simple` or `cnn_dilated`
- `gru_simple` for speed
- Shorter sequence_length (10-20)

**Swing Trading (Medium-term patterns):**
- `cnn_attention` or `lstm_bidirectional`
- `lstm_cnn_hybrid` for multi-scale
- Medium sequence_length (30-50)

**Long-term Investment (Long-term patterns):**
- `lstm_stacked` or `lstm_attention`
- `cnn_residual` for deep patterns
- Longer sequence_length (50-100)

**Experimental/Research:**
- `cnn_residual` for deep learning
- `lstm_attention` for interpretability
- GAN for data augmentation

## Performance Considerations

### Memory Usage:
- **Lowest**: `gru_simple`, `cnn_simple`
- **Medium**: `lstm_simple`, `cnn_dilated`
- **Highest**: `lstm_stacked`, `cnn_deep`

### Training Speed:
- **Fastest**: GRU variants, simple CNN
- **Medium**: Simple LSTM, attention models
- **Slowest**: Deep/stacked models, hybrid models

### Accuracy Potential:
- **Good**: Simple models for basic patterns
- **Better**: Attention models for complex patterns
- **Best**: Hybrid and deep models for complex data

## Hyperparameter Tuning

### Key Parameters:

**sequence_length**: 
- Shorter (10-20): Fast, less context
- Medium (30-50): Balanced
- Longer (50-100): More context, slower

**epochs**:
- Few (10-25): Fast training, may underfit
- Medium (50-100): Balanced
- Many (100+): Better accuracy, risk overfitting

**batch_size**:
- Small (8-16): More stable, slower
- Medium (32-64): Balanced
- Large (128+): Faster, less stable

### Optimization Tips:

1. **Start Simple**: Begin with `cnn_simple` or `lstm_simple`
2. **Monitor Overfitting**: Use validation loss, early stopping
3. **Experiment**: Try different architectures for your data
4. **Ensemble**: Combine multiple neural networks
5. **Feature Engineering**: Good features help all models

## Troubleshooting

### Common Issues:

**"No module named 'tensorflow'"**
```bash
pip install tensorflow keras
```

**Out of Memory Errors:**
- Reduce batch_size (32→16→8)
- Reduce sequence_length
- Use simpler models

**Poor Performance:**
- Check data quality and preprocessing
- Try different sequence lengths
- Experiment with feature selection
- Consider data augmentation with GAN

**Slow Training:**
- Use GPU if available
- Reduce model complexity
- Use GRU instead of LSTM
- Reduce epochs for initial testing

## File Structure

```
src/
├── NeuralNetworksManager.py     # Main neural networks implementation
├── ModelsManager.py             # Updated with neural network support
└── ...

examples/
├── neural_networks_demo.py      # Demonstration script
└── ...

README_NEURAL_NETWORKS.md       # This documentation
test_neural_networks.py         # Quick test script
```

## Next Steps

1. **Install Dependencies**: `pip install tensorflow keras`
2. **Run Demo**: `python examples/neural_networks_demo.py`
3. **Test Integration**: `python test_neural_networks.py`
4. **Integrate**: Update your training pipeline
5. **Experiment**: Try different models and hyperparameters
6. **Monitor**: Track performance and adjust accordingly

## Advanced Features

### Custom Model Creation:
You can extend the NeuralNetworksManager to add your own architectures by following the existing pattern.

### GAN Data Augmentation:
Use the GAN to generate additional training data for imbalanced datasets.

### Transfer Learning:
Pre-train models on one cryptocurrency and fine-tune on others.

### Ensemble Methods:
Combine predictions from multiple neural networks for better accuracy.

---

**Note**: Neural networks require more computational resources and data than traditional ML models. Start with smaller configurations and scale up based on your hardware and data availability.
