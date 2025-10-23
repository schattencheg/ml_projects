# ML Models for Price Change Prediction

This project now includes a comprehensive suite of machine learning models for predicting cryptocurrency price changes.

## Available Models

### Traditional ML Models
1. **Logistic Regression** - Linear model for binary classification
2. **Ridge Classifier** - Regularized linear classifier
3. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
4. **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
5. **Decision Tree** - Tree-based classifier with interpretable rules
6. **Random Forest** - Ensemble of decision trees
7. **Gradient Boosting** - Sequential ensemble method
8. **Support Vector Machine (SVM)** - Margin-based classifier

### Advanced ML Models
9. **XGBoost** - Optimized gradient boosting framework
10. **LightGBM** - Fast gradient boosting framework

### Neural Network Models
11. **LSTM** - Long Short-Term Memory network for sequential data
12. **CNN** - 1D Convolutional Neural Network for pattern recognition
13. **LSTM-CNN Hybrid** - Combined approach using both CNN and LSTM layers

## Model Descriptions

### LSTM (Long Short-Term Memory)
- **Purpose**: Captures long-term dependencies in sequential price data
- **Architecture**: 2 LSTM layers (50 units each) + Dense layers
- **Best for**: Time series patterns, trend analysis
- **Sequence length**: 30 time steps

### CNN (Convolutional Neural Network)
- **Purpose**: Detects local patterns and features in price sequences
- **Architecture**: 3 Conv1D layers + MaxPooling + Dense layers
- **Best for**: Pattern recognition, technical analysis signals
- **Features**: Batch normalization, dropout for regularization

### LSTM-CNN Hybrid
- **Purpose**: Combines pattern detection (CNN) with sequence modeling (LSTM)
- **Architecture**: Conv1D → LSTM → Dense layers
- **Best for**: Complex pattern recognition with temporal dependencies

### XGBoost & LightGBM
- **Purpose**: High-performance gradient boosting for tabular data
- **Advantages**: Fast training, feature importance, handles missing values
- **Best for**: Feature-rich datasets, ensemble methods

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The models are automatically integrated into the existing training pipeline. Simply run:

```python
python run_me.py
```

## Model Selection

The system automatically:
1. Trains all available models
2. Evaluates performance on validation data
3. Selects the best model based on accuracy
4. Tests on held-out data
5. Generates visualizations

## Performance Considerations

- **Neural Networks**: Require more training time but can capture complex patterns
- **Tree-based models**: Fast training, good for feature importance analysis
- **Linear models**: Fast, interpretable, good baseline performance

## Feature Engineering

The models use engineered features including:
- Price returns (log returns for OHLC)
- Simple Moving Averages (SMA 5, 10, 15)
- SMA crossovers
- Technical indicators

## Hyperparameters

Key hyperparameters are pre-configured for optimal performance:
- **Neural Networks**: 50 epochs, early stopping, learning rate scheduling
- **Tree models**: 100 estimators, controlled depth to prevent overfitting
- **Regularization**: Dropout (0.3) for neural networks, max_depth limits for trees

## Output

The system generates:
1. **Model Performance Metrics**: Accuracy, F1-score, Precision, Recall, ROC AUC
2. **Visualizations**: Comprehensive performance comparison charts, prediction plots
3. **Saved Models**: Best performing model saved to `models/` directory
4. **Classification Reports**: Detailed per-class performance

## Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)  
- **ROC AUC**: Area Under the Receiver Operating Characteristic curve - measures model's ability to distinguish between classes

## Suggestions for Further Improvement

1. **Ensemble Methods**: Combine predictions from multiple models
2. **Feature Selection**: Use feature importance from tree models
3. **Hyperparameter Tuning**: Grid search or Bayesian optimization
4. **Cross-Validation**: More robust model evaluation
5. **Additional Features**: Volume indicators, volatility measures
6. **Attention Mechanisms**: Transformer-based models for sequence data
