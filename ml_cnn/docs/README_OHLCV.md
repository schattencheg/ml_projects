# OHLCV Time Series Classification for Price Movement Prediction

A comprehensive research-grade Jupyter notebook for predicting financial price movements using machine learning on OHLCV (Open, High, Low, Close, Volume) time series data.

## 🎯 Overview

This project implements a complete ML pipeline that:
- Takes OHLCV time series data as input
- Engineers 100+ features from technical indicators and statistical transformations
- Labels timestamps according to future price movement over N bars (up ≥ p%, down ≤ –p%, or flat)
- Trains and evaluates multiple ML models for 3-class directional movement prediction
- Includes extensive visualizations, model diagnostics, and trading strategy simulation
- Provides comprehensive model interpretation using SHAP and feature importance analysis

## 📁 Project Structure

```
ml_cnn/
├── ohlcv_time_series_classification.ipynb  # Main research notebook (13 cells)
├── generate_sample_data.py                 # Sample data generation script
├── requirements_ohlcv.txt                  # Python dependencies
├── data/                                   # Data directory
│   ├── sample_data_1000.csv              # 1000-day sample dataset
│   ├── sample_data_2000.csv              # 2000-day sample dataset
│   └── high_vol_data.csv                 # High volatility sample dataset
└── README_OHLCV.md                       # This file
```

## 🛠 Installation & Setup

### 1. Clone Repository
```bash
git clone <repository_url>
cd ml_projects/ml_cnn
```

### 2. Install Dependencies
```bash
pip install -r requirements_ohlcv.txt
```

### 3. Generate Sample Data (Optional)
```bash
python generate_sample_data.py
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook ohlcv_time_series_classification.ipynb
```

## 📊 Key Features

### Data Processing
- **Temporal splitting**: Maintains chronological order (no data leakage)
- **Feature engineering**: 100+ features including technical indicators, lags, rolling statistics
- **Label generation**: 3-class classification (Up/Down/Flat) with configurable thresholds
- **Data validation**: Comprehensive data quality checks and visualizations

### Machine Learning Models
- **Traditional ML**: Logistic Regression, Random Forest, XGBoost
- **Deep Learning**: Multi-layer Perceptron (MLP) with TensorFlow/Keras
- **Hyperparameter Tuning**: Optuna-based optimization for best performance
- **Model Comparison**: Comprehensive evaluation with multiple metrics

### Analysis & Interpretation
- **Feature Importance**: Tree-based and permutation importance analysis
- **SHAP Analysis**: Model-agnostic explanations for predictions
- **Performance Metrics**: Accuracy, Balanced Accuracy, F1-Macro scores
- **Confusion Matrix**: Detailed class-wise performance analysis

### Trading Strategy Simulation
- **Backtesting**: Simple long/short strategy based on predictions
- **Performance Metrics**: Returns, Sharpe ratio, maximum drawdown
- **Benchmark Comparison**: Strategy vs. buy-and-hold performance
- **Visualization**: Equity curve plots and trade analysis

## 🔧 Configuration

### Global Parameters (Cell 1)
```python
N = 10           # prediction horizon (bars)
P_PCT = 1.0      # threshold in percent (e.g., 1%)
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2   # of remaining after test split
```

### Customization Options
- **Prediction Horizon**: Adjust `N` for different forecasting periods
- **Movement Threshold**: Modify `P_PCT` for sensitivity tuning
- **Feature Selection**: Enable/disable feature categories in Cell 4
- **Model Selection**: Add/remove models in baseline evaluation
- **Hyperparameter Ranges**: Customize Optuna search spaces

## 📈 Notebook Structure

### Cell 1: Imports and Global Config
- All library imports and global constants
- Reproducibility settings and display options

### Cell 2: Load and Inspect Raw OHLCV Data
- Data loading with sample generation function
- Basic statistics and time series visualization

### Cell 3: Label Generation (Target Definition)
- Future return calculation and movement labeling
- Class distribution analysis and visualization

### Cell 4: Feature Engineering
- Raw transformations (ratios, shadows, positions)
- Rolling statistics (multiple windows)
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Lag features and time-based features

### Cell 5: Exploratory Data Analysis (EDA)
- Feature correlation heatmaps
- Feature vs. price relationship analysis
- Class-wise feature distributions

### Cell 6: Train/Validation/Test Split (Temporal)
- Chronological data splitting
- Class balance verification across splits

### Cell 7: Feature Scaling
- StandardScaler fitting on training data only
- Scaling verification and visualization

### Cell 8: Baseline Models
- Logistic Regression, Random Forest, XGBoost
- Comprehensive model comparison table
- Performance visualization

### Cell 9: Neural Network Prototype (MLP)
- Multi-layer perceptron architecture
- Training history visualization
- Updated model comparison

### Cell 10: Hyperparameter Tuning
- Optuna-based optimization for XGBoost and MLP
- Best parameter identification and model retraining

### Cell 11: Model Interpretation
- Feature importance analysis
- SHAP value computation and visualization
- Neural network interpretation via permutation importance

### Cell 12: Final Evaluation & Strategy Simulation
- Best model selection and test set evaluation
- Confusion matrix and classification report
- Trading strategy backtesting and performance analysis

### Cell 13: Conclusion & Next Steps
- Comprehensive results summary
- Limitations and future work discussion
- Research conclusions and recommendations

## 📊 Expected Results

### Model Performance
- **Baseline Models**: 50-65% accuracy typical for 3-class prediction
- **Optimized Models**: 55-70% accuracy with hyperparameter tuning
- **Class Balance**: Performance varies by market regime and threshold

### Feature Importance
- **Technical Indicators**: RSI, MACD, Bollinger Bands typically important
- **Rolling Statistics**: Price momentum and volatility measures
- **Lag Features**: Recent price and volume patterns

### Trading Strategy
- **Performance**: Varies significantly based on market conditions
- **Risk Metrics**: Sharpe ratio, maximum drawdown analysis
- **Benchmark**: Comparison against buy-and-hold strategy

## ⚠️ Important Notes

### Data Requirements
- **Format**: CSV with columns: date, open, high, low, close, volume
- **Frequency**: Daily data recommended (can adapt for other frequencies)
- **Quality**: Clean data without gaps or errors essential
- **Size**: Minimum 500+ samples for meaningful results

### Limitations
- **Synthetic Data**: Sample data may not reflect real market behavior
- **Transaction Costs**: Strategy simulation doesn't include realistic costs
- **Market Regimes**: Model performance varies across different market conditions
- **Overfitting Risk**: Extensive feature engineering requires careful validation

### Best Practices
- **Temporal Validation**: Always use chronological splits for time series
- **Feature Selection**: Consider domain knowledge in feature engineering
- **Model Validation**: Use multiple metrics for comprehensive evaluation
- **Risk Management**: Implement proper position sizing and stop-losses

## 🔮 Future Enhancements

### Advanced Models
- **LSTM/GRU**: Sequential neural networks for time series
- **CNN**: Convolutional networks for pattern recognition
- **Transformer**: Attention-based architectures
- **Ensemble Methods**: Combining multiple model predictions

### Feature Engineering
- **Alternative Data**: Sentiment, news, social media signals
- **Cross-Asset**: Correlations with other financial instruments
- **Macro Indicators**: Economic data integration
- **Regime Detection**: Adaptive features based on market conditions

### Strategy Development
- **Risk Management**: Position sizing, stop-losses, portfolio allocation
- **Transaction Costs**: Realistic execution modeling
- **Multi-Asset**: Portfolio-level strategy implementation
- **Online Learning**: Model adaptation to changing market conditions

## 📚 References

- **Technical Analysis**: Murphy, J.J. "Technical Analysis of the Financial Markets"
- **Machine Learning**: Hastie, T. "The Elements of Statistical Learning"
- **Time Series**: Hamilton, J.D. "Time Series Analysis"
- **Quantitative Finance**: Jansen, S. "Machine Learning for Algorithmic Trading"

## 📄 License

This project is provided for educational and research purposes. Please ensure compliance with relevant financial regulations when using real trading data.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional model implementations
- Enhanced feature engineering techniques
- Improved visualization and analysis
- Real-world data integration examples
- Performance optimization

---

**Note**: This is a research framework. Always validate results with out-of-sample data and consider transaction costs, market impact, and regulatory requirements before implementing any trading strategies.
