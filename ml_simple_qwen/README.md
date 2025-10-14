# Simple ML Project - Continuous Training Demo

This educational project demonstrates how to implement a continuous machine learning training pipeline where models are:
- Trained on historical data
- Evaluated on new data as it arrives
- Retrained periodically every N values
- Performance metrics are collected and analyzed

## Features

- **Modular Design**: Clean separation of data generation, model management, metrics collection, and training logic
- **Multiple Models**: Support for different types of ML models with consistent interface
- **Continuous Learning**: Models adapt to new data patterns over time
- **Performance Tracking**: Comprehensive metrics collection and analysis
- **Educational**: Well-commented code with explanations of ML concepts

## Project Structure

```
ml_simple_qwen/
├── src/                    # Source code files
│   ├── __init__.py
│   ├── data_generator.py   # Generates synthetic time series data
│   ├── model_manager.py    # Manages different ML models
│   ├── metrics_collector.py # Collects and stores model metrics
│   ├── training_loop.py    # Main training loop with retraining logic
│   └── models/             # Folder for different ML models
│       ├── __init__.py
│       ├── linear_model.py
│       └── random_forest_model.py
├── outputs/                # Output files (created during execution)
│   ├── metrics/            # Metrics CSV files
│   └── visualizations/     # Performance charts and plots
├── data/                   # Generated datasets (if saved)
├── main.py                # Main execution script with examples
├── run_demo.py            # Non-interactive demo script
├── requirements.txt       # Project dependencies
├── README.md             # This file
└── __init__.py           # Package initialization
```

## Installation

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install individually:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

## Usage

### Run the full demonstration:
```bash
python main.py
```

### Run without interactive prompts:
```bash
python run_demo.py
```

The script will:
1. Generate synthetic data with known patterns
2. Set up multiple ML models
3. Run continuous training with periodic retraining
4. Collect and display comprehensive metrics
5. Create visualizations of model performance over time
6. Export metrics to CSV in `outputs/metrics/` and charts to `outputs/visualizations/`

## Key Concepts Demonstrated

### 1. Continuous Learning
- Models are not trained once but continuously updated as new data arrives
- This is essential for real-world systems where data distributions change over time (concept drift)

### 2. Periodic Retraining
- Models are periodically retrained on recent data to maintain performance
- Retraining interval can be tuned based on data volatility and computational resources

### 3. Performance Monitoring
- Multiple metrics (MSE, MAE, R²) are tracked to monitor model performance
- Metrics help detect when models start to degrade and need retraining

### 4. Model Comparison
- Multiple models can be compared side-by-side on the same data
- Performance metrics help identify which algorithms work best for specific problems

## Files and Modules

- `src/data_generator.py`: Creates synthetic datasets for training and testing
- `src/model_manager.py`: Handles model registration, training, and prediction
- `src/metrics_collector.py`: Calculates and stores performance metrics
- `src/training_loop.py`: Orchestrates the continuous training process
- `src/models/`: Contains specific model implementations (sklearn, custom, etc.)
- `main.py`: Entry point with full demonstration
- `run_demo.py`: Non-interactive demo script

## Educational Value

This project is designed to be educational and includes:
- Comprehensive comments explaining ML concepts
- Modular design for easy understanding and modification
- Examples of different ML algorithms
- Performance monitoring and analysis
- Visualization of model performance over time
- Best practices for ML project organization

## Sample Output

When running the demo, you'll see:
- Model training and evaluation metrics at each step
- Periodic retraining messages
- Final performance summary comparing all models
- Exported CSV with all metrics to `outputs/metrics/model_metrics.csv`
- Generated performance visualization charts in `outputs/visualizations/model_performance.png`

## Customization

### Adding New Models
To add a new model:
1. Create a new class inheriting from `BaseModel` in the `src/model_manager.py`
2. Implement the required methods: `train()`, `predict()`, and `get_params()`
3. Register your model with the `ModelManager`

### Adjusting Training Parameters
The main training function accepts several parameters:
- `initial_train_size`: Number of initial samples for training
- `retrain_interval`: How often to retrain (every N samples)
- `batch_size`: Size of data batches for evaluation

### Different Data Patterns
The `DataGenerator` can create different types of synthetic data:
- Linear relationships
- Non-linear relationships
- Custom patterns for specific use cases

## License

This project is open source and available under the MIT License.