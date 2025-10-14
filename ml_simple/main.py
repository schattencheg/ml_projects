import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout

# --- Data Preparation ---
def create_dataset(X, y, look_back=1):
    """Create a dataset with a lookback window for time series."""
    dataX, dataY = [], []
    for i in range(len(X) - look_back - 1):
        a = X[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(y[i + look_back])
    return np.array(dataX), np.array(dataY)

# --- Model Definitions ---
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        Flatten(),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Load and prepare data
data = pd.read_csv('data/6e_2007_2019.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
unique_years = sorted(data['Year'].unique())

features = ['Open', 'High', 'Low', 'Close']
target = 'Close' # Predicting the next day's close

# --- Main Loop ---
look_back = 3
all_results = {year: {} for year in unique_years[1:]}

for i in range(len(unique_years) - 1):
    train_year = unique_years[i]
    eval_year = unique_years[i+1]

    print(f"\n{'='*60}")
    print(f"--- Training on {train_year}, Evaluating on {eval_year} ---")
    print(f"{'='*60}\n")

    train_df = data[data['Year'] == train_year]
    eval_df = data[data['Year'] == eval_year]

    # Create sequences
    X_train, y_train = create_dataset(train_df[features].values, train_df[target].values, look_back)
    X_eval, y_eval = create_dataset(eval_df[features].values, eval_df[target].values, look_back)

    # Reshape for scikit-learn models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_eval_flat = X_eval.reshape(X_eval.shape[0], -1)

    # --- Model Initialization ---
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
        "LSTM": create_lstm_model((X_train.shape[1], X_train.shape[2])),
        "CNN": create_cnn_model((X_train.shape[1], X_train.shape[2]))
    }

    for name, model in models.items():
        print(f"--- Training {name} ---")

        if name in ["LSTM", "CNN"]:
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
            predictions = model.predict(X_eval, verbose=0).flatten()
        else:
            model.fit(X_train_flat, y_train)
            predictions = model.predict(X_eval_flat)

        # --- Evaluate and Store Results ---
        rmse = np.sqrt(mean_squared_error(y_eval, predictions))
        mae = mean_absolute_error(y_eval, predictions)
        r2 = r2_score(y_eval, predictions)
        all_results[eval_year][name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}

        print(f"Results for {name}:")
        print(f"  RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")

# --- Plotting Results ---
fig, axs = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
metrics = ['RMSE', 'MAE', 'R2']
model_names = list(models.keys())

for i, metric in enumerate(metrics):
    for model_name in model_names:
        metric_values = [all_results[year][model_name][metric] for year in unique_years[1:]]
        axs[i].plot(unique_years[1:], metric_values, marker='o', linestyle='-', label=model_name)
    axs[i].set_title(f'{metric} per Year')
    axs[i].set_ylabel(metric)
    axs[i].legend()
    axs[i].grid(True)

plt.xlabel('Year')
plt.tight_layout()
plt.show()

print("\nWalk-forward analysis complete.")
