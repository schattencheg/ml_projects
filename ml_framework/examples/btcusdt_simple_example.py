"""
BTCUSDT Simple ML Workflow Example

A simplified standalone example that demonstrates:
1. Downloading BTCUSDT data
2. Generating features
3. Training ML models
4. Testing and evaluation

This version uses direct imports to avoid dependency issues.

Author: ML Framework
Date: 2024-11-24
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("BTCUSDT SIMPLE ML WORKFLOW")
print("="*80)

# ========================================================================
# STEP 1: Download BTCUSDT Data
# ========================================================================
print("\n[STEP 1] DOWNLOADING BTCUSDT DATA")
print("-" * 80)

try:
    import yfinance as yf
    
    TICKER = 'BTC-USD'
    START_DATE = '2020-01-01'
    END_DATE = '2024-11-24'
    
    print(f"Downloading {TICKER} from {START_DATE} to {END_DATE}...")
    
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, interval='1d', progress=False)
    df.columns = df.columns.get_level_values(0)
    df.columns = [col.lower() for col in df.columns]
    
    print(f"✓ Downloaded {len(df)} rows")
    print(f"✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"\nLatest prices:")
    print(f"  Open:  ${df['open'].iloc[-1]:,.2f}")
    print(f"  High:  ${df['high'].iloc[-1]:,.2f}")
    print(f"  Low:   ${df['low'].iloc[-1]:,.2f}")
    print(f"  Close: ${df['close'].iloc[-1]:,.2f}")
    
except ImportError:
    print("❌ yfinance not installed. Install with: pip install yfinance")
    exit(1)

# ========================================================================
# STEP 2: Generate Technical Features
# ========================================================================
print("\n[STEP 2] GENERATING TECHNICAL FEATURES")
print("-" * 80)

def generate_features(df):
    """Generate technical indicator features."""
    data = df.copy()
    
    # Price-based features
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    
    # Moving Averages
    for period in [5, 10, 20, 50]:
        data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
        data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
    
    # Price relative to moving averages
    data['price_to_sma_20'] = data['close'] / data['sma_20']
    data['price_to_sma_50'] = data['close'] / data['sma_50']
    
    # Volatility
    data['volatility_10'] = data['returns'].rolling(window=10).std()
    data['volatility_20'] = data['returns'].rolling(window=20).std()
    
    # RSI (Relative Strength Index)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_diff'] = data['macd'] - data['macd_signal']
    
    # Bollinger Bands
    data['bb_middle'] = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    
    # Volume features
    data['volume_sma_20'] = data['volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_sma_20']
    
    # Price momentum
    for period in [5, 10, 20]:
        data[f'momentum_{period}'] = data['close'] - data['close'].shift(period)
        data[f'roc_{period}'] = (data['close'] - data['close'].shift(period)) / data['close'].shift(period) * 100
    
    # ATR (Average True Range)
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['atr_14'] = true_range.rolling(14).mean()
    
    return data

df_features = generate_features(df)
print(f"✓ Generated {len(df_features.columns)} total columns")

# ========================================================================
# STEP 3: Create Target Variable
# ========================================================================
print("\n[STEP 3] CREATING TARGET VARIABLE")
print("-" * 80)

FUTURE_BARS = 5
THRESHOLD = 0.02

# Calculate future return
df_features['future_return'] = df_features['close'].shift(-FUTURE_BARS) / df_features['close'] - 1

# Create binary target: 1 if price increases by THRESHOLD%, 0 otherwise
df_features['target'] = (df_features['future_return'] > THRESHOLD).astype(int)

print(f"Target: Predict if price will increase by {THRESHOLD*100}% in {FUTURE_BARS} days")

# Remove NaN values
df_features = df_features.dropna()

# Check class distribution
target_counts = df_features['target'].value_counts()
print(f"\nClass Distribution:")
print(f"  Class 0 (No increase): {target_counts[0]} samples ({target_counts[0]/len(df_features)*100:.1f}%)")
print(f"  Class 1 (Increase):    {target_counts[1]} samples ({target_counts[1]/len(df_features)*100:.1f}%)")
print(f"  Total samples: {len(df_features)}")

# ========================================================================
# STEP 4: Prepare Data for ML
# ========================================================================
print("\n[STEP 4] PREPARING DATA FOR ML")
print("-" * 80)

# Define feature columns (exclude OHLCV, target, and helper columns)
exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'future_return']
feature_cols = [col for col in df_features.columns if col not in exclude_cols]

print(f"Feature columns: {len(feature_cols)}")

# Split data
train_size = int(len(df_features) * 0.7)
val_size = int(len(df_features) * 0.15)

train_df = df_features.iloc[:train_size]
val_df = df_features.iloc[train_size:train_size+val_size]
test_df = df_features.iloc[train_size+val_size:]

print(f"\nData split:")
print(f"  Train: {len(train_df)} samples (70%)")
print(f"  Val:   {len(val_df)} samples (15%)")
print(f"  Test:  {len(test_df)} samples (15%)")

# Prepare X and y
X_train = train_df[feature_cols]
y_train = train_df['target']
X_val = val_df[feature_cols]
y_val = val_df['target']
X_test = test_df[feature_cols]
y_test = test_df['target']

# Scale features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")

# ========================================================================
# STEP 5: Train ML Models
# ========================================================================
print("\n[STEP 5] TRAINING ML MODELS")
print("-" * 80)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

trained_models = {}
results = {}

print(f"Training {len(models)} models...\n")

for name, model in models.items():
    print(f"Training {name}...")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict on validation set
    y_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    trained_models[name] = model
    
    print(f"  ✓ Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['f1_score'])
best_model = trained_models[best_model_name]

print(f"\n✓ Best model: {best_model_name} (F1: {results[best_model_name]['f1_score']:.4f})")

# ========================================================================
# STEP 6: Test on Test Set
# ========================================================================
print("\n[STEP 6] TESTING ON TEST SET")
print("-" * 80)

print(f"Testing best model ({best_model_name}) on test set...\n")

y_test_pred = best_model.predict(X_test_scaled)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

print(f"Test Set Results:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['No Increase', 'Increase']))

# ========================================================================
# STEP 7: Simple Backtest
# ========================================================================
print("\n[STEP 7] SIMPLE BACKTESTING")
print("-" * 80)

# Get predictions with probabilities
y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Create backtest dataframe
backtest_df = test_df.copy()
backtest_df['prediction'] = y_test_pred
backtest_df['prediction_proba'] = y_test_proba

# Simple strategy: buy when prediction = 1
initial_capital = 10000
capital = initial_capital
position = 0
trades = []

for i in range(len(backtest_df) - FUTURE_BARS):
    if backtest_df['prediction'].iloc[i] == 1 and position == 0:
        # Buy
        entry_price = backtest_df['close'].iloc[i]
        position = capital / entry_price
        capital = 0
        
    elif position > 0 and i >= len(backtest_df) - FUTURE_BARS - 1:
        # Sell at end
        exit_price = backtest_df['close'].iloc[i]
        capital = position * exit_price
        profit = capital - initial_capital
        trades.append(profit)
        position = 0

# Calculate final capital
if position > 0:
    capital = position * backtest_df['close'].iloc[-1]

total_return = (capital - initial_capital) / initial_capital * 100
buy_hold_return = (backtest_df['close'].iloc[-1] - backtest_df['close'].iloc[0]) / backtest_df['close'].iloc[0] * 100

print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"Final Capital:   ${capital:,.2f}")
print(f"Total Return:    {total_return:.2f}%")
print(f"Buy & Hold:      {buy_hold_return:.2f}%")
print(f"Difference:      {total_return - buy_hold_return:.2f}%")

# ========================================================================
# SUMMARY
# ========================================================================
print("\n" + "="*80)
print("WORKFLOW COMPLETE - SUMMARY")
print("="*80)

print(f"\n📊 Data:")
print(f"  ✓ Ticker: {TICKER}")
print(f"  ✓ Total samples: {len(df_features)}")
print(f"  ✓ Features: {len(feature_cols)}")
print(f"  ✓ Date range: {df_features.index[0].date()} to {df_features.index[-1].date()}")

print(f"\n🤖 Models:")
print(f"  ✓ Models trained: {len(models)}")
print(f"  ✓ Best model: {best_model_name}")
print(f"  ✓ Validation F1: {results[best_model_name]['f1_score']:.4f}")
print(f"  ✓ Test F1: {test_f1:.4f}")

print(f"\n💰 Backtest:")
print(f"  ✓ Strategy return: {total_return:.2f}%")
print(f"  ✓ Buy & Hold: {buy_hold_return:.2f}%")

print("\n" + "="*80)
print("🎉 BTCUSDT ML workflow completed successfully!")
print("="*80 + "\n")
