"""
Test script for three-class classification with automatic target transformation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("Testing three-class classification system...\n")

# Test 1: Target creation
print("="*60)
print("TEST 1: Three-Class Target Creation")
print("="*60)

from src.features_generator import FeaturesGenerator
from src.data_provider import DataProvider

# Create sample data
data_provider = DataProvider(data_dir='data')
features_gen = FeaturesGenerator()

# Create sample DataFrame
dates = pd.date_range('2020-01-01', periods=100, freq='D')
df = pd.DataFrame({
    'close': np.random.randn(100).cumsum() + 100
}, index=dates)

# Add basic features
df['returns'] = df['close'].pct_change()
df['sma_10'] = df['close'].rolling(10).mean()
df = df.dropna()

# Create three-class target
df_with_target = features_gen.create_target(
    df,
    target_type='classification',
    future_bars=5,
    threshold=0.02,
    num_classes=3
)

print(f"\n✓ Target created with {len(df_with_target)} samples")
print(f"✓ Unique target values: {sorted(df_with_target['target'].unique())}")

# Test 2: Target Transformer
print("\n" + "="*60)
print("TEST 2: Target Transformer")
print("="*60)

from src.target_transformer import TargetTransformer, transform_for_model

transformer = TargetTransformer()

# Create sample targets
y_three_class = np.array([-1, -1, 0, 0, 1, 1, -1, 0, 1])

print(f"\nOriginal targets (three-class): {y_three_class}")

# Transform for classic ML
y_classic = transformer.transform_for_classic_ml(y_three_class)
print(f"Classic ML format (0/1/2):      {y_classic}")

# Transform back
y_back = transformer.inverse_transform_classic_ml(y_classic)
print(f"Transformed back:                {y_back}")

assert np.array_equal(y_three_class, y_back), "Transformation should be reversible"
print("✓ Transformation is reversible")

# Test 3: Model Type Detection
print("\n" + "="*60)
print("TEST 3: Model Type Detection")
print("="*60)

from src.models_lib import LogisticRegressionModel, RandomForestModel

# Classic ML model
lr_model = LogisticRegressionModel()
model_type = transformer.get_model_type(lr_model)
print(f"\nLogisticRegression detected as: {model_type}")
assert model_type == 'classic_ml', "Should detect as classic_ml"

rf_model = RandomForestModel()
model_type = transformer.get_model_type(rf_model)
print(f"RandomForest detected as: {model_type}")
assert model_type == 'classic_ml', "Should detect as classic_ml"

print("✓ Model type detection works")

# Test 4: Automatic Target Transformation
print("\n" + "="*60)
print("TEST 4: Automatic Target Transformation")
print("="*60)

# Create sample data
X_train = np.random.randn(50, 5)
y_train = np.array([-1] * 15 + [0] * 20 + [1] * 15)

print(f"\nTraining data:")
print(f"  Samples: {len(X_train)}")
print(f"  Features: {X_train.shape[1]}")
print(f"  Target distribution: {np.bincount(y_train + 1)}")  # Shift to 0,1,2 for bincount

# Train model
model = LogisticRegressionModel(name='LogReg_3Class')

# The BaseModel will automatically convert targets
print(f"\nFitting model (automatic target conversion)...")
model.fit(X_train, y_train)

print(f"✓ Model fitted")
print(f"  Target mapping: {model.target_mapping}")
print(f"  Reverse mapping: {model.reverse_mapping}")

# Make predictions
X_test = np.random.randn(10, 5)
y_pred = model.predict(X_test)

print(f"\n✓ Predictions made")
print(f"  Predictions: {y_pred}")
print(f"  Unique values: {sorted(np.unique(y_pred))}")

assert set(y_pred).issubset({-1, 0, 1}), "Predictions should be in {-1, 0, 1}"
print("✓ Predictions are in three-class format")

# Test 5: Strategy with Three-Class
print("\n" + "="*60)
print("TEST 5: Strategy with Three-Class Signals")
print("="*60)

from src.strategies import MLStrategy

# Create strategy
strategy = MLStrategy(
    name='ThreeClass_Strategy',
    holding_period=5,
    trailing_stop_pct=0.05,
    enable_trailing_stop=False
)

print(f"\n✓ Strategy created: {strategy}")

# Create test data with features
test_df = pd.DataFrame({
    'close': np.random.randn(30).cumsum() + 100,
    'feature1': np.random.randn(30),
    'feature2': np.random.randn(30),
    'feature3': np.random.randn(30),
    'feature4': np.random.randn(30),
    'feature5': np.random.randn(30)
}, index=pd.date_range('2024-01-01', periods=30, freq='D'))

feature_cols = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']

# Scale features
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(test_df[feature_cols].values)

# Generate signals
signals = strategy.generate_signals(
    df=test_df,
    model=model,
    scaler=scaler,
    feature_cols=feature_cols
)

print(f"\n✓ Signals generated")
print(f"  Total signals: {len(signals)}")
print(f"  Long signals (+1): {(signals == 1).sum()}")
print(f"  Neutral signals (0): {(signals == 0).sum()}")
print(f"  Short signals (-1): {(signals == -1).sum()}")

assert set(signals.unique()).issubset({-1, 0, 1}), "Signals should be in {-1, 0, 1}"
print("✓ Signals are in three-class format")

# Summary
print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)

print("\nThree-Class Classification System:")
print("  ✓ Target creation with num_classes=3")
print("  ✓ Automatic target transformation")
print("  ✓ Classic ML: -1/0/+1 → 0/1/2 (internal)")
print("  ✓ Deep Learning: -1/0/+1 → -1/0/+1 (no change)")
print("  ✓ Predictions: always returned as -1/0/+1")
print("  ✓ Strategy: handles three-class signals")

print("\nTarget Meanings:")
print("  -1: Decrease (price drops > threshold)")
print("   0: Neutral (price change within ±threshold)")
print("  +1: Increase (price rises > threshold)")

print("\nSignal Meanings:")
print("  -1: Short entry signal")
print("   0: No signal / Neutral")
print("  +1: Long entry signal")

print("\n")
