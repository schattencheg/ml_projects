"""
Feature Quality Analysis Script

This script analyzes the quality of generated features for ML prediction.
Use this to evaluate if your feature set is good before training models.
"""

import pandas as pd
import sys
import importlib

# Force reload to get latest changes
if 'src.FeaturesGenerator' in sys.modules:
    importlib.reload(sys.modules['src.FeaturesGenerator'])

from src.FeaturesGenerator import FeaturesGenerator
from src.data_preparation import prepare_data

# Data path
PATH_TRAIN = "data/hour/btc.csv"

print("="*80)
print("FEATURE QUALITY ANALYSIS")
print("="*80)

# Load data
print("\nLoading data...")
df_train = pd.read_csv(PATH_TRAIN)
print(f"Data shape: {df_train.shape}")

# Prepare data with features and target
print("\nPreparing features and target...")
target_bars = 15
target_pct = 3.0

X, y = prepare_data(df_train, target_bars=target_bars, target_pct=target_pct)

# Combine features and target for analysis
df_with_target = X.copy()
df_with_target['target'] = y

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts().sort_index()}")

# Analyze feature quality
print("\n" + "="*80)
print("ANALYZING FEATURE QUALITY...")
print("="*80 + "\n")

fg = FeaturesGenerator()
results = fg.test_features(df_with_target, target_col='target', top_n=20)

# Print summary
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nFeature Set Rating: {results['rating']}")
print(f"Score: {results['score']}/{results['max_score']}")
print(f"\nKey Metrics:")
print(f"  Average |correlation|: {results['metrics']['avg_abs_corr']:.4f}")
print(f"  Maximum |correlation|: {results['metrics']['max_abs_corr']:.4f}")
print(f"  Top 10 avg |corr|:     {results['metrics']['top10_avg_corr']:.4f}")
print(f"  Redundancy ratio:      {results['metrics']['redundancy_ratio']:.4f}")

# Save top features to file
if results['correlation_df'] is not None:
    top_features = results['correlation_df'].head(20)
    top_features.to_csv('feature_analysis_top_features.csv', index=False)
    print(f"\n✓ Top 20 features saved to: feature_analysis_top_features.csv")

# Save redundant features if any
if results['redundancy_df'] is not None:
    results['redundancy_df'].to_csv('feature_analysis_redundant_pairs.csv', index=False)
    print(f"✓ Redundant feature pairs saved to: feature_analysis_redundant_pairs.csv")

print("\n" + "="*80)
print("Use this information to improve your feature engineering!")
print("="*80)
