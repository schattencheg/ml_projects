# Feature Testing Guide

## Overview

The `test_features()` method in `FeaturesGenerator` helps you evaluate the quality of your feature set **before** training models.

## Important Notes

⚠️ **`test_features()` is for ANALYSIS ONLY, not for data transformation!**

- It analyzes feature quality and provides recommendations
- It does NOT transform or modify your data
- Use it to evaluate features, then train models separately

## Usage

### Method 1: Using the Analysis Script (Recommended)

```bash
python analyze_features.py
```

This will:
1. Load your data
2. Generate features
3. Analyze feature quality
4. Save results to CSV files
5. Print recommendations

### Method 2: In Your Own Script

```python
from src.FeaturesGenerator import FeaturesGenerator
from src.data_preparation import prepare_data
import pandas as pd

# Load data
df = pd.read_csv('data/hour/btc.csv')

# Prepare features and target
X, y = prepare_data(df, target_bars=15, target_pct=1.0)

# Combine for analysis
df_with_target = X.copy()
df_with_target['target'] = y

# Analyze features
fg = FeaturesGenerator()
results = fg.test_features(df_with_target, target_col='target', top_n=20)

# Check results
print(f"Rating: {results['rating']}")
print(f"Score: {results['score']}/{results['max_score']}")
```

### Method 3: In Jupyter Notebook

```python
# Cell 1: Import and load data
from src.FeaturesGenerator import FeaturesGenerator
from src.data_preparation import prepare_data
import pandas as pd

df_train = pd.read_csv('data/hour/btc.csv')

# Cell 2: Prepare data
X, y = prepare_data(df_train, target_bars=15, target_pct=1.0)
df_with_target = X.copy()
df_with_target['target'] = y

# Cell 3: Analyze features
fg = FeaturesGenerator()
results = fg.test_features(df_with_target, target_col='target', top_n=20)

# Cell 4: View results
print(f"Rating: {results['rating']}")
results['correlation_df'].head(10)  # Top correlated features
```

## What Gets Analyzed

### 1. Missing Values
- Identifies features with missing data
- Shows percentage missing

### 2. Feature-Target Correlation
- Linear correlation with target
- Top N most predictive features

### 3. Mutual Information
- Non-linear relationships
- Captures complex patterns

### 4. Feature Redundancy
- Highly correlated feature pairs (>0.8)
- Multicollinearity issues

### 5. Feature Statistics
- Mean, std, min, max
- Low variance features
- Zero value percentages

### 6. Overall Rating
- Scores on 5 criteria (0-5)
- Rating: EXCELLENT, GOOD, FAIR, or POOR
- Actionable recommendations

## Results Dictionary

```python
results = {
    'correlation_df': DataFrame,      # Feature-target correlations
    'mutual_info_df': DataFrame,      # Mutual information scores
    'redundancy_df': DataFrame,       # Redundant pairs (or None)
    'missing_df': DataFrame,          # Missing values (or None)
    'stats_df': DataFrame,            # Feature statistics
    'score': int,                     # Overall score (0-5)
    'max_score': int,                 # Maximum score (5)
    'rating': str,                    # EXCELLENT/GOOD/FAIR/POOR
    'metrics': {
        'avg_abs_corr': float,        # Average |correlation|
        'max_abs_corr': float,        # Maximum |correlation|
        'top10_avg_corr': float,      # Top 10 avg |correlation|
        'avg_mi': float,              # Average mutual information
        'max_mi': float,              # Maximum mutual information
        'redundancy_ratio': float     # Redundancy ratio
    }
}
```

## Rating Criteria

### EXCELLENT (Score: 4-5)
✓ Strong predictive features (top 10 avg |corr| > 0.15)
✓ At least one very strong feature (max |corr| > 0.25)
✓ Low redundancy (<10%)
✓ No missing values
✓ All features have sufficient variance

### GOOD (Score: 3)
- Most criteria met
- Minor issues to address

### FAIR (Score: 2)
- Some weak features
- Moderate redundancy or missing values

### POOR (Score: 0-1)
- Weak predictive power
- High redundancy
- Many missing values or low variance features

## Common Workflow

```python
# 1. Analyze features
results = fg.test_features(df_with_target)

# 2. Check rating
if results['rating'] in ['EXCELLENT', 'GOOD']:
    print("✓ Features look good! Proceed with training")
else:
    print("⚠ Consider improving features before training")

# 3. Identify top features
top_features = results['correlation_df'].head(10)['Feature'].tolist()
print(f"Top features: {top_features}")

# 4. Remove redundant features if needed
if results['redundancy_df'] is not None:
    redundant = results['redundancy_df']['Feature_2'].unique()
    print(f"Consider removing: {list(redundant)}")

# 5. Train models with good features
models, scaler, train_results, best_model, label_encoder = train(X, y)
```

## Troubleshooting

### Error: 'FeaturesGenerator' object has no attribute 'test_features'

**Solution 1**: Restart Python kernel
- Jupyter: Kernel → Restart Kernel
- VS Code/PyCharm: Stop and restart

**Solution 2**: Force reload module
```python
import sys
import importlib

if 'src.FeaturesGenerator' in sys.modules:
    importlib.reload(sys.modules['src.FeaturesGenerator'])

from src.FeaturesGenerator import FeaturesGenerator
```

### Error: pd.concat() missing required positional argument

**Wrong**:
```python
df_combined = pd.concat(X, y)  # ❌ Wrong!
```

**Correct**:
```python
df_combined = X.copy()
df_combined['target'] = y  # ✓ Correct!
```

## Example Output

```
================================================================================
FEATURE QUALITY ANALYSIS
================================================================================

Dataset Info:
  Total samples: 10000
  Total features: 45
  Target classes: [-1, 0, 1]

================================================================================
2. FEATURE-TARGET CORRELATION
================================================================================

Top 20 features by correlation with target:
           Feature  Correlation
      RSI_return       0.2341
     MACD_return       0.1987
  BB_width_pct        0.1654
       ...

================================================================================
FEATURE SET RATING:
--------------------------------------------------------------------------------
✓ Strong predictive features (top 10 avg |corr| > 0.15)
✓ At least one very strong feature (max |corr| > 0.25)
✓ Low feature redundancy (<10%)
✓ No missing values
✓ All features have sufficient variance

Overall Score: 5/5
Rating: EXCELLENT

================================================================================
RECOMMENDATIONS
================================================================================
✓ Feature set looks good! No major issues found.
```

## Best Practices

1. **Analyze features BEFORE training**
   - Saves time by identifying issues early
   - Helps you improve feature engineering

2. **Compare different feature sets**
   - Try different technical indicators
   - Compare ratings and scores

3. **Remove redundant features**
   - Reduces multicollinearity
   - Speeds up training
   - May improve model performance

4. **Focus on top features**
   - Use top 10-20 features for faster training
   - Less risk of overfitting

5. **Iterate and improve**
   - If rating is POOR or FAIR, add more features
   - Try feature transformations (log, sqrt, etc.)
   - Create interaction terms

## Files Created

- `analyze_features.py` - Ready-to-use analysis script
- `feature_analysis_top_features.csv` - Top 20 features by correlation
- `feature_analysis_redundant_pairs.csv` - Redundant feature pairs (if any)

## Next Steps

After analyzing features:

1. If rating is GOOD or EXCELLENT → Train models
2. If rating is FAIR → Consider adding more features
3. If rating is POOR → Redesign feature engineering

Then proceed with training:

```python
from src.model_training import train

models, scaler, results, best_model, label_encoder = train(X, y)
```
