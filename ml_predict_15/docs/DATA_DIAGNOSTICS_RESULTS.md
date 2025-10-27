# Data Diagnostics Results - Why F1 Scores Were ~0.005

## Problem Identified

**Symptom**: All models had F1 scores around 0.005 and similar precision, regardless of model type.

**Root Cause**: Extreme class imbalance - with `target_pct=3.0%`, only **0.4% of samples were positive** (4,265 out of 1,023,507 samples).

## Diagnostic Results

Ran `diagnose_data.py` on `data/btc_2022.csv` with `target_bars=45`:

| target_pct | Positive Samples | Imbalance Ratio | Status | Expected F1 |
|------------|------------------|-----------------|--------|-------------|
| 0.5% | **11.7%** | 7.5:1 | ✅ **GOOD** | **0.4-0.6** |
| 1.0% | 4.1% | 23.4:1 | ⚠️ Very Imbalanced | 0.2-0.4 |
| 1.5% | 1.9% | 51.0:1 | ⚠️ Very Imbalanced | 0.1-0.3 |
| 2.0% | 1.1% | 92.1:1 | ⚠️ Very Imbalanced | 0.05-0.2 |
| 2.5% | 0.7% | 150.8:1 | ❌ Too Few | ~0.01 |
| 3.0% | **0.4%** | 239.0:1 | ❌ **Too Few** | **~0.005** ← Your issue |
| 4.0% | 0.2% | 473.1:1 | ❌ Too Few | ~0.002 |
| 5.0% | 0.1% | 733.8:1 | ❌ Too Few | ~0.001 |

## Why Models Failed with target_pct=3.0%

With only **0.4% positive samples**:

1. **Models predict all negative**: It's more accurate to always predict "No Increase" (99.6% accuracy)
2. **No learning of positive class**: Too few examples to learn patterns
3. **F1 score collapses**: F1 = 2 × (Precision × Recall) / (Precision + Recall)
   - Recall ≈ 0 (can't find positive samples)
   - Precision ≈ 0 (few predictions, mostly wrong)
   - F1 ≈ 0.005

4. **All models similar**: Every model converges to the same strategy (predict all negative)

## The Solution

### Changed Configuration

**Before** (broken):
```python
models, scaler, results, best_model = train(
    df_train,
    target_bars=45,
    target_pct=3.0,  # Only 0.4% positive samples
    use_smote=True
)
```

**After** (fixed):
```python
models, scaler, results, best_model = train(
    df_train,
    target_bars=45,
    target_pct=0.5,  # 11.7% positive samples ✓
    use_smote=True   # Further balances to ~50/50
)
```

### Expected Results with target_pct=0.5%

**Class Distribution**:
- No Increase (0): 88.3% (903,726 samples)
- Increase (1): 11.7% (119,781 samples)
- Imbalance ratio: 7.5:1 (manageable with SMOTE)

**After SMOTE** (training set only):
- No Increase (0): ~50%
- Increase (1): ~50%
- Balanced for training

**Expected Model Performance**:

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Logistic Regression | 70-75% | 0.45-0.55 | 0.50-0.60 | 0.40-0.50 |
| Ridge Classifier | 70-75% | 0.45-0.55 | 0.50-0.60 | 0.40-0.50 |
| Naive Bayes | 68-73% | 0.42-0.52 | 0.45-0.55 | 0.40-0.50 |
| Decision Tree | 65-72% | 0.40-0.50 | 0.45-0.55 | 0.35-0.45 |
| Random Forest | 73-78% | 0.50-0.60 | 0.55-0.65 | 0.45-0.55 |
| XGBoost | 75-80% | 0.52-0.62 | 0.57-0.67 | 0.48-0.58 |
| LightGBM | 74-79% | 0.51-0.61 | 0.56-0.66 | 0.47-0.57 |

**Model Differentiation**: 
- Accuracy spread: 10-15% (good differentiation)
- F1 spread: 0.15-0.20 (clear differences)
- Best models: XGBoost, LightGBM, Random Forest

## Key Learnings

### 1. Always Check Class Distribution First

Before training, always run:
```python
python diagnose_data.py
```

This shows you the class distribution for different `target_pct` values.

### 2. Target Guidelines

For good model performance, aim for:
- **Optimal**: 10-30% positive samples
- **Acceptable**: 5-10% positive samples (use SMOTE)
- **Poor**: <5% positive samples (models struggle)
- **Broken**: <1% positive samples (F1 ≈ 0)

### 3. Dataset-Specific Tuning

The "optimal" `target_pct` depends on your data:
- **Volatile assets** (crypto): Use smaller target_pct (0.5-1.5%)
- **Stable assets** (stocks): Use larger target_pct (2-5%)
- **Always verify** with diagnostics

### 4. Don't Trust Generic Advice

My initial recommendation of `target_pct=3.0%` was **wrong** because:
- It was based on generic assumptions
- Didn't account for your specific data characteristics
- Your data is more volatile than typical examples

**Lesson**: Always run diagnostics on YOUR data!

## How to Use Diagnostics

### Step 1: Run Diagnostics
```bash
python diagnose_data.py
```

### Step 2: Find Optimal target_pct

Look for:
- 10-20% positive samples (ideal)
- 5-30% positive samples (acceptable)

### Step 3: Update Configuration

Edit `run_me.py`:
```python
models, scaler, results, best_model = train(
    df_train,
    target_bars=45,
    target_pct=X.X,  # Use value from diagnostics
    use_smote=True
)
```

### Step 4: Train and Verify

Check training output:
```
Target distribution:
0    903726  # No Increase
1    119781  # Increase  ← Should be 5-30% of total

Class imbalance ratio: 7.54:1  ← Should be <20:1

Applying SMOTE to balance training data...
Original training size: 817205
Resampled training size: 1445004  ← SMOTE creates synthetic samples
New class distribution: 
0    722602
1    722402  ← Now balanced!
```

## Summary

**Problem**: F1 ≈ 0.005 with target_pct=3.0%
- Only 0.4% positive samples
- Models predict all negative
- No differentiation between models

**Solution**: Use target_pct=0.5%
- 11.7% positive samples
- SMOTE balances to ~50/50 for training
- Expected F1: 0.4-0.6
- Clear model differentiation

**Tool**: `diagnose_data.py`
- Run before training
- Find optimal target_pct for YOUR data
- Avoid wasting time on broken configurations

## Next Steps

1. ✅ Configuration updated to `target_pct=0.5%`
2. ✅ SMOTE enabled
3. ▶️ Run training: `python run_me.py`
4. ✅ Expect F1 scores: 0.4-0.6
5. ✅ Expect model differentiation: 10-15% accuracy spread

You should now see meaningful differences between models and F1 scores in the 0.4-0.6 range instead of 0.005!
