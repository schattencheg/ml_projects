"""
Data Diagnostics Script

Run this to understand why models have poor F1 scores.
"""

import pandas as pd
from src.data_preparation import prepare_data

# Load your training data
PATH_TRAIN = "data/btc_2022.csv"
df_train = pd.read_csv(PATH_TRAIN)

print("="*80)
print("DATA DIAGNOSTICS")
print("="*80)

# Test different target_pct values
for target_pct in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    print(f"\n{'='*80}")
    print(f"Testing target_pct = {target_pct}%")
    print(f"{'='*80}")
    
    try:
        X, y = prepare_data(df_train, target_bars=45, target_pct=target_pct)
        
        # Calculate class distribution
        class_counts = y.value_counts()
        total = len(y)
        
        if 0 in class_counts and 1 in class_counts:
            no_increase = class_counts[0]
            increase = class_counts[1]
            pct_increase = (increase / total) * 100
            imbalance_ratio = no_increase / increase if increase > 0 else float('inf')
            
            print(f"Total samples: {total}")
            print(f"No Increase (0): {no_increase} ({(no_increase/total)*100:.1f}%)")
            print(f"Increase (1):    {increase} ({pct_increase:.1f}%)")
            print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
            
            # Diagnosis
            if pct_increase < 1:
                print("⚠️  WARNING: Less than 1% positive samples - TOO FEW!")
                print("   → Models will predict all negative (F1 ≈ 0)")
                print("   → SOLUTION: Decrease target_pct")
            elif pct_increase < 5:
                print("⚠️  WARNING: Less than 5% positive samples - VERY IMBALANCED")
                print("   → Models struggle to learn positive class")
                print("   → SOLUTION: Use SMOTE or decrease target_pct")
            elif pct_increase < 15:
                print("⚠️  CAUTION: Less than 15% positive samples - IMBALANCED")
                print("   → SMOTE recommended")
                print("   → Should work with class weights")
            elif pct_increase < 30:
                print("✓  GOOD: 15-30% positive samples - ACCEPTABLE")
                print("   → SMOTE will help further")
            else:
                print("✓  EXCELLENT: >30% positive samples - BALANCED")
                print("   → No SMOTE needed")
                
        else:
            print("❌ ERROR: Only one class present!")
            if 1 not in class_counts:
                print("   → No positive samples found!")
                print("   → SOLUTION: Decrease target_pct significantly")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

print(f"\n{'='*80}")
print("RECOMMENDATIONS")
print(f"{'='*80}")
print("""
Based on the diagnostics above:

1. If positive samples < 5%:
   → Use target_pct between 1.0-2.0%
   → Enable SMOTE
   → Expect F1 scores: 0.3-0.5

2. If positive samples 5-15%:
   → Use target_pct between 2.0-3.0%
   → Enable SMOTE
   → Expect F1 scores: 0.4-0.6

3. If positive samples 15-30%:
   → Use target_pct between 3.0-5.0%
   → SMOTE optional
   → Expect F1 scores: 0.5-0.7

4. If positive samples > 30%:
   → Use target_pct > 5.0%
   → No SMOTE needed
   → Expect F1 scores: 0.6-0.8

Current issue (F1 ≈ 0.005):
→ Almost certainly means positive samples < 1%
→ Models predict all negative class
→ Need to DECREASE target_pct to 1.5-2.0%
""")

print(f"\n{'='*80}")
print("NEXT STEPS")
print(f"{'='*80}")
print("""
1. Look at the diagnostics above
2. Find target_pct with 10-20% positive samples
3. Update run_me.py with that target_pct
4. Enable use_smote=True
5. Re-run training

Example:
    If 2.0% gives 15% positive samples:
    
    models, scaler, results, best_model = train(
        df_train,
        target_bars=45,
        target_pct=2.0,  # ← Use this value
        use_smote=True,
        use_gpu=False,
        n_jobs=-1
    )
""")
