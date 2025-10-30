"""
Example: Using Comprehensive Cryptocurrency Features for ML Prediction

This script demonstrates how to use the add_crypto_features method to create
a rich set of technical indicators and features for cryptocurrency price prediction.

Features included:
- Price-based: Returns, momentum, spreads
- Moving averages: SMA, EMA, crossovers
- Volatility: Bollinger Bands, ATR, rolling std
- Momentum indicators: RSI, MACD, Stochastic, ROC
- Volume features: OBV, MFI, volume ratios
- Pattern features: Candlestick patterns
- Time-based: Cyclical encoding
- Statistical: Skewness, kurtosis, percentiles

Total: 150+ features automatically created!
"""

import pandas as pd
import numpy as np
from src.crypto_features import create_crypto_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_sample_data():
    """Load or create sample cryptocurrency data"""
    # For demonstration, create synthetic data
    # In practice, load your actual crypto data here
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=5000, freq='1H')
    
    # Simulate price movement
    price = 40000
    prices = []
    for _ in range(len(dates)):
        price *= (1 + np.random.normal(0, 0.01))
        prices.append(price)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
        'volume': [np.random.uniform(100, 1000) for _ in prices]
    })
    
    return df

def analyze_feature_importance(X_train, y_train, feature_names, top_n=20):
    """Analyze and visualize feature importance"""
    print(f"\n{'='*70}")
    print(f"FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*70}\n")
    
    # Train Random Forest to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Print top features
    print(f"Top {top_n} Most Important Features:")
    print(f"{'-'*70}")
    for idx, row in importance_df.head(top_n).iterrows():
        print(f"{row['feature']:40s} {row['importance']:.6f}")
    
    # Visualize
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
    plt.title(f'Top {top_n} Most Important Features')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Feature importance plot saved to: feature_importance.png")
    
    return importance_df

def analyze_feature_correlation(X_train, feature_names, threshold=0.8):
    """Analyze feature correlations to identify redundant features"""
    print(f"\n{'='*70}")
    print(f"FEATURE CORRELATION ANALYSIS")
    print(f"{'='*70}\n")
    
    # Calculate correlation matrix
    corr_matrix = X_train.corr().abs()
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        print(f"Found {len(high_corr_pairs)} highly correlated feature pairs (>{threshold}):")
        print(f"{'-'*70}")
        for pair in high_corr_pairs[:20]:  # Show top 20
            print(f"{pair['feature1']:30s} <-> {pair['feature2']:30s} : {pair['correlation']:.3f}")
        
        if len(high_corr_pairs) > 20:
            print(f"... and {len(high_corr_pairs)-20} more pairs")
    else:
        print(f"No highly correlated features found (threshold: {threshold})")
    
    # Visualize correlation heatmap for top features
    top_features = X_train.var().nlargest(30).index
    plt.figure(figsize=(14, 12))
    sns.heatmap(X_train[top_features].corr(), annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Heatmap (Top 30 Features by Variance)')
    plt.tight_layout()
    plt.savefig('feature_correlation.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Correlation heatmap saved to: feature_correlation.png")
    
    return high_corr_pairs

def main():
    """Main execution function"""
    print("="*70)
    print("CRYPTOCURRENCY FEATURE ENGINEERING EXAMPLE")
    print("="*70)
    
    # Step 1: Load data
    print("\nStep 1: Loading cryptocurrency data...")
    #df = load_sample_data()
    df = pd.read_csv("data/minute/btc_2025.csv")
    df.columns = df.columns.str.lower()
    
    print(f"✓ Loaded {len(df):,} rows of data")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Step 2: Generate features
    print("\nStep 2: Generating comprehensive crypto features...")
    
    # Use 2% price change as threshold (you can adjust this)
    result = create_crypto_features(df, price_change_threshold=0.02)
    
    X_train = result['X_train']
    y_train = result['y_train']
    X_val = result['X_val']
    y_val = result['y_val']
    X_test = result['X_test']
    y_test = result['y_test']
    feature_names = result['feature_names']
    
    # Step 3: Feature importance analysis
    print("\nStep 3: Analyzing feature importance...")
    importance_df = analyze_feature_importance(X_train, y_train, feature_names, top_n=30)
    
    # Step 4: Correlation analysis
    print("\nStep 4: Analyzing feature correlations...")
    high_corr_pairs = analyze_feature_correlation(X_train, feature_names, threshold=0.85)
    
    # Step 5: Train a model with all features
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL WITH ALL FEATURES")
    print(f"{'='*70}\n")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("Training Random Forest classifier...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test_scaled)
    
    print("\nTest Set Results:")
    print(f"{'-'*70}")
    print(classification_report(y_test, y_pred, target_names=['No Rise', 'Rise']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Step 6: Train model with top features only
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL WITH TOP 30 FEATURES ONLY")
    print(f"{'='*70}\n")
    
    top_30_features = importance_df.head(30)['feature'].tolist()
    X_train_top = X_train[top_30_features]
    X_test_top = X_test[top_30_features]
    
    X_train_top_scaled = scaler.fit_transform(X_train_top)
    X_test_top_scaled = scaler.transform(X_test_top)
    
    print("Training Random Forest with top 30 features...")
    rf_top = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_top.fit(X_train_top_scaled, y_train)
    
    y_pred_top = rf_top.predict(X_test_top_scaled)
    
    print("\nTest Set Results (Top 30 Features):")
    print(f"{'-'*70}")
    print(classification_report(y_test, y_pred_top, target_names=['No Rise', 'Rise']))
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total features created: {len(feature_names)}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"\nFeature categories:")
    print(f"  - Price-based: Returns, momentum, spreads")
    print(f"  - Moving averages: SMA, EMA (multiple periods)")
    print(f"  - Volatility: Bollinger Bands, ATR")
    print(f"  - Momentum: RSI, MACD, Stochastic, ROC")
    print(f"  - Volume: OBV, MFI, volume ratios")
    print(f"  - Patterns: Candlestick patterns")
    print(f"  - Time: Cyclical encoding")
    print(f"  - Statistical: Skewness, kurtosis, percentiles")
    print(f"\nRecommendations:")
    print(f"  1. Use top 30-50 features for best performance")
    print(f"  2. Remove highly correlated features (>0.85)")
    print(f"  3. Experiment with different price change thresholds")
    print(f"  4. Try ensemble methods combining multiple models")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
