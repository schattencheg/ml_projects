import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.FeaturesGenerator import FeaturesGenerator


path_1 = "data/btc_2024.csv"
path_2 = "data/btc_2025.csv"

def create_visualizations(results, close_test, y_test, idx_test):
    """
    Create and save visualizations for model performance.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results with keys: accuracy, f1, precision, recall, y_pred, model
    close_test : pd.Series
        Test set Close prices (not scaled)
    y_test : pd.Series
        Test set actual target labels
    idx_test : pd.Index
        Test set indices
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # 1. Overall Accuracy Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    f1_scores = [results[m]['f1'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='steelblue')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8, color='coral')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in model_names], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for rect in rects1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plots/overall_accuracy_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: plots/overall_accuracy_comparison.png")
    plt.close()
    
    # 2. Individual Model Predictions with Real Close Prices
    for model_name in model_names:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        y_pred = results[model_name]['y_pred']
        accuracy = results[model_name]['accuracy']
        
        # Create a DataFrame for easier plotting
        plot_df = pd.DataFrame({
            'Close': close_test.values,
            'Actual': y_test.values,
            'Predicted': y_pred
        }, index=idx_test)
        plot_df = plot_df.sort_index()
        
        # Top plot: Close prices with prediction markers
        ax1.plot(plot_df.index, plot_df['Close'], 'k-', linewidth=1.5, label='Close Price', alpha=0.7)
        
        # Mark correct predictions
        correct_mask = plot_df['Actual'] == plot_df['Predicted']
        increase_mask = plot_df['Predicted'] == 1
        
        # Correct predictions where model predicted increase
        correct_increase = correct_mask & increase_mask
        ax1.scatter(plot_df[correct_increase].index, plot_df[correct_increase]['Close'], 
                    color='green', marker='^', s=100, label='Correct: Predicted Increase', 
                    alpha=0.7, edgecolors='darkgreen', linewidths=1.5, zorder=5)
        
        # Correct predictions where model predicted no increase
        correct_no_increase = correct_mask & ~increase_mask
        ax1.scatter(plot_df[correct_no_increase].index, plot_df[correct_no_increase]['Close'], 
                    color='blue', marker='o', s=50, label='Correct: Predicted No Increase', 
                    alpha=0.5, edgecolors='darkblue', linewidths=1, zorder=4)
        
        # Incorrect predictions
        incorrect_mask = ~correct_mask
        ax1.scatter(plot_df[incorrect_mask].index, plot_df[incorrect_mask]['Close'], 
                    color='red', marker='x', s=100, label='Incorrect Prediction', 
                    alpha=0.8, linewidths=2, zorder=6)
        
        ax1.set_xlabel('Data Point Index', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Close Price', fontsize=11, fontweight='bold')
        ax1.set_title(f'{model_name.replace("_", " ").title()} - Predictions on Real Close Prices\nAccuracy: {accuracy:.4f}', 
                      fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Actual vs Predicted labels
        x_pos = np.arange(len(plot_df))
        ax2.scatter(x_pos, plot_df['Actual'], color='blue', marker='o', s=50, 
                    label='Actual', alpha=0.6, edgecolors='darkblue')
        ax2.scatter(x_pos, plot_df['Predicted'], color='red', marker='x', s=50, 
                    label='Predicted', alpha=0.6)
        
        ax2.set_xlabel('Sample Index (sorted by time)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Target Label', fontsize=11, fontweight='bold')
        ax2.set_title('Actual vs Predicted Labels (0=No Increase, 1=Increase)', fontsize=12, fontweight='bold')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['No Increase (0)', 'Increase (1)'])
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_filename = f'plots/{model_name}_predictions.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {plot_filename}")
        plt.close()
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETED")
    print("="*80)
    print(f"\nPlots saved in: plots/")
    print(f"  - overall_accuracy_comparison.png")
    for model_name in model_names:
        print(f"  - {model_name}_predictions.png")


# read data (Timestamp,Open,High,Low,Close,Volume)
df_original_1 = pd.read_csv(path_1)
#
df_prepared_1 = df_original_1.copy()
df_prepared_1.drop(['Timestamp', 'Volume'], axis=1, inplace=True)

fg = FeaturesGenerator()
# Add target: 3% increase in 45 bars
df_prepared_1 = fg.add_target(df_prepared_1, 45, 3)
# Add features
df_prepared_1 = fg.add_features(df_prepared_1)
# Add returns
df_prepared_1 = fg.returnificate(df_prepared_1)
# Drop rows with NaN
df_prepared_1.dropna(inplace=True)

# Store Close prices and index for plotting later (before dropping)
close_prices = df_prepared_1['Close'].copy()
original_index = df_prepared_1.index.copy()

# Drop Open, High, Low, Close columns
df_prepared_1.drop(['Open', 'High', 'Low', 'Close'], axis=1, inplace=True)

# Separate features (X) from target (y)
X = df_prepared_1.drop(['target', 'pct_change_N'], axis=1)
y = df_prepared_1['target']

print(f"Dataset shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")
print(f"Target balance: {y.value_counts(normalize=True)}")
print()

# Split data (also split close prices and indices for plotting)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define classification models to train
models = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        {"max_iter": 1000}
    ),
    #"decision_tree": (DecisionTreeClassifier(max_depth=10, random_state=42),{"max_depth": 10}),
    #"random_forest": (RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),{"n_estimators": 50, "max_depth": 8}),
    #"gradient_boosting": (GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),{"n_estimators": 50, "max_depth": 3})
}

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate models
print("="*80)
print("TRAINING AND EVALUATING MODELS")
print("="*80)

best_model = None
best_score = 0
best_model_name = ""

# Store results for plotting
results = {}

for model_name, model_data in models.items():
    print(f"\n{'='*80}")
    print(f"Model: {model_name.upper()}")
    print(f"{'='*80}")
    
    model = model_data[0]
    params = model_data[1]
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    
    # Cross validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"\nCross Validation (5-fold):")
    print(f"  Scores: {cv_scores}")
    print(f"  Mean:   {cv_scores.mean():.4f}")
    print(f"  Std:    {cv_scores.std():.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Increase', 'Increase']))
    
    # Store results for plotting
    results[model_name] = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'y_pred': y_pred,
        'model': model
    }
    
    # Track best model
    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        best_model_name = model_name

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_model_name.upper()} with accuracy: {best_score:.4f}")
print(f"{'='*80}")

# Save the best model
os.makedirs('models', exist_ok=True)
model_path = f'models/{best_model_name}_best.joblib'
joblib.dump(best_model, model_path)
print(f"\nBest model saved to: {model_path}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# Call the visualization function
#create_visualizations(results, close_test, y_test, idx_test)


# NOW TEST ON NEXT YEAR DATA
df_original_2 = pd.read_csv(path_2)
df_prepared_2 = df_original_2.copy()
df_prepared_2.drop(['Timestamp', 'Volume'], axis=1, inplace=True)

fg = FeaturesGenerator()
# Add target: 3% increase in 45 bars
df_prepared_2 = fg.add_target(df_prepared_2, 45, 3)
# Add features
df_prepared_2 = fg.add_features(df_prepared_2)
# Add returns
df_prepared_2 = fg.returnificate(df_prepared_2)
# Drop rows with NaN
df_prepared_2.dropna(inplace=True)

# Drop Open, High, Low, Close columns
df_prepared_2.drop(['Open', 'High', 'Low', 'Close'], axis=1, inplace=True)
df_prepared_2 = df_prepared_2.drop(['pct_change_N'], axis=1)

#scaler = StandardScaler()
#scaler.transform(df_prepared_2)
df_prepared_2_scaled = scaler.transform(df_prepared_2)

# Separate features (X) from target (y)
y_test_scaled = df_prepared_2['target']
X_test_scaled = df_prepared_2.drop(['target'], axis=1)


print(f"\n{'='*80}")
print(f"TESTING ON NEXT YEAR DATA")
print(f"{'='*80}")

for model_name, model_data in models.items():
    print(f"\n{'='*80}")
    print(f"Model: {model_name.upper()}")
    print(f"{'='*80}")
    
    model = model_data[0]
    params = model_data[1]
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_scaled, y_pred_scaled)
    f1 = f1_score(y_test_scaled, y_pred_scaled)
    precision = precision_score(y_test_scaled, y_pred_scaled)
    recall = recall_score(y_test_scaled, y_pred_scaled)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    
    # Cross validation
    cv_scores = cross_val_score(model, X_test_scaled, y_test_scaled, cv=5, scoring='accuracy')
    print(f"\nCross Validation (5-fold):")
    print(f"  Scores: {cv_scores}")
    print(f"  Mean:   {cv_scores.mean():.4f}")
    print(f"  Std:    {cv_scores.std():.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test_scaled, y_pred_scaled, target_names=['No Increase', 'Increase']))
    
    # Store results for plotting
    results[model_name] = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'y_pred': y_pred_scaled,
        'model': model
    }
    
    # Track best model
    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        best_model_name = model_name
