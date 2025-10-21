import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.FeaturesGenerator import FeaturesGenerator


path_1 = "data/btc_2022.csv"
path_2 = "data/btc_2023.csv"

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


def prepare_data(df_raw: pd.DataFrame, target_bars: int = 45, target_pct: float = 3.0):
    """
    Prepare raw OHLCV data by adding features, target, and returns.
    
    Parameters:
    -----------
    df_raw : pd.DataFrame
        Raw dataframe with columns: Timestamp, Open, High, Low, Close, Volume (optional)
    target_bars : int
        Number of bars to look ahead for target
    target_pct : float
        Percentage increase threshold for target
    
    Returns:
    --------
    X : pd.DataFrame
        Features (without target and pct_change_N)
    y : pd.Series
        Target labels
    """
    df_prepared = df_raw.copy()
    
    # Drop unnecessary columns
    if 'Timestamp' in df_prepared.columns:
        df_prepared.drop(['Timestamp'], axis=1, inplace=True)
    if 'Volume' in df_prepared.columns:
        df_prepared.drop(['Volume'], axis=1, inplace=True)
    
    # Feature engineering
    fg = FeaturesGenerator()
    df_prepared = fg.add_target(df_prepared, target_bars, target_pct)
    df_prepared = fg.add_features(df_prepared)
    df_prepared = fg.returnificate(df_prepared)
    df_prepared.dropna(inplace=True)
    
    # Drop OHLC columns (features are based on them, but we don't use raw prices)
    df_prepared.drop(['Open', 'High', 'Low', 'Close'], axis=1, inplace=True)
    
    # Separate features from target
    X = df_prepared.drop(['target', 'pct_change_N'], axis=1)
    y = df_prepared['target']
    
    return X, y


def fit_scaler(X_train: pd.DataFrame):
    """
    Fit a StandardScaler on training data.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    
    Returns:
    --------
    scaler : StandardScaler
        Fitted scaler
    X_train_scaled : np.ndarray
        Scaled training features
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return scaler, X_train_scaled


def train(df_train: pd.DataFrame, target_bars: int = 45, target_pct: float = 3.0):
    """
    Train models on the provided training dataframe.
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Training dataframe with OHLCV data
    target_bars : int
        Number of bars to look ahead for target
    target_pct : float
        Percentage increase threshold for target
    
    Returns:
    --------
    models : dict
        Dictionary of trained models
    scaler : StandardScaler
        Fitted scaler for feature normalization
    results : dict
        Training results with metrics
    best_model_name : str
        Name of the best performing model
    """
    # Prepare data
    X, y = prepare_data(df_train, target_bars, target_pct)

    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Target balance: {y.value_counts(normalize=True)}")
    print()

    # Split data into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define classification models to train (5 lightweight models)
    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"max_iter": 1000}
        ),
        "ridge_classifier": (
            RidgeClassifier(random_state=42),
            {}
        ),
        "naive_bayes": (
            GaussianNB(),
            {}
        ),
        "knn": (
            KNeighborsClassifier(n_neighbors=5),
            {"n_neighbors": 5}
        ),
        "decision_tree": (
            DecisionTreeClassifier(max_depth=10, random_state=42),
            {"max_depth": 10}
        ),
    }

    # Fit scaler on training data
    scaler, X_train_scaled = fit_scaler(X_train)
    X_val_scaled = scaler.transform(X_val)

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
        
        # Make predictions on validation set
        y_pred = model.predict(X_val_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        
        print(f"\nValidation Set Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        
        # Cross validation on training set
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"\nCross Validation (5-fold):")
        print(f"  Scores: {cv_scores}")
        print(f"  Mean:   {cv_scores.mean():.4f}")
        print(f"  Std:    {cv_scores.std():.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['No Increase', 'Increase']))
        
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
    #create_visualizations(results, close_val, y_val, idx_val)

    return models, scaler, results, best_model_name


def test(models, scaler, df_test: pd.DataFrame, target_bars: int = 45, target_pct: float = 3.0):
    """
    Test trained models on new data (e.g., next year).
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    scaler : StandardScaler
        Fitted scaler from training
    df_test : pd.DataFrame
        Test dataframe with OHLCV data
    target_bars : int
        Number of bars to look ahead for target (should match training)
    target_pct : float
        Percentage increase threshold for target (should match training)
    
    Returns:
    --------
    results_test : dict
        Dictionary with test metrics for each model
    """
    # Prepare test data using the same pipeline as training
    X_test, y_test = prepare_data(df_test, target_bars, target_pct)
    
    # Scale using the fitted scaler from training (DO NOT refit!)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n{'='*80}")
    print(f"TESTING ON HELD-OUT TEST DATA")
    print(f"{'='*80}")
    print(f"Test dataset shape: {X_test.shape}")
    print(f"Test target distribution:\n{y_test.value_counts()}")
    print()

    results_test = {}

    for model_name, model_data in models.items():
        print(f"\n{'='*80}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*80}")
        
        model = model_data[0]
        
        # Make predictions on test set
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        
        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_test, target_names=['No Increase', 'Increase']))
        
        # Store results
        results_test[model_name] = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'y_pred': y_pred_test,
        }

    return results_test


if __name__ == "__main__":
    # Load data
    print("Loading training data...")
    df_train = pd.read_csv(path_1)
    print("Loading test data...")
    df_test = pd.read_csv(path_2)
    
    # Train models
    models, scaler, train_results, best_model_name = train(df_train)
    
    # Test on next year data
    test_metrics = test(models, scaler, df_test)
    
    print("\n" + "="*80)
    print("SUMMARY: Test Metrics (Next Year Data)")
    print("="*80)
    for model_name, metrics in test_metrics.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
