import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.FeaturesGenerator import FeaturesGenerator
from src.model_loader import save_all_models

# Neural Network imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Neural network models will be skipped.")

# Additional ML models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available.")


path_1 = "data/btc_2022.csv"
path_2 = "data/btc_2023.csv"


def create_sequences(X, y, sequence_length=60):
    """
    Create sequences for LSTM/CNN models from feature data.
    
    Parameters:
    -----------
    X : np.ndarray or pd.DataFrame
        Feature data
    y : np.ndarray or pd.Series
        Target data
    sequence_length : int
        Length of each sequence
    
    Returns:
    --------
    X_seq : np.ndarray
        Sequences of shape (samples, sequence_length, features)
    y_seq : np.ndarray
        Corresponding targets
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    X_seq, y_seq = [], []
    
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)


def create_lstm_model(input_shape, dropout_rate=0.2):
    """
    Create a simple LSTM model for binary classification.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    dropout_rate : float
        Dropout rate for regularization
    
    Returns:
    --------
    model : Sequential
        Compiled LSTM model
    """
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(50, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_cnn_model(input_shape, dropout_rate=0.2):
    """
    Create a 1D CNN model for pattern recognition in sequential data.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    dropout_rate : float
        Dropout rate for regularization
    
    Returns:
    --------
    model : Sequential
        Compiled CNN model
    """
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        Conv1D(filters=16, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_hybrid_lstm_cnn_model(input_shape, dropout_rate=0.2):
    """
    Create a hybrid LSTM-CNN model combining both approaches.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    dropout_rate : float
        Dropout rate for regularization
    
    Returns:
    --------
    model : Sequential
        Compiled hybrid model
    """
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = Sequential([
        # CNN layers for pattern extraction
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # LSTM layers for sequence modeling
        LSTM(50, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(25, return_sequences=False),
        Dropout(dropout_rate),
        
        # Dense layers for final prediction
        Dense(25, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


class KerasClassifierWrapper:
    """
    Wrapper to make Keras models compatible with sklearn-style interface.
    """
    def __init__(self, model_builder, input_shape, sequence_length=60, **kwargs):
        self.model_builder = model_builder
        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.model = None
        self.kwargs = kwargs
        self.scaler = MinMaxScaler()
        
    def fit(self, X, y):
        # Create sequences
        X_seq, y_seq = create_sequences(X, y, self.sequence_length)
        
        if len(X_seq) == 0:
            raise ValueError(f"Not enough data to create sequences of length {self.sequence_length}")
        
        # Scale the sequences
        original_shape = X_seq.shape
        X_seq_reshaped = X_seq.reshape(-1, X_seq.shape[-1])
        X_seq_scaled = self.scaler.fit_transform(X_seq_reshaped)
        X_seq = X_seq_scaled.reshape(original_shape)
        
        # Build model
        self.model = self.model_builder(self.input_shape, **self.kwargs)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)
        ]
        
        # Train model
        self.model.fit(
            X_seq, y_seq,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Create sequences
        X_seq, _ = create_sequences(X, np.zeros(len(X)), self.sequence_length)
        
        if len(X_seq) == 0:
            # If not enough data for sequences, return predictions for available data
            return np.zeros(len(X))
        
        # Scale the sequences
        original_shape = X_seq.shape
        X_seq_reshaped = X_seq.reshape(-1, X_seq.shape[-1])
        X_seq_scaled = self.scaler.transform(X_seq_reshaped)
        X_seq = X_seq_scaled.reshape(original_shape)
        
        # Predict
        predictions = self.model.predict(X_seq, verbose=0)
        predictions_binary = (predictions > 0.5).astype(int).flatten()
        
        # Pad predictions to match original length
        full_predictions = np.zeros(len(X))
        full_predictions[self.sequence_length:self.sequence_length + len(predictions_binary)] = predictions_binary
        
        return full_predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities for ROC AUC calculation.
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Create sequences
        X_seq, _ = create_sequences(X, np.zeros(len(X)), self.sequence_length)
        
        if len(X_seq) == 0:
            # If not enough data for sequences, return default probabilities
            return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])
        
        # Scale the sequences
        original_shape = X_seq.shape
        X_seq_reshaped = X_seq.reshape(-1, X_seq.shape[-1])
        X_seq_scaled = self.scaler.transform(X_seq_reshaped)
        X_seq = X_seq_scaled.reshape(original_shape)
        
        # Predict probabilities
        predictions = self.model.predict(X_seq, verbose=0)
        predictions_proba = predictions.flatten()
        
        # Pad predictions to match original length
        full_predictions = np.ones(len(X)) * 0.5  # Default probability
        full_predictions[self.sequence_length:self.sequence_length + len(predictions_proba)] = predictions_proba
        
        # Return as [prob_class_0, prob_class_1]
        return np.column_stack([1 - full_predictions, full_predictions])

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
    
    # 1. Overall Performance Comparison
    fig, ax = plt.subplots(figsize=(15, 8))
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    f1_scores = [results[m]['f1'] for m in model_names]
    roc_aucs = [results[m].get('roc_auc', 0.0) for m in model_names]
    precisions = [results[m]['precision'] for m in model_names]
    recalls = [results[m]['recall'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.15
    
    rects1 = ax.bar(x - 2*width, accuracies, width, label='Accuracy', alpha=0.8, color='steelblue')
    rects2 = ax.bar(x - width, f1_scores, width, label='F1 Score', alpha=0.8, color='coral')
    rects3 = ax.bar(x, roc_aucs, width, label='ROC AUC', alpha=0.8, color='green')
    rects4 = ax.bar(x + width, precisions, width, label='Precision', alpha=0.8, color='orange')
    rects5 = ax.bar(x + 2*width, recalls, width, label='Recall', alpha=0.8, color='purple')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comprehensive Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in model_names], rotation=45, ha='right')
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars (only for accuracy and ROC AUC to avoid clutter)
    for rect in rects1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for rect in rects3:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
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


def print_model_summary(results):
    """
    Print a comprehensive summary of all model performances.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    """
    print("\n" + "="*100)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*100)
    
    # Sort models by accuracy
    sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"{'Rank':<4} {'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'ROC AUC':<10}")
    print("-" * 80)
    
    for i, (model_name, metrics) in enumerate(sorted_models, 1):
        roc_auc = metrics.get('roc_auc', 0.0)
        print(f"{i:<4} {model_name:<20} {metrics['accuracy']:<10.4f} "
              f"{metrics['f1']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {roc_auc:<10.4f}")
    
    print("\n" + "="*100)
    
    # Model type analysis
    neural_models = [name for name in results.keys() if name in ['lstm', 'cnn', 'lstm_cnn_hybrid']]
    tree_models = [name for name in results.keys() if name in ['decision_tree', 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']]
    linear_models = [name for name in results.keys() if name in ['logistic_regression', 'ridge_classifier', 'svm']]
    
    if neural_models:
        avg_neural = np.mean([results[name]['accuracy'] for name in neural_models])
        print(f"Average Neural Network Performance: {avg_neural:.4f}")
    
    if tree_models:
        avg_tree = np.mean([results[name]['accuracy'] for name in tree_models])
        print(f"Average Tree-based Model Performance: {avg_tree:.4f}")
    
    if linear_models:
        avg_linear = np.mean([results[name]['accuracy'] for name in linear_models])
        print(f"Average Linear Model Performance: {avg_linear:.4f}")
    
    print("="*100)


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

    # Define classification models to train
    models = {
        # Traditional ML models
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
        "random_forest": (
            RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            {"n_estimators": 100, "max_depth": 10}
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            {"n_estimators": 100, "max_depth": 5}
        ),
        "svm": (
            SVC(kernel='rbf', probability=True, random_state=42),
            {"kernel": "rbf"}
        ),
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models["xgboost"] = (
            xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            {"n_estimators": 100, "max_depth": 6}
        )
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models["lightgbm"] = (
            lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            {"n_estimators": 100, "max_depth": 6}
        )
    
    # Add Neural Network models if TensorFlow is available
    if TENSORFLOW_AVAILABLE:
        sequence_length = 30  # Reduced for faster training
        input_shape = (sequence_length, X_train.shape[1])
        
        models["lstm"] = (
            KerasClassifierWrapper(
                create_lstm_model, 
                input_shape, 
                sequence_length=sequence_length,
                dropout_rate=0.3
            ),
            {"sequence_length": sequence_length}
        )
        
        models["cnn"] = (
            KerasClassifierWrapper(
                create_cnn_model, 
                input_shape, 
                sequence_length=sequence_length,
                dropout_rate=0.3
            ),
            {"sequence_length": sequence_length}
        )
        
        models["lstm_cnn_hybrid"] = (
            KerasClassifierWrapper(
                create_hybrid_lstm_cnn_model, 
                input_shape, 
                sequence_length=sequence_length,
                dropout_rate=0.3
            ),
            {"sequence_length": sequence_length}
        )

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
        
        # Calculate ROC AUC
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X_val_scaled)
            else:
                # For models without probability prediction, use predictions as scores
                y_pred_proba = y_pred.astype(float)
            
            roc_auc = roc_auc_score(y_val, y_pred_proba)
        except Exception as e:
            print(f"  Warning: Could not calculate ROC AUC for {model_name}: {str(e)}")
            roc_auc = 0.0
        
        print(f"\nValidation Set Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")
        
        # Cross validation on training set (skip for neural networks due to complexity)
        if not isinstance(model, KerasClassifierWrapper):
            try:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                print(f"\nCross Validation (5-fold):")
                print(f"  Scores: {cv_scores}")
                print(f"  Mean:   {cv_scores.mean():.4f}")
                print(f"  Std:    {cv_scores.std():.4f}")
            except Exception as e:
                print(f"\nCross Validation: Skipped due to error: {str(e)}")
        else:
            print(f"\nCross Validation: Skipped for neural network models")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['No Increase', 'Increase']))
        
        # Store results for plotting
        results[model_name] = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
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

    # Save all models and scaler
    save_all_models(models, scaler, models_dir='models')
    
    # Also save the best model separately
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{best_model_name}_best.joblib'
    joblib.dump(best_model, model_path)
    print(f"\nBest model also saved separately to: {model_path}")

    # ============================================================================
    # MODEL SUMMARY
    # ============================================================================
    print_model_summary(results)

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
        
        # Calculate ROC AUC
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_pred_proba_test = model.decision_function(X_test_scaled)
            else:
                # For models without probability prediction, use predictions as scores
                y_pred_proba_test = y_pred_test.astype(float)
            
            roc_auc = roc_auc_score(y_test, y_pred_proba_test)
        except Exception as e:
            print(f"  Warning: Could not calculate ROC AUC for {model_name}: {str(e)}")
            roc_auc = 0.0
        
        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_test, target_names=['No Increase', 'Increase']))
        
        # Store results
        results_test[model_name] = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'y_pred': y_pred_test,
        }

    # Print test results summary
    print_model_summary(results_test)
    
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
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
