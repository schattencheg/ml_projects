"""
Enhanced CNN Hyperparameter Optimizer with:
- Custom model building using optimized neuron counts
- Optimization results visualization
- Model comparison and evaluation
- Complete metrics tracking
"""

import optuna
import tensorflow as tf
from tensorflow.keras import optimizers, layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import joblib


class EnhancedCNNOptimizer:
    """
    Enhanced CNN optimizer with custom architectures, visualization, and comparison.
    """
    
    def __init__(self, X_train, y_train, X_val, y_val, input_shape):
        """
        Initialize optimizer with training and validation data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            input_shape: Shape of input data (sequence_length, num_features)
        """
        # Prepare data
        self.X_train = self._prepare_features(X_train)
        self.y_train = np.array(y_train, dtype=np.int32).flatten()
        self.X_val = self._prepare_features(X_val)
        self.y_val = np.array(y_val, dtype=np.int32).flatten()
        self.input_shape = input_shape
        
        # Store optimization history
        self.optimization_history = []
        
    def _prepare_features(self, X):
        """Prepare features by handling non-numeric data."""
        if isinstance(X, pd.DataFrame):
            return X.select_dtypes(include=[np.number]).values
        
        elif isinstance(X, np.ndarray):
            if X.dtype == object:
                if X.ndim == 3:
                    cleaned_sequences = []
                    for sequence in X:
                        df = pd.DataFrame(sequence)
                        for col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        numeric_df = df.select_dtypes(include=[np.number])
                        numeric_df = numeric_df.dropna(axis=1, how='all')
                        cleaned_sequences.append(numeric_df.values)
                    return np.array(cleaned_sequences, dtype=np.float32)
        
        return np.array(X, dtype=np.float32)
    
    def _build_simple_cnn(self, filters_1, filters_2, dense_units):
        """Build simple CNN with custom neuron counts."""
        model = models.Sequential([
            layers.Conv1D(filters=filters_1, kernel_size=3, padding='same', 
                         activation='relu', input_shape=self.input_shape),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=filters_2, kernel_size=3, padding='same', 
                         activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(dense_units, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _build_deeper_cnn(self, filters_1, filters_2, dense_units_1, dense_units_2, dropout_1, dropout_2):
        """Build deeper CNN with custom neuron counts."""
        model = models.Sequential([
            layers.Conv1D(filters=filters_1, kernel_size=3, padding='same',
                         activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv1D(filters=filters_1, kernel_size=3, padding='same',
                         activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(dropout_1),
            
            layers.Conv1D(filters=filters_2, kernel_size=3, padding='same',
                         activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(filters=filters_2, kernel_size=3, padding='same',
                         activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(dropout_1),
            
            layers.Dense(dense_units_1, activation='relu'),
            layers.Dropout(dropout_2),
            layers.Dense(dense_units_2, activation='relu'),
            layers.Dropout(dropout_2),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _build_cnn_lstm(self, filters_1, filters_2, lstm_units_1, lstm_units_2, 
                        dense_units_1, dense_units_2, dropout_1, dropout_2, dropout_3):
        """Build CNN-LSTM with custom neuron counts."""
        model = models.Sequential([
            layers.Conv1D(filters=filters_1, kernel_size=3, padding='same',
                         activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv1D(filters=filters_2, kernel_size=3, padding='same',
                         activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(dropout_1),
            
            layers.LSTM(lstm_units_1, return_sequences=True),
            layers.Dropout(dropout_2),
            layers.LSTM(lstm_units_2),
            layers.Dropout(dropout_2),
            
            layers.Dense(dense_units_1, activation='relu'),
            layers.Dropout(dropout_3),
            layers.Dense(dense_units_2, activation='relu'),
            layers.Dropout(dropout_3),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def objective(self, trial):
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation recall to maximize
        """
        # Suggest architecture type
        architecture_type = trial.suggest_categorical('architecture', ['simple', 'deeper', 'cnn_lstm'])
        
        # Suggest hyperparameters based on architecture
        if architecture_type == 'simple':
            filters_1 = trial.suggest_int('filters_1', 16, 128, step=16)
            filters_2 = trial.suggest_int('filters_2', 32, 128, step=16)
            dense_units = trial.suggest_int('dense_units', 32, 128, step=16)
            model = self._build_simple_cnn(filters_1, filters_2, dense_units)
            
        elif architecture_type == 'deeper':
            filters_1 = trial.suggest_int('filters_1', 16, 64, step=16)
            filters_2 = trial.suggest_int('filters_2', 32, 128, step=16)
            dense_units_1 = trial.suggest_int('dense_units_1', 128, 512, step=64)
            dense_units_2 = trial.suggest_int('dense_units_2', 64, 256, step=32)
            dropout_1 = trial.suggest_float('dropout_1', 0.2, 0.5)
            dropout_2 = trial.suggest_float('dropout_2', 0.3, 0.6)
            model = self._build_deeper_cnn(filters_1, filters_2, dense_units_1, dense_units_2, dropout_1, dropout_2)
            
        elif architecture_type == 'cnn_lstm':
            filters_1 = trial.suggest_int('filters_1', 32, 128, step=16)
            filters_2 = trial.suggest_int('filters_2', 32, 128, step=16)
            lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 128, step=16)
            lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 128, step=16)
            dense_units_1 = trial.suggest_int('dense_units_1', 64, 256, step=32)
            dense_units_2 = trial.suggest_int('dense_units_2', 32, 128, step=16)
            dropout_1 = trial.suggest_float('dropout_1', 0.2, 0.5)
            dropout_2 = trial.suggest_float('dropout_2', 0.2, 0.5)
            dropout_3 = trial.suggest_float('dropout_3', 0.2, 0.5)
            model = self._build_cnn_lstm(filters_1, filters_2, lstm_units_1, lstm_units_2,
                                         dense_units_1, dense_units_2, dropout_1, dropout_2, dropout_3)
        
        # Suggest optimizer and learning rate
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        if optimizer_name == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Suggest batch size and epochs
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        epochs = trial.suggest_int('epochs', 20, 100, step=10)
        
        # Train model with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        model.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate model
        val_predictions = model.predict(self.X_val, verbose=0)
        val_predictions_binary = (val_predictions > 0.5).astype(int).flatten()
        
        # Calculate all metrics
        accuracy = accuracy_score(self.y_val, val_predictions_binary)
        precision = precision_score(self.y_val, val_predictions_binary, zero_division=0)
        recall = recall_score(self.y_val, val_predictions_binary, zero_division=0)
        f1 = f1_score(self.y_val, val_predictions_binary, zero_division=0)
        
        # Store optimization history
        self.optimization_history.append({
            'trial': trial.number,
            'architecture': architecture_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'params': trial.params.copy()
        })
        
        # Return recall (maximize true positive rate)
        return recall
    
    def optimize(self, n_trials=50, show_progress=True):
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of optimization trials
            show_progress: Whether to show progress bar
            
        Returns:
            Optuna study object
        """
        print(f"\n{'='*80}")
        print("STARTING CNN HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Number of trials: {n_trials}")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")
        print(f"Input shape: {self.input_shape}")
        print(f"{'='*80}\n")
        
        study = optuna.create_study(direction='maximize', study_name='cnn_optimization')
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=show_progress)
        
        print(f"\n{'='*80}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best Recall: {study.best_value:.4f}")
        print(f"\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"{'='*80}\n")
        
        return study
    
    def plot_optimization_results(self, study, save_path=None):
        """
        Plot optimization results.
        
        Args:
            study: Optuna study object
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CNN Hyperparameter Optimization Results (Maximizing Recall)', fontsize=16, fontweight='bold')
        
        # Convert history to DataFrame
        df = pd.DataFrame(self.optimization_history)
        
        # 1. Recall progress
        axes[0, 0].plot(df['trial'], df['recall'], 'b-', alpha=0.6, label='Recall')
        axes[0, 0].axhline(y=study.best_value, color='r', linestyle='--', label=f'Best: {study.best_value:.4f}')
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel('Recall')
        axes[0, 0].set_title('Recall Progress (Optimization Target)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Best model metrics
        best_idx = df['recall'].idxmax()
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [df.loc[best_idx, m] for m in metrics]
        colors = ['blue', 'green', 'red', 'orange']  # Red for recall (primary metric)
        
        axes[0, 1].bar(metrics, values, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Best Model Metrics')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(values):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. Architecture performance
        arch_perf = df.groupby('architecture')['recall'].agg(['mean', 'std', 'max'])
        x = np.arange(len(arch_perf))
        width = 0.25
        
        axes[0, 2].bar(x - width, arch_perf['mean'], width, label='Mean', alpha=0.8)
        axes[0, 2].bar(x, arch_perf['std'], width, label='Std', alpha=0.8)
        axes[0, 2].bar(x + width, arch_perf['max'], width, label='Max', alpha=0.8)
        axes[0, 2].set_xlabel('Architecture')
        axes[0, 2].set_ylabel('Recall')
        axes[0, 2].set_title('Architecture Comparison (by Recall)')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(arch_perf.index, rotation=45)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # 4. Learning rate vs Recall
        learning_rates = [p.get('learning_rate', 0) for p in df['params']]
        scatter = axes[1, 0].scatter(learning_rates, df['recall'], alpha=0.6, c=df['trial'], cmap='viridis')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_xlabel('Learning Rate')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Learning Rate vs Recall')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Trial')
        
        # 5. Batch size impact
        batch_sizes = [p.get('batch_size', 0) for p in df['params']]
        batch_df = pd.DataFrame({'batch_size': batch_sizes, 'recall': df['recall']})
        batch_stats = batch_df.groupby('batch_size')['recall'].agg(['mean', 'std'])
        
        axes[1, 1].bar(batch_stats.index.astype(str), batch_stats['mean'], 
                      yerr=batch_stats['std'], capsize=5, alpha=0.7, color='steelblue')
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Batch Size Impact on Recall')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 6. Top 10 trials
        top_10 = df.nlargest(10, 'recall')[['trial', 'recall', 'architecture']]
        colors_map = {'simple': 'red', 'deeper': 'green', 'cnn_lstm': 'blue'}
        colors = [colors_map[arch] for arch in top_10['architecture']]
        
        axes[1, 2].barh(range(len(top_10)), top_10['recall'], color=colors, alpha=0.7)
        axes[1, 2].set_yticks(range(len(top_10)))
        axes[1, 2].set_yticklabels([f"Trial {t}" for t in top_10['trial']])
        axes[1, 2].set_xlabel('Recall')
        axes[1, 2].set_title('Top 10 Trials (by Recall)')
        axes[1, 2].grid(True, alpha=0.3, axis='x')
        axes[1, 2].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Optimization plot saved to: {save_path}")
        
        plt.show()
    
    def compare_optimized_models(self, study, X_test, y_test, top_n=5):
        """
        Compare top N optimized models on test data.
        
        Args:
            study: Optuna study object
            X_test: Test features
            y_test: Test labels
            top_n: Number of top models to compare
            
        Returns:
            DataFrame with comparison results
        """
        print(f"\n{'='*80}")
        print(f"COMPARING TOP {top_n} MODELS ON TEST DATA")
        print(f"{'='*80}\n")
        
        # Prepare test data
        X_test = self._prepare_features(X_test)
        y_test = np.array(y_test, dtype=np.int32).flatten()
        
        # Get top trials
        top_trials = sorted([t for t in study.trials if t.value is not None], 
                           key=lambda t: t.value, reverse=True)[:top_n]
        
        results = []
        
        for i, trial in enumerate(top_trials):
            print(f"Evaluating Model {i+1}/{top_n} (Trial {trial.number})...")
            
            # Rebuild model
            arch = trial.params['architecture']
            if arch == 'simple':
                model = self._build_simple_cnn(
                    trial.params['filters_1'],
                    trial.params['filters_2'],
                    trial.params['dense_units']
                )
            elif arch == 'deeper':
                model = self._build_deeper_cnn(
                    trial.params['filters_1'],
                    trial.params['filters_2'],
                    trial.params['dense_units_1'],
                    trial.params['dense_units_2'],
                    trial.params['dropout_1'],
                    trial.params['dropout_2']
                )
            else:  # cnn_lstm
                model = self._build_cnn_lstm(
                    trial.params['filters_1'],
                    trial.params['filters_2'],
                    trial.params['lstm_units_1'],
                    trial.params['lstm_units_2'],
                    trial.params['dense_units_1'],
                    trial.params['dense_units_2'],
                    trial.params['dropout_1'],
                    trial.params['dropout_2'],
                    trial.params['dropout_3']
                )
            
            # Compile
            opt_name = trial.params['optimizer']
            lr = trial.params['learning_rate']
            optimizer = optimizers.Adam(lr) if opt_name == 'adam' else optimizers.RMSprop(lr)
            
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            
            # Train
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            model.fit(
                self.X_train, self.y_train,
                batch_size=trial.params['batch_size'],
                epochs=trial.params['epochs'],
                validation_data=(self.X_val, self.y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate on test
            test_pred = model.predict(X_test, verbose=0)
            test_pred_binary = (test_pred > 0.5).astype(int).flatten()
            
            accuracy = accuracy_score(y_test, test_pred_binary)
            precision = precision_score(y_test, test_pred_binary, zero_division=0)
            recall = recall_score(y_test, test_pred_binary, zero_division=0)
            f1 = f1_score(y_test, test_pred_binary, zero_division=0)
            
            results.append({
                'Rank': i + 1,
                'Trial': trial.number,
                'Architecture': arch,
                'Val_F1': trial.value,
                'Test_Accuracy': accuracy,
                'Test_Precision': precision,
                'Test_Recall': recall,
                'Test_F1': f1,
                'Learning_Rate': lr,
                'Batch_Size': trial.params['batch_size']
            })
            
            print(f"  Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        
        comparison_df = pd.DataFrame(results)
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON RESULTS")
        print(f"{'='*80}")
        print(comparison_df.to_string(index=False))
        print(f"{'='*80}\n")
        
        # Plot comparison
        self._plot_model_comparison(comparison_df)
        
        return comparison_df
    
    def _plot_model_comparison(self, comparison_df):
        """Plot model comparison results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Top Models Comparison on Test Data', fontsize=14, fontweight='bold')
        
        # Metrics comparison
        metrics = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1']
        x = np.arange(len(comparison_df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            axes[0].bar(x + i * width, comparison_df[metric], width, 
                       label=metric.replace('Test_', ''), alpha=0.8)
        
        axes[0].set_xlabel('Model Rank')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Test Metrics Comparison')
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels([f"#{r}" for r in comparison_df['Rank']])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, 1])
        
        # Architecture distribution
        arch_counts = comparison_df['Architecture'].value_counts()
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        axes[1].pie(arch_counts.values, labels=arch_counts.index, autopct='%1.1f%%',
                   colors=colors[:len(arch_counts)], startangle=90)
        axes[1].set_title('Architecture Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def save_best_model(self, study, model_name='cnn_model', base_dir='models'):
        """
        Save the best model with plots, logs, and metadata in a timestamped subfolder.
        
        Args:
            study: Optuna study object
            model_name: Name for the saved model
            base_dir: Base directory for saving (default: 'models')
            
        Returns:
            str: Path to the saved model directory
        """
        # Create timestamped directory
        timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        save_dir = os.path.join(base_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"SAVING BEST MODEL TO: {save_dir}")
        print(f"{'='*80}\n")
        
        # Get best trial
        best_trial = study.best_trial
        best_params = best_trial.params
        
        # Rebuild best model
        print("Step 1: Rebuilding best model...")
        arch = best_params['architecture']
        if arch == 'simple':
            model = self._build_simple_cnn(
                best_params['filters_1'],
                best_params['filters_2'],
                best_params['dense_units']
            )
        elif arch == 'deeper':
            model = self._build_deeper_cnn(
                best_params['filters_1'],
                best_params['filters_2'],
                best_params['dense_units_1'],
                best_params['dense_units_2'],
                best_params['dropout_1'],
                best_params['dropout_2']
            )
        else:  # cnn_lstm
            model = self._build_cnn_lstm(
                best_params['filters_1'],
                best_params['filters_2'],
                best_params['lstm_units_1'],
                best_params['lstm_units_2'],
                best_params['dense_units_1'],
                best_params['dense_units_2'],
                best_params['dropout_1'],
                best_params['dropout_2'],
                best_params['dropout_3']
            )
        
        # Compile model
        opt_name = best_params['optimizer']
        lr = best_params['learning_rate']
        optimizer = optimizers.Adam(lr) if opt_name == 'adam' else optimizers.RMSprop(lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        print("✓ Model rebuilt")
        
        # Train model
        print("\nStep 2: Training best model...")
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = model.fit(
            self.X_train, self.y_train,
            batch_size=best_params['batch_size'],
            epochs=best_params['epochs'],
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        print("✓ Model trained")
        
        # Save model
        print("\nStep 3: Saving model files...")
        model_path = os.path.join(save_dir, f'{model_name}.h5')
        model.save(model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Save model architecture as JSON
        arch_path = os.path.join(save_dir, f'{model_name}_architecture.json')
        with open(arch_path, 'w') as f:
            f.write(model.to_json())
        print(f"✓ Architecture saved to: {arch_path}")
        
        # Save metadata
        print("\nStep 4: Saving metadata...")
        metadata = {
            'timestamp': timestamp,
            'model_name': model_name,
            'best_trial_number': best_trial.number,
            'best_recall': float(best_trial.value),
            'architecture': arch,
            'hyperparameters': best_params,
            'input_shape': self.input_shape,
            'training_samples': len(self.X_train),
            'validation_samples': len(self.X_val),
            'total_trials': len(study.trials),
            'optimization_metric': 'recall'
        }
        
        metadata_path = os.path.join(save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✓ Metadata saved to: {metadata_path}")
        
        # Save optimization history
        print("\nStep 5: Saving optimization history...")
        history_df = pd.DataFrame(self.optimization_history)
        history_path = os.path.join(save_dir, 'optimization_history.csv')
        history_df.to_csv(history_path, index=False)
        print(f"✓ Optimization history saved to: {history_path}")
        
        # Save training history
        training_history = pd.DataFrame(history.history)
        training_history_path = os.path.join(save_dir, 'training_history.csv')
        training_history.to_csv(training_history_path, index=False)
        print(f"✓ Training history saved to: {training_history_path}")
        
        # Save optimization plot
        print("\nStep 6: Saving optimization plot...")
        opt_plot_path = os.path.join(save_dir, 'optimization_results.png')
        self.plot_optimization_results(study, save_path=opt_plot_path)
        plt.close('all')  # Close to avoid display
        
        # Save training history plot
        print("\nStep 7: Saving training history plot...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Training History', fontsize=14, fontweight='bold')
        
        # Loss plot
        axes[0].plot(history.history['loss'], label='Training Loss', alpha=0.8)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(history.history['accuracy'], label='Training Accuracy', alpha=0.8)
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        training_plot_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(training_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Training plot saved to: {training_plot_path}")
        
        # Save model summary
        print("\nStep 8: Saving model summary...")
        summary_path = os.path.join(save_dir, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"✓ Model summary saved to: {summary_path}")
        
        # Save README
        print("\nStep 9: Creating README...")
        readme_content = f"""# CNN Model - {timestamp}

## Model Information

- **Model Name**: {model_name}
- **Architecture**: {arch}
- **Optimization Metric**: Recall
- **Best Recall**: {best_trial.value:.4f}
- **Best Trial**: {best_trial.number}
- **Total Trials**: {len(study.trials)}

## Hyperparameters

```json
{json.dumps(best_params, indent=2)}
```

## Input Shape

- Sequence Length: {self.input_shape[0]}
- Number of Features: {self.input_shape[1]}

## Training Data

- Training Samples: {len(self.X_train)}
- Validation Samples: {len(self.X_val)}

## Files in This Directory

1. **{model_name}.h5** - Trained Keras model
2. **{model_name}_architecture.json** - Model architecture in JSON format
3. **metadata.json** - Complete metadata and hyperparameters
4. **optimization_history.csv** - All trials and their metrics
5. **training_history.csv** - Training/validation loss and accuracy per epoch
6. **optimization_results.png** - 6-panel optimization visualization
7. **training_history.png** - Training loss and accuracy plots
8. **model_summary.txt** - Model architecture summary
9. **README.md** - This file

## Usage

### Load Model

```python
from tensorflow.keras.models import load_model

model = load_model('{model_name}.h5')
```

### Make Predictions

```python
import numpy as np

# Prepare your data (shape: samples, {self.input_shape[0]}, {self.input_shape[1]})
X_test = np.array(...)

# Predict
probabilities = model.predict(X_test)
predictions = (probabilities > 0.5).astype(int)
```

### Load Metadata

```python
import json

with open('metadata.json', 'r') as f:
    metadata = json.load(f)
    
print(f"Best Recall: {{metadata['best_recall']}}")
print(f"Architecture: {{metadata['architecture']}}")
```

## Notes

- This model was optimized to maximize **Recall** (minimize false negatives)
- Higher recall means fewer missed positive cases
- May have more false positives (lower precision)
- Adjust prediction threshold (default 0.5) to balance precision/recall

## Generated

- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Optimizer: EnhancedCNNOptimizer
"""
        
        readme_path = os.path.join(save_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"✓ README saved to: {readme_path}")
        
        print(f"\n{'='*80}")
        print("MODEL SAVED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"\nDirectory: {save_dir}")
        print(f"\nFiles saved:")
        print(f"  1. {model_name}.h5 - Trained model")
        print(f"  2. {model_name}_architecture.json - Model architecture")
        print(f"  3. metadata.json - Metadata and hyperparameters")
        print(f"  4. optimization_history.csv - All trials")
        print(f"  5. training_history.csv - Training metrics")
        print(f"  6. optimization_results.png - Optimization plot")
        print(f"  7. training_history.png - Training plot")
        print(f"  8. model_summary.txt - Model summary")
        print(f"  9. README.md - Documentation")
        print(f"\n{'='*80}\n")
        
        return save_dir
