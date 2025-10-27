"""
Example: How to Enable/Disable Models Using MODEL_ENABLED_CONFIG

This example demonstrates how to control which models are trained by modifying
the MODEL_ENABLED_CONFIG dictionary in src/model_training.py
"""

# To enable/disable models, edit the MODEL_ENABLED_CONFIG dictionary in src/model_training.py
# 
# Example configurations:
#
# 1. Train only fast models (disable slow ones):
#    MODEL_ENABLED_CONFIG = {
#        'logistic_regression': True,
#        'ridge_classifier': True,
#        'naive_bayes': True,
#        'knn_k_neighbours': False,  # Slow on large datasets
#        'decision_tree': True,
#        'random_forest': True,
#        'gradient_boosting': False,  # Slow, use XGBoost instead
#        'svm_support_vector_classification': False,  # Very slow
#        'xgboost': True,
#        'lightgbm': True,
#        'lstm': False,  # Neural networks are slow
#        'cnn': False,
#        'hybrid_lstm_cnn': False,
#    }
#
# 2. Train only tree-based models:
#    MODEL_ENABLED_CONFIG = {
#        'logistic_regression': False,
#        'ridge_classifier': False,
#        'naive_bayes': False,
#        'knn_k_neighbours': False,
#        'decision_tree': True,
#        'random_forest': True,
#        'gradient_boosting': True,
#        'svm_support_vector_classification': False,
#        'xgboost': True,
#        'lightgbm': True,
#        'lstm': False,
#        'cnn': False,
#        'hybrid_lstm_cnn': False,
#    }
#
# 3. Train only neural networks:
#    MODEL_ENABLED_CONFIG = {
#        'logistic_regression': False,
#        'ridge_classifier': False,
#        'naive_bayes': False,
#        'knn_k_neighbours': False,
#        'decision_tree': False,
#        'random_forest': False,
#        'gradient_boosting': False,
#        'svm_support_vector_classification': False,
#        'xgboost': False,
#        'lightgbm': False,
#        'lstm': True,
#        'cnn': True,
#        'hybrid_lstm_cnn': True,
#    }
#
# 4. Train only one specific model for testing:
#    MODEL_ENABLED_CONFIG = {
#        'logistic_regression': False,
#        'ridge_classifier': False,
#        'naive_bayes': False,
#        'knn_k_neighbours': False,
#        'decision_tree': False,
#        'random_forest': False,
#        'gradient_boosting': False,
#        'svm_support_vector_classification': False,
#        'xgboost': True,  # Only train XGBoost
#        'lightgbm': False,
#        'lstm': False,
#        'cnn': False,
#        'hybrid_lstm_cnn': False,
#    }

import pandas as pd
from src.model_training import train, MODEL_ENABLED_CONFIG
from src.FeaturesGenerator import FeaturesGenerator

# Display current configuration
print("="*80)
print("CURRENT MODEL CONFIGURATION")
print("="*80)
print("\nEnabled models:")
for model_name, enabled in MODEL_ENABLED_CONFIG.items():
    status = "✓ ENABLED" if enabled else "✗ DISABLED"
    print(f"  {model_name:40s} {status}")
print()

# Example: Train models with current configuration
if __name__ == "__main__":
    print("="*80)
    print("TRAINING WITH CURRENT CONFIGURATION")
    print("="*80)
    print()
    
    # Load your data
    # df = pd.read_csv('your_data.csv')
    # df = FeaturesGenerator.generate_features(df)
    
    # Train models - only enabled models will be trained
    # models, scaler, results, best_model = train(
    #     df_train,
    #     target_bars=45,
    #     target_pct=3.0,
    #     use_smote=True,
    #     use_gpu=False,
    #     n_jobs=-1
    # )
    
    print("To modify which models are trained:")
    print("1. Open: src/model_training.py")
    print("2. Find: MODEL_ENABLED_CONFIG dictionary (around line 116)")
    print("3. Set models to True (enabled) or False (disabled)")
    print("4. Save the file")
    print("5. Run your training script")
    print()
    print("The training will automatically skip disabled models!")
