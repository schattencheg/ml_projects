"""
Example: Using Centralized Model Configuration

This script demonstrates how to use the centralized ModelConfig class
to manage model enable/disable settings for both traditional ML models
and neural network models.
"""

from src.ModelConfig import get_model_config
from src.ModelsManager import ModelsManager

def example_basic_usage():
    """Example 1: Basic configuration usage."""
    print("\n" + "="*80)
    print("EXAMPLE 1: BASIC USAGE")
    print("="*80)
    
    # Get singleton instance
    config = get_model_config()
    
    # View current configuration
    config.print_config()
    
    # Enable/disable specific models
    print("\nEnabling random_forest...")
    config.enable_model('random_forest', True)
    
    print("Disabling cnn_simple...")
    config.enable_model('cnn_simple', False)
    
    # View updated configuration
    print("\nUpdated configuration:")
    config.print_config(show_disabled=False)


def example_category_control():
    """Example 2: Category-based control for neural networks."""
    print("\n" + "="*80)
    print("EXAMPLE 2: CATEGORY-BASED CONTROL")
    print("="*80)
    
    config = get_model_config()
    
    # Disable all neural networks first
    config.enable_all_neural_networks(False)
    
    # Enable only CNN models
    print("\nEnabling only CNN models...")
    config.enable_by_category('CNN', True)
    
    # Check enabled models
    enabled_nn = config.get_enabled_neural_network_models()
    print(f"\nEnabled neural networks: {enabled_nn}")
    
    # Enable LSTM models too
    print("\nEnabling LSTM models...")
    config.enable_by_category('LSTM', True)
    
    enabled_nn = config.get_enabled_neural_network_models()
    print(f"Enabled neural networks: {enabled_nn}")


def example_preset_fast_training():
    """Example 3: Fast training preset (< 1 minute)."""
    print("\n" + "="*80)
    print("EXAMPLE 3: FAST TRAINING PRESET")
    print("="*80)
    
    config = get_model_config()
    
    # Disable all models
    config.enable_all_traditional(False)
    config.enable_all_neural_networks(False)
    
    # Enable only fast models
    print("\nConfiguring for fast training...")
    config.enable_model('logistic_regression', True)
    config.enable_model('xgboost', True)
    config.enable_model('lightgbm', True)
    
    # View configuration
    enabled = config.get_all_enabled_models()
    print(f"\nEnabled models: {enabled}")
    print("Expected training time: ~10-15 seconds")
    
    config.print_config(show_disabled=False)


def example_preset_tree_based():
    """Example 4: Tree-based models only."""
    print("\n" + "="*80)
    print("EXAMPLE 4: TREE-BASED MODELS PRESET")
    print("="*80)
    
    config = get_model_config()
    
    # Disable all
    config.enable_all_traditional(False)
    config.enable_all_neural_networks(False)
    
    # Enable tree-based models
    print("\nConfiguring for tree-based models...")
    tree_models = ['decision_tree', 'random_forest', 'gradient_boosting', 
                   'xgboost', 'lightgbm']
    
    for model in tree_models:
        config.enable_model(model, True)
    
    # View configuration
    enabled = config.get_enabled_traditional_models()
    print(f"\nEnabled traditional models: {enabled}")
    
    config.print_config(show_disabled=False)


def example_preset_neural_networks():
    """Example 5: Neural networks only."""
    print("\n" + "="*80)
    print("EXAMPLE 5: NEURAL NETWORKS ONLY PRESET")
    print("="*80)
    
    config = get_model_config()
    
    # Disable traditional models
    config.enable_all_traditional(False)
    
    # Enable all neural networks
    print("\nConfiguring for neural networks only...")
    config.enable_all_neural_networks(True)
    
    # View configuration
    enabled_nn = config.get_enabled_neural_network_models()
    print(f"\nEnabled neural networks: {len(enabled_nn)} models")
    print(f"Models: {enabled_nn}")
    
    config.print_config(show_disabled=False)


def example_models_manager_integration():
    """Example 6: Using with ModelsManager."""
    print("\n" + "="*80)
    print("EXAMPLE 6: MODELS MANAGER INTEGRATION")
    print("="*80)
    
    # Configure models
    config = get_model_config()
    config.enable_all_traditional(False)
    config.enable_all_neural_networks(False)
    config.enable_model('logistic_regression', True)
    config.enable_model('xgboost', True)
    
    # Create ModelsManager (uses centralized config)
    print("\nCreating ModelsManager...")
    manager = ModelsManager(models_dir='models', include_neural_networks=False)
    
    # View configuration through manager
    print("\nConfiguration via ModelsManager:")
    manager.print_config()
    
    # Get enabled models
    enabled = manager.get_enabled_models(include_neural_networks=False)
    print(f"\nEnabled models: {enabled}")
    
    # Create model instances
    print("\nCreating model instances...")
    models = manager.create_models(enabled_only=True, include_neural_networks=False)
    print(f"Created {len(models)} models: {list(models.keys())}")


def example_query_model_info():
    """Example 7: Query model information."""
    print("\n" + "="*80)
    print("EXAMPLE 7: QUERY MODEL INFORMATION")
    print("="*80)
    
    config = get_model_config()
    
    # Get info for specific models
    models_to_check = ['xgboost', 'cnn_simple', 'lstm_bidirectional']
    
    for model_name in models_to_check:
        info = config.get_model_info(model_name)
        if info:
            print(f"\n{model_name}:")
            print(f"  Type: {info['type']}")
            print(f"  Enabled: {info['config']['enabled']}")
            print(f"  Description: {info['config']['description']}")
            if 'training_time' in info['config']:
                print(f"  Training time: {info['config']['training_time']}")
            if 'category' in info['config']:
                print(f"  Category: {info['config']['category']}")


def example_export_config():
    """Example 8: Export configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 8: EXPORT CONFIGURATION")
    print("="*80)
    
    config = get_model_config()
    
    # Export configuration
    print("\nExporting configuration...")
    config_dict = config.export_config()
    
    # Display summary
    trad_enabled = sum(1 for m in config_dict['traditional_models'].values() if m['enabled'])
    trad_total = len(config_dict['traditional_models'])
    nn_enabled = sum(1 for m in config_dict['neural_network_models'].values() if m['enabled'])
    nn_total = len(config_dict['neural_network_models'])
    
    print(f"\nConfiguration Summary:")
    print(f"  Traditional ML: {trad_enabled}/{trad_total} enabled")
    print(f"  Neural Networks: {nn_enabled}/{nn_total} enabled")
    print(f"  Total: {trad_enabled + nn_enabled}/{trad_total + nn_total} enabled")
    
    # Save to file (optional)
    import json
    output_file = 'model_config_export.json'
    with open(output_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"\n✓ Configuration exported to: {output_file}")


def reset_to_defaults():
    """Reset configuration to defaults."""
    print("\n" + "="*80)
    print("RESETTING TO DEFAULTS")
    print("="*80)
    
    config = get_model_config()
    
    # Reset traditional models (only fast ones enabled)
    config.enable_all_traditional(False)
    config.enable_model('logistic_regression', True)
    config.enable_model('xgboost', True)
    config.enable_model('lightgbm', True)
    
    # Reset neural networks (all enabled by default)
    config.enable_all_neural_networks(True)
    
    print("\n✓ Configuration reset to defaults")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("CENTRALIZED MODEL CONFIGURATION EXAMPLES")
    print("="*80)
    
    # Run examples
    example_basic_usage()
    example_category_control()
    example_preset_fast_training()
    example_preset_tree_based()
    example_preset_neural_networks()
    example_models_manager_integration()
    example_query_model_info()
    example_export_config()
    
    # Reset to defaults
    reset_to_defaults()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. Use get_model_config() to get the singleton instance")
    print("  2. Use enable_model() to enable/disable any model")
    print("  3. Use enable_by_category() for neural network categories")
    print("  4. Use print_config() to view current configuration")
    print("  5. ModelsManager automatically uses centralized config")
    print("\nFor more information, see: docs/CENTRALIZED_MODEL_CONFIG.md")


if __name__ == "__main__":
    main()
