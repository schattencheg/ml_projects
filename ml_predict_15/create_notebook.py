import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ML Price Prediction - Training and Testing Demo\n",
                "\n",
                "This notebook demonstrates the complete machine learning pipeline for cryptocurrency price prediction.\n",
                "\n",
                "## Features:\n",
                "- **3-class classification**: Short (-1), Flat (0), Long (1)\n",
                "- **Multiple ML models**: Logistic Regression, XGBoost, Random Forest, LightGBM, etc.\n",
                "- **Feature engineering**: Technical indicators (RSI, MACD, Bollinger Bands, etc.)\n",
                "- **Model evaluation**: Confusion matrices, classification reports\n",
                "- **Training on historical data**, testing on future data"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Import Libraries"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Import training functions\n",
                "from src.model_training import train, test\n",
                "from src.visualization import print_model_summary\n",
                "\n",
                "# Set display options\n",
                "pd.set_option('display.max_columns', None)\n",
                "pd.set_option('display.width', None)\n",
                "\n",
                "print(\"✓ Libraries imported successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Load Data"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Data paths\n",
                "PATH_TRAIN = 'data/hour/btc.csv'\n",
                "PATH_TEST = 'data/hour/btc_2025.csv'\n",
                "\n",
                "# Load training data\n",
                "print('Loading training data...')\n",
                "df_train = pd.read_csv(PATH_TRAIN)\n",
                "print(f'Training data shape: {df_train.shape}')\n",
                "print(f'Columns: {list(df_train.columns)}')\n",
                "\n",
                "# Load test data\n",
                "print('\\nLoading test data...')\n",
                "df_test = pd.read_csv(PATH_TEST)\n",
                "print(f'Test data shape: {df_test.shape}')\n",
                "\n",
                "# Display first few rows\n",
                "print('\\nFirst 5 rows of training data:')\n",
                "df_train.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Train Models\n",
                "\n",
                "### Configuration:\n",
                "- **target_bars=15**: Look ahead 15 hours for price movement\n",
                "- **target_pct=3.0**: ±3% threshold for class labels\n",
                "  - Price increase ≥3%: Class 1 (Long)\n",
                "  - Price decrease ≥3%: Class -1 (Short)\n",
                "  - Price change <3%: Class 0 (Flat)\n",
                "- **use_smote=False**: No oversampling (optional)\n",
                "- **n_jobs=-1**: Use all CPU cores for parallel training"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train models\n",
                "print('Starting model training...')\n",
                "models, scaler, train_results, best_model_name, label_encoder = train(\n",
                "    df_train,\n",
                "    target_bars=15,\n",
                "    target_pct=3.0,\n",
                "    use_smote=False,\n",
                "    use_gpu=False,\n",
                "    n_jobs=-1,\n",
                "    use_mlflow=False  # Disable MLflow for notebook\n",
                ")\n",
                "\n",
                "print(f'\\n✓ Training complete! Best model: {best_model_name}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Training Results Summary"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Print model summary\n",
                "print_model_summary(train_results)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Confusion Matrix - Training Set\n",
                "\n",
                "Shows how well each model classifies the three classes on the validation set."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.model_evaluation import print_confusion_matrix_summary\n",
                "\n",
                "# Print confusion matrix summary for training\n",
                "print_confusion_matrix_summary(train_results)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. Test Models on Future Data"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test on future data\n",
                "print('Testing models on future data...')\n",
                "test_results = test(models, scaler, df_test, label_encoder)\n",
                "print('\\n✓ Testing complete!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. Confusion Matrix - Test Set"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Print confusion matrix summary for test\n",
                "print_confusion_matrix_summary(test_results)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 8. Compare Training vs Test Results"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create comparison dataframe\n",
                "comparison_data = []\n",
                "for model_name in train_results.keys():\n",
                "    if model_name in test_results:\n",
                "        comparison_data.append({\n",
                "            'Model': model_name.replace('_', ' ').title(),\n",
                "            'Train Acc': f\"{train_results[model_name]['accuracy']:.4f}\",\n",
                "            'Test Acc': f\"{test_results[model_name]['accuracy']:.4f}\",\n",
                "            'Train F1': f\"{train_results[model_name]['f1']:.4f}\",\n",
                "            'Test F1': f\"{test_results[model_name]['f1']:.4f}\",\n",
                "            'Acc Diff': f\"{train_results[model_name]['accuracy'] - test_results[model_name]['accuracy']:.4f}\"\n",
                "        })\n",
                "\n",
                "df_comparison = pd.DataFrame(comparison_data)\n",
                "print('\\nTraining vs Test Performance:')\n",
                "print(df_comparison.to_string(index=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 9. Visualize Results"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Extract numeric values for plotting\n",
                "train_acc = [train_results[m]['accuracy'] for m in train_results.keys() if m in test_results]\n",
                "test_acc = [test_results[m]['accuracy'] for m in train_results.keys() if m in test_results]\n",
                "train_f1 = [train_results[m]['f1'] for m in train_results.keys() if m in test_results]\n",
                "test_f1 = [test_results[m]['f1'] for m in train_results.keys() if m in test_results]\n",
                "model_names = [m.replace('_', ' ').title() for m in train_results.keys() if m in test_results]\n",
                "\n",
                "# Plot comparison\n",
                "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
                "\n",
                "# Accuracy comparison\n",
                "x = np.arange(len(model_names))\n",
                "width = 0.35\n",
                "axes[0].bar(x - width/2, train_acc, width, label='Train', alpha=0.8, color='steelblue')\n",
                "axes[0].bar(x + width/2, test_acc, width, label='Test', alpha=0.8, color='coral')\n",
                "axes[0].set_xlabel('Model', fontsize=12)\n",
                "axes[0].set_ylabel('Accuracy', fontsize=12)\n",
                "axes[0].set_title('Accuracy: Training vs Test', fontsize=14, fontweight='bold')\n",
                "axes[0].set_xticks(x)\n",
                "axes[0].set_xticklabels(model_names, rotation=45, ha='right')\n",
                "axes[0].legend(fontsize=11)\n",
                "axes[0].grid(axis='y', alpha=0.3)\n",
                "\n",
                "# F1 Score comparison\n",
                "axes[1].bar(x - width/2, train_f1, width, label='Train', alpha=0.8, color='steelblue')\n",
                "axes[1].bar(x + width/2, test_f1, width, label='Test', alpha=0.8, color='coral')\n",
                "axes[1].set_xlabel('Model', fontsize=12)\n",
                "axes[1].set_ylabel('F1 Score', fontsize=12)\n",
                "axes[1].set_title('F1 Score: Training vs Test', fontsize=14, fontweight='bold')\n",
                "axes[1].set_xticks(x)\n",
                "axes[1].set_xticklabels(model_names, rotation=45, ha='right')\n",
                "axes[1].legend(fontsize=11)\n",
                "axes[1].grid(axis='y', alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 10. Best Model Analysis"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(f'Best Model (Training): {best_model_name.upper()}')\n",
                "print(f'\\nBest Model Metrics:')\n",
                "print(f'  Training Accuracy:  {train_results[best_model_name][\"accuracy\"]:.4f}')\n",
                "print(f'  Training F1 Score:  {train_results[best_model_name][\"f1\"]:.4f}')\n",
                "print(f'  Training Precision: {train_results[best_model_name][\"precision\"]:.4f}')\n",
                "print(f'  Training Recall:    {train_results[best_model_name][\"recall\"]:.4f}')\n",
                "print(f'  Training ROC AUC:   {train_results[best_model_name][\"roc_auc\"]:.4f}')\n",
                "print(f'\\n  Test Accuracy:      {test_results[best_model_name][\"accuracy\"]:.4f}')\n",
                "print(f'  Test F1 Score:      {test_results[best_model_name][\"f1\"]:.4f}')\n",
                "print(f'  Test Precision:     {test_results[best_model_name][\"precision\"]:.4f}')\n",
                "print(f'  Test Recall:        {test_results[best_model_name][\"recall\"]:.4f}')\n",
                "print(f'  Test ROC AUC:       {test_results[best_model_name][\"roc_auc\"]:.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. Label Encoding Information\n",
                "\n",
                "The models use encoded labels internally for sklearn compatibility."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Label Encoding Mapping:')\n",
                "print(f'  Original classes: {sorted(label_encoder.classes_)}')\n",
                "print(f'  Encoded classes:  {sorted(label_encoder.transform(label_encoder.classes_))}')\n",
                "print(f'\\nMapping:')\n",
                "for orig, enc in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):\n",
                "    class_name = {-1: 'Short', 0: 'Flat', 1: 'Long'}[orig]\n",
                "    print(f'  {orig:2d} ({class_name:5s}) -> {enc}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Summary\n",
                "\n",
                "This notebook demonstrated:\n",
                "1. ✓ Loading training and test data\n",
                "2. ✓ Training multiple ML models with 3-class classification\n",
                "3. ✓ Evaluating models with confusion matrices\n",
                "4. ✓ Testing on future data\n",
                "5. ✓ Comparing training vs test performance\n",
                "6. ✓ Visualizing results\n",
                "\n",
                "All models are now trained and ready for predictions!"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save notebook
with open('ml_training_demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("✓ Notebook created successfully: ml_training_demo.ipynb")
