"""
Test script to verify model imports work correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing model imports...\n")

try:
    # Test importing from models_lib package
    from src.models_lib import (
        BaseModel,
        LogisticRegressionModel,
        RandomForestModel,
        LinearRegressionModel,
        XGBoostModel,
        CatBoostModel,
        SimpleCNN,
        DeepCNN,
        ResidualCNN
    )
    print("✓ All imports from src.models_lib successful!")
    
    # Test importing directly from linear_model
    from src.models_lib.linear_model import (
        LogisticRegressionModel as LR,
        RandomForestModel as RF,
        LinearRegressionModel as LinReg
    )
    print("✓ Direct imports from src.models_lib.linear_model successful!")
    
    # Test creating model instances
    print("\nTesting model instantiation...\n")
    
    lr_model = LogisticRegressionModel(name="TestLogistic")
    print(f"✓ Created LogisticRegressionModel: {lr_model.name}")
    
    rf_model = RandomForestModel(name="TestRandomForest")
    print(f"✓ Created RandomForestModel: {rf_model.name}")
    
    lin_model = LinearRegressionModel(name="TestLinear")
    print(f"✓ Created LinearRegressionModel: {lin_model.name}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYou can now use these models in your code:")
    print("  from src.models_lib import LogisticRegressionModel, RandomForestModel")
    print("\n")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nPlease check:")
    print("  1. src/models_lib/__init__.py exports the models")
    print("  2. src/models_lib/linear_model.py contains the model classes")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
