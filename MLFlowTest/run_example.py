"""Example script to demonstrate the complete OHLC prediction pipeline."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from train import ModelTrainer
from evaluate import ModelEvaluator
from utils import get_logger

logger = get_logger(__name__)

def run_complete_example():
    """Run a complete example of the OHLC prediction pipeline."""
    
    print("\n" + "="*60)
    print("OHLC PREDICTION PIPELINE EXAMPLE")
    print("="*60)
    
    # Configuration
    symbol = "AAPL"
    model_types = ["rf", "xgb"]  # Start with faster models
    period = "1y"  # Use 1 year of data for quick demo
    
    print(f"Symbol: {symbol}")
    print(f"Models: {model_types}")
    print(f"Data Period: {period}")
    print("="*60)
    
    trained_models = []
    
    # Step 1: Train models
    print("\n🚀 STEP 1: TRAINING MODELS")
    print("-" * 30)
    
    for model_type in model_types:
        try:
            print(f"\n📊 Training {model_type.upper()} model...")
            
            trainer = ModelTrainer(symbol, model_type)
            results = trainer.run_training_pipeline(
                period=period,
                force_fetch=True,
                save_model=True,
                track_mlflow=True
            )
            
            trained_models.append({
                'model_type': model_type,
                'model_path': results.get('model_path'),
                'metrics': results.get('evaluation', {}),
                'mlflow_run_id': results.get('mlflow_run_id')
            })
            
            print(f"✅ {model_type.upper()} training completed!")
            if 'evaluation' in results:
                for metric, value in list(results['evaluation'].items())[:3]:
                    print(f"   {metric}: {value:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error training {model_type}: {str(e)}")
            continue
    
    if not trained_models:
        print("❌ No models were trained successfully!")
        return
    
    # Step 2: Evaluate models
    print(f"\n🔍 STEP 2: EVALUATING MODELS")
    print("-" * 30)
    
    evaluation_results = []
    
    for model_info in trained_models:
        if model_info['model_path']:
            try:
                print(f"\n📈 Evaluating {model_info['model_type'].upper()} model...")
                
                evaluator = ModelEvaluator(model_info['model_path'], symbol)
                eval_results = evaluator.run_evaluation(
                    period="3mo",  # Use 3 months for evaluation
                    save_plots=True
                )
                
                evaluation_results.append({
                    'model_type': model_info['model_type'],
                    'metrics': eval_results['metrics']
                })
                
                print(f"✅ {model_info['model_type'].upper()} evaluation completed!")
                
            except Exception as e:
                logger.error(f"❌ Error evaluating {model_info['model_type']}: {str(e)}")
                continue
    
    # Step 3: Compare results
    print(f"\n📊 STEP 3: MODEL COMPARISON")
    print("-" * 30)
    
    if evaluation_results:
        print(f"\n{'Model':<15} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
        print("-" * 50)
        
        for result in evaluation_results:
            model_type = result['model_type'].upper()
            metrics = result['metrics']
            mae = metrics.get('MAE', 0)
            rmse = metrics.get('RMSE', 0)
            r2 = metrics.get('R2', 0)
            
            print(f"{model_type:<15} {mae:<10.4f} {rmse:<10.4f} {r2:<10.4f}")
        
        # Find best model
        best_model = min(evaluation_results, key=lambda x: x['metrics'].get('MAE', float('inf')))
        print(f"\n🏆 Best Model: {best_model['model_type'].upper()} (lowest MAE)")
    
    # Step 4: API Demo (if models exist)
    print(f"\n🌐 STEP 4: API DEMONSTRATION")
    print("-" * 30)
    
    try:
        # Import API components
        from api.app import app
        import requests
        import json
        
        print("📡 Starting API server for demonstration...")
        print("   (In production, run: python api/app.py)")
        print("   API endpoints available:")
        print("   - POST /predict - Make predictions")
        print("   - GET /models - List available models") 
        print("   - GET /health - Health check")
        
    except Exception as e:
        logger.warning(f"API demo skipped: {str(e)}")
    
    # Summary
    print(f"\n✨ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("📁 Generated Files:")
    print("   - models/ - Trained model files")
    print("   - evaluation_plots/ - Model evaluation plots")
    print("   - evaluation_report.md - Detailed evaluation report")
    print("   - data/ - Downloaded market data")
    
    print(f"\n🔗 MLflow Tracking:")
    print("   - Open http://127.0.0.1:5000 to view experiments")
    print("   - Compare model performance and metrics")
    print("   - View training artifacts and plots")
    
    print(f"\n🚀 Next Steps:")
    print("   1. Explore MLflow UI for detailed analysis")
    print("   2. Try different symbols and model types")
    print("   3. Tune hyperparameters for better performance")
    print("   4. Deploy models using the API endpoints")
    print("="*60)

def main():
    """Main function."""
    try:
        run_complete_example()
    except KeyboardInterrupt:
        print("\n\n⏹️  Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
