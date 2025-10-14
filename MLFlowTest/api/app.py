"""FastAPI application for OHLC price prediction service."""

import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import DataFetcher, DataPreprocessor
from features import FeatureEngineer
from models import LSTMModel, RandomForestModel, XGBoostModel, EnsembleModel
from utils import get_logger, EnvConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OHLC Price Prediction API",
    description="Machine Learning API for predicting stock OHLC prices",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models and components
loaded_models = {}
data_fetcher = DataFetcher()
preprocessor = DataPreprocessor()
feature_engineer = FeatureEngineer()

# Pydantic models for API
class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    model_type: str = Field("lstm", description="Model type (lstm, rf, xgb, ensemble)")
    days: int = Field(5, ge=1, le=30, description="Number of days to predict")
    use_latest_data: bool = Field(True, description="Whether to fetch latest data")

class PredictionResponse(BaseModel):
    symbol: str
    model_type: str
    predictions: List[Dict[str, float]]
    dates: List[str]
    confidence_interval: Optional[List[Dict[str, float]]] = None
    metadata: Dict[str, Any]

class ModelInfo(BaseModel):
    model_type: str
    symbol: str
    is_loaded: bool
    last_updated: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    loaded_models: int
    api_version: str

# Helper functions
def load_model(model_path: str, model_type: str):
    """Load a model from file."""
    try:
        if model_type == 'lstm':
            model = LSTMModel({})
        elif model_type in ['rf', 'random_forest']:
            model = RandomForestModel({})
        elif model_type in ['xgb', 'xgboost']:
            model = XGBoostModel({})
        elif model_type == 'ensemble':
            model = EnsembleModel({})
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load_model(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {str(e)}")
        raise

def prepare_prediction_data(symbol: str, model_type: str, use_latest_data: bool = True):
    """Prepare data for prediction."""
    try:
        # Fetch data
        if use_latest_data:
            raw_data = data_fetcher.fetch_symbol_data(
                symbol=symbol,
                period="1y",  # Get enough data for feature engineering
                save_to_file=False
            )
        else:
            raw_data = data_fetcher.load_saved_data(symbol)
        
        # Preprocess data
        cleaned_data = preprocessor.clean_data(raw_data)
        processed_data = feature_engineer.engineer_features(cleaned_data)
        processed_data = processed_data.dropna()
        
        # Prepare for model
        target_columns = ['Open', 'High', 'Low', 'Close']
        feature_columns = [col for col in processed_data.columns 
                          if col not in ['Symbol'] + target_columns]
        
        if model_type == 'lstm':
            # Load scalers and create sequences
            try:
                preprocessor.load_scalers('models/scalers.pkl')
                scaled_data = preprocessor.scale_data(processed_data, fit_scaler=False)
            except:
                scaled_data = preprocessor.scale_data(processed_data, fit_scaler=True)
            
            # Create sequences (take last sequence for prediction)
            sequence_length = 60  # Default sequence length
            X, _ = preprocessor.create_sequences(
                scaled_data,
                sequence_length=sequence_length,
                target_columns=target_columns,
                prediction_horizon=1
            )
            
            # Return last sequence
            return X[-1:], processed_data.index[-1], scaled_data
        else:
            # For non-sequence models
            X = processed_data[feature_columns].values
            
            # Scale data
            try:
                preprocessor.load_scalers('models/scalers.pkl')
                X_scaled = preprocessor.scale_data(
                    pd.DataFrame(X, columns=feature_columns), 
                    fit_scaler=False
                ).values
            except:
                X_scaled = preprocessor.scale_data(
                    pd.DataFrame(X, columns=feature_columns), 
                    fit_scaler=True
                ).values
            
            # Return last row
            return X_scaled[-1:], processed_data.index[-1], processed_data
            
    except Exception as e:
        logger.error(f"Error preparing data for {symbol}: {str(e)}")
        raise

def generate_future_dates(start_date: pd.Timestamp, days: int) -> List[str]:
    """Generate future business dates."""
    dates = []
    current_date = start_date
    
    for _ in range(days):
        # Add one business day
        current_date = current_date + pd.Timedelta(days=1)
        while current_date.weekday() >= 5:  # Skip weekends
            current_date = current_date + pd.Timedelta(days=1)
        dates.append(current_date.strftime('%Y-%m-%d'))
    
    return dates

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "OHLC Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        loaded_models=len(loaded_models),
        api_version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_prices(request: PredictionRequest):
    """Predict future OHLC prices."""
    try:
        logger.info(f"Prediction request: {request.symbol}, {request.model_type}, {request.days} days")
        
        # Check if model is loaded
        model_key = f"{request.symbol}_{request.model_type}"
        if model_key not in loaded_models:
            # Try to load model
            model_path = f"models/{request.symbol}_{request.model_type}_model.pkl"
            if not Path(model_path).exists():
                raise HTTPException(
                    status_code=404, 
                    detail=f"Model not found for {request.symbol} with type {request.model_type}"
                )
            
            loaded_models[model_key] = load_model(model_path, request.model_type)
            logger.info(f"Loaded model: {model_key}")
        
        model = loaded_models[model_key]
        
        # Prepare data
        X, last_date, processed_data = prepare_prediction_data(
            request.symbol, 
            request.model_type, 
            request.use_latest_data
        )
        
        # Make predictions
        if request.model_type == 'lstm' and hasattr(model, 'predict_sequence'):
            # Multi-step prediction for LSTM
            predictions = model.predict_sequence(X, request.days)
        else:
            # Single-step predictions (repeated)
            predictions = []
            current_X = X.copy()
            
            for _ in range(request.days):
                pred = model.predict(current_X)
                predictions.append(pred[0])
                
                # Update input for next prediction (simple approach)
                if len(current_X.shape) == 3:  # LSTM case
                    # Shift sequence and add prediction
                    new_step = current_X[0, -1, :].copy()
                    new_step[:len(pred[0])] = pred[0]
                    current_X = np.roll(current_X, -1, axis=1)
                    current_X[0, -1, :] = new_step
                else:
                    # For non-sequence models, use same input
                    pass
            
            predictions = np.array(predictions)
        
        # Inverse transform predictions if needed
        if request.model_type == 'lstm':
            try:
                target_columns = ['Open', 'High', 'Low', 'Close']
                predictions = preprocessor.inverse_scale_data(
                    predictions, 
                    scaler_type='standard',
                    columns=target_columns
                )
            except:
                pass
        
        # Generate future dates
        future_dates = generate_future_dates(last_date, request.days)
        
        # Format predictions
        target_names = ['Open', 'High', 'Low', 'Close']
        formatted_predictions = []
        
        for i, date in enumerate(future_dates):
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                pred_dict = {}
                for j, name in enumerate(target_names[:predictions.shape[1]]):
                    pred_dict[name] = float(predictions[i, j])
            else:
                pred_dict = {"Close": float(predictions[i] if predictions.ndim == 1 else predictions[i, 0])}
            
            formatted_predictions.append(pred_dict)
        
        # Calculate confidence intervals (if model supports uncertainty)
        confidence_intervals = None
        if hasattr(model, 'predict_with_uncertainty'):
            try:
                _, uncertainty = model.predict_with_uncertainty(X)
                # Format uncertainty as confidence intervals
                confidence_intervals = []
                for i in range(len(formatted_predictions)):
                    ci_dict = {}
                    for key in formatted_predictions[i].keys():
                        std = uncertainty[i] if uncertainty.ndim == 1 else uncertainty[i, 0]
                        ci_dict[f"{key}_lower"] = formatted_predictions[i][key] - 1.96 * std
                        ci_dict[f"{key}_upper"] = formatted_predictions[i][key] + 1.96 * std
                    confidence_intervals.append(ci_dict)
            except:
                pass
        
        # Prepare metadata
        metadata = {
            "model_info": model.get_model_info(),
            "data_last_updated": last_date.isoformat(),
            "prediction_timestamp": datetime.now().isoformat(),
            "data_points_used": len(processed_data) if hasattr(processed_data, '__len__') else "unknown"
        }
        
        return PredictionResponse(
            symbol=request.symbol,
            model_type=request.model_type,
            predictions=formatted_predictions,
            dates=future_dates,
            confidence_interval=confidence_intervals,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models."""
    models = []
    models_dir = Path("models")
    
    if models_dir.exists():
        for model_file in models_dir.glob("*.pkl"):
            # Parse model info from filename
            name_parts = model_file.stem.split('_')
            if len(name_parts) >= 3:
                symbol = name_parts[0]
                model_type = '_'.join(name_parts[1:-1])
                
                model_key = f"{symbol}_{model_type}"
                is_loaded = model_key in loaded_models
                
                models.append(ModelInfo(
                    model_type=model_type,
                    symbol=symbol,
                    is_loaded=is_loaded,
                    last_updated=datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                ))
    
    return models

@app.post("/models/load")
async def load_model_endpoint(symbol: str, model_type: str):
    """Load a specific model."""
    try:
        model_path = f"models/{symbol}_{model_type}_model.pkl"
        if not Path(model_path).exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Model file not found: {model_path}"
            )
        
        model_key = f"{symbol}_{model_type}"
        loaded_models[model_key] = load_model(model_path, model_type)
        
        return {"message": f"Model {model_key} loaded successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/unload")
async def unload_model_endpoint(symbol: str, model_type: str):
    """Unload a specific model."""
    model_key = f"{symbol}_{model_type}"
    
    if model_key in loaded_models:
        del loaded_models[model_key]
        return {"message": f"Model {model_key} unloaded successfully"}
    else:
        raise HTTPException(
            status_code=404, 
            detail=f"Model {model_key} not currently loaded"
        )

@app.get("/symbols")
async def get_available_symbols():
    """Get list of symbols with trained models."""
    symbols = set()
    models_dir = Path("models")
    
    if models_dir.exists():
        for model_file in models_dir.glob("*.pkl"):
            name_parts = model_file.stem.split('_')
            if len(name_parts) >= 3:
                symbols.add(name_parts[0])
    
    return {"symbols": sorted(list(symbols))}

@app.get("/data/{symbol}")
async def get_latest_data(symbol: str, period: str = "1mo"):
    """Get latest market data for a symbol."""
    try:
        data = data_fetcher.fetch_symbol_data(
            symbol=symbol,
            period=period,
            save_to_file=False
        )
        
        # Convert to JSON-serializable format
        data_dict = data.reset_index().to_dict('records')
        
        return {
            "symbol": symbol,
            "period": period,
            "data": data_dict,
            "shape": data.shape,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background task for model training
@app.post("/train")
async def trigger_training(
    background_tasks: BackgroundTasks,
    symbol: str,
    model_type: str,
    period: str = "2y"
):
    """Trigger model training in the background."""
    def train_model_task():
        try:
            # Import training module
            from train import ModelTrainer
            
            trainer = ModelTrainer(symbol, model_type)
            results = trainer.run_training_pipeline(
                period=period,
                force_fetch=True,
                save_model=True,
                track_mlflow=True
            )
            
            logger.info(f"Training completed for {symbol}_{model_type}")
            
        except Exception as e:
            logger.error(f"Training failed for {symbol}_{model_type}: {str(e)}")
    
    background_tasks.add_task(train_model_task)
    
    return {
        "message": f"Training started for {symbol} with {model_type} model",
        "status": "training_started",
        "symbol": symbol,
        "model_type": model_type
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "app:app",
        host=EnvConfig.API_HOST,
        port=EnvConfig.API_PORT,
        reload=True,
        log_level="info"
    )
