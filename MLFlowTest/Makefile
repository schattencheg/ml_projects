# Makefile for OHLC Price Prediction Project

.PHONY: help install setup-mlflow train evaluate api test clean

# Default target
help:
	@echo "OHLC Price Prediction Project"
	@echo "============================="
	@echo ""
	@echo "Available commands:"
	@echo "  install      - Install Python dependencies"
	@echo "  setup-mlflow - Setup MLflow tracking server"
	@echo "  train        - Train models (specify SYMBOL and MODEL)"
	@echo "  evaluate     - Evaluate trained model"
	@echo "  api          - Start prediction API server"
	@echo "  example      - Run complete pipeline example"
	@echo "  notebook     - Start Jupyter notebook server"
	@echo "  test         - Run tests"
	@echo "  clean        - Clean generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make train SYMBOL=AAPL MODEL=lstm"
	@echo "  make evaluate SYMBOL=AAPL MODEL=lstm"
	@echo "  make api"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully!"

# Setup MLflow
setup-mlflow:
	@echo "Setting up MLflow tracking server..."
	python scripts/setup_mlflow.py
	@echo "✅ MLflow setup completed!"

# Train model
train:
	@if [ -z "$(SYMBOL)" ] || [ -z "$(MODEL)" ]; then \
		echo "❌ Please specify SYMBOL and MODEL"; \
		echo "Usage: make train SYMBOL=AAPL MODEL=lstm"; \
		echo "Available models: lstm, rf, xgb, ensemble"; \
		exit 1; \
	fi
	@echo "Training $(MODEL) model for $(SYMBOL)..."
	python src/train.py --symbol $(SYMBOL) --model $(MODEL)
	@echo "✅ Training completed!"

# Evaluate model
evaluate:
	@if [ -z "$(SYMBOL)" ] || [ -z "$(MODEL)" ]; then \
		echo "❌ Please specify SYMBOL and MODEL"; \
		echo "Usage: make evaluate SYMBOL=AAPL MODEL=lstm"; \
		exit 1; \
	fi
	@echo "Evaluating $(MODEL) model for $(SYMBOL)..."
	python src/evaluate.py --model-path models/$(SYMBOL)_$(MODEL)_model.pkl --symbol $(SYMBOL)
	@echo "✅ Evaluation completed!"

# Start API server
api:
	@echo "Starting prediction API server..."
	@echo "API will be available at: http://127.0.0.1:8000"
	@echo "API docs at: http://127.0.0.1:8000/docs"
	cd api && python app.py

# Run complete example
example:
	@echo "Running complete pipeline example..."
	python scripts/run_example.py
	@echo "✅ Example completed!"

# Start Jupyter notebook
notebook:
	@echo "Starting Jupyter notebook server..."
	@echo "Notebooks available at: http://127.0.0.1:8888"
	jupyter notebook notebooks/

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v
	@echo "✅ Tests completed!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf models/*.pkl
	rm -rf data/*.csv
	rm -rf evaluation_plots/
	rm -rf mlruns/
	rm -f evaluation_report.md
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "✅ Cleanup completed!"

# Development commands
dev-setup: install setup-mlflow
	@echo "✅ Development environment setup completed!"

# Quick start for new users
quickstart: dev-setup example
	@echo "🚀 Quickstart completed!"
	@echo "   - MLflow UI: http://127.0.0.1:5000"
	@echo "   - Check evaluation_plots/ for results"
	@echo "   - Run 'make api' to start the prediction service"
