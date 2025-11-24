# ML Framework - Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        ML FRAMEWORK                              │
│                  Financial Data Analysis & Backtesting           │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │              DataProvider                                │        │
│  │  • load_csv()         • validate_data()                 │        │
│  │  • load_yahoo()       • clean_data()                    │        │
│  │  • save_data()        • split_data()                    │        │
│  └─────────────────────────────────────────────────────────┘        │
│                            │                                          │
│                            ▼                                          │
│                    OHLCV DataFrame                                    │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FEATURE LAYER                                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │           FeaturesGenerator                              │        │
│  │  • generate_features()  • create_target()               │        │
│  │  • get_feature_names()  • select_features()             │        │
│  │                                                          │        │
│  │  Features:                                               │        │
│  │  ├─ Moving Averages (SMA, EMA)                          │        │
│  │  ├─ Momentum (RSI, MACD)                                │        │
│  │  ├─ Volatility (Bollinger Bands, ATR)                   │        │
│  │  └─ Volume Indicators                                   │        │
│  └─────────────────────────────────────────────────────────┘        │
│                            │                                          │
│                            ▼                                          │
│              Features DataFrame + Target                              │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────┐  ┌──────────────────────────────┐          │
│  │   ModelManager      │  │      ML_Trainer              │          │
│  │  • get_models()     │  │  • train()                   │          │
│  │  • enable_model()   │──▶  • get_trained_models()      │          │
│  │  • save_models()    │  │  • get_results()             │          │
│  │  • load_models()    │  │  • print_results()           │          │
│  │  • print_config()   │  │                              │          │
│  └─────────────────────┘  └──────────────────────────────┘          │
│           │                            │                              │
│           │                            ▼                              │
│           │                  Trained Models + Scaler                 │
│           │                            │                              │
│           │                            ▼                              │
│           │               ┌──────────────────────────────┐           │
│           │               │      ML_Tester               │           │
│           │               │  • evaluate()                │           │
│           │               │  • get_predictions()         │           │
│           │               │  • compare_models()          │           │
│           │               │  • print_classification_     │           │
│           │               │    report()                  │           │
│           │               └──────────────────────────────┘           │
│           │                            │                              │
│           │                            ▼                              │
│           │                   Test Results + Metrics                 │
│           │                                                           │
│           ▼                                                           │
│  ┌─────────────────────────────────────────────────────┐            │
│  │         Timestamped Model Storage                    │            │
│  │  models/YYYY-MM-DD_HH-MM-SS/                        │            │
│  │  ├─ model_name.joblib                               │            │
│  │  ├─ scaler.joblib                                   │            │
│  │  └─ metadata.joblib                                 │            │
│  └─────────────────────────────────────────────────────┘            │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    BACKTESTING LAYER                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │                Backtester                                │        │
│  │  • run()                                                 │        │
│  │  • plot_results()                                        │        │
│  │  • get_results()                                         │        │
│  │                                                          │        │
│  │  Features:                                               │        │
│  │  ├─ Position Sizing                                     │        │
│  │  ├─ Commission Modeling                                 │        │
│  │  ├─ Performance Metrics                                 │        │
│  │  └─ Equity Curve Visualization                          │        │
│  └─────────────────────────────────────────────────────────┘        │
│                            │                                          │
│                            ▼                                          │
│                  Backtest Results + Plots                             │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌─────────────┐
│   Raw Data  │
│  (CSV/API)  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  DataProvider   │
│  • Load         │
│  • Validate     │
│  • Clean        │
│  • Split        │
└──────┬──────────┘
       │
       ▼
┌──────────────────────┐
│  FeaturesGenerator   │
│  • Generate Features │
│  • Create Target     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────┐
│  ML_Trainer      │
│  • Train Models  │
│  • Scale Data    │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  ML_Tester       │
│  • Evaluate      │
│  • Compare       │
└──────┬───────────┘
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌──────────────┐    ┌────────────────┐
│ ModelManager │    │  Backtester    │
│ • Save       │    │  • Run         │
│ • Load       │    │  • Visualize   │
└──────────────┘    └────────────────┘
```

## Class Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application                          │
└─────────────────────────────────────────────────────────────┘
       │
       │ uses
       ▼
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  DataProvider ──▶ FeaturesGenerator ──▶ ML_Trainer           │
│       │                                      │                │
│       │                                      ▼                │
│       │                                 ML_Tester             │
│       │                                      │                │
│       │                                      ▼                │
│       └──────────────────────────────▶ Backtester            │
│                                             │                 │
│                                             ▼                 │
│                                      ModelManager             │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Workflow Sequence

```
1. LOAD DATA
   ┌─────────────────┐
   │  DataProvider   │
   │  load_yahoo()   │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  OHLCV Data     │
   └────────┬────────┘
            │
2. GENERATE FEATURES
            │
            ▼
   ┌─────────────────────┐
   │ FeaturesGenerator   │
   │ generate_features() │
   │ create_target()     │
   └────────┬────────────┘
            │
            ▼
   ┌─────────────────┐
   │ Features + Target│
   └────────┬────────┘
            │
3. SPLIT DATA
            │
            ▼
   ┌─────────────────┐
   │  DataProvider   │
   │  split_data()   │
   └────────┬────────┘
            │
            ▼
   ┌──────────────────────────────┐
   │ Train │ Val │ Test           │
   └────┬───────┬──────┬──────────┘
        │       │      │
4. TRAIN MODELS
        │       │      │
        ▼       │      │
   ┌─────────────────┐│
   │  ML_Trainer     ││
   │  train()        ││
   └────────┬────────┘│
            │         │
            ▼         │
   ┌─────────────────┐│
   │ Trained Models  ││
   └────────┬────────┘│
            │         │
5. TEST MODELS       │
            │         │
            ▼         ▼
   ┌─────────────────────┐
   │    ML_Tester        │
   │    evaluate()       │
   └────────┬────────────┘
            │
            ▼
   ┌─────────────────┐
   │  Test Results   │
   └────────┬────────┘
            │
6. SAVE MODELS
            │
            ▼
   ┌─────────────────┐
   │  ModelManager   │
   │  save_models()  │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────────────┐
   │ models/YYYY-MM-DD_.../  │
   └─────────────────────────┘
            │
7. BACKTEST
            │
            ▼
   ┌─────────────────┐
   │  Backtester     │
   │  run()          │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ Backtest Results│
   │ + Visualization │
   └─────────────────┘
```

## Module Dependencies

```
┌─────────────────────────────────────────────────────────┐
│                   External Libraries                     │
├─────────────────────────────────────────────────────────┤
│  pandas │ numpy │ sklearn │ matplotlib │ yfinance │ joblib│
└────┬────────────────────────────────────────────────────┘
     │
     │ imported by
     ▼
┌─────────────────────────────────────────────────────────┐
│                    Core Modules                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  DataProvider        FeaturesGenerator                   │
│  (pandas, yfinance)  (pandas, numpy)                     │
│                                                          │
│  ModelManager        ML_Trainer                          │
│  (joblib, pathlib)   (sklearn, time)                     │
│                                                          │
│  ML_Tester           Backtester                          │
│  (sklearn, pandas)   (pandas, matplotlib)                │
│                                                          │
└─────────────────────────────────────────────────────────┘
     │
     │ used by
     ▼
┌─────────────────────────────────────────────────────────┐
│                  User Applications                       │
├─────────────────────────────────────────────────────────┤
│  • basic_workflow.py                                     │
│  • Custom scripts                                        │
│  • Jupyter notebooks                                     │
└─────────────────────────────────────────────────────────┘
```

## File Structure

```
ml_framework/
│
├── src/                          # Core modules (1,230 lines)
│   ├── __init__.py              # Package init
│   ├── data_provider.py         # 200 lines
│   ├── features_generator.py   # 250 lines
│   ├── model_manager.py         # 200 lines
│   ├── ml_trainer.py            # 180 lines
│   ├── ml_tester.py             # 180 lines
│   └── backtester.py            # 220 lines
│
├── examples/                     # Example scripts
│   └── basic_workflow.py        # 180 lines
│
├── models/                       # Saved models (gitignored)
│   └── YYYY-MM-DD_HH-MM-SS/
│       ├── *.joblib
│       ├── scaler.joblib
│       └── metadata.joblib
│
├── data/                         # Data files (gitignored)
│
├── README.md                     # Overview
├── STRUCTURE.md                  # Architecture docs (400 lines)
├── QUICKSTART.md                 # Quick start (300 lines)
├── PROJECT_SUMMARY.md            # Summary (400 lines)
├── ARCHITECTURE.md               # This file
├── requirements.txt              # Dependencies
└── .gitignore                   # Git ignore
```

## Extension Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Base Framework                          │
│  (Current Implementation - Functional & Simple)          │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ extend
                     ▼
┌─────────────────────────────────────────────────────────┐
│                Custom Extensions                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  • Add new data sources (databases, APIs)                │
│  • Add new features (custom indicators)                  │
│  • Add new models (XGBoost, LightGBM, Neural Networks)   │
│  • Add new metrics (custom performance measures)         │
│  • Add new backtest strategies (complex logic)           │
│  • Add MLflow tracking                                   │
│  • Add hyperparameter optimization                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Design Patterns Used

### 1. Single Responsibility Principle
```
Each class has ONE clear purpose:
• DataProvider     → Data management
• FeaturesGenerator → Feature engineering
• ModelManager     → Model lifecycle
• ML_Trainer       → Training
• ML_Tester        → Evaluation
• Backtester       → Backtesting
```

### 2. Dependency Injection
```
Classes accept dependencies as parameters:
• ML_Trainer.train(df, model_configs)
• ML_Tester.evaluate(df, models, scaler)
• Backtester.run(df, model, scaler)
```

### 3. Factory Pattern
```
ModelManager creates model instances:
• get_models() → Returns configured models
• ML_Trainer._create_model() → Creates model instances
```

### 4. Strategy Pattern
```
Different cleaning strategies:
• clean_data(method='drop')
• clean_data(method='ffill')
• clean_data(method='interpolate')
```

---

## Summary

This architecture provides:

✅ **Clear separation of concerns** - Each module has a specific role  
✅ **Modular design** - Easy to understand and extend  
✅ **Flexible workflow** - Can use modules independently  
✅ **Simple API** - Intuitive method names and parameters  
✅ **Extensible** - Clear extension points for customization  
✅ **Production-ready** - Includes versioning, metadata, error handling  

**Perfect foundation for building advanced ML trading systems!** 🚀
