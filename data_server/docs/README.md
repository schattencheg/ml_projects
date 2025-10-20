# Data Provider Server - Documentation

Welcome to the Data Provider Server documentation! This directory contains comprehensive guides and references for using the server.

## 📚 Documentation Index

### Getting Started

- **[Quick Start Guide](QUICK_START.md)** - Get up and running in minutes
- **[Main README](../README.md)** - Project overview and basic usage

### API Documentation

- **[API Guide](API_GUIDE.md)** - Complete REST API reference with examples
- **[Detailed Instruments API](DETAILED_INSTRUMENTS_API.md)** - Advanced instrument data endpoints
- **[WebSocket Guide](WEBSOCKET_GUIDE.md)** - Real-time data streaming with WebSocket

### MLflow Integration

- **[MLflow Integration](MLFLOW_INTEGRATION.md)** - Experiment tracking setup and usage
- **[MLflow Troubleshooting](MLFLOW_TROUBLESHOOTING.md)** - Common issues and solutions
- **[MLflow Windows Fix](MLFLOW_WINDOWS_FIX.md)** - Windows-specific path configuration
- **[MLflow Fix Summary](MLFLOW_FIX_SUMMARY.md)** - Recent fixes and updates

### Advanced Topics

- **[Smart Update Feature](SMART_UPDATE_FEATURE.md)** - Intelligent data update mechanisms
- **[Refactoring Summary](REFACTORING_SUMMARY.md)** - Code improvements and changes

### Project Information

- **[Changelog](CHANGELOG.md)** - Version history and release notes

## 🚀 Quick Links

### For New Users
1. Start with [Quick Start Guide](QUICK_START.md)
2. Read the [Main README](../README.md)
3. Explore [API Guide](API_GUIDE.md)

### For API Users
1. [API Guide](API_GUIDE.md) - REST API endpoints
2. [WebSocket Guide](WEBSOCKET_GUIDE.md) - Real-time streaming
3. [Detailed Instruments API](DETAILED_INSTRUMENTS_API.md) - Advanced features

### For ML Developers
1. [MLflow Integration](MLFLOW_INTEGRATION.md) - Setup experiment tracking
2. [MLflow Troubleshooting](MLFLOW_TROUBLESHOOTING.md) - Solve common issues
3. [MLflow Windows Fix](MLFLOW_WINDOWS_FIX.md) - Windows configuration

## 📖 Documentation by Topic

### REST API
- Basic endpoints and usage
- Data retrieval and filtering
- Batch operations
- CSV exports

**See:** [API Guide](API_GUIDE.md)

### WebSocket
- Real-time data streaming
- Connection management
- Callbacks and events
- Batch requests with progress tracking

**See:** [WebSocket Guide](WEBSOCKET_GUIDE.md)

### MLflow
- Automatic server startup
- Experiment tracking
- Model logging
- Troubleshooting

**See:** [MLflow Integration](MLFLOW_INTEGRATION.md), [MLflow Troubleshooting](MLFLOW_TROUBLESHOOTING.md)

### Data Management
- Local caching
- Data updates
- Instrument management
- Resolution and period options

**See:** [Main README](../README.md), [Detailed Instruments API](DETAILED_INSTRUMENTS_API.md)

## 🔧 Troubleshooting

Having issues? Check these guides:

1. **MLflow not starting?** → [MLflow Troubleshooting](MLFLOW_TROUBLESHOOTING.md)
2. **Windows path errors?** → [MLflow Windows Fix](MLFLOW_WINDOWS_FIX.md)
3. **API questions?** → [API Guide](API_GUIDE.md)
4. **WebSocket issues?** → [WebSocket Guide](WEBSOCKET_GUIDE.md)

## 📝 Contributing

When adding new documentation:

1. Place `.md` files in this `docs/` directory
2. Update this README.md index
3. Update the main [README.md](../README.md) if needed
4. Follow the existing documentation style

## 🆘 Support

- Check the relevant guide above
- Review [Changelog](CHANGELOG.md) for recent changes
- Look at example files in the parent directory
- Check server logs: `data_server.log`

## 📂 File Organization

```
docs/
├── README.md                      # This file - documentation index
├── API_GUIDE.md                   # REST API reference
├── WEBSOCKET_GUIDE.md             # WebSocket documentation
├── MLFLOW_INTEGRATION.md          # MLflow setup guide
├── MLFLOW_TROUBLESHOOTING.md      # MLflow problem solving
├── MLFLOW_WINDOWS_FIX.md          # Windows-specific fixes
├── MLFLOW_FIX_SUMMARY.md          # Recent MLflow updates
├── QUICK_START.md                 # Quick start guide
├── DETAILED_INSTRUMENTS_API.md    # Advanced API features
├── SMART_UPDATE_FEATURE.md        # Smart update documentation
├── REFACTORING_SUMMARY.md         # Code changes summary
└── CHANGELOG.md                   # Version history
```

## 🔗 External Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Flask-SocketIO Documentation](https://flask-socketio.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [pandas Documentation](https://pandas.pydata.org/docs/)

---

**Last Updated:** 2025-10-20  
**Server Version:** 1.0.0  
**MLflow Version:** 3.5.0+
