"""
Test script to verify backtesting module works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing backtesting module imports...\n")

try:
    # Test importing from backtesting package
    from src.backtesting import (
        BaseBacktest,
        BacktestNoLib,
        BacktestBacktrader,
        BacktestBacktestingPy
    )
    print("✓ All imports from src.backtesting successful!")
    
    # Test importing from main src package
    from src import (
        BaseBacktest as BB,
        BacktestNoLib as BNL,
        BacktestBacktrader as BBT,
        BacktestBacktestingPy as BBP
    )
    print("✓ All imports from src successful!")
    
    # Test creating instances
    print("\nTesting backtest instantiation...\n")
    
    nolib = BacktestNoLib(
        initial_capital=10000,
        commission=0.001,
        position_size=1.0,
        stop_loss=0.05,
        take_profit=0.10
    )
    print(f"✓ Created BacktestNoLib: {nolib.__class__.__name__}")
    
    backtrader = BacktestBacktrader(
        initial_capital=10000,
        commission=0.001,
        position_size=1.0
    )
    print(f"✓ Created BacktestBacktrader: {backtrader.__class__.__name__}")
    
    backtesting_py = BacktestBacktestingPy(
        initial_capital=10000,
        commission=0.001,
        position_size=1.0
    )
    print(f"✓ Created BacktestBacktestingPy: {backtesting_py.__class__.__name__}")
    
    # Test inheritance
    print("\nTesting inheritance...\n")
    
    assert isinstance(nolib, BaseBacktest), "BacktestNoLib should inherit from BaseBacktest"
    print("✓ BacktestNoLib inherits from BaseBacktest")
    
    assert isinstance(backtrader, BaseBacktest), "BacktestBacktrader should inherit from BaseBacktest"
    print("✓ BacktestBacktrader inherits from BaseBacktest")
    
    assert isinstance(backtesting_py, BaseBacktest), "BacktestBacktestingPy should inherit from BaseBacktest"
    print("✓ BacktestBacktestingPy inherits from BaseBacktest")
    
    # Test methods exist
    print("\nTesting methods exist...\n")
    
    methods = ['run', 'calculate_metrics', 'get_results', 'get_metrics', 'get_trades', 'print_results', 'save_results']
    
    for method in methods:
        assert hasattr(nolib, method), f"BacktestNoLib should have {method} method"
        assert hasattr(backtrader, method), f"BacktestBacktrader should have {method} method"
        assert hasattr(backtesting_py, method), f"BacktestBacktestingPy should have {method} method"
    
    print(f"✓ All {len(methods)} methods exist on all backends")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYou can now use the backtesting module:")
    print("  from src.backtesting import BacktestNoLib, BacktestBacktrader, BacktestBacktestingPy")
    print("\nRun the complete example:")
    print("  python btcusdt_backtest_comparison.py")
    print("\n")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nPlease check:")
    print("  1. src/backtesting/__init__.py exports the classes")
    print("  2. All backtest implementation files exist")
    sys.exit(1)
    
except AssertionError as e:
    print(f"❌ Assertion Error: {e}")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
