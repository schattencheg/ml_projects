"""
Test script to verify strategy module works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing strategy module...\n")

try:
    # Test importing strategies
    from src.strategies import BaseStrategy, MLStrategy
    print("✓ Strategy imports successful!")
    
    # Test importing from main src package
    from src import BaseStrategy as BS, MLStrategy as MS
    print("✓ Imports from src successful!")
    
    # Test creating strategy instance
    print("\nTesting strategy instantiation...\n")
    
    strategy = MLStrategy(
        name='Test_Strategy',
        holding_period=15,
        trailing_stop_pct=0.05,
        enable_trailing_stop=False
    )
    print(f"✓ Created MLStrategy: {strategy}")
    
    # Test inheritance
    print("\nTesting inheritance...\n")
    assert isinstance(strategy, BaseStrategy), "MLStrategy should inherit from BaseStrategy"
    print("✓ MLStrategy inherits from BaseStrategy")
    
    # Test methods exist
    print("\nTesting methods exist...\n")
    
    methods = [
        'generate_signals',
        'should_enter_long',
        'should_enter_short',
        'should_exit',
        'open_position',
        'close_position',
        'backtest',
        'get_statistics',
        'get_config'
    ]
    
    for method in methods:
        assert hasattr(strategy, method), f"MLStrategy should have {method} method"
    
    print(f"✓ All {len(methods)} methods exist")
    
    # Test configuration
    print("\nTesting configuration...\n")
    
    config = strategy.get_config()
    assert config['name'] == 'Test_Strategy'
    assert config['holding_period'] == 15
    assert config['trailing_stop_pct'] == 0.05
    assert config['enable_trailing_stop'] == False
    print("✓ Configuration correct")
    
    # Test position management
    print("\nTesting position management...\n")
    
    strategy.reset()
    assert len(strategy.get_open_positions()) == 0
    print("✓ Reset works")
    
    # Open a position
    position = strategy.open_position('long', 0, 100.0, 10.0)
    assert len(strategy.get_open_positions()) == 1
    assert strategy.has_open_position()
    print("✓ Open position works")
    
    # Update tracking
    strategy.update_position_tracking(position, 110.0)
    assert position['highest_price'] == 110.0
    print("✓ Position tracking works")
    
    # Close position
    closed = strategy.close_position(position, 10, 105.0, 'test')
    assert len(strategy.get_open_positions()) == 0
    assert len(strategy.get_closed_positions()) == 1
    assert closed['pnl'] == 50.0  # (105 - 100) * 10
    print("✓ Close position works")
    
    # Test statistics
    stats = strategy.get_statistics()
    assert stats['total_trades'] == 1
    assert stats['winning_trades'] == 1
    assert stats['win_rate'] == 1.0
    print("✓ Statistics calculation works")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYou can now use the strategy module:")
    print("  from src.strategies import MLStrategy")
    print("\nStrategy features:")
    print("  ✓ Long and short positions")
    print("  ✓ Fixed holding period exit")
    print("  ✓ Optional trailing stop loss")
    print("  ✓ Position tracking")
    print("  ✓ Comprehensive statistics")
    print("\n")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nPlease check:")
    print("  1. src/strategies/__init__.py exports the classes")
    print("  2. All strategy implementation files exist")
    sys.exit(1)
    
except AssertionError as e:
    print(f"❌ Assertion Error: {e}")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
