# Backtest Visualization Improvements

## Summary

Fixed three critical issues with backtest visualization:
1. ✅ OHLC candlestick bars now display correctly (case-insensitive column detection)
2. ✅ Backtrader equity curve now shows actual values (not linear approximation)
3. ✅ Added support for SHORT entries with RED ARROW DOWN markers

---

## Issue 1: No OHLC Bars on Price Chart

### Problem
OHLC candlestick chart was not displaying, falling back to line chart instead.

### Root Cause
Column name detection was case-sensitive. If DataFrame had columns like `'Open'`, `'High'`, `'Low'`, `'Close'` (capitalized), the check for `['open', 'high', 'low', 'close']` (lowercase) would fail.

### Solution

**Before:**
```python
# Only checked for lowercase
has_ohlc = all(col in df.columns for col in ['open', 'high', 'low', 'close'])

fig.add_trace(go.Candlestick(
    open=df['open'],  # Would fail if column is 'Open'
    high=df['high'],
    low=df['low'],
    close=df['close']
))
```

**After:**
```python
# Case-insensitive detection
df_cols_lower = [col.lower() for col in df.columns]
has_ohlc = all(col in df_cols_lower for col in ['open', 'high', 'low', 'close'])

# Create column mapping
col_map = {}
for col in df.columns:
    col_map[col.lower()] = col

# Use mapped column names
fig.add_trace(go.Candlestick(
    open=df[col_map['open']],   # Works with 'open' or 'Open'
    high=df[col_map['high']],   # Works with 'high' or 'High'
    low=df[col_map['low']],     # Works with 'low' or 'Low'
    close=df[col_map['close']]  # Works with 'close' or 'Close'
))
```

### Benefits
✅ Works with lowercase: `open`, `high`, `low`, `close`  
✅ Works with capitalized: `Open`, `High`, `Low`, `Close`  
✅ Works with uppercase: `OPEN`, `HIGH`, `LOW`, `CLOSE`  
✅ Works with mixed case: `Open`, `HIGH`, `low`, `Close`  

---

## Issue 2: Backtrader Equity Curve Approximation

### Problem
Backtrader equity curve was using linear interpolation between initial and final capital, not showing actual equity progression.

### Root Cause
```python
# Old implementation - linear approximation
equity_curve = pd.Series(
    np.linspace(self.initial_capital, final_value, len(df)),
    index=df.index
)
```

This created a straight line from start to end, completely ignoring actual trade performance over time.

### Impact
- ❌ Couldn't see drawdowns
- ❌ Couldn't see equity volatility
- ❌ Couldn't see actual performance progression
- ❌ Misleading visualization

### Solution

**Step 1: Track Equity at Each Bar**

Added equity tracking to strategy:
```python
def __init__(strategy_self):
    strategy_self.equity_curve = []  # Track equity at each bar

def next(strategy_self):
    # Track equity at each bar
    strategy_self.equity_curve.append(strategy_self.broker.getvalue())
    
    # ... rest of strategy logic
```

**Step 2: Use Actual Equity Values**

```python
# Get actual equity curve from strategy
equity_values = strategy.equity_curve

# Ensure equity curve has same length as df
if len(equity_values) < len(df):
    # Pad with initial capital if needed
    equity_values = [self.initial_capital] * (len(df) - len(equity_values)) + equity_values
elif len(equity_values) > len(df):
    # Trim if too long
    equity_values = equity_values[:len(df)]

equity_curve = pd.Series(equity_values, index=df.index)
```

### Before vs After

**Before (Linear Approximation):**
```
Initial: $10,000
Final:   $12,000

Equity Curve:
Bar 0:   $10,000
Bar 1:   $10,020  (linear)
Bar 2:   $10,040  (linear)
...
Bar 99:  $11,980  (linear)
Bar 100: $12,000

Result: Straight line from $10k to $12k ❌
```

**After (Actual Values):**
```
Initial: $10,000
Final:   $12,000

Equity Curve:
Bar 0:   $10,000
Bar 1:   $10,000  (no position)
Bar 2:   $10,500  (trade profit)
Bar 3:   $10,200  (drawdown)
Bar 4:   $10,800  (recovery)
...
Bar 99:  $11,900  (actual value)
Bar 100: $12,000

Result: Realistic equity progression ✅
```

### Benefits
✅ **Accurate drawdown visualization** - See actual maximum drawdown  
✅ **Realistic performance** - Shows true equity volatility  
✅ **Better risk assessment** - Identify risky periods  
✅ **Proper comparison** - Compare with other backends accurately  

---

## Issue 3: No SHORT Entry Support

### Problem
Only LONG entries were supported with RED ARROW UP markers. SHORT entries had no visual distinction.

### Requirements
- **LONG entry**: RED ARROW UP (▲)
- **SHORT entry**: RED ARROW DOWN (▼)

### Solution

**Step 1: Detect LONG vs SHORT**

```python
# Determine if LONG or SHORT based on shares sign
shares = trade.get('shares', 0)
is_long = shares > 0  # Positive = LONG, Negative = SHORT
```

**Step 2: Separate Entry Markers**

```python
# Separate long and short entries
long_entry_indices = []
long_entry_prices = []
long_entry_hover = []

short_entry_indices = []
short_entry_prices = []
short_entry_hover = []

for i, trade in enumerate(trades):
    shares = trade.get('shares', 0)
    is_long = shares > 0
    
    if entry_idx is not None and entry_idx < len(df):
        hover_text = (
            f"<b>Trade #{i+1} - {'LONG' if is_long else 'SHORT'} ENTRY</b><br>"
            f"Price: ${entry_price:.2f}<br>"
            f"Shares: {abs(shares):.2f}"
        )
        
        if is_long:
            # LONG entry: RED ARROW UP
            long_entry_indices.append(df.index[entry_idx])
            long_entry_prices.append(entry_price)
            long_entry_hover.append(hover_text)
        else:
            # SHORT entry: RED ARROW DOWN
            short_entry_indices.append(df.index[entry_idx])
            short_entry_prices.append(entry_price)
            short_entry_hover.append(hover_text)
```

**Step 3: Add Both Marker Types**

```python
# Add LONG entry markers (RED arrows UP)
if long_entry_indices:
    fig.add_trace(go.Scatter(
        x=long_entry_indices,
        y=long_entry_prices,
        mode='markers',
        name='Long Entry',
        marker=dict(
            symbol='triangle-up',    # ▲
            size=12,
            color='red',
            line=dict(color='darkred', width=1)
        ),
        hovertext=long_entry_hover,
        hovertemplate='%{hovertext}<extra></extra>'
    ))

# Add SHORT entry markers (RED arrows DOWN)
if short_entry_indices:
    fig.add_trace(go.Scatter(
        x=short_entry_indices,
        y=short_entry_prices,
        mode='markers',
        name='Short Entry',
        marker=dict(
            symbol='triangle-down',  # ▼
            size=12,
            color='red',
            line=dict(color='darkred', width=1)
        ),
        hovertext=short_entry_hover,
        hovertemplate='%{hovertext}<extra></extra>'
    ))
```

### Visual Result

```
OHLC Chart with Trades:

Price
  |
  |     ▼ SHORT entry (RED)
  |    /
  |   /
  |  /
  | ▲ LONG entry (RED)
  |/
  +-------------------------> Time
  
Legend:
  ▲ = LONG entry (RED)
  ▼ = SHORT entry (RED)
  ✕ = Exit (GREEN=profit, RED=loss)
```

### Hover Information

**LONG Entry:**
```
Trade #5 - LONG ENTRY
Price: $45,000.00
Shares: 0.50
```

**SHORT Entry:**
```
Trade #7 - SHORT ENTRY
Price: $47,000.00
Shares: 0.30
```

### Benefits
✅ **Clear visual distinction** - LONG vs SHORT immediately visible  
✅ **Consistent color scheme** - Both use RED for entries  
✅ **Intuitive symbols** - UP for LONG, DOWN for SHORT  
✅ **Complete information** - Hover shows trade type and details  

---

## Files Modified

### 1. `src/backtesting/backtest_backtrader.py`

**Changes:**
- Added `equity_curve` list to track equity at each bar
- Updated `next()` to append equity value at each bar
- Replaced linear interpolation with actual equity values
- Added padding/trimming logic to match DataFrame length

**Lines modified:** ~20 lines

### 2. `src/managers/visualization_manager.py`

**Changes:**
- Added case-insensitive OHLC column detection
- Created column mapping for flexible column names
- Separated LONG and SHORT entry tracking
- Added SHORT entry markers with triangle-down symbol
- Updated hover text to show LONG/SHORT type

**Lines modified:** ~50 lines

---

## Compatibility

### OHLC Column Names Supported

| Format | Example | Supported |
|--------|---------|-----------|
| Lowercase | `open`, `high`, `low`, `close` | ✅ |
| Capitalized | `Open`, `High`, `Low`, `Close` | ✅ |
| Uppercase | `OPEN`, `HIGH`, `LOW`, `CLOSE` | ✅ |
| Mixed | `Open`, `HIGH`, `low`, `Close` | ✅ |

### Trade Types Supported

| Type | Shares Sign | Entry Marker | Color |
|------|-------------|--------------|-------|
| LONG | Positive (+) | Triangle UP (▲) | RED |
| SHORT | Negative (-) | Triangle DOWN (▼) | RED |

### Backtest Backends

| Backend | Equity Curve | OHLC Chart | LONG/SHORT |
|---------|-------------|------------|------------|
| NoLib | ✅ Actual | ✅ Works | ✅ Supported |
| Backtrader | ✅ Actual (Fixed) | ✅ Works | ✅ Supported |
| BacktestingPy | ✅ Actual | ✅ Works | ✅ Supported |

---

## Testing

### Test Case 1: OHLC Column Detection

```python
# Test with different column names
df1 = pd.DataFrame({'open': [...], 'high': [...], 'low': [...], 'close': [...]})
df2 = pd.DataFrame({'Open': [...], 'High': [...], 'Low': [...], 'Close': [...]})
df3 = pd.DataFrame({'OPEN': [...], 'HIGH': [...], 'LOW': [...], 'CLOSE': [...]})

# All should display candlestick chart ✅
```

### Test Case 2: Backtrader Equity Curve

```python
# Run Backtrader backtest
backtest = BacktestBacktrader(initial_capital=10000)
results = backtest.run(df, model, scaler, feature_cols)

# Check equity curve
equity = results['equity_curve']
print(f"Initial: ${equity.iloc[0]:.2f}")
print(f"Max: ${equity.max():.2f}")
print(f"Min: ${equity.min():.2f}")
print(f"Final: ${equity.iloc[-1]:.2f}")

# Should show realistic progression, not linear ✅
```

### Test Case 3: LONG/SHORT Entries

```python
# Create trades with LONG and SHORT
trades = [
    {'entry_idx': 10, 'shares': 0.5, ...},   # LONG (positive shares)
    {'entry_idx': 30, 'shares': -0.3, ...},  # SHORT (negative shares)
    {'entry_idx': 50, 'shares': 0.4, ...},   # LONG (positive shares)
]

# Visualization should show:
# - 2 RED arrows UP (LONG entries)
# - 1 RED arrow DOWN (SHORT entry)
# ✅
```

---

## Before vs After Comparison

### Issue 1: OHLC Chart

**Before:**
```
DataFrame columns: ['Open', 'High', 'Low', 'Close']
Detection: has_ohlc = False (case mismatch)
Result: Line chart displayed ❌
```

**After:**
```
DataFrame columns: ['Open', 'High', 'Low', 'Close']
Detection: has_ohlc = True (case-insensitive)
Result: Candlestick chart displayed ✅
```

### Issue 2: Equity Curve

**Before:**
```
Backtrader Equity Curve:
  $12,000 |                    ╱
          |                 ╱
          |              ╱
          |           ╱
  $10,000 |________╱
          +-------------------> Time
          
Linear interpolation ❌
```

**After:**
```
Backtrader Equity Curve:
  $12,000 |        ╱╲    ╱╲
          |       ╱  ╲  ╱  ╲
          |      ╱    ╲╱    ╲
          |     ╱            ╲
  $10,000 |____╱              ╲
          +-------------------> Time
          
Actual values with drawdowns ✅
```

### Issue 3: Entry Markers

**Before:**
```
All entries: ▲ (RED arrow UP)
No distinction between LONG and SHORT ❌
```

**After:**
```
LONG entries:  ▲ (RED arrow UP)
SHORT entries: ▼ (RED arrow DOWN)
Clear visual distinction ✅
```

---

## Benefits Summary

### 1. **Accurate Visualization**
✅ OHLC candlesticks display correctly  
✅ Equity curves show actual performance  
✅ Trade markers distinguish LONG/SHORT  

### 2. **Better Analysis**
✅ See actual drawdowns and volatility  
✅ Identify risky trading periods  
✅ Compare strategies accurately  

### 3. **Improved UX**
✅ Works with any column name format  
✅ Intuitive visual symbols  
✅ Clear hover information  

### 4. **Robust Implementation**
✅ Case-insensitive column detection  
✅ Handles edge cases (padding/trimming)  
✅ Backward compatible  

---

## Future Enhancements

### 1. Additional Entry Types

Support more entry types:
```python
# Based on entry_type field
if entry_type == 'long':
    symbol = 'triangle-up'
    color = 'red'
elif entry_type == 'short':
    symbol = 'triangle-down'
    color = 'red'
elif entry_type == 'scale_in':
    symbol = 'circle'
    color = 'orange'
```

### 2. Position Size Visualization

Scale marker size by position size:
```python
marker_size = 8 + (abs(shares) / max_shares) * 8  # 8-16 range
```

### 3. Equity Curve Smoothing

Option to smooth equity curve:
```python
equity_smooth = equity_curve.rolling(window=10).mean()
```

---

## Summary

**Issues Fixed:**
1. ✅ OHLC bars not displaying (case-sensitive columns)
2. ✅ Backtrader equity curve linear approximation
3. ✅ No SHORT entry support

**Solutions Applied:**
1. ✅ Case-insensitive column detection with mapping
2. ✅ Track actual equity at each bar
3. ✅ Separate LONG (▲) and SHORT (▼) markers

**Files Modified:** 2 files  
**Lines Changed:** ~70 lines total  
**Breaking Changes:** None  
**Backward Compatible:** Yes  

**Result:** Professional, accurate backtest visualization with complete LONG/SHORT support! 🎉

---

**Status: ✅ ALL ISSUES FIXED**

All three visualization issues resolved. Charts now display correctly with accurate data and complete trade type support.
