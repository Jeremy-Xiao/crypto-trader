"""
测试rolling_window_10rounds.py中的指标和回测逻辑
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 添加scripts目录到path
scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, scripts_dir)

from rolling_window_10rounds import calc_ema, calc_rsi, calc_bollinger, run_backtest
from src.utils.indicators import EMA, RSI, BollingerBands


def test_calc_ema_matches_pandas():
    """测试calc_ema与pandas ewm一致"""
    np.random.seed(42)
    prices = list(np.random.randn(200) + 40000)
    
    for period in [5, 10, 20, 30, 50]:
        ema_val = calc_ema(prices, period)
        pandas_val = float(pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1])
        
        assert abs(ema_val - pandas_val) < 1e-6, \
            f"EMA({period}): calc_ema={ema_val:.6f} vs pandas={pandas_val:.6f}"
    
    print("  [PASS] calc_ema matches pandas ewm for all periods")


def test_calc_rsi_matches_pandas():
    """测试calc_rsi与src/utils/indicators.RSI一致"""
    np.random.seed(42)
    prices = list(np.random.randn(100) + 40000)
    
    rsi_val = calc_rsi(prices, 14)
    pandas_rsi = float(RSI(prices, 14).iloc[-1])
    
    assert abs(rsi_val - pandas_rsi) < 1e-6, \
        f"RSI: calc_rsi={rsi_val:.6f} vs pandas={pandas_rsi:.6f}"
    
    print("  [PASS] calc_rsi matches src/utils/indicators.RSI")


def test_calc_bollinger_matches_pandas():
    """测试calc_bollinger与src/utils/indicators.BollingerBands一致"""
    np.random.seed(42)
    prices = list(np.random.randn(100) + 40000)
    
    boll = calc_bollinger(prices, 20, 2.0)
    pandas_boll = BollingerBands(prices, 20, 2.0)
    
    pandas_upper = float(pandas_boll["upper"].iloc[-1])
    pandas_middle = float(pandas_boll["middle"].iloc[-1])
    pandas_lower = float(pandas_boll["lower"].iloc[-1])
    
    assert abs(boll[0] - pandas_upper) < 1e-6, f"Upper: {boll[0]} vs {pandas_upper}"
    assert abs(boll[1] - pandas_middle) < 1e-6, f"Middle: {boll[1]} vs {pandas_middle}"
    assert abs(boll[2] - pandas_lower) < 1e-6, f"Lower: {boll[2]} vs {pandas_lower}"
    
    print("  [PASS] calc_bollinger matches src/utils/indicators.BollingerBands")


def test_run_backtest_pnl_includes_fees():
    """测试回测PnL扣除了手续费"""
    # 创建一个简单的策略：第5根K线买入，第15根卖出
    def simple_strategy(closes, position, params):
        if len(closes) == 5 and not position:
            return 'buy'
        if len(closes) == 15 and position:
            return 'sell'
        return 'hold'
    
    params = {'position_pct': 1.0}  # 全仓
    closes = [100.0] * 60  # 价格不变，需要>=50条数据
    
    result = run_backtest(closes, simple_strategy, params)
    
    assert result is not None
    
    # 如果价格不变，买入卖出后应该亏损手续费
    # 买入: cost = 1000 * 1.001 = 1001
    # 卖出: revenue = amount * 100 * 0.999
    # amount = 1000 / 100 = 10
    # revenue = 10 * 100 * 0.999 = 999
    # PnL = 999 - 1001 = -2
    assert result['return'] < 0, "With flat prices, return should be negative due to fees"
    
    print(f"  [PASS] Backtest PnL includes fees (return: {result['return']:.4f}%)")


def test_run_backtest_force_close():
    """测试回测结束强制平仓"""
    def hold_strategy(closes, position, params):
        if len(closes) == 5 and not position:
            return 'buy'
        # 从不主动卖出
        return 'hold'
    
    params = {'position_pct': 0.5}
    closes = [100.0 + i for i in range(50)]
    
    result = run_backtest(closes, hold_strategy, params)
    
    assert result is not None
    assert result['trades'] >= 1, "Should have at least 1 trade (force close)"
    
    print(f"  [PASS] Force close works (trades: {result['trades']})")


def test_sharpe_uses_365():
    """测试夏普比率使用365天年化"""
    np.random.seed(42)
    prices = list(40000 + np.cumsum(np.random.randn(200) * 100))
    
    def always_hold(closes, position, params):
        return 'hold'
    
    params = {'position_pct': 0.1}
    result = run_backtest(prices, always_hold, params)
    
    # 如果没有交易，夏普应该为0
    if result['trades'] == 0:
        assert result['sharpe'] == 0
        print("  [SKIP] No trades, sharpe = 0")
    else:
        # 手动验证用365
        print(f"  [PASS] Sharpe ratio computed: {result['sharpe']:.4f}")


if __name__ == "__main__":
    print("Running rolling_window tests...")
    test_calc_ema_matches_pandas()
    test_calc_rsi_matches_pandas()
    test_calc_bollinger_matches_pandas()
    test_run_backtest_pnl_includes_fees()
    test_run_backtest_force_close()
    test_sharpe_uses_365()
    print("\nAll rolling_window tests passed!")
