"""
测试策略逻辑
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.base import BaseStrategy, Signal, SignalType, PositionSide, Position
from src.strategies.double_ma import DoubleMAStrategy


def test_ema_uses_full_history():
    """测试EMA使用全部历史数据而非截断"""
    strategy = DoubleMAStrategy(
        instId="BTC-USDT",
        fast_period=10,
        slow_period=30,
    )
    
    # 生成100个数据点
    prices = list(np.random.randn(100) + 40000)
    
    # 使用策略的calculate_ema
    ema_short = strategy.calculate_ema(prices[:50], 10)
    ema_full = strategy.calculate_ema(prices, 10)
    
    # 如果截断的话，两者会相同。正确实现下应该不同
    assert ema_short != ema_full, "EMA should differ when using different history lengths"
    
    # 验证与pandas ewm一致
    s = pd.Series(prices)
    pandas_ema = float(s.ewm(span=10, adjust=False).mean().iloc[-1])
    assert abs(ema_full - pandas_ema) < 1e-6, f"EMA should match pandas ewm: {ema_full} vs {pandas_ema}"
    
    print("  [PASS] EMA uses full history, matches pandas ewm")


def test_stop_loss_short():
    """测试SHORT仓位止损"""
    from src.strategies.base import BaseStrategy
    
    # 创建一个简单的策略实例用于测试
    class TestStrategy(BaseStrategy):
        def generate_signal(self, data):
            return Signal(SignalType.HOLD, self.instId, 0, 0, "")
        
        def calculate_position_size(self, account_balance, price):
            return 1.0
    
    strategy = TestStrategy("test", "BTC-USDT", {"stop_loss_pct": 0.05})
    
    # 测试LONG止损
    strategy.open_position(100, 1, "2024-01-01", PositionSide.LONG)
    assert not strategy.should_stop_loss(97), "LONG should not stop loss at 3% drop"
    assert strategy.should_stop_loss(94), "LONG should stop loss at 6% drop"
    
    # 测试SHORT止损
    strategy.position = None
    strategy.open_position(100, 1, "2024-01-01", PositionSide.SHORT)
    assert not strategy.should_stop_loss(103), "SHORT should not stop loss at 3% rise"
    assert strategy.should_stop_loss(106), "SHORT should stop loss at 6% rise"
    
    print("  [PASS] Stop loss works for both LONG and SHORT")


def test_take_profit_short():
    """测试SHORT仓位止盈"""
    class TestStrategy(BaseStrategy):
        def generate_signal(self, data):
            return Signal(SignalType.HOLD, self.instId, 0, 0, "")
        
        def calculate_position_size(self, account_balance, price):
            return 1.0
    
    strategy = TestStrategy("test", "BTC-USDT", {"take_profit_pct": 0.10})
    
    # 测试LONG止盈
    strategy.open_position(100, 1, "2024-01-01", PositionSide.LONG)
    assert not strategy.should_take_profit(108), "LONG should not take profit at 8% gain"
    assert strategy.should_take_profit(111), "LONG should take profit at 11% gain"
    
    # 测试SHORT止盈
    strategy.position = None
    strategy.open_position(100, 1, "2024-01-01", PositionSide.SHORT)
    assert not strategy.should_take_profit(92), "SHORT should not take profit at 8% drop"
    assert strategy.should_take_profit(89), "SHORT should take profit at 11% drop"
    
    print("  [PASS] Take profit works for both LONG and SHORT")


def test_open_position_side():
    """测试open_position可以接受side参数"""
    class TestStrategy(BaseStrategy):
        def generate_signal(self, data):
            return Signal(SignalType.HOLD, self.instId, 0, 0, "")
        
        def calculate_position_size(self, account_balance, price):
            return 1.0
    
    strategy = TestStrategy("test", "BTC-USDT")
    
    # 默认LONG
    strategy.open_position(100, 1, "2024-01-01")
    assert strategy.position.side == PositionSide.LONG
    
    # 明确SHORT
    strategy.position = None
    strategy.open_position(100, 1, "2024-01-01", PositionSide.SHORT)
    assert strategy.position.side == PositionSide.SHORT
    
    print("  [PASS] open_position accepts side parameter")


def test_close_position_pnl_short():
    """测试SHORT仓位平仓PnL"""
    class TestStrategy(BaseStrategy):
        def generate_signal(self, data):
            return Signal(SignalType.HOLD, self.instId, 0, 0, "")
        
        def calculate_position_size(self, account_balance, price):
            return 1.0
    
    strategy = TestStrategy("test", "BTC-USDT")
    
    # SHORT: 入场100，平仓90 -> 盈利10
    strategy.open_position(100, 1, "2024-01-01", PositionSide.SHORT)
    strategy.close_position(90, "2024-01-02", "test")
    
    close_trade = [t for t in strategy.trades if t["action"] == "close"][0]
    assert close_trade["realized_pnl"] == 10, f"SHORT PnL should be 10, got {close_trade['realized_pnl']}"
    
    # SHORT: 入场100，平仓110 -> 亏损10
    strategy.position = None
    strategy.open_position(100, 1, "2024-01-01", PositionSide.SHORT)
    strategy.close_position(110, "2024-01-02", "test")
    
    close_trade = [t for t in strategy.trades if t["action"] == "close"][-1]
    assert close_trade["realized_pnl"] == -10, f"SHORT PnL should be -10, got {close_trade['realized_pnl']}"
    
    print("  [PASS] SHORT position PnL calculated correctly")


def test_strategy_uses_account_balance():
    """测试策略使用回测引擎传入的余额而非硬编码"""
    strategy = DoubleMAStrategy(
        instId="BTC-USDT",
        fast_period=5,
        slow_period=10,
        position_pct=0.1,
    )
    
    # 设置不同的账户余额
    for balance in [1000, 10000, 50000]:
        strategy.account_balance = balance
        size = strategy.calculate_position_size(balance, 40000)
        expected = balance * 0.1 / 40000
        assert abs(size - expected) < 1e-10, f"Position size should be {expected}, got {size}"
    
    print("  [PASS] Strategy uses dynamic account balance, not hardcoded 10000")


if __name__ == "__main__":
    print("Running strategy tests...")
    test_ema_uses_full_history()
    test_stop_loss_short()
    test_take_profit_short()
    test_open_position_side()
    test_close_position_pnl_short()
    test_strategy_uses_account_balance()
    print("\nAll strategy tests passed!")
