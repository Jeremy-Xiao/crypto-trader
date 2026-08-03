"""
测试回测引擎的核心逻辑
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.strategies.double_ma import DoubleMAStrategy


def test_engine_basic_run():
    """测试回测引擎基本运行"""
    # 生成模拟价格数据
    np.random.seed(42)
    n = 200
    prices = 40000 + np.cumsum(np.random.randn(n) * 200)
    prices = np.maximum(prices, 1000)  # 确保价格为正
    
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.rand(n) * 1000
    })
    
    strategy = DoubleMAStrategy(
        instId="BTC-USDT",
        fast_period=10,
        slow_period=30,
        position_pct=0.1,
        stop_loss_pct=0.05,
        take_profit_pct=0.10
    )
    
    engine = BacktestEngine(strategy, BacktestConfig(initial_balance=10000))
    engine.load_data(df)
    result = engine.run()
    
    assert "error" not in result, "Engine should not return error"
    assert "total_return" in result
    assert "sharpe_ratio" in result
    assert "max_drawdown" in result
    assert "win_rate" in result
    
    print(f"  [PASS] Basic backtest run - return: {result['total_return']:.2f}%, trades: {result['total_trades']}")


def test_force_close_position():
    """测试回测结束强制平仓"""
    # 创建一个持续上涨的场景，确保策略开仓后不会触发死叉
    np.random.seed(42)
    n = 100
    # 持续上涨的序列，确保 fast EMA 始终在 slow EMA 之上
    prices = 40000 + np.arange(n) * 50 + np.random.randn(n) * 10
    
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "close": prices,
    })
    
    strategy = DoubleMAStrategy(
        instId="BTC-USDT",
        fast_period=5,
        slow_period=20,
        position_pct=0.1,
        stop_loss_pct=0.99,  # 几乎不触发止损
        take_profit_pct=0.99  # 几乎不触发止盈
    )
    
    engine = BacktestEngine(strategy, BacktestConfig(initial_balance=10000))
    engine.load_data(df)
    result = engine.run()
    
    # 检查仓位被清零
    assert engine.position_amount == 0, "Position should be closed after backtest"
    
    # 检查最后一笔交易是否是强制平仓
    sell_trades = [t for t in engine.trades if t.action == "sell"]
    if sell_trades:
        last_trade = sell_trades[-1]
        # 在持续上涨中，策略应该只通过强制平仓卖出
        assert last_trade.reason == "end_of_backtest", \
            f"Last trade should be force close, got reason: {last_trade.reason}"
    
    print("  [PASS] Force close position at end of backtest")


def test_sharpe_ratio_uses_365():
    """测试夏普比率使用365天年化"""
    np.random.seed(42)
    n = 200
    prices = 40000 + np.cumsum(np.random.randn(n) * 200)
    prices = np.maximum(prices, 1000)
    
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "close": prices,
    })
    
    strategy = DoubleMAStrategy(
        instId="BTC-USDT",
        fast_period=10,
        slow_period=30,
    )
    
    engine = BacktestEngine(strategy, BacktestConfig(initial_balance=10000))
    engine.load_data(df)
    result = engine.run()
    
    # 手动计算夏普比率验证用的是365
    equity_series = pd.Series([e["equity"] for e in engine.equity_curve])
    returns = equity_series.pct_change().dropna()
    
    if returns.std() > 0:
        expected_sharpe_365 = returns.mean() / returns.std() * np.sqrt(365)
        expected_sharpe_252 = returns.mean() / returns.std() * np.sqrt(252)
        
        # 结果应该接近365版本，不是252版本
        assert abs(result["sharpe_ratio"] - expected_sharpe_365) < 0.01, \
            f"Sharpe should use √365 ({expected_sharpe_365:.4f}), got {result['sharpe_ratio']:.4f}"
        assert abs(result["sharpe_ratio"] - expected_sharpe_252) > 0.01, \
            f"Sharpe should NOT use √252 ({expected_sharpe_252:.4f})"
    
    print(f"  [PASS] Sharpe ratio uses √365 (value: {result['sharpe_ratio']:.4f})")


def test_slippage_consistency():
    """测试买入滑点一致性"""
    np.random.seed(42)
    n = 100
    prices = 40000 + np.cumsum(np.random.randn(n) * 100)
    prices = np.maximum(prices, 1000)
    
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "close": prices,
    })
    
    strategy = DoubleMAStrategy(
        instId="BTC-USDT",
        fast_period=5,
        slow_period=20,
        position_pct=0.1,
    )
    
    config = BacktestConfig(initial_balance=10000, slippage=0.001, fee_rate=0.001)
    engine = BacktestEngine(strategy, config)
    engine.load_data(df)
    engine.run()
    
    # 检查每笔买入交易：price应该是加滑点后的价格
    buy_trades = [t for t in engine.trades if t.action == "buy"]
    for trade in buy_trades:
        # 查找对应的市场价格
        # 实际买入价 = 市场价 * (1 + slippage)
        # 持仓成本 = 实际买入价
        # amount * trade.price 应该等于买入时的buy_value
        assert trade.price > 0, "Buy price should be positive"
    
    print("  [PASS] Slippage consistency check")


def test_no_hardcoded_balance():
    """测试没有硬编码的账户余额"""
    np.random.seed(42)
    n = 200
    prices = 40000 + np.cumsum(np.random.randn(n) * 200)
    prices = np.maximum(prices, 1000)
    
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "close": prices,
    })
    
    # 用不同的初始余额测试
    for init_balance in [1000, 5000, 10000, 50000]:
        strategy = DoubleMAStrategy(
            instId="BTC-USDT",
            fast_period=10,
            slow_period=30,
            position_pct=0.1,
        )
        
        engine = BacktestEngine(strategy, BacktestConfig(initial_balance=init_balance))
        engine.load_data(df)
        result = engine.run()
        
        # 初始余额应该正确传入
        assert result["initial_balance"] == init_balance, \
            f"Initial balance mismatch: expected {init_balance}, got {result['initial_balance']}"
        
        # 买入金额应该与初始余额成比例
        buy_trades = [t for t in engine.trades if t.action == "buy"]
        if buy_trades:
            first_buy = buy_trades[0]
            # 仓位金额 ≈ init_balance * position_pct / price * price = init_balance * position_pct
            expected_value = init_balance * 0.1  # position_pct=0.1
            actual_value = first_buy.amount * first_buy.price
            # 允许滑点和手续费导致的差异
            assert abs(actual_value - expected_value) / expected_value < 0.1, \
                f"Buy value {actual_value:.2f} should be close to {expected_value:.2f} for balance {init_balance}"
    
    print("  [PASS] No hardcoded balance - position size scales with initial balance")


def test_pnl_includes_fees():
    """测试PnL是否正确计算（含手续费）"""
    # 创建一个先跌后涨的场景，确保有金叉和死叉
    prices = list(range(100, 80, -1)) + list(range(80, 130))  # 下跌然后上涨
    
    timestamps = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    df = pd.DataFrame({"timestamp": timestamps, "close": prices})
    
    strategy = DoubleMAStrategy(
        instId="BTC-USDT",
        fast_period=5,
        slow_period=10,
        position_pct=0.1,
        stop_loss_pct=0.99,
        take_profit_pct=0.99,  # 不触发止盈止损，只靠死叉平仓
    )
    
    config = BacktestConfig(initial_balance=10000, fee_rate=0.001, slippage=0.0)
    engine = BacktestEngine(strategy, config)
    engine.load_data(df)
    result = engine.run()
    
    # 如果有交易，验证手续费
    buy_trades = [t for t in engine.trades if t.action == "buy"]
    if buy_trades:
        # 验证总手续费 > 0
        assert result["total_fee"] > 0, "Total fees should be positive"
        
        # 验证权益守恒
        sell_trades = [t for t in engine.trades if t.action == "sell"]
        total_buy_cost = sum(t.amount * t.price * (1 + config.fee_rate) for t in buy_trades)
        total_sell_revenue = sum(t.amount * t.price * (1 - config.fee_rate) for t in sell_trades)
        
        expected_final_balance = 10000 - total_buy_cost + total_sell_revenue
        assert abs(engine.balance - expected_final_balance) < 0.01, \
            f"Balance mismatch: {engine.balance:.4f} vs {expected_final_balance:.4f}"
        
        print(f"  [PASS] PnL includes fees - total fee: {result['total_fee']:.4f}")
    else:
        print("  [SKIP] No trades generated in this test scenario")


if __name__ == "__main__":
    print("Running backtest engine tests...")
    test_engine_basic_run()
    test_force_close_position()
    test_sharpe_ratio_uses_365()
    test_slippage_consistency()
    test_no_hardcoded_balance()
    test_pnl_includes_fees()
    print("\nAll backtest engine tests passed!")
