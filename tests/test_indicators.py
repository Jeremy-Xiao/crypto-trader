"""
测试技术指标计算的正确性
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.indicators import SMA, EMA, RSI, MACD, BollingerBands, ATR


def test_ema():
    """测试EMA计算"""
    # 使用已知数据验证
    prices = [10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10]
    ema = EMA(prices, 5)
    
    # EMA不应该有NaN（adjust=False时）
    assert not ema.isna().any(), "EMA should not have NaN values with adjust=False"
    
    # EMA第一个值应该等于价格第一个值
    assert ema.iloc[0] == 10, f"EMA first value should be 10, got {ema.iloc[0]}"
    
    # 手动验证最后一个值
    k = 2 / (5 + 1)
    manual_ema = prices[0]
    for p in prices[1:]:
        manual_ema = p * k + manual_ema * (1 - k)
    assert abs(ema.iloc[-1] - manual_ema) < 1e-10, f"EMA mismatch: {ema.iloc[-1]} vs {manual_ema}"
    
    # 测试长序列 - 确保不截断
    long_prices = list(np.random.randn(500) + 100)
    ema_long = EMA(long_prices, 20)
    assert len(ema_long) == len(long_prices), "EMA length should match input length"
    
    print("  [PASS] EMA tests")


def test_rsi():
    """测试RSI计算"""
    # 纯上涨序列RSI应该接近100
    prices = list(range(1, 50))
    rsi = RSI(prices, 14)
    assert rsi.iloc[-1] > 95, f"RSI for pure uptrend should be > 95, got {rsi.iloc[-1]}"
    
    # 纯下跌序列RSI应该接近0
    prices = list(range(50, 1, -1))
    rsi = RSI(prices, 14)
    assert rsi.iloc[-1] < 5, f"RSI for pure downtrend should be < 5, got {rsi.iloc[-1]}"
    
    print("  [PASS] RSI tests")


def test_macd():
    """测试MACD计算"""
    prices = list(np.random.randn(100) + 100)
    macd = MACD(prices, 12, 26, 9)
    
    assert "macd" in macd
    assert "signal" in macd
    assert "histogram" in macd
    
    # histogram = macd - signal
    diff = (macd["macd"] - macd["signal"]).dropna()
    hist = macd["histogram"].dropna()
    assert np.allclose(diff.values, hist.values, atol=1e-10), "Histogram should equal MACD - Signal"
    
    print("  [PASS] MACD tests")


def test_bollinger_bands():
    """测试布林带"""
    prices = list(np.random.randn(50) + 100)
    bb = BollingerBands(prices, 20, 2.0)
    
    # 上轨 > 中轨 > 下轨
    valid = bb["upper"].dropna()
    middle = bb["middle"].dropna()
    lower = bb["lower"].dropna()
    
    assert (valid > middle).all(), "Upper band should be above middle"
    assert (middle > lower).all(), "Middle band should be above lower"
    
    print("  [PASS] Bollinger Bands tests")


def test_indicators_no_dict_error():
    """测试Dict类型标注不会导致NameError"""
    # 如果Dict没有import，这行会抛NameError
    prices = list(np.random.randn(50) + 100)
    macd = MACD(prices)
    bb = BollingerBands(prices)
    assert macd is not None
    assert bb is not None
    print("  [PASS] No Dict import error")


if __name__ == "__main__":
    print("Running indicator tests...")
    test_ema()
    test_rsi()
    test_macd()
    test_bollinger_bands()
    test_indicators_no_dict_error()
    print("\nAll indicator tests passed!")
