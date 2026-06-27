"""
技术指标计算模块
常用交易指标的Python实现
"""

import pandas as pd
import numpy as np
from typing import Union


def SMA(prices: Union[pd.Series, list], period: int) -> pd.Series:
    """
    简单移动平均
    
    Args:
        prices: 价格序列
        period: 周期
    
    Returns:
        SMA序列
    """
    return pd.Series(prices).rolling(window=period).mean()


def EMA(prices: Union[pd.Series, list], period: int) -> pd.Series:
    """
    指数移动平均
    
    Args:
        prices: 价格序列
        period: 周期
    
    Returns:
        EMA序列
    """
    return pd.Series(prices).ewm(span=period, adjust=False).mean()


def RSI(prices: Union[pd.Series, list], period: int = 14) -> pd.Series:
    """
    相对强弱指数
    
    Args:
        prices: 价格序列
        period: 周期
    
    Returns:
        RSI序列
    """
    prices = pd.Series(prices)
    
    # 计算价格变动
    delta = prices.diff()
    
    # 分离上涨和下跌
    gain = delta.where(delta > 0, 0)
    loss = delta.where(delta < 0, 0)
    
    # 计算平均上涨和下跌
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # 计算RS
    rs = avg_gain / avg_loss
    
    # 计算RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def MACD(
    prices: Union[pd.Series, list],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Dict[str, pd.Series]:
    """
    MACD指标
    
    Args:
        prices: 价格序列
        fast_period: 快线周期
        slow_period: 慢线周期
        signal_period: 信号线周期
    
    Returns:
        {"macd": MACD线, "signal": 信号线, "histogram": 柱状图}
    """
    prices = pd.Series(prices)
    
    # 计算EMA
    ema_fast = EMA(prices, fast_period)
    ema_slow = EMA(prices, slow_period)
    
    # MACD线
    macd_line = ema_fast - ema_slow
    
    # 信号线
    signal_line = EMA(macd_line, signal_period)
    
    # 柱状图
    histogram = macd_line - signal_line
    
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }


def BollingerBands(
    prices: Union[pd.Series, list],
    period: int = 20,
    std_dev: float = 2.0
) -> Dict[str, pd.Series]:
    """
    布林带
    
    Args:
        prices: 价格序列
        period: 周期
        std_dev: 标准差倍数
    
    Returns:
        {"upper": 上轨, "middle": 中轨, "lower": 下轨}
    """
    prices = pd.Series(prices)
    
    # 中轨（SMA）
    middle = SMA(prices, period)
    
    # 标准差
    std = prices.rolling(window=period).std()
    
    # 上轨和下轨
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    return {
        "upper": upper,
        "middle": middle,
        "lower": lower
    }


def ATR(
    high: Union[pd.Series, list],
    low: Union[pd.Series, list],
    close: Union[pd.Series, list],
    period: int = 14
) -> pd.Series:
    """
    平均真实波幅
    
    Args:
        high: 最高价
        low: 最低价
        close: 收盘价
        period: 周期
    
    Returns:
        ATR序列
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    # 计算真实波幅
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 计算ATR
    atr = tr.rolling(window=period).mean()
    
    return atr


def KDJ(
    high: Union[pd.Series, list],
    low: Union[pd.Series, list],
    close: Union[pd.Series, list],
    n: int = 9,
    m1: int = 3,
    m2: int = 3
) -> Dict[str, pd.Series]:
    """
    KDJ指标
    
    Args:
        high: 最高价
        low: 最低价
        close: 收盘价
        n: RSV周期
        m1: K值平滑周期
        m2: D值平滑周期
    
    Returns:
        {"k": K值, "d": D值, "j": J值}
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    # 计算RSV
    rsv = (close - low.rolling(window=n).min()) / (high.rolling(window=n).max() - low.rolling(window=n).min()) * 100
    
    # 计算K值
    k = rsv.ewm(span=m1, adjust=False).mean()
    
    # 计算D值
    d = k.ewm(span=m2, adjust=False).mean()
    
    # 计算J值
    j = 3 * k - 2 * d
    
    return {"k": k, "d": d, "j": j}


def VolumeProfile(prices: list, volumes: list, bins: int = 20) -> Dict:
    """
    成交量分布
    
    Args:
        prices: 价格序列
        volumes: 成交量序列
        bins: 分组数量
    
    Returns:
        {"price_levels": 价格区间, "volume": 成交量分布}
    """
    # 创建价格区间
    price_min = min(prices)
    price_max = max(prices)
    step = (price_max - price_min) / bins
    
    levels = [price_min + i * step for i in range(bins + 1)]
    
    # 计算每个价格区间的成交量
    volume_profile = {}
    for i in range(bins):
        lower = levels[i]
        upper = levels[i + 1]
        volume = sum([v for p, v in zip(prices, volumes) if lower <= p < upper])
        volume_profile[f"{lower:.2f}-{upper:.2f}"] = volume
    
    return {
        "price_levels": levels,
        "volume_profile": volume_profile
    }


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有常用指标
    
    Args:
        df: DataFrame with columns: open, high, low, close, volume
    
    Returns:
        DataFrame with all indicators
    """
    result = df.copy()
    
    # SMA
    result['sma_10'] = SMA(df['close'], 10)
    result['sma_20'] = SMA(df['close'], 20)
    result['sma_50'] = SMA(df['close'], 50)
    
    # EMA
    result['ema_10'] = EMA(df['close'], 10)
    result['ema_20'] = EMA(df['close'], 20)
    result['ema_50'] = EMA(df['close'], 50)
    
    # RSI
    result['rsi_14'] = RSI(df['close'], 14)
    
    # MACD
    macd = MACD(df['close'])
    result['macd'] = macd['macd']
    result['macd_signal'] = macd['signal']
    result['macd_hist'] = macd['histogram']
    
    # Bollinger Bands
    bb = BollingerBands(df['close'])
    result['bb_upper'] = bb['upper']
    result['bb_middle'] = bb['middle']
    result['bb_lower'] = bb['lower']
    
    # ATR
    result['atr_14'] = ATR(df['high'], df['low'], df['close'], 14)
    
    # KDJ
    kdj = KDJ(df['high'], df['low'], df['close'])
    result['kdj_k'] = kdj['k']
    result['kdj_d'] = kdj['d']
    result['kdj_j'] = kdj['j']
    
    return result