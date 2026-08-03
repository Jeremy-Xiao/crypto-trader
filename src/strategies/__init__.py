"""
策略模块
"""

from .base import BaseStrategy, Signal, SignalType, Position, PositionSide, MarketRegime, detect_market_regime
from .double_ma import DoubleMAStrategy
from .rsi_bollinger import RSIBollingerStrategy
from .macd_cross import MACDCrossStrategy
from .breakout import BreakoutStrategy

__all__ = [
    'BaseStrategy',
    'Signal',
    'SignalType',
    'Position',
    'PositionSide',
    'MarketRegime',
    'detect_market_regime',
    'DoubleMAStrategy',
    'RSIBollingerStrategy',
    'MACDCrossStrategy',
    'BreakoutStrategy',
]
