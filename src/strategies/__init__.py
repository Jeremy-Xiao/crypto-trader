"""
策略模块
"""

from .base import BaseStrategy, Signal, SignalType, Position, PositionSide
from .double_ma import DoubleMAStrategy

__all__ = [
    'BaseStrategy',
    'Signal',
    'SignalType',
    'Position',
    'PositionSide',
    'DoubleMAStrategy'
]