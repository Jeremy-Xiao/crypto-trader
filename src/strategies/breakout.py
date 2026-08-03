"""
突破策略（趋势跟随）
价格突破N日高点买入，使用 src/utils/indicators.py 统一指标。
"""

from typing import Dict, Optional
from .base import BaseStrategy, Signal, SignalType


class BreakoutStrategy(BaseStrategy):
    """突破策略（趋势跟随，适用于 trending 市场）"""

    applicable_regime = "trending"

    def __init__(
        self,
        instId: str,
        lookback: int = 20,
        position_pct: float = 0.2,
        risk_pct: float = 0.02,
        atr_multiplier: float = 2.0,
        atr_sl_multiplier: float = 2.0,
        atr_tp_multiplier: float = 4.0,
        params: Optional[Dict] = None
    ):
        all_params = {
            "lookback": lookback,
            "position_pct": position_pct,
            "risk_pct": risk_pct,
            "atr_multiplier": atr_multiplier,
            "atr_sl_multiplier": atr_sl_multiplier,
            "atr_tp_multiplier": atr_tp_multiplier,
        }
        if params:
            all_params.update(params)

        super().__init__(name="Breakout", instId=instId, params=all_params)

    def generate_signal(self, data: Dict) -> Signal:
        price = data.get("price", 0)
        timestamp = data.get("timestamp", "")

        lookback = self.params["lookback"]

        if len(self.price_history) < lookback + 2:
            return Signal(SignalType.HOLD, self.instId, price, 0, timestamp, "数据不足")

        prices = self.price_history

        # 过去 lookback 根K线的最高价（不含当前K线）
        recent_high = max(prices[-lookback-1:-1])

        signal = Signal(SignalType.HOLD, self.instId, price, 0, timestamp,
                        f"突破位={recent_high:.0f}, 当前={price:.0f}")

        # 突破买入
        if price > recent_high and not self.position:
            signal.signal_type = SignalType.BUY
            signal.amount = self.calculate_position_size(self.account_balance, price)
            signal.reason = f"突破{lookback}日高点: {price:.0f} > {recent_high:.0f}"

        # 突破策略无额外卖出条件，由引擎 ATR 止损止盈管理
        if self.position:
            self.position.update_price(price)

        return signal

    def calculate_position_size(self, account_balance: float, price: float) -> float:
        position_pct = self.params["position_pct"]
        return (account_balance * position_pct) / price
