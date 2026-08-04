"""
MACD 交叉策略（趋势跟随）
使用 src/utils/indicators.py 统一指标。
"""

from typing import Dict, Optional
from .base import BaseStrategy, Signal, SignalType, PositionSide
from src.utils.indicators import MACD


class MACDCrossStrategy(BaseStrategy):
    """MACD 金叉/死叉策略（趋势跟随，适用于 trending 市场）"""

    applicable_regime = "trending"

    def __init__(
        self,
        instId: str,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        position_pct: float = 0.2,
        risk_pct: float = 0.02,
        atr_multiplier: float = 2.5,
        atr_sl_multiplier: float = 2.5,
        atr_tp_multiplier: float = 5.0,
        params: Optional[Dict] = None
    ):
        all_params = {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period,
            "position_pct": position_pct,
            "risk_pct": risk_pct,
            "atr_multiplier": atr_multiplier,
            "atr_sl_multiplier": atr_sl_multiplier,
            "atr_tp_multiplier": atr_tp_multiplier,
        }
        if params:
            all_params.update(params)

        super().__init__(name="MACDCross", instId=instId, params=all_params)
        self.macd_history: list = []

    def generate_signal(self, data: Dict) -> Signal:
        price = data.get("price", 0)
        timestamp = data.get("timestamp", "")

        fast = self.params["fast_period"]
        slow = self.params["slow_period"]
        signal_period = self.params["signal_period"]

        if len(self.price_history) < slow + signal_period + 2:
            return Signal(SignalType.HOLD, self.instId, price, 0, timestamp, "数据不足")

        prices = self.price_history

        # 使用 src/utils/indicators.py 统一计算
        macd_result = MACD(prices, fast, slow, signal_period)
        macd_line = float(macd_result["macd"].iloc[-1])
        signal_line = float(macd_result["signal"].iloc[-1])

        self.macd_history.append(macd_line)

        if len(self.macd_history) < 2:
            return Signal(SignalType.HOLD, self.instId, price, 0, timestamp,
                          f"MACD={macd_line:.2f}, Signal={signal_line:.2f}")

        prev_macd = float(macd_result["macd"].iloc[-2])
        prev_signal = float(macd_result["signal"].iloc[-2])

        # MACD 金叉
        golden = prev_macd <= prev_signal and macd_line > signal_line
        # MACD 死叉
        death = prev_macd >= prev_signal and macd_line < signal_line

        signal = Signal(SignalType.HOLD, self.instId, price, 0, timestamp,
                        f"MACD={macd_line:.2f}, Signal={signal_line:.2f}")

        if golden:
            # 金叉：开多（若已持空，引擎自动翻转平空开多）
            signal.signal_type = SignalType.OPEN_LONG
            signal.amount = self.calculate_position_size(self.account_balance, price)
            signal.reason = f"MACD金叉: {macd_line:.2f} > {signal_line:.2f}"

        elif death:
            if self.allow_short:
                # 死叉：开空（若已持多，引擎自动翻转平多开空）
                signal.signal_type = SignalType.OPEN_SHORT
                signal.amount = self.calculate_position_size(self.account_balance, price)
                signal.reason = f"MACD死叉做空: {macd_line:.2f} < {signal_line:.2f}"
            elif self.position and self.position.side == PositionSide.LONG:
                # 不允许做空时，死叉仅平多
                signal.signal_type = SignalType.CLOSE_LONG
                signal.amount = self.position.amount
                signal.reason = f"MACD死叉平多: {macd_line:.2f} < {signal_line:.2f}"

        if self.position:
            self.position.update_price(price)

        return signal

    def calculate_position_size(self, account_balance: float, price: float) -> float:
        position_pct = self.params["position_pct"]
        return (account_balance * position_pct) / price
