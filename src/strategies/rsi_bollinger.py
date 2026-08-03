"""
RSI + 布林带策略（均值回归，适用于震荡市）
使用 src/utils/indicators.py 统一指标。
"""

from typing import Dict, Optional
from .base import BaseStrategy, Signal, SignalType
from src.utils.indicators import RSI, BollingerBands


class RSIBollingerStrategy(BaseStrategy):
    """RSI + 布林带策略（均值回归，适用于 ranging 市场）"""

    applicable_regime = "ranging"

    def __init__(
        self,
        instId: str,
        rsi_period: int = 14,
        boll_period: int = 20,
        boll_std: float = 2.0,
        rsi_low: float = 30,
        rsi_high: float = 70,
        position_pct: float = 0.2,
        risk_pct: float = 0.015,
        atr_multiplier: float = 2.0,
        atr_sl_multiplier: float = 2.0,
        atr_tp_multiplier: float = 4.0,
        params: Optional[Dict] = None
    ):
        all_params = {
            "rsi_period": rsi_period,
            "boll_period": boll_period,
            "boll_std": boll_std,
            "rsi_low": rsi_low,
            "rsi_high": rsi_high,
            "position_pct": position_pct,
            "risk_pct": risk_pct,
            "atr_multiplier": atr_multiplier,
            "atr_sl_multiplier": atr_sl_multiplier,
            "atr_tp_multiplier": atr_tp_multiplier,
        }
        if params:
            all_params.update(params)

        super().__init__(name="RSIBollinger", instId=instId, params=all_params)

    def generate_signal(self, data: Dict) -> Signal:
        price = data.get("price", 0)
        timestamp = data.get("timestamp", "")

        rsi_period = self.params["rsi_period"]
        boll_period = self.params["boll_period"]

        if len(self.price_history) < max(rsi_period, boll_period) + 2:
            return Signal(SignalType.HOLD, self.instId, price, 0, timestamp, "数据不足")

        prices = self.price_history

        # 使用 src/utils/indicators.py 统一计算
        rsi = float(RSI(prices, rsi_period).iloc[-1])
        bb = BollingerBands(prices, boll_period, self.params["boll_std"])
        upper = float(bb["upper"].iloc[-1])
        middle = float(bb["middle"].iloc[-1])
        lower = float(bb["lower"].iloc[-1])

        if any(pd_val != pd_val for pd_val in [rsi, upper, middle, lower]):  # NaN check
            return Signal(SignalType.HOLD, self.instId, price, 0, timestamp, "指标含NaN")

        signal = Signal(SignalType.HOLD, self.instId, price, 0, timestamp,
                        f"RSI={rsi:.1f}, BB下={lower:.0f}, BB上={upper:.0f}")

        # 买入：RSI 超卖 + 价格触及布林带下轨
        if rsi < self.params["rsi_low"] and price <= lower and not self.position:
            signal.signal_type = SignalType.BUY
            signal.amount = self.calculate_position_size(self.account_balance, price)
            signal.reason = f"超卖: RSI={rsi:.1f}<{self.params['rsi_low']}, 价格触下轨"

        # 卖出：RSI 超买 + 价格触及布林带上轨
        elif self.position:
            if rsi > self.params["rsi_high"] and price >= upper:
                signal.signal_type = SignalType.SELL
                signal.amount = self.position.amount
                signal.reason = f"超买: RSI={rsi:.1f}>{self.params['rsi_high']}, 价格触上轨"
            elif price >= middle:
                signal.signal_type = SignalType.SELL
                signal.amount = self.position.amount
                signal.reason = "回归中轨"

        if self.position:
            self.position.update_price(price)

        return signal

    def calculate_position_size(self, account_balance: float, price: float) -> float:
        position_pct = self.params["position_pct"]
        return (account_balance * position_pct) / price
