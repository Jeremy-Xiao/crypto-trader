"""
双均线交叉策略（趋势跟随）
使用 src/utils/indicators.py 统一指标，支持趋势过滤。
"""

from typing import Dict, Optional
from .base import BaseStrategy, Signal, SignalType, PositionSide
from src.utils.indicators import EMA


class DoubleMAStrategy(BaseStrategy):
    """双均线交叉策略（趋势跟随，适用于 trending 市场）"""

    applicable_regime = "trending"

    def __init__(
        self,
        instId: str,
        fast_period: int = 10,
        slow_period: int = 30,
        trend_period: int = 60,
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
            "trend_period": trend_period,
            "position_pct": position_pct,
            "risk_pct": risk_pct,
            "atr_multiplier": atr_multiplier,
            "atr_sl_multiplier": atr_sl_multiplier,
            "atr_tp_multiplier": atr_tp_multiplier,
        }
        if params:
            all_params.update(params)

        super().__init__(name="DoubleMA", instId=instId, params=all_params)

        self.fast_ma_history: list = []
        self.slow_ma_history: list = []

    def generate_signal(self, data: Dict) -> Signal:
        price = data.get("price", 0)
        timestamp = data.get("timestamp", "")

        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]
        trend_period = self.params["trend_period"]

        if len(self.price_history) < slow_period + 1:
            return Signal(SignalType.HOLD, self.instId, price, 0, timestamp, "数据不足")

        # 使用 src/utils/indicators.py 统一计算
        prices = self.price_history
        fast_ma = float(EMA(prices, fast_period).iloc[-1])
        slow_ma = float(EMA(prices, slow_period).iloc[-1])

        self.fast_ma_history.append(fast_ma)
        self.slow_ma_history.append(slow_ma)

        if len(self.fast_ma_history) < 2:
            return Signal(SignalType.HOLD, self.instId, price, 0, timestamp,
                          f"EMA{fast_period}={fast_ma:.0f}, EMA{slow_period}={slow_ma:.0f}")

        prev_fast = self.fast_ma_history[-2]
        prev_slow = self.slow_ma_history[-2]

        # 金叉
        golden_cross = prev_fast <= prev_slow and fast_ma > slow_ma
        # 死叉
        death_cross = prev_fast >= prev_slow and fast_ma < slow_ma

        # 趋势过滤：价格需在趋势线上方才买入
        in_uptrend = True
        if len(prices) >= trend_period:
            trend_ema = float(EMA(prices, trend_period).iloc[-1])
            in_uptrend = price > trend_ema

        signal = Signal(SignalType.HOLD, self.instId, price, 0, timestamp,
                        f"EMA{fast_period}={fast_ma:.0f}, EMA{slow_period}={slow_ma:.0f}")

        # 金叉 + 趋势向上 → 开多（若已持空，引擎自动翻转平空开多）
        if golden_cross and in_uptrend:
            signal.signal_type = SignalType.OPEN_LONG
            signal.amount = self.calculate_position_size(self.account_balance, price)
            signal.reason = f"金叉: EMA{fast_period}({fast_ma:.0f}) > EMA{slow_period}({slow_ma:.0f})"

        # 死叉 → 开空（允许做空时）；不允许做空且持多时仅平多
        elif death_cross:
            if self.allow_short:
                signal.signal_type = SignalType.OPEN_SHORT
                signal.amount = self.calculate_position_size(self.account_balance, price)
                signal.reason = f"死叉做空: EMA{fast_period}({fast_ma:.0f}) < EMA{slow_period}({slow_ma:.0f})"
            elif self.position and self.position.side == PositionSide.LONG:
                signal.signal_type = SignalType.CLOSE_LONG
                signal.amount = self.position.amount
                signal.reason = f"死叉平多: EMA{fast_period}({fast_ma:.0f}) < EMA{slow_period}({slow_ma:.0f})"

        # 持多且趋势破位 → 平多（持空时趋势破位是利好，不动）
        elif self.position and self.position.side == PositionSide.LONG and not in_uptrend and len(prices) >= trend_period:
            signal.signal_type = SignalType.CLOSE_LONG
            signal.amount = self.position.amount
            signal.reason = "趋势破位: 价格跌破趋势线"

        # 更新持仓价格
        if self.position:
            self.position.update_price(price)

        return signal

    def calculate_position_size(self, account_balance: float, price: float) -> float:
        position_pct = self.params["position_pct"]
        return (account_balance * position_pct) / price

    def describe(self) -> str:
        desc = super().describe()
        desc += "\n策略逻辑:\n"
        desc += f"  1. 金叉买入: EMA{self.params['fast_period']} 上穿 EMA{self.params['slow_period']}\n"
        desc += f"  2. 趋势过滤: 价格需在 EMA{self.params['trend_period']} 上方\n"
        desc += f"  3. 死叉卖出 / 趋势破位卖出\n"
        desc += f"  4. ATR动态止损: {self.params['atr_sl_multiplier']}x ATR\n"
        desc += f"  5. ATR动态止盈: {self.params['atr_tp_multiplier']}x ATR (盈亏比2:1)\n"
        desc += f"  6. 移动止损: 盈利{self.params.get('risk_pct', 0.02)*100:.0f}%后激活\n"
        return desc
