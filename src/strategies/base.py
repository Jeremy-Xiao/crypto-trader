"""
策略基类
定义策略的基本结构和接口，支持 ATR 动态风控、移动止损、市场状态自适应。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from src.utils.indicators import ATR, ADX


class SignalType(Enum):
    """信号类型（支持双向交易：做多 / 做空）"""
    HOLD = "hold"
    OPEN_LONG = "open_long"       # 买入开多
    OPEN_SHORT = "open_short"     # 卖出开空（做空）
    CLOSE_LONG = "close_long"     # 卖出平多
    CLOSE_SHORT = "close_short"   # 买入平空
    # 兼容别名（旧脚本仍可用 BUY / SELL）
    BUY = OPEN_LONG
    SELL = CLOSE_LONG


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"
    NONE = "none"


class MarketRegime(Enum):
    """市场状态"""
    TRENDING = "trending"
    RANGING = "ranging"
    UNKNOWN = "unknown"


@dataclass
class Signal:
    """交易信号"""
    signal_type: SignalType
    instId: str
    price: float
    amount: float
    timestamp: str
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "signal": self.signal_type.value,
            "instId": self.instId,
            "price": self.price,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "reason": self.reason
        }


@dataclass
class Position:
    """持仓信息（支持移动止损）"""
    instId: str
    side: PositionSide
    amount: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    # 移动止损相关
    highest_price: float = 0.0
    lowest_price: float = 0.0
    trailing_active: bool = False
    trailing_stop: float = 0.0

    def update_price(self, current_price: float):
        """更新当前价格和盈亏"""
        self.current_price = current_price
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.amount
            self.unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.entry_price - current_price) * self.amount
            self.unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price * 100

    def to_dict(self) -> Dict:
        return {
            "instId": self.instId,
            "side": self.side.value,
            "amount": self.amount,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price,
            "trailing_active": self.trailing_active,
            "trailing_stop": self.trailing_stop
        }


def detect_market_regime(
    highs: list,
    lows: list,
    closes: list,
    adx_period: int = 14,
    lookback: int = 50,
    adx_val: Optional[float] = None
) -> tuple:
    """
    检测市场状态：trending（趋势）/ ranging（震荡）

    综合两个指标：
    1. ADX > 25 → 趋势市；ADX < 20 → 震荡市；20-25 → 中性
    2. 波动率分位数：近期波动率在历史中的位置，辅助判断

    Args:
        adx_val: 外部已算好的 ADX 值。回测引擎会预计算整条 ADX 序列并传入，
                 避免在每根K线上重复全量计算（结果完全一致，仅为提速）。

    Returns:
        (MarketRegime, {'adx': float, 'vol_pct': float})
    """
    n = len(closes)
    if n < adx_period + 5 or not highs or not lows:
        return MarketRegime.UNKNOWN, {'adx': 0, 'vol_pct': 0.5}

    # ADX
    if adx_val is None:
        adx_series = ADX(highs, lows, closes, adx_period)
        adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0
    else:
        adx_val = float(adx_val) if not pd.isna(adx_val) else 0

    # 波动率分位数
    # 原实现用列表推导逐窗口调用 .std()（且每个窗口算两遍），复杂度 O(n²) 且常数极大；
    # 这里改为向量化 rolling(20).std()，数值完全等价（同为 ddof=1，同样的窗口切分）。
    returns = pd.Series(closes).pct_change().dropna()
    m = len(returns)
    if m >= 20:
        roll = returns.rolling(20).std()
        recent_vol = roll.iloc[-1]
    else:
        roll = None
        recent_vol = returns.std()

    if m >= lookback and roll is not None and not pd.isna(recent_vol):
        # 对应原来的 i ∈ [0, m-21]，窗口 [i, i+20) 的右端点下标为 i+19 ∈ [19, m-2]
        hist_vols = roll.iloc[19:m - 1].dropna().to_numpy()
        if hist_vols.size > 0:
            vol_pct = float((hist_vols <= recent_vol).sum()) / hist_vols.size
        else:
            vol_pct = 0.5
    else:
        vol_pct = 0.5

    # 综合判断
    if adx_val >= 25:
        regime = MarketRegime.TRENDING
    elif adx_val <= 20:
        regime = MarketRegime.RANGING
    else:
        regime = MarketRegime.TRENDING if vol_pct > 0.6 else MarketRegime.RANGING

    return regime, {'adx': round(adx_val, 2), 'vol_pct': round(vol_pct, 3)}


class BaseStrategy(ABC):
    """
    策略基类（支持 ATR 动态风控 + 移动止损 + 市场状态自适应）

    子类需要实现：
    - generate_signal(): 根据市场数据生成买卖信号
    - calculate_position_size(): 计算仓位大小

    引擎层负责：
    - ATR 动态止损/止盈
    - 移动止损
    - 多时间框架确认
    """

    # 策略适用的市场状态，子类可覆盖
    applicable_regime: str = "both"  # "trending" / "ranging" / "both"

    def __init__(
        self,
        name: str,
        instId: str,
        params: Optional[Dict] = None
    ):
        self.name = name
        self.instId = instId
        self.params = params or {}

        # 状态
        self.position: Optional[Position] = None
        self.signals: List[Signal] = []
        self.trades: List[Dict] = []
        self.account_balance: float = 10000  # 由回测引擎更新

        # 是否允许做空（双向交易）。默认开启；回测对比时设为 False 即模拟仅做多
        self.allow_short = (params or {}).get("allow_short", True)

        # 历史数据缓存
        self.price_history: list = []
        self.high_history: list = []
        self.low_history: list = []

    @abstractmethod
    def generate_signal(self, data: Dict) -> Signal:
        """
        生成交易信号

        Args:
            data: 市场数据 {"price": float, "timestamp": str, "high": float, "low": float}

        Returns:
            交易信号
        """
        pass

    @abstractmethod
    def calculate_position_size(self, account_balance: float, price: float) -> float:
        """计算仓位大小"""
        pass

    # ====== ATR 动态风控（由引擎调用） ======

    def get_atr(self, period: int = 14) -> float:
        """获取当前 ATR 值"""
        if len(self.high_history) < period + 1 or len(self.low_history) < period + 1:
            return 0.0
        atr_series = ATR(
            pd.Series(self.high_history),
            pd.Series(self.low_history),
            pd.Series(self.price_history),
            period
        )
        val = atr_series.iloc[-1]
        return float(val) if not pd.isna(val) else 0.0

    def calculate_atr_position_size(
        self,
        account_balance: float,
        price: float,
        atr_val: float,
        risk_pct: float = 0.02,
        atr_multiplier: float = 2.0,
        max_pct: float = 0.5
    ) -> float:
        """
        ATR 动态仓位计算

        仓位 = (balance × risk_pct) / (ATR × atr_multiplier)
        波动大 → ATR大 → 仓位小；波动小 → 仓位大

        Args:
            account_balance: 账户余额
            price: 当前价格
            atr_val: ATR 值
            risk_pct: 每笔交易风险占比（如 0.02 = 2%）
            atr_multiplier: ATR 倍数作为止损距离
            max_pct: 单笔仓位上限（占账户比例）

        Returns:
            仓位大小（币种数量）
        """
        if atr_val <= 0 or price <= 0:
            pct = self.params.get("position_pct", 0.2)
            return (account_balance * pct) / price

        risk_amount = account_balance * risk_pct
        stop_distance = atr_val * atr_multiplier
        amount = risk_amount / stop_distance

        # 单笔仓位上限
        max_amount = account_balance * max_pct / price
        amount = min(amount, max_amount)

        return amount

    def check_atr_stop_loss(self, atr_val: float, atr_sl_multiplier: float = 2.0) -> bool:
        """ATR 动态止损检查（双向：多头看下破，空头看上破）"""
        if not self.position or atr_val <= 0:
            return False

        if self.position.side == PositionSide.LONG:
            stop_price = self.position.entry_price - atr_val * atr_sl_multiplier
            return self.position.current_price <= stop_price
        else:  # SHORT
            stop_price = self.position.entry_price + atr_val * atr_sl_multiplier
            return self.position.current_price >= stop_price

    def check_atr_take_profit(self, atr_val: float, atr_tp_multiplier: float = 4.0) -> bool:
        """ATR 动态止盈检查（双向：多头看上破，空头看下破）"""
        if not self.position or atr_val <= 0:
            return False

        if self.position.side == PositionSide.LONG:
            tp_price = self.position.entry_price + atr_val * atr_tp_multiplier
            return self.position.current_price >= tp_price
        else:  # SHORT
            tp_price = self.position.entry_price - atr_val * atr_tp_multiplier
            return self.position.current_price <= tp_price

    def update_trailing_stop(
        self,
        current_price: float,
        atr_val: float,
        atr_sl_multiplier: float = 2.0,
        trailing_pct: float = 0.02
    ) -> Optional[str]:
        """
        更新移动止损（双向支持），返回触发原因或 None

        多头：盈利达到 trailing_pct 后激活，止损线 = 最高价 - ATR × mult，只上移
        空头：盈利达到 trailing_pct 后激活，止损线 = 最低价 + ATR × mult，只下移

        Returns:
            'trailing_stop' 如果触发移动止损，否则 None
        """
        if not self.position or atr_val <= 0:
            return None

        pos = self.position

        if pos.side == PositionSide.LONG:
            pos.highest_price = max(pos.highest_price, current_price)
            if not pos.trailing_active:
                if current_price >= pos.entry_price * (1 + trailing_pct):
                    pos.trailing_active = True
                    pos.trailing_stop = pos.highest_price - atr_val * atr_sl_multiplier
            if pos.trailing_active:
                new_trail = pos.highest_price - atr_val * atr_sl_multiplier
                pos.trailing_stop = max(pos.trailing_stop, new_trail)
                if current_price <= pos.trailing_stop:
                    return 'trailing_stop'

        else:  # SHORT
            pos.lowest_price = min(pos.lowest_price, current_price)
            if not pos.trailing_active:
                if current_price <= pos.entry_price * (1 - trailing_pct):
                    pos.trailing_active = True
                    pos.trailing_stop = pos.lowest_price + atr_val * atr_sl_multiplier
            if pos.trailing_active:
                new_trail = pos.lowest_price + atr_val * atr_sl_multiplier
                pos.trailing_stop = min(pos.trailing_stop, new_trail)  # 只下移
                if current_price >= pos.trailing_stop:
                    return 'trailing_stop'

        return None

    # ====== 基础风控（兼容旧策略） ======

    def should_stop_loss(self, current_price: float) -> bool:
        """固定百分比止损（兼容旧逻辑）"""
        if not self.position:
            return False
        stop_loss_pct = self.params.get("stop_loss_pct", 0.05)
        if self.position.side == PositionSide.LONG:
            loss_pct = (self.position.entry_price - current_price) / self.position.entry_price
            return loss_pct >= stop_loss_pct
        elif self.position.side == PositionSide.SHORT:
            loss_pct = (current_price - self.position.entry_price) / self.position.entry_price
            return loss_pct >= stop_loss_pct
        return False

    def should_take_profit(self, current_price: float) -> bool:
        """固定百分比止盈（兼容旧逻辑）"""
        if not self.position:
            return False
        take_profit_pct = self.params.get("take_profit_pct", 0.10)
        if self.position.side == PositionSide.LONG:
            profit_pct = (current_price - self.position.entry_price) / self.position.entry_price
            return profit_pct >= take_profit_pct
        elif self.position.side == PositionSide.SHORT:
            profit_pct = (self.position.entry_price - current_price) / self.position.entry_price
            return profit_pct >= take_profit_pct
        return False

    # ====== 仓位管理 ======

    def open_position(self, price: float, amount: float, timestamp: str, side: PositionSide = PositionSide.LONG):
        """开仓"""
        self.position = Position(
            instId=self.instId,
            side=side,
            amount=amount,
            entry_price=price,
            current_price=price,
            unrealized_pnl=0,
            unrealized_pnl_pct=0,
            highest_price=price,
            lowest_price=price,
            trailing_active=False,
            trailing_stop=0.0
        )
        self.trades.append({
            "action": "open",
            "side": "buy" if side == PositionSide.LONG else "sell",
            "price": price,
            "amount": amount,
            "timestamp": timestamp
        })

    def close_position(self, price: float, timestamp: str, reason: str = ""):
        """平仓"""
        if not self.position:
            return

        if self.position.side == PositionSide.LONG:
            realized_pnl = (price - self.position.entry_price) * self.position.amount
        else:
            realized_pnl = (self.position.entry_price - price) * self.position.amount

        self.trades.append({
            "action": "close",
            "side": "sell",
            "price": price,
            "amount": self.position.amount,
            "realized_pnl": realized_pnl,
            "reason": reason,
            "timestamp": timestamp
        })
        self.position = None

    def get_performance_stats(self) -> Dict:
        """获取策略绩效"""
        if not self.trades:
            return {}
        total_trades = len([t for t in self.trades if t["action"] == "close"])
        winning_trades = len([t for t in self.trades if t.get("realized_pnl", 0) > 0])
        total_pnl = sum([t.get("realized_pnl", 0) for t in self.trades])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "current_position": self.position.to_dict() if self.position else None
        }

    def describe(self) -> str:
        """描述策略"""
        desc = f"策略名称: {self.name}\n"
        desc += f"交易产品: {self.instId}\n"
        desc += f"适用市场: {self.applicable_regime}\n"
        desc += f"参数: {self.params}\n"
        return desc
