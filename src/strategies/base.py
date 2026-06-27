"""
策略基类
定义策略的基本结构和接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"
    NONE = "none"


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
    """持仓信息"""
    instId: str
    side: PositionSide
    amount: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
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
            "unrealized_pnl_pct": self.unrealized_pnl_pct
        }


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(
        self,
        name: str,
        instId: str,
        params: Optional[Dict] = None
    ):
        """
        初始化策略
        
        Args:
            name: 策略名称
            instId: 交易产品ID
            params: 策略参数
        """
        self.name = name
        self.instId = instId
        self.params = params or {}
        
        # 状态
        self.position: Optional[Position] = None
        self.signals: List[Signal] = []
        self.trades: List[Dict] = []
        
    @abstractmethod
    def generate_signal(self, data: Dict) -> Signal:
        """
        生成交易信号
        
        Args:
            data: 市场数据
        
        Returns:
            交易信号
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, account_balance: float, price: float) -> float:
        """
        计算仓位大小
        
        Args:
            account_balance: 账户余额
            price: 当前价格
        
        Returns:
            仓位大小
        """
        pass
    
    def should_stop_loss(self, current_price: float) -> bool:
        """
        是否止损
        
        Args:
            current_price: 当前价格
        
        Returns:
            是否止损
        """
        if not self.position:
            return False
        
        stop_loss_pct = self.params.get("stop_loss_pct", 0.05)  # 默认5%止损
        
        if self.position.side == PositionSide.LONG:
            loss_pct = (self.position.entry_price - current_price) / self.position.entry_price
            return loss_pct >= stop_loss_pct
        
        return False
    
    def should_take_profit(self, current_price: float) -> bool:
        """
        是否止盈
        
        Args:
            current_price: 当前价格
        
        Returns:
            是否止盈
        """
        if not self.position:
            return False
        
        take_profit_pct = self.params.get("take_profit_pct", 0.10)  # 默认10%止盈
        
        if self.position.side == PositionSide.LONG:
            profit_pct = (current_price - self.position.entry_price) / self.position.entry_price
            return profit_pct >= take_profit_pct
        
        return False
    
    def open_position(self, price: float, amount: float, timestamp: str):
        """开仓"""
        self.position = Position(
            instId=self.instId,
            side=PositionSide.LONG,
            amount=amount,
            entry_price=price,
            current_price=price,
            unrealized_pnl=0,
            unrealized_pnl_pct=0
        )
        
        self.trades.append({
            "action": "open",
            "side": "buy",
            "price": price,
            "amount": amount,
            "timestamp": timestamp
        })
    
    def close_position(self, price: float, timestamp: str, reason: str = ""):
        """平仓"""
        if not self.position:
            return
        
        realized_pnl = (price - self.position.entry_price) * self.position.amount
        
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
        desc += f"参数: {self.params}\n"
        return desc