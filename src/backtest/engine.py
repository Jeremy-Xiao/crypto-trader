"""
回测引擎
模拟策略在历史数据上的表现
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

from src.strategies.base import BaseStrategy, Signal, SignalType, PositionSide


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_balance: float = 10000  # 初始资金
    fee_rate: float = 0.001  # 手续费率（0.1%）
    slippage: float = 0.0005  # 滑点（0.05%）
    leverage: float = 1.0  # 杠杆倍数


@dataclass
class TradeRecord:
    """交易记录"""
    timestamp: str
    instId: str
    action: str  # "buy" / "sell"
    price: float
    amount: float
    fee: float
    pnl: float
    reason: str


class BacktestEngine:
    """回测引擎"""
    
    def __init__(
        self,
        strategy: BaseStrategy,
        config: Optional[BacktestConfig] = None
    ):
        """
        初始化回测引擎
        
        Args:
            strategy: 策略实例
            config: 回测配置
        """
        self.strategy = strategy
        self.config = config or BacktestConfig()
        
        # 状态
        self.balance = self.config.initial_balance
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Dict] = []
        self.position_amount = 0
        self.position_price = 0
        
    def load_data(self, data: pd.DataFrame) -> None:
        """
        加载历史数据
        
        Args:
            data: DataFrame with columns: timestamp, open, high, low, close, volume
        """
        self.data = data
    
    def run(self) -> Dict:
        """
        运行回测
        
        Returns:
            回测结果
        """
        if self.data is None or len(self.data) == 0:
            return {"error": "无数据"}
        
        print(f"开始回测: {self.strategy.name}")
        print(f"数据范围: {len(self.data)} 条")
        
        # 遍历每条数据
        for idx, row in self.data.iterrows():
            timestamp = row.get("timestamp", row.get("ts", ""))
            price = row["close"]  # 使用收盘价
            
            # 生成信号
            signal = self.strategy.generate_signal({
                "price": price,
                "timestamp": timestamp
            })
            
            # 执行交易
            if signal.signal_type == SignalType.BUY:
                self._execute_buy(price, signal.amount, timestamp, signal.reason)
            
            elif signal.signal_type == SignalType.SELL:
                self._execute_sell(price, timestamp, signal.reason)
            
            # 记录权益曲线
            equity = self._calculate_equity(price)
            self.equity_curve.append({
                "timestamp": timestamp,
                "equity": equity,
                "balance": self.balance,
                "position_value": self.position_amount * price,
                "price": price
            })
        
        # 计算绩效
        results = self._calculate_performance()
        
        print(f"\n回测完成!")
        print(f"总收益: {results['total_return']:.2f}%")
        print(f"最大回撤: {results['max_drawdown']:.2f}%")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"胜率: {results['win_rate']:.2f}%")
        
        return results
    
    def _execute_buy(self, price: float, amount: float, timestamp: str, reason: str):
        """执行买入"""
        if self.position_amount > 0:
            return  # 已有持仓
        
        # 计算实际买入金额
        buy_value = min(amount * price, self.balance)
        actual_amount = buy_value / price
        
        # 计算手续费
        fee = buy_value * self.config.fee_rate
        
        # 计算滑点
        actual_price = price * (1 + self.config.slippage)
        
        # 更新状态
        self.balance -= (buy_value + fee)
        self.position_amount = actual_amount
        self.position_price = actual_price
        
        # 开仓
        self.strategy.open_position(actual_price, actual_amount, timestamp)
        
        # 记录交易
        self.trades.append(TradeRecord(
            timestamp=timestamp,
            instId=self.strategy.instId,
            action="buy",
            price=actual_price,
            amount=actual_amount,
            fee=fee,
            pnl=0,
            reason=reason
        ))
    
    def _execute_sell(self, price: float, timestamp: str, reason: str):
        """执行卖出"""
        if self.position_amount <= 0:
            return  # 无持仓
        
        # 计算滑点
        actual_price = price * (1 - self.config.slippage)
        
        # 计算卖出金额
        sell_value = self.position_amount * actual_price
        
        # 计算手续费
        fee = sell_value * self.config.fee_rate
        
        # 计算盈亏
        pnl = (actual_price - self.position_price) * self.position_amount
        
        # 更新状态
        self.balance += (sell_value - fee)
        
        # 平仓
        self.strategy.close_position(actual_price, timestamp, reason)
        
        # 记录交易
        self.trades.append(TradeRecord(
            timestamp=timestamp,
            instId=self.strategy.instId,
            action="sell",
            price=actual_price,
            amount=self.position_amount,
            fee=fee,
            pnl=pnl,
            reason=reason
        ))
        
        # 清空持仓
        self.position_amount = 0
        self.position_price = 0
    
    def _calculate_equity(self, current_price: float) -> float:
        """计算总权益"""
        position_value = self.position_amount * current_price
        return self.balance + position_value
    
    def _calculate_performance(self) -> Dict:
        """计算绩效指标"""
        if not self.equity_curve:
            return {}
        
        equity_series = pd.Series([e["equity"] for e in self.equity_curve])
        
        # 总收益率
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0] * 100
        
        # 最大回撤
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # 收益率序列
        returns = equity_series.pct_change().dropna()
        
        # 夏普比率（假设无风险利率为0）
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # 交易统计
        sell_trades = [t for t in self.trades if t.action == "sell"]
        winning_trades = [t for t in sell_trades if t.pnl > 0]
        
        total_trades = len(sell_trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # 盈亏比
        if winning_trades and len([t for t in sell_trades if t.pnl < 0]) > 0:
            avg_win = np.mean([t.pnl for t in winning_trades])
            avg_loss = np.mean([abs(t.pnl) for t in sell_trades if t.pnl < 0])
            profit_ratio = avg_win / avg_loss
        else:
            profit_ratio = 0
        
        # 总盈亏
        total_pnl = sum([t.pnl for t in self.trades])
        
        return {
            "initial_balance": self.config.initial_balance,
            "final_equity": equity_series.iloc[-1],
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "win_rate": win_rate,
            "profit_ratio": profit_ratio,
            "total_pnl": total_pnl,
            "total_fee": sum([t.fee for t in self.trades])
        }
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        """获取权益曲线DataFrame"""
        return pd.DataFrame(self.equity_curve)
    
    def get_trades_df(self) -> pd.DataFrame:
        """获取交易记录DataFrame"""
        return pd.DataFrame([{
            "timestamp": t.timestamp,
            "instId": t.instId,
            "action": t.action,
            "price": t.price,
            "amount": t.amount,
            "fee": t.fee,
            "pnl": t.pnl,
            "reason": t.reason
        } for t in self.trades])


# 使用示例
if __name__ == "__main__":
    from src.strategies.double_ma import DoubleMAStrategy
    
    # 创建策略
    strategy = DoubleMAStrategy(
        instId="BTC-USDT",
        fast_period=10,
        slow_period=30,
        position_pct=0.1,
        stop_loss_pct=0.05,
        take_profit_pct=0.10
    )
    
    # 创建回测引擎
    engine = BacktestEngine(
        strategy=strategy,
        config=BacktestConfig(
            initial_balance=10000,
            fee_rate=0.001,
            slippage=0.0005
        )
    )
    
    print(strategy.describe())