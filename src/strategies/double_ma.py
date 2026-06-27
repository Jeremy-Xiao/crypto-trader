"""
双均线交叉策略
经典的趋势跟随策略
"""

from typing import Dict, Optional
from .base import BaseStrategy, Signal, SignalType


class DoubleMAStrategy(BaseStrategy):
    """双均线交叉策略"""
    
    def __init__(
        self,
        instId: str,
        fast_period: int = 10,
        slow_period: int = 30,
        position_pct: float = 0.02,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        params: Optional[Dict] = None
    ):
        """
        初始化双均线策略
        
        Args:
            instId: 交易产品ID
            fast_period: 快线周期（短期均线）
            slow_period: 慢线周期（长期均线）
            position_pct: 仓位比例（账户权益的百分比）
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
        """
        # 合并参数
        all_params = {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "position_pct": position_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct
        }
        if params:
            all_params.update(params)
        
        super().__init__(
            name="DoubleMA",
            instId=instId,
            params=all_params
        )
        
        # 历史数据缓存
        self.price_history: list = []
        self.fast_ma_history: list = []
        self.slow_ma_history: list = []
        
        # 上一次信号
        self.last_cross: str = "none"  # "golden" / "death" / "none"
    
    def calculate_ema(self, prices: list, period: int) -> float:
        """
        计算EMA
        
        Args:
            prices: 价格列表
            period: 周期
        
        Returns:
            EMA值
        """
        if len(prices) < period:
            return 0
        
        # 使用指数平滑
        k = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = price * k + ema * (1 - k)
        
        return ema
    
    def detect_cross(self, fast_ma: float, slow_ma: float) -> str:
        """
        检测交叉
        
        Args:
            fast_ma: 快线值
            slow_ma: 慢线值
        
        Returns:
            交叉类型: "golden" / "death" / "none"
        """
        if len(self.fast_ma_history) < 2:
            return "none"
        
        prev_fast = self.fast_ma_history[-1]
        prev_slow = self.slow_ma_history[-1]
        
        # 金叉：快线从下方穿越慢线
        if prev_fast <= prev_slow and fast_ma > slow_ma:
            return "golden"
        
        # 死叉：快线从上方穿越慢线
        if prev_fast >= prev_slow and fast_ma < slow_ma:
            return "death"
        
        return "none"
    
    def generate_signal(self, data: Dict) -> Signal:
        """
        生成交易信号
        
        Args:
            data: 市场数据 {"price": float, "timestamp": str}
        
        Returns:
            交易信号
        """
        price = data.get("price", 0)
        timestamp = data.get("timestamp", "")
        
        # 记录价格
        self.price_history.append(price)
        
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]
        
        # 计算均线
        fast_ma = self.calculate_ema(self.price_history, fast_period)
        slow_ma = self.calculate_ema(self.price_history, slow_period)
        
        # 记录均线历史
        self.fast_ma_history.append(fast_ma)
        self.slow_ma_history.append(slow_ma)
        
        # 检测交叉
        cross = self.detect_cross(fast_ma, slow_ma)
        
        # 默认持有信号
        signal = Signal(
            signal_type=SignalType.HOLD,
            instId=self.instId,
            price=price,
            amount=0,
            timestamp=timestamp,
            reason=f"fast_ma={fast_ma:.2f}, slow_ma={slow_ma:.2f}"
        )
        
        # 金叉买入
        if cross == "golden" and not self.position:
            signal.signal_type = SignalType.BUY
            signal.reason = f"金叉买入: fast_ma({fast_ma:.2f}) > slow_ma({slow_ma:.2f})"
        
        # 死叉卖出
        elif cross == "death" and self.position:
            signal.signal_type = SignalType.SELL
            signal.reason = f"死叉卖出: fast_ma({fast_ma:.2f}) < slow_ma({slow_ma:.2f})"
        
        # 止损
        elif self.should_stop_loss(price):
            signal.signal_type = SignalType.SELL
            signal.reason = f"止损: 亏损{self.params['stop_loss_pct']*100}%"
        
        # 止盈
        elif self.should_take_profit(price):
            signal.signal_type = SignalType.SELL
            signal.reason = f"止盈: 盈利{self.params['take_profit_pct']*100}%"
        
        # 更新持仓价格
        if self.position:
            self.position.update_price(price)
        
        self.last_cross = cross
        
        return signal
    
    def calculate_position_size(self, account_balance: float, price: float) -> float:
        """
        计算仓位大小
        
        Args:
            account_balance: 账户余额
            price: 当前价格
        
        Returns:
            仓位大小（币种数量）
        """
        position_pct = self.params["position_pct"]
        
        # 计算可用金额
        available_amount = account_balance * position_pct
        
        # 计算币种数量
        amount = available_amount / price
        
        return amount
    
    def describe(self) -> str:
        """描述策略"""
        desc = super().describe()
        desc += "\n策略逻辑:\n"
        desc += f"  1. 金叉买入: EMA{self.params['fast_period']} 上穿 EMA{self.params['slow_period']}\n"
        desc += f"  2. 死叉卖出: EMA{self.params['fast_period']} 下穿 EMA{self.params['slow_period']}\n"
        desc += f"  3. 止损: {self.params['stop_loss_pct']*100}%\n"
        desc += f"  4. 止盈: {self.params['take_profit_pct']*100}%\n"
        desc += f"  5. 仓位: {self.params['position_pct']*100}%账户权益\n"
        return desc


# 使用示例
if __name__ == "__main__":
    strategy = DoubleMAStrategy(
        instId="BTC-USDT",
        fast_period=10,
        slow_period=30,
        position_pct=0.02,
        stop_loss_pct=0.05,
        take_profit_pct=0.10
    )
    
    print(strategy.describe())
    
    # 模拟数据测试
    prices = [40000, 40500, 41000, 40800, 41200, 41500, 42000, 41800, 42500]
    
    for i, price in enumerate(prices):
        signal = strategy.generate_signal({
            "price": price,
            "timestamp": f"2024-01-01T{i}:00:00Z"
        })
        
        print(f"价格{price}: {signal.signal_type.value} - {signal.reason}")