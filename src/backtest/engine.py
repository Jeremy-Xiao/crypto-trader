"""
回测引擎
支持 ATR 动态止盈止损、移动止损、市场状态自适应、手续费/滑点感知。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from src.strategies.base import (
    BaseStrategy, Signal, SignalType, PositionSide,
    MarketRegime, detect_market_regime
)
from src.utils.indicators import ADX


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_balance: float = 10000       # 初始资金
    fee_rate: float = 0.001              # 手续费率（0.1%）
    slippage: float = 0.0005             # 滑点（0.05%）
    leverage: float = 1.0                # 杠杆倍数

    # ATR 动态风控参数
    use_atr_risk: bool = True            # 启用 ATR 动态风控
    atr_period: int = 14                 # ATR 计算周期
    risk_pct: float = 0.02               # 每笔交易风险占比（2%）
    atr_multiplier: float = 2.0          # 仓位计算 ATR 倍数
    atr_sl_multiplier: float = 2.0       # 止损 ATR 倍数
    atr_tp_multiplier: float = 4.0       # 止盈 ATR 倍数（盈亏比 2:1）

    # 移动止损
    use_trailing: bool = True            # 启用移动止损
    trailing_pct: float = 0.02           # 盈利达到此比例后激活

    # 多时间框架确认
    use_mtf: bool = False                # 启用多时间框架确认
    tf_confirm: Optional[list] = None    # 外部传入的 MTF 确认数组

    # 市场状态自适应
    use_regime: bool = True              # 启用市场状态检测

    # 入场过滤（针对震荡市空耗的核心修复）
    min_adx_for_entry: float = 0.0       # ADX 低于此值禁止入场（0=不限制）。趋势策略设为 20~25 可避免震荡市被反复扫损
    max_position_pct: float = 0.5        # 单笔仓位上限（占账户比例），替代硬编码 0.5


class BacktestEngine:
    """
    回测引擎（ATR 动态风控 + 移动止损 + 市场状态自适应）

    引擎层职责：
    1. ATR 动态止损/止盈（覆盖策略层的固定止损止盈）
    2. 移动止损（追踪最高价，止损线跟随上移）
    3. 多时间框架确认（买入信号需通过高级别趋势确认）
    4. 市场状态检测（用于策略适配和统计）
    5. 手续费/滑点模拟
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        config: Optional[BacktestConfig] = None
    ):
        self.strategy = strategy
        self.config = config or BacktestConfig()

        # 状态
        self.balance = self.config.initial_balance
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.position_amount = 0.0
        self.position_price = 0.0
        self.position_side = PositionSide.NONE  # 持仓方向（支持做空）

        # 统计
        self.stop_stats = {
            'stop_loss': 0, 'take_profit': 0,
            'trailing_stop': 0, 'signal_sell': 0,
            'fixed_sl': 0, 'fixed_tp': 0
        }
        self.mtf_blocked = 0
        self.adx_blocked = 0
        self.regime_history: List[str] = []

    def load_data(self, data: pd.DataFrame) -> None:
        """
        加载历史数据

        Args:
            data: DataFrame with columns: timestamp, open, high, low, close, volume
        """
        self.data = data

    def run(self) -> Dict:
        """运行回测"""
        if self.data is None or len(self.data) == 0:
            return {"error": "无数据"}

        print(f"开始回测: {self.strategy.name}")
        print(f"数据范围: {len(self.data)} 条")
        print(f"ATR风控: {'开启' if self.config.use_atr_risk else '关闭'} | "
              f"移动止损: {'开启' if self.config.use_trailing else '关闭'} | "
              f"MTF确认: {'开启' if self.config.use_mtf else '关闭'}")

        for idx, row in self.data.iterrows():
            timestamp = row.get("timestamp", row.get("ts", ""))
            price = float(row["close"])
            high = float(row.get("high", price))
            low = float(row.get("low", price))

            # 更新策略的账户余额和历史数据
            self.strategy.account_balance = self.balance
            self.strategy.price_history.append(price)
            self.strategy.high_history.append(high)
            self.strategy.low_history.append(low)

            # 市场状态检测（统计用）
            if self.config.use_regime and len(self.strategy.price_history) >= 25:
                regime, info = detect_market_regime(
                    self.strategy.high_history,
                    self.strategy.low_history,
                    self.strategy.price_history
                )
                self.regime_history.append(regime.value)
            else:
                self.regime_history.append("unknown")

            # ====== 引擎层：持仓时优先检查退出条件（止损/止盈/移动止损）======
            if self.position_side != PositionSide.NONE:
                exit_reason = self._check_exit_conditions(price, high, low, timestamp)

                if exit_reason:
                    self._close_current_position(price, timestamp, exit_reason)
                    self._record_equity(price, timestamp)
                    continue

            # ====== 策略层：生成信号 ======
            signal = self.strategy.generate_signal({
                "price": price,
                "high": high,
                "low": low,
                "timestamp": timestamp
            })
            st = signal.signal_type

            # 出场信号（平多 / 平空）
            if st in (SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT):
                if self.position_side != PositionSide.NONE:
                    self._close_current_position(price, timestamp, st.value)
                    self._record_equity(price, timestamp)
                    continue

            # 入场信号（开多 / 开空，支持翻转）
            elif st in (SignalType.OPEN_LONG, SignalType.OPEN_SHORT):
                target = PositionSide.LONG if st == SignalType.OPEN_LONG else PositionSide.SHORT

                # 多时间框架确认（仅对开多/开空方向有效）
                if self.config.use_mtf and self.config.tf_confirm:
                    data_idx = len(self.strategy.price_history) - 1
                    if data_idx < len(self.config.tf_confirm) and not self.config.tf_confirm[data_idx]:
                        self.mtf_blocked += 1
                        self._record_equity(price, timestamp)
                        continue

                if self.position_side == PositionSide.NONE:
                    self._open_position(target, price, signal.amount, timestamp, signal.reason)
                elif self.position_side == target:
                    pass  # 已持有同向，忽略
                else:
                    # 反向信号 → 先平后开（翻转）
                    self._close_current_position(price, timestamp, "reverse")
                    self._open_position(target, price, signal.amount, timestamp, signal.reason)

            # HOLD：不动作

            self._record_equity(price, timestamp)

        # 强制平仓
        if self.position_side != PositionSide.NONE:
            last_row = self.data.iloc[-1]
            last_price = float(last_row["close"])
            last_ts = last_row.get("timestamp", last_row.get("ts", ""))
            self._close_current_position(last_price, last_ts, "end_of_backtest")

        # 计算绩效
        results = self._calculate_performance()

        print(f"\n回测完成!")
        print(f"总收益: {results['total_return']:.2f}%")
        print(f"最大回撤: {results['max_drawdown']:.2f}%")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"胜率: {results['win_rate']:.2f}%")
        print(f"总交易: {results['total_trades']} | 盈利: {results['winning_trades']}")
        if sum(self.stop_stats.values()) > 0:
            ss = self.stop_stats
            total = sum(ss.values())
            print(f"退出方式: 止盈{ss['take_profit']}({ss['take_profit']/total*100:.0f}%) | "
                  f"止损{ss['stop_loss']}({ss['stop_loss']/total*100:.0f}%) | "
                  f"移动止损{ss['trailing_stop']}({ss['trailing_stop']/total*100:.0f}%) | "
                  f"信号卖出{ss['signal_sell']}({ss['signal_sell']/total*100:.0f}%)")
        if self.mtf_blocked > 0:
            print(f"MTF拦截: {self.mtf_blocked}次")

        return results

    def _check_exit_conditions(self, price: float, high: float, low: float, timestamp: str) -> Optional[str]:
        """
        检查退出条件（引擎层优先于策略层，支持双向）

        优先级：
        1. ATR 动态止损/止盈（如果启用，按持仓方向计算）
        2. 移动止损（如果启用，按持仓方向计算）
        3. 固定止损/止盈（兼容旧策略，should_* 已双向）
        """
        pos = self.strategy.position
        if pos is None:
            return None

        if self.config.use_atr_risk:
            atr_val = self.strategy.get_atr(self.config.atr_period)

            if atr_val > 0:
                pos.update_price(price)
                entry = pos.entry_price
                sl = self.config.atr_sl_multiplier
                tp = self.config.atr_tp_multiplier

                if self.config.use_trailing:
                    trailing_reason = self.strategy.update_trailing_stop(
                        price, atr_val, sl, self.config.trailing_pct
                    )
                    if trailing_reason:
                        self.stop_stats['trailing_stop'] += 1
                        return trailing_reason

                if pos.side == PositionSide.LONG:
                    if price <= entry - atr_val * sl:
                        self.stop_stats['stop_loss'] += 1
                        return 'stop_loss'
                    if price >= entry + atr_val * tp:
                        self.stop_stats['take_profit'] += 1
                        return 'take_profit'
                else:  # SHORT
                    if price >= entry + atr_val * sl:   # 价格上涨 → 空头亏损
                        self.stop_stats['stop_loss'] += 1
                        return 'stop_loss'
                    if price <= entry - atr_val * tp:   # 价格下跌 → 空头盈利
                        self.stop_stats['take_profit'] += 1
                        return 'take_profit'

                return None  # ATR 风控启用但未触发

        # 固定止损止盈（兼容旧逻辑，should_* 已双向支持）
        if self.strategy.should_stop_loss(price):
            self.stop_stats['fixed_sl'] += 1
            return 'fixed_stop_loss'
        if self.strategy.should_take_profit(price):
            self.stop_stats['fixed_tp'] += 1
            return 'fixed_take_profit'

        return None

    def _current_adx(self) -> float:
        """计算当前 ADX（用于入场过滤）"""
        n = len(self.strategy.price_history)
        if n < 20:
            return 0.0
        adx_series = ADX(
            pd.Series(self.strategy.high_history),
            pd.Series(self.strategy.low_history),
            pd.Series(self.strategy.price_history),
            14
        )
        val = adx_series.iloc[-1]
        return float(val) if not pd.isna(val) else 0.0

    def _execute_buy(self, price: float, amount: float, timestamp: str, reason: str):
        """执行买入开多"""
        if self.position_amount > 0:
            return

        actual_price = price * (1 + self.config.slippage)

        # ATR 动态仓位
        if self.config.use_atr_risk:
            atr_val = self.strategy.get_atr(self.config.atr_period)
            if atr_val > 0:
                amount = self.strategy.calculate_atr_position_size(
                    self.balance, actual_price, atr_val,
                    risk_pct=self.config.risk_pct,
                    atr_multiplier=self.config.atr_multiplier,
                    max_pct=self.config.max_position_pct
                )

        buy_value = min(amount * actual_price, self.balance)
        actual_amount = buy_value / actual_price
        fee = buy_value * self.config.fee_rate

        self.balance -= (buy_value + fee)
        self.position_amount = actual_amount
        self.position_price = actual_price

        self.strategy.open_position(actual_price, actual_amount, timestamp)

        self.trades.append({
            "timestamp": timestamp,
            "instId": self.strategy.instId,
            "action": "buy",
            "price": actual_price,
            "amount": actual_amount,
            "fee": fee,
            "pnl": 0,
            "reason": reason
        })

    def _open_position(self, side: PositionSide, price: float, amount: float, timestamp: str, reason: str):
        """开仓（多或空），含 ADX 入场过滤"""
        if self.position_side != PositionSide.NONE:
            return
        # ADX 入场过滤：震荡市（ADX 过低）禁止入场，多空都适用
        if self.config.min_adx_for_entry > 0:
            adx_now = self._current_adx()
            if adx_now < self.config.min_adx_for_entry:
                self.adx_blocked += 1
                return
        if side == PositionSide.LONG:
            self._execute_buy(price, amount, timestamp, reason)
        else:
            self._execute_open_short(price, amount, timestamp, reason)
        self.position_side = side

    def _execute_open_short(self, price: float, amount: float, timestamp: str, reason: str):
        """执行卖出开空（做空）"""
        if self.position_amount > 0:
            return
        actual_price = price * (1 - self.config.slippage)  # 做空卖出，滑点不利方向→成交价略低
        sell_value = amount * actual_price
        fee = sell_value * self.config.fee_rate
        self.balance += (sell_value - fee)                # 收到现金，同时背负 amount 币的负债
        self.position_amount = amount
        self.position_price = actual_price
        self.strategy.open_position(actual_price, amount, timestamp, side=PositionSide.SHORT)
        self.trades.append({
            "timestamp": timestamp,
            "instId": self.strategy.instId,
            "action": "sell",
            "side": "short_open",
            "price": actual_price,
            "amount": amount,
            "fee": fee,
            "pnl": 0,
            "reason": reason
        })

    def _execute_buy_to_cover(self, price: float, timestamp: str, reason: str):
        """执行买入平空（买回还债）"""
        if self.position_amount <= 0:
            return
        actual_price = price * (1 + self.config.slippage)  # 买回，滑点不利方向→成交价略高
        buy_value = self.position_amount * actual_price
        fee = buy_value * self.config.fee_rate
        pnl = (self.position_price - actual_price) * self.position_amount  # 空头盈利 = 开仓价 - 平仓价
        self.balance -= (buy_value + fee)
        self.strategy.close_position(actual_price, timestamp, reason)
        self.trades.append({
            "timestamp": timestamp,
            "instId": self.strategy.instId,
            "action": "buy",
            "side": "short_close",
            "price": actual_price,
            "amount": self.position_amount,
            "fee": fee,
            "pnl": pnl,
            "reason": reason
        })
        self.position_amount = 0.0
        self.position_price = 0.0

    def _close_current_position(self, price: float, timestamp: str, reason: str):
        """平掉当前持仓（按方向自动选平多/平空）"""
        if self.position_side == PositionSide.LONG:
            self._execute_sell(price, timestamp, reason)
        elif self.position_side == PositionSide.SHORT:
            self._execute_buy_to_cover(price, timestamp, reason)
        self.position_side = PositionSide.NONE

    def _execute_sell(self, price: float, timestamp: str, reason: str):
        """执行卖出平多"""
        if self.position_amount <= 0:
            return

        actual_price = price * (1 - self.config.slippage)
        sell_value = self.position_amount * actual_price
        fee = sell_value * self.config.fee_rate
        pnl = (actual_price - self.position_price) * self.position_amount

        self.balance += (sell_value - fee)

        self.strategy.close_position(actual_price, timestamp, reason)

        self.trades.append({
            "timestamp": timestamp,
            "instId": self.strategy.instId,
            "action": "sell",
            "price": actual_price,
            "amount": self.position_amount,
            "fee": fee,
            "pnl": pnl,
            "reason": reason
        })

        self.position_amount = 0.0
        self.position_price = 0.0

    def _record_equity(self, price: float, timestamp: str):
        """记录权益曲线（空头持仓的市值记为负数）"""
        if self.position_side == PositionSide.SHORT:
            pos_value = -self.position_amount * price
        else:
            pos_value = self.position_amount * price
        equity = self.balance + pos_value
        self.equity_curve.append({
            "timestamp": timestamp,
            "equity": equity,
            "balance": self.balance,
            "position_value": pos_value,
            "price": price
        })

    def _calculate_performance(self) -> Dict:
        """计算绩效指标"""
        if not self.equity_curve:
            return {}

        equity_series = pd.Series([e["equity"] for e in self.equity_curve])

        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0] * 100

        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min() * 100

        returns = equity_series.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0

        sell_trades = [t for t in self.trades if t["action"] == "sell"]
        winning_trades = [t for t in sell_trades if t["pnl"] > 0]
        total_trades = len(sell_trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        if winning_trades and len([t for t in sell_trades if t["pnl"] < 0]) > 0:
            avg_win = np.mean([t["pnl"] for t in winning_trades])
            avg_loss = np.mean([abs(t["pnl"]) for t in sell_trades if t["pnl"] < 0])
            profit_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            profit_ratio = 0

        total_pnl = sum([t["pnl"] for t in self.trades])

        # 市场状态统计
        regime_stats = {}
        if self.regime_history:
            for r in self.regime_history:
                regime_stats[r] = regime_stats.get(r, 0) + 1

        return {
            "initial_balance": self.config.initial_balance,
            "final_equity": float(equity_series.iloc[-1]),
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "win_rate": win_rate,
            "profit_ratio": profit_ratio,
            "total_pnl": total_pnl,
            "total_fee": sum([t["fee"] for t in self.trades]),
            "stop_stats": self.stop_stats,
            "mtf_blocked": self.mtf_blocked,
            "regime_stats": regime_stats
        }

    def get_equity_curve_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve)

    def get_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "timestamp": t["timestamp"],
            "instId": t["instId"],
            "action": t["action"],
            "price": t["price"],
            "amount": t["amount"],
            "fee": t["fee"],
            "pnl": t["pnl"],
            "reason": t["reason"]
        } for t in self.trades])
