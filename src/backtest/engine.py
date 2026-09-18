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

    # ===== 杠杆与保证金 =====
    leverage: float = 1.0                # 杠杆倍数（1.0 = 无杠杆/现货口径）
    max_leverage: float = 3.0            # 硬上限，配置超过会被强制截断，防手滑
    maintenance_margin_rate: float = 0.005   # 维持保证金率（占名义价值），低于则强平
    liquidation_buffer: float = 0.25     # 提前强平缓冲：权益 < 初始保证金×此值 时主动平仓（早于交易所强平）
    borrow_rate_daily: float = 0.0003    # 借币日利率（≈11%年化），只对借入部分 notional×(1-1/L) 计息
    risk_scales_with_leverage: bool = False  # False=杠杆只放开资金约束、不放大单笔风险（推荐）
    max_drawdown_halt: float = 0.0       # 回撤熔断：权益自峰值回撤超过此比例时禁止新开仓（0=关闭）

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

        # 杠杆硬约束：任何配置都不可能突破 max_leverage，且不得低于 1
        self.leverage = min(max(float(self.config.leverage), 1.0),
                            float(self.config.max_leverage))

        # 状态
        self.balance = self.config.initial_balance   # 可用现金（保证金模式下为「空闲保证金」）
        self.margin_used = 0.0                       # 当前持仓占用的保证金
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.position_amount = 0.0
        self.position_price = 0.0
        self.position_side = PositionSide.NONE  # 持仓方向（支持做空）

        # 统计
        self.stop_stats = {
            'stop_loss': 0, 'take_profit': 0,
            'trailing_stop': 0, 'signal_sell': 0,
            'fixed_sl': 0, 'fixed_tp': 0,
            'reverse': 0, 'end_of_backtest': 0,
            'liquidation': 0
        }
        self.mtf_blocked = 0
        self.adx_blocked = 0
        self.dd_halt_blocked = 0        # 回撤熔断拦截次数
        self.margin_blocked = 0         # 保证金不足导致无法开仓次数
        self.liquidations = 0           # 强平次数
        self.total_borrow_cost = 0.0    # 累计借币利息
        self.peak_equity = self.config.initial_balance
        self.max_gross_leverage = 0.0   # 实际用到的最大名义杠杆（名义价值/权益）
        self.regime_history: List[str] = []
        self.data: Optional[pd.DataFrame] = None
        self._adx_series = None         # load_data 时预计算，避免逐根重算 ADX

    # ==================== 保证金记账 ====================

    def _unrealized_pnl(self, price: float) -> float:
        """当前持仓的未实现盈亏"""
        if self.position_side == PositionSide.LONG:
            return (price - self.position_price) * self.position_amount
        if self.position_side == PositionSide.SHORT:
            return (self.position_price - price) * self.position_amount
        return 0.0

    def _current_equity(self, price: float) -> float:
        """账户权益 = 空闲现金 + 占用保证金 + 未实现盈亏"""
        return self.balance + self.margin_used + self._unrealized_pnl(price)

    def load_data(self, data: pd.DataFrame) -> None:
        """
        加载历史数据

        Args:
            data: DataFrame with columns: timestamp, open, high, low, close, volume
        """
        self.data = data
        self._adx_series = None
        # 预计算 ADX 序列：ADX 由 diff/shift(1)/ewm(adjust=False) 构成，全部是因果算子，
        # 因此「全量算一次按下标取」与「每根K线用前缀重算」结果完全一致，不存在未来函数。
        # 这一步把入场过滤从 O(n²) 降到 O(n)，多年回测提速几十倍。
        try:
            if len(data) > 0 and {"high", "low", "close"}.issubset(data.columns):
                self._adx_series = ADX(
                    pd.Series(data["high"].astype(float).values),
                    pd.Series(data["low"].astype(float).values),
                    pd.Series(data["close"].astype(float).values),
                    14
                ).values
        except Exception:
            self._adx_series = None

    def run(self) -> Dict:
        """运行回测"""
        if self.data is None or len(self.data) == 0:
            return {"error": "无数据"}

        print(f"开始回测: {self.strategy.name}")
        print(f"数据范围: {len(self.data)} 条")
        print(f"ATR风控: {'开启' if self.config.use_atr_risk else '关闭'} | "
              f"移动止损: {'开启' if self.config.use_trailing else '关闭'} | "
              f"MTF确认: {'开启' if self.config.use_mtf else '关闭'} | "
              f"杠杆: {self.leverage}x")

        for idx, row in self.data.iterrows():
            timestamp = row.get("timestamp", row.get("ts", ""))
            price = float(row["close"])
            high = float(row.get("high", price))
            low = float(row.get("low", price))
            open_p = float(row.get("open", price))

            # 更新策略的历史数据
            self.strategy.price_history.append(price)
            self.strategy.high_history.append(high)
            self.strategy.low_history.append(low)

            # 杠杆持仓：每根K线计提借币利息
            self._accrue_borrow_cost(price)
            # 策略的仓位计算基准用「权益」而非「空闲现金」（保证金模式下现金会被占用）
            self.strategy.account_balance = self._current_equity(price)

            # 市场状态检测（统计用）
            if self.config.use_regime and len(self.strategy.price_history) >= 25:
                regime, info = detect_market_regime(
                    self.strategy.high_history,
                    self.strategy.low_history,
                    self.strategy.price_history,
                    adx_val=self._current_adx()   # 复用预计算的 ADX，避免重复全量计算
                )
                self.regime_history.append(regime.value)
            else:
                self.regime_history.append("unknown")

            # ====== 引擎层：持仓时优先检查强平，再检查退出条件（止损/止盈/移动止损）======
            if self.position_side != PositionSide.NONE:
                # 强平优先级最高：账户先爆仓，止损再漂亮也没用
                if self._check_liquidation(high, low, timestamp, open_p):
                    self._record_equity(price, timestamp)
                    continue

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
                "timestamp": timestamp,
                "adx": self._current_adx(),   # 预计算 ADX，供策略（如元策略）做市场状态判断，避免重复全量计算
                "market_bias": row.get("market_bias", None)  # 市场状态过滤序列（回测回放用，无则 None）
            })
            st = signal.signal_type

            # 出场信号（平多 / 平空）
            if st in (SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT):
                if self.position_side != PositionSide.NONE:
                    self.stop_stats['signal_sell'] += 1
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
                    # 注意：平仓无条件执行（趋势反转即离场），但新仓仍需过 ADX 过滤，
                    # 因此可能出现「只平不开」——这是预期行为，被拦截时 adx_blocked 会累加
                    self.stop_stats['reverse'] += 1
                    self._close_current_position(price, timestamp, "reverse")
                    self._open_position(target, price, signal.amount, timestamp, signal.reason)

            # HOLD：不动作

            self._record_equity(price, timestamp)

        # 强制平仓
        if self.position_side != PositionSide.NONE:
            last_row = self.data.iloc[-1]
            last_price = float(last_row["close"])
            last_ts = last_row.get("timestamp", last_row.get("ts", ""))
            self.stop_stats['end_of_backtest'] += 1
            self._close_current_position(last_price, last_ts, "end_of_backtest")

        # 计算绩效
        results = self._calculate_performance()

        print(f"\n回测完成!")
        print(f"总收益: {results['total_return']:.2f}%")
        print(f"最大回撤: {results['max_drawdown']:.2f}%")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"胜率: {results['win_rate']:.2f}%")
        print(f"总交易: {results['total_trades']} | 盈利: {results['winning_trades']}")
        ls, ss_ = results['long_stats'], results['short_stats']
        print(f"多头: {ls['trades']}笔 盈利{ls['wins']}笔 PnL {ls['pnl']:.2f} | "
              f"空头: {ss_['trades']}笔 盈利{ss_['wins']}笔 PnL {ss_['pnl']:.2f}")
        if sum(self.stop_stats.values()) > 0:
            ss = self.stop_stats
            total = sum(ss.values())
            print(f"退出方式: 止盈{ss['take_profit']}({ss['take_profit']/total*100:.0f}%) | "
                  f"止损{ss['stop_loss']}({ss['stop_loss']/total*100:.0f}%) | "
                  f"移动止损{ss['trailing_stop']}({ss['trailing_stop']/total*100:.0f}%) | "
                  f"信号卖出{ss['signal_sell']}({ss['signal_sell']/total*100:.0f}%)")
        if self.mtf_blocked > 0:
            print(f"MTF拦截: {self.mtf_blocked}次")
        if self.leverage > 1.0:
            print(f"杠杆: {self.leverage}x | 实际最大名义杠杆: {self.max_gross_leverage:.2f}x | "
                  f"强平: {self.liquidations}次 | 借币利息: {self.total_borrow_cost:.2f} | "
                  f"回撤熔断拦截: {self.dd_halt_blocked}次")

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
        """当前 ADX（用于入场过滤）。优先读预计算序列，否则回退到实时计算。"""
        n = len(self.strategy.price_history)
        if n < 20:
            return 0.0

        series = getattr(self, "_adx_series", None)
        if series is not None and n - 1 < len(series):
            val = series[n - 1]
            return float(val) if not pd.isna(val) else 0.0

        adx_series = ADX(
            pd.Series(self.strategy.high_history),
            pd.Series(self.strategy.low_history),
            pd.Series(self.strategy.price_history),
            14
        )
        val = adx_series.iloc[-1]
        return float(val) if not pd.isna(val) else 0.0

    def _size_notional(self, price: float, actual_price: float, amount: float) -> float:
        """
        统一的仓位定价（多空共用），三重约束取最小：

        1. 策略/ATR 给出的目标仓位
        2. 名义价值上限 = 权益 × max_position_pct × 杠杆
        3. 可用保证金约束：margin + fee ≤ 空闲现金

        安全要点：默认 risk_scales_with_leverage=False，即 ATR 单笔风险始终按「权益」
        计算，杠杆只放开资金约束、不放大每笔亏损。3倍杠杆下单笔风险仍是 risk_pct。
        """
        equity = self._current_equity(price)
        if equity <= 0:
            return 0.0

        if self.config.use_atr_risk:
            atr_val = self.strategy.get_atr(self.config.atr_period)
            if atr_val > 0:
                risk_base = equity * (self.leverage if self.config.risk_scales_with_leverage else 1.0)
                amount = self.strategy.calculate_atr_position_size(
                    risk_base, actual_price, atr_val,
                    risk_pct=self.config.risk_pct,
                    atr_multiplier=self.config.atr_multiplier,
                    max_pct=self.config.max_position_pct * self.leverage
                )

        notional = amount * actual_price
        # 约束2：名义价值上限
        notional = min(notional, equity * self.config.max_position_pct * self.leverage)
        # 约束3：保证金 + 手续费不得超过空闲现金
        #   notional/L + notional*fee ≤ balance  →  notional ≤ balance / (1/L + fee)
        denom = 1.0 / self.leverage + self.config.fee_rate
        notional = min(notional, max(self.balance, 0.0) / denom)
        return max(notional, 0.0)

    def _execute_buy(self, price: float, amount: float, timestamp: str, reason: str):
        """执行买入开多（保证金记账）"""
        if self.position_amount > 0:
            return

        actual_price = price * (1 + self.config.slippage)
        notional = self._size_notional(price, actual_price, amount)
        if notional <= 0:
            self.margin_blocked += 1
            return

        actual_amount = notional / actual_price
        margin = notional / self.leverage
        fee = notional * self.config.fee_rate

        self.balance -= (margin + fee)
        self.margin_used = margin
        self.position_amount = actual_amount
        self.position_price = actual_price
        self._track_gross_leverage(price)

        self.strategy.open_position(actual_price, actual_amount, timestamp)

        self.trades.append({
            "timestamp": timestamp,
            "instId": self.strategy.instId,
            "action": "buy",
            "side": "long_open",
            "price": actual_price,
            "amount": actual_amount,
            "notional": notional,
            "margin": margin,
            "fee": fee,
            "pnl": 0,
            "reason": reason
        })

    def _open_position(self, side: PositionSide, price: float, amount: float, timestamp: str, reason: str):
        """开仓（多或空），含 ADX 入场过滤 + 回撤熔断"""
        if self.position_side != PositionSide.NONE:
            return
        # 回撤熔断：权益自峰值回撤过大时停止新开仓，防止杠杆下的连续亏损螺旋
        if self.config.max_drawdown_halt > 0 and self.peak_equity > 0:
            dd = 1.0 - self._current_equity(price) / self.peak_equity
            if dd >= self.config.max_drawdown_halt:
                self.dd_halt_blocked += 1
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
        """
        执行卖出开空（做空，保证金记账）

        与 _execute_buy 完全对等：同样的 ATR 仓位、同样的名义上限、同样的保证金占用。
        做空同样需要压保证金（不再是「凭空收到现金」），这才符合真实杠杆账户。
        """
        if self.position_amount > 0:
            return

        actual_price = price * (1 - self.config.slippage)  # 做空卖出，滑点不利方向→成交价略低
        notional = self._size_notional(price, actual_price, amount)
        if notional <= 0:
            self.margin_blocked += 1
            return

        actual_amount = notional / actual_price
        margin = notional / self.leverage
        fee = notional * self.config.fee_rate

        self.balance -= (margin + fee)     # 压保证金，卖出所得计入持仓盈亏而非现金
        self.margin_used = margin
        self.position_amount = actual_amount
        self.position_price = actual_price
        self._track_gross_leverage(price)

        self.strategy.open_position(actual_price, actual_amount, timestamp, side=PositionSide.SHORT)
        self.trades.append({
            "timestamp": timestamp,
            "instId": self.strategy.instId,
            "action": "sell",
            "side": "short_open",
            "price": actual_price,
            "amount": actual_amount,
            "notional": notional,
            "margin": margin,
            "fee": fee,
            "pnl": 0,
            "reason": reason
        })

    def _execute_buy_to_cover(self, price: float, timestamp: str, reason: str):
        """执行买入平空（保证金释放 + 盈亏结算）"""
        if self.position_amount <= 0:
            return
        actual_price = price * (1 + self.config.slippage)  # 买回，滑点不利方向→成交价略高
        exit_notional = self.position_amount * actual_price
        fee = exit_notional * self.config.fee_rate
        pnl = (self.position_price - actual_price) * self.position_amount  # 空头盈利 = 开仓价 - 平仓价

        self.balance += (self.margin_used + pnl - fee)
        self.margin_used = 0.0

        self.strategy.close_position(actual_price, timestamp, reason)
        self.trades.append({
            "timestamp": timestamp,
            "instId": self.strategy.instId,
            "action": "buy",
            "side": "short_close",
            "price": actual_price,
            "amount": self.position_amount,
            "notional": exit_notional,
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
        """执行卖出平多（保证金释放 + 盈亏结算）"""
        if self.position_amount <= 0:
            return

        actual_price = price * (1 - self.config.slippage)
        exit_notional = self.position_amount * actual_price
        fee = exit_notional * self.config.fee_rate
        pnl = (actual_price - self.position_price) * self.position_amount

        self.balance += (self.margin_used + pnl - fee)
        self.margin_used = 0.0

        self.strategy.close_position(actual_price, timestamp, reason)

        self.trades.append({
            "timestamp": timestamp,
            "instId": self.strategy.instId,
            "action": "sell",
            "side": "long_close",
            "price": actual_price,
            "amount": self.position_amount,
            "notional": exit_notional,
            "fee": fee,
            "pnl": pnl,
            "reason": reason
        })

        self.position_amount = 0.0
        self.position_price = 0.0

    def _record_equity(self, price: float, timestamp: str):
        """记录权益曲线（保证金口径：现金 + 占用保证金 + 未实现盈亏）"""
        if self.position_side == PositionSide.SHORT:
            pos_value = -self.position_amount * price
        else:
            pos_value = self.position_amount * price
        equity = self._current_equity(price)
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.equity_curve.append({
            "timestamp": timestamp,
            "equity": equity,
            "balance": self.balance,
            "margin_used": self.margin_used,
            "position_value": pos_value,
            "price": price
        })

    # ==================== 杠杆风控 ====================

    def _track_gross_leverage(self, price: float):
        """记录实际达到的最大名义杠杆（名义价值 / 权益），用于事后核查是否越界"""
        equity = self._current_equity(price)
        if equity > 0:
            gross = (self.position_amount * price) / equity
            self.max_gross_leverage = max(self.max_gross_leverage, gross)

    def _accrue_borrow_cost(self, price: float):
        """
        计提借币利息：只对「借入部分」计息 = notional × (1 - 1/L) × 日利率

        杠杆1倍时借入为0，成本自然为0，保证 1x 结果与无杠杆口径可比。
        """
        if self.position_side == PositionSide.NONE or self.leverage <= 1.0:
            return
        notional = self.position_amount * price
        borrowed = notional * (1.0 - 1.0 / self.leverage)
        cost = borrowed * self.config.borrow_rate_daily
        if cost > 0:
            self.balance -= cost
            self.total_borrow_cost += cost

    def _liquidation_price(self) -> float:
        """
        解出精确的强平价：权益恰好等于触发线时的价格。

        触发线取两者较高者，保证「先于交易所强平」：
        - 交易所维持保证金：名义价值 × maintenance_margin_rate
        - 自设安全缓冲：初始保证金 × liquidation_buffer
        """
        amt = self.position_amount
        entry = self.position_price
        cash = self.balance + self.margin_used
        mmr = self.config.maintenance_margin_rate
        buf = self.margin_used * self.config.liquidation_buffer

        if self.position_side == PositionSide.LONG:
            # 缓冲线（阈值与价格无关）: cash + (P-entry)*amt = buf
            p_buf = entry + (buf - cash) / amt
            # 维持保证金线: cash + (P-entry)*amt = amt*P*mmr
            p_mm = (entry * amt - cash) / (amt * (1 - mmr))
            return max(p_buf, p_mm)   # 下跌途中先碰到的是较高者
        else:
            p_buf = entry + (cash - buf) / amt
            p_mm = (cash + entry * amt) / (amt * (1 + mmr))
            return min(p_buf, p_mm)   # 上涨途中先碰到的是较低者

    def _check_liquidation(self, high: float, low: float, timestamp: str,
                           open_price: Optional[float] = None) -> bool:
        """
        强平检查（用K线内对持仓最不利的价格判定是否触及）

        成交价处理：
        - 正常情况按精确强平价成交（交易所在触线瞬间平仓，不会等到当根K线最低/最高点）
        - 若开盘就跳空穿过强平价，则按开盘价成交（真实的跳空损失，必须体现）
        """
        if self.position_side == PositionSide.NONE:
            return False
        # 现货多头（1倍）无借贷，不可能被强平；空头即使1倍也借了币，亏损无上限，仍需检查
        if self.position_side == PositionSide.LONG and self.leverage <= 1.0:
            return False
        if self.position_amount <= 0:
            return False

        liq_price = self._liquidation_price()

        if self.position_side == PositionSide.LONG:
            if low > liq_price:
                return False
            fill = liq_price
            if open_price is not None and open_price < liq_price:
                fill = open_price          # 跳空低开，只能在更差的价格成交
            fill = max(fill, low)
        else:
            if high < liq_price:
                return False
            fill = liq_price
            if open_price is not None and open_price > liq_price:
                fill = open_price          # 跳空高开
            fill = min(fill, high)

        self.stop_stats['liquidation'] += 1
        self.liquidations += 1
        self._close_current_position(fill, timestamp, 'liquidation')
        return True

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

        # 平仓交易统计（必须同时覆盖平多 action=sell 和平空 action=buy）
        # 旧逻辑只取 action=="sell"，会把「开空」误算成一笔交易、且漏掉所有「平空」盈亏
        def _is_close(t):
            side = t.get("side")
            if side is not None:
                return side in ("long_close", "short_close")
            return t["action"] == "sell"  # 兼容无 side 字段的旧记录

        close_trades = [t for t in self.trades if _is_close(t)]
        winning_trades = [t for t in close_trades if t["pnl"] > 0]
        losing_trades = [t for t in close_trades if t["pnl"] < 0]
        total_trades = len(close_trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        if winning_trades and losing_trades:
            avg_win = np.mean([t["pnl"] for t in winning_trades])
            avg_loss = np.mean([abs(t["pnl"]) for t in losing_trades])
            profit_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            profit_ratio = 0

        # 多空分别统计
        long_closes = [t for t in close_trades if t.get("side") == "long_close"]
        short_closes = [t for t in close_trades if t.get("side") == "short_close"]
        long_stats = {
            "trades": len(long_closes),
            "pnl": float(sum(t["pnl"] for t in long_closes)),
            "wins": len([t for t in long_closes if t["pnl"] > 0]),
        }
        short_stats = {
            "trades": len(short_closes),
            "pnl": float(sum(t["pnl"] for t in short_closes)),
            "wins": len([t for t in short_closes if t["pnl"] > 0]),
        }

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
            "adx_blocked": self.adx_blocked,
            "regime_stats": regime_stats,
            "long_stats": long_stats,
            "short_stats": short_stats,
            # 杠杆相关
            "leverage": self.leverage,
            "liquidations": self.liquidations,
            "borrow_cost": self.total_borrow_cost,
            "max_gross_leverage": self.max_gross_leverage,
            "dd_halt_blocked": self.dd_halt_blocked,
            "margin_blocked": self.margin_blocked,
        }

    def get_equity_curve_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve)

    def get_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "timestamp": t["timestamp"],
            "instId": t["instId"],
            "action": t["action"],
            "side": t.get("side", ""),
            "price": t["price"],
            "amount": t["amount"],
            "fee": t["fee"],
            "pnl": t["pnl"],
            "reason": t["reason"]
        } for t in self.trades])
