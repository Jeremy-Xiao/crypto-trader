"""
外层调度器（LiveOrchestrator）：把「回测被动逐根喂数据」升级为「小时级主动循环」。

每个周期做四件事：
  1. 抓最新 K 线（OKX 公共接口，1H）+ 由 K 线算 ADX
  2. 抓市场情绪（MarketMonitor：恐惧贪婪 + 资金费率）→ trading_bias
  3. 把情绪注入策略 set_market_bias，再把当前行情喂给 MetaStrategy.generate_signal
  4. 把信号交给 Executor 执行（默认 DryRun 模拟，不下真单）

设计要点（回应「气囊是主动还是被动」）：
  - 过去的实现是被动的——只有你调 generate_signal 那一下才生效，且情绪得你手动喂。
  - 本调度器就是那个「主动循环」：定时(默认1小时)自己去抓情绪+行情、喂给策略、
    把信号落成操作。这样情绪一变，下一个周期立刻反应，不必干等。
  - 仍是「周期级」反应（每小时一次），不是 tick 级毫秒触发；要更密就调小 interval。
"""
import os
import sys
import time
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.api.okx_rest import OKXPublicAPI
from src.monitor.monitor import MarketMonitor
from src.strategies.meta import MetaStrategy
from src.strategies.base import Signal, SignalType, PositionSide
from src.utils.indicators import ADX

logger = logging.getLogger("live.orchestrator")


# ============================ 数据抓取辅助 ============================

def _parse_candles(raw) -> List[tuple]:
    """OKX /market/candles 返回最新在前(list of [ts_ms,o,h,l,c,vol,...])，
    解析为升序的 (ts, o, h, l, close) 列表。"""
    rows = []
    for c in raw:
        try:
            ts = int(c[0]); o = float(c[1]); h = float(c[2]); l = float(c[3]); cl = float(c[4])
            rows.append((ts, o, h, l, cl))
        except (IndexError, ValueError, TypeError):
            continue
    rows.sort(key=lambda x: x[0])
    return rows


def compute_adx(rows, period: int = 14) -> np.ndarray:
    """由 K 线序列算 ADX(period) 序列，与回测引擎同款(因果算子，无未来函数)。"""
    n = len(rows)
    if n < period + 1:
        return np.zeros(n, dtype=float)
    high = pd.Series([r[2] for r in rows], dtype=float)
    low = pd.Series([r[3] for r in rows], dtype=float)
    close = pd.Series([r[4] for r in rows], dtype=float)
    try:
        vals = np.array(ADX(high, low, close, period).values, dtype=float)
        if len(vals) != n:
            vals = np.pad(vals, (n - len(vals), 0), constant_values=0.0)
        return vals
    except Exception as e:  # 极端情况下退化为中性
        logger.warning(f"ADX 计算失败: {e}")
        return np.zeros(n, dtype=float)


def _ts_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()


# ============================ 执行器 ============================

class Executor(ABC):
    """执行器抽象：把策略信号落成操作。实盘/模拟各自实现。"""

    @abstractmethod
    def get_equity(self, symbol: str, strategy, price: float) -> float:
        """当前权益（用于策略仓位计算）。"""

    @abstractmethod
    def execute(self, strategy, symbol: str, signal: Signal, price: float,
                timestamp: str) -> dict:
        """执行一个信号，返回操作记录 dict。"""


class DryRunExecutor(Executor):
    """模拟执行器：只记账、不下真单。安全默认。

    维护每币「已实现权益」，持仓期间按市价算未实现盈亏，平仓时落袋。
    同时把持仓同步回 strategy（open/close_position），保证策略状态不脱节。
    """

    def __init__(self, initial_balance: float = 10000.0):
        self.initial = initial_balance
        self.realized: Dict[str, float] = {}
        self.orders: List[dict] = []

    def get_equity(self, symbol: str, strategy, price: float) -> float:
        if symbol not in self.realized:
            self.realized[symbol] = self.initial
        pos = strategy.position
        unreal = 0.0
        if pos is not None:
            if pos.side == PositionSide.LONG:
                unreal = (price - pos.entry_price) * pos.amount
            elif pos.side == PositionSide.SHORT:
                unreal = (pos.entry_price - price) * pos.amount
        return self.realized[symbol] + unreal

    def execute(self, strategy, symbol: str, signal: Signal, price: float,
                timestamp: str) -> dict:
        st = signal.signal_type
        rec = {"symbol": symbol, "action": st.value, "price": price,
               "amount": signal.amount, "ts": timestamp, "reason": signal.reason,
               "executed": False}
        if symbol not in self.realized:
            self.realized[symbol] = self.initial

        if st == SignalType.OPEN_LONG:
            strategy.open_position(price, signal.amount, timestamp, PositionSide.LONG)
            rec["executed"] = True
        elif st == SignalType.OPEN_SHORT:
            strategy.open_position(price, signal.amount, timestamp, PositionSide.SHORT)
            rec["executed"] = True
        elif st == SignalType.CLOSE_LONG:
            if strategy.position is not None:
                pnl = (price - strategy.position.entry_price) * strategy.position.amount
                self.realized[symbol] += pnl
                strategy.close_position(price, timestamp, signal.reason)
                rec["pnl"] = pnl
                rec["executed"] = True
        elif st == SignalType.CLOSE_SHORT:
            if strategy.position is not None:
                pnl = (strategy.position.entry_price - price) * strategy.position.amount
                self.realized[symbol] += pnl
                strategy.close_position(price, timestamp, signal.reason)
                rec["pnl"] = pnl
                rec["executed"] = True
        # HOLD → 不操作

        rec["equity_after"] = self.get_equity(symbol, strategy, price)
        self.orders.append(rec)
        return rec


class OKXExecutor(Executor):
    """真实下单执行器（占位桩）。

    ⚠ 默认不启用。接真钱需要：
      1) 配置 OKX API key/secret/passphrase（环境变量或配置文件）；
      2) 实现 _place_order 调用 src/api/okx_rest.OKXClient 的下单接口；
      3) 启动时显式传入 executor=OKXExecutor(...)，且你清楚这是在动真钱。
    出于安全，本文件不直接实现下单，避免误触发真实交易。
    """

    def __init__(self, api_key=None, api_secret=None, passphrase=None):
        raise NotImplementedError(
            "OKXExecutor 未实现：真实下单需显式配置 API key 并自行实现 _place_order。"
            "当前请使用 DryRunExecutor（默认）。"
        )

    def get_equity(self, symbol, strategy, price):
        raise NotImplementedError

    def execute(self, strategy, symbol, signal, price, timestamp):
        raise NotImplementedError


# ============================ 调度器 ============================

class LiveOrchestrator:
    """小时级主动循环：抓行情+情绪 → 喂策略 → 出信号 → 执行。"""

    def __init__(self,
                 symbols: List[str],
                 mode: str = "adaptive",
                 meta_params: Optional[Dict] = None,
                 bar: str = "1H",
                 interval_seconds: int = 3600,
                 warmup_bars: int = 120,
                 executor: Optional[Executor] = None,
                 monitor: Optional[MarketMonitor] = None,
                 okx: Optional[OKXPublicAPI] = None,
                 use_bias: bool = True,
                 report_path: Optional[str] = None,
                 decision_cadence: str = "bar"):
        self.symbols = symbols
        self.bar = bar
        self.interval = interval_seconds
        self.warmup_bars = warmup_bars
        # 决策节奏：
        #   "bar"   = 每根 K 线决策一次（1H 数据下即每小时决策——2026-09-14 一周复盘
        #             证实该节奏与日线回测语义背离，震荡市反复翻仓，已证伪）
        #   "daily" = 每天只在「日K收盘后」决策一次，与回测引擎逐日决策语义对齐
        #             （第23章 A方案 +36%/3年 即此节奏）；每小时仍做 ATR 止损止盈
        #             巡检 + 对账 + 权益快照，风控粒度不变
        self.decision_cadence = decision_cadence
        self.executor = executor or DryRunExecutor()
        self.monitor = monitor or MarketMonitor()
        self.okx = okx or OKXPublicAPI()
        self.adx_period = 14
        self.use_bias = use_bias          # 气囊①开关：False=不抓情绪、不做过滤
        self.report_path = report_path    # 常驻循环时每周期落盘路径（None=不落盘）

        full = dict(min_adx=0.0)
        if meta_params:
            full.update(meta_params)
        self.strategies = {
            s: MetaStrategy(instId=s, mode=mode, params=dict(full), allow_short=True)
            for s in symbols
        }
        self._warmed = False
        self._last_bias = None
        self._last_ts: Dict[str, int] = {}   # 每币已处理的最新 bar 时间戳（区分新/旧 K 线）

    # ---------- 抓取 ----------
    def _fetch_candles(self, symbol: str, limit: int):
        raw = self.okx.get_candles(symbol, bar=self.bar, limit=limit)
        rows = _parse_candles(raw.get("data", []) if isinstance(raw, dict) else [])
        return rows

    def _current_market(self, symbol: str):
        """返回 (rows, adx_series)。失败返回 (None, None) 由调用方兜底。"""
        rows = self._fetch_candles(symbol, max(self.warmup_bars, 60))
        if not rows:
            return None, None
        return rows, compute_adx(rows, self.adx_period)

    @staticmethod
    def _feed_bar(strat, row):
        """复刻回测引擎的喂数据顺序：先把 K 线并入策略历史，再让它决策。

        必须先 append，否则策略的 price_history 为空、所有指标算不出来 → 永远 hold
        （engine.py:169 就是这么做的，实盘循环必须保持一致）。
        """
        strat.price_history.append(row[4])
        strat.high_history.append(row[2])
        strat.low_history.append(row[3])

    # 引擎层保护参数（与回测 build_engine_config 实际使用值一致）
    ATR_SL_MULTIPLIER = 3.0   # 止损 ATR 倍数（meta_backtest.build_engine_config 显式覆盖）
    ATR_TP_MULTIPLIER = 6.0   # 止盈 ATR 倍数（盈亏比 2:1）

    def _check_atr_exit(self, strat, price: float) -> Optional[str]:
        """复刻回测引擎 _check_exit_conditions 的 ATR 止损/止盈（engine.py:287）。

        回测中持仓时每根 K 线先查止损/止盈再生成策略信号；实盘若缺这层，
        亏损单会一直扛着不砍——重大风险缺口。use_trailing=False，与回测一致。
        """
        if strat.position is None:
            return None
        atr = strat.get_atr(14)
        if atr <= 0:
            return None
        pos = strat.position
        pos.update_price(price)
        entry = pos.entry_price
        sl, tp = self.ATR_SL_MULTIPLIER, self.ATR_TP_MULTIPLIER
        if pos.side == PositionSide.LONG:
            if price <= entry - atr * sl:
                return "stop_loss"
            if price >= entry + atr * tp:
                return "take_profit"
        else:
            if price >= entry + atr * sl:
                return "stop_loss"
            if price <= entry - atr * tp:
                return "take_profit"
        return None

    def _fetch_bias(self) -> Optional[str]:
        if not self.use_bias:
            return None
        try:
            snap = self.monitor.fetch_all()
            bias = snap.trading_bias().value  # 'risk_off'|'risk_on'|'neutral'
            logger.info(f"情绪快照 crowd_score={snap.crowd_score():+.2f} → bias={bias} | {snap.risk_note()}")
            return bias
        except Exception as e:
            logger.warning(f"情绪抓取失败，沿用上次 bias={self._last_bias}: {e}")
            return self._last_bias

    # ---------- 预热 ----------
    def warmup(self):
        if self.decision_cadence == "daily":
            # 日线节奏：策略的「历史」必须是日K（EMA/ATR/ADX 语义 = 天），不能用 1H 填充
            logger.info(f"预热：用最近 {self.warmup_bars} 根 1D 日K填充策略状态（日线决策节奏）")
            for sym, strat in self.strategies.items():
                try:
                    rows = self._fetch_candles_bar(sym, "1D", self.warmup_bars)
                    if not rows:
                        logger.warning(f"{sym} 预热失败：无日K")
                        continue
                    now_ms = time.time() * 1000.0
                    # 只喂「已收盘」的日K（OKX 返回的最后一根是今天未收盘的）
                    completed = [r for r in rows if r[0] + 86400000 <= now_ms]
                    hist = completed[:-1]  # 最新一根已收盘日K留给首个 tick 决策
                    adx_d = compute_adx(hist, self.adx_period)
                    for i, r in enumerate(hist):
                        self._feed_bar(strat, r)
                        data = {"price": r[4], "high": r[2], "low": r[3],
                                "timestamp": _ts_iso(r[0]), "adx": adx_d[i]}
                        strat.generate_signal(data)  # 预热期忽略信号，只建状态
                    if completed:
                        self._last_ts[sym] = hist[-1][0] if hist else 0
                    logger.info(f"  {sym} 预热完成，price_history={len(strat.price_history)} 根日K"
                                f"（{len(completed)} 根已收盘）")
                except Exception as e:
                    logger.warning(f"{sym} 预热异常: {e}")
        else:
            logger.info(f"预热：用最近 {self.warmup_bars} 根 {self.bar} K线填充策略状态")
            for sym, strat in self.strategies.items():
                try:
                    rows = self._fetch_candles(sym, self.warmup_bars)
                    if not rows:
                        logger.warning(f"{sym} 预热失败：无K线")
                        continue
                    adx = compute_adx(rows, self.adx_period)
                    # 用除最后一根外的全部建状态，最后一根留给首个 tick 处理（避免重复）
                    hist = rows[:-1]
                    for i, r in enumerate(hist):
                        self._feed_bar(strat, r)
                        data = {"price": r[4], "high": r[2], "low": r[3],
                                "timestamp": _ts_iso(r[0]), "adx": adx[i]}
                        strat.generate_signal(data)  # 预热期忽略信号，只建状态
                    self._last_ts[sym] = hist[-1][0] if hist else 0
                    logger.info(f"  {sym} 预热完成，price_history={len(strat.price_history)} 根")
                except Exception as e:
                    logger.warning(f"{sym} 预热异常: {e}")
        # 模拟盘：预热完成后恢复上次落盘的持仓（重启续命）
        if hasattr(self.executor, "restore_into"):
            self.executor.restore_into(self.strategies)
        self._warmed = True

    # ---------- 单周期 ----------
    def tick(self) -> Dict:
        if not self._warmed:
            self.warmup()
        bias = self._fetch_bias()
        self._last_bias = bias
        report = {"ts": datetime.now(timezone.utc).isoformat(), "bias": bias,
                  "cadence": self.decision_cadence, "symbols": {}}
        for sym, strat in self.strategies.items():
            try:
                if self.decision_cadence == "daily":
                    self._tick_symbol_daily(sym, strat, bias, report)
                else:
                    self._tick_symbol_bar(sym, strat, bias, report)
            except Exception as e:
                logger.exception(f"{sym} tick 异常: {e}")
        return report

    def _tick_symbol_bar(self, sym: str, strat, bias, report: Dict):
        """逐 K 线决策（原行为：每根 1H 收盘即决策）。"""
        try:
            rows, adx = self._current_market(sym)
            if not rows:
                logger.warning(f"{sym} 取行情失败，跳过本周期")
                return
            last = rows[-1]
            ts_ms, price = last[0], last[4]
            # 每周期对账：本地持仓 vs 交易所，以交易所为准（防手动干预/部分成交）
            if hasattr(self.executor, "reconcile"):
                self.executor.reconcile(strat, sym, price)
            equity = self.executor.get_equity(sym, strat, price)
            strat.account_balance = equity
            side = strat.position.side.value if strat.position else "none"

            prev = self._last_ts.get(sym)
            if prev is not None and ts_ms <= prev:
                # 没出新 K 线：只刷权益快照，不重复决策（避免同一根 bar 反复交易）
                logger.info(f"[{sym}] price={price:.2f} 无新K线，跳过决策 "
                            f"side={side} equity={equity:.2f}")
                report["symbols"][sym] = {
                    "price": price, "bias": bias, "signal": "no_new_bar",
                    "position": side, "equity": equity, "action": "skip",
                }
                if hasattr(self.executor, "snapshot"):
                    self.executor.snapshot(strat, sym, price, equity,
                                           "no_new_bar", _ts_iso(ts_ms))
                return

            self._feed_bar(strat, last)
            self._last_ts[sym] = ts_ms
            if bias is not None:
                strat.set_market_bias(bias)

            # 引擎层保护：ATR 止损/止盈优先于策略信号（复刻回测 continue 语义）
            exit_reason = self._check_atr_exit(strat, price)
            if exit_reason and strat.position is not None:
                side_now = strat.position.side
                close_type = (SignalType.CLOSE_LONG if side_now == PositionSide.LONG
                              else SignalType.CLOSE_SHORT)
                close_sig = Signal(close_type, sym, price, strat.position.amount,
                                   _ts_iso(ts_ms), f"引擎保护触发: {exit_reason}")
                rec = self.executor.execute(strat, sym, close_sig, price,
                                            _ts_iso(ts_ms))
                eq = rec.get("equity_after", equity)
                side_after = strat.position.side.value if strat.position else "none"
                logger.warning(f"[{sym}] {exit_reason.upper()} 触发强制离场 "
                               f"@{price:.2f} equity={eq:.2f} "
                               f"executed={rec.get('executed')}")
                report["symbols"][sym] = {
                    "price": price, "bias": bias, "adx": round(float(adx[-1]), 2),
                    "signal": close_type.value, "position": side_after,
                    "equity": eq, "action": exit_reason,
                    "reason": f"引擎保护: {exit_reason}",
                }
                if hasattr(self.executor, "snapshot"):
                    self.executor.snapshot(strat, sym, price, eq,
                                           close_type.value, _ts_iso(ts_ms))
                return

            data = {"price": price, "high": last[2], "low": last[3],
                    "timestamp": _ts_iso(ts_ms), "adx": adx[-1]}
            signal = strat.generate_signal(data)
            rec = self.executor.execute(strat, sym, signal, price, _ts_iso(ts_ms))
            side = strat.position.side.value if strat.position else "none"
            logger.info(f"[{sym}] price={price:.2f} adx={adx[-1]:.1f} bias={bias} "
                        f"signal={signal.signal_type.value} side={side} "
                        f"equity={rec.get('equity_after', equity):.2f} | {signal.reason}")
            report["symbols"][sym] = {
                "price": price, "bias": bias, "adx": round(float(adx[-1]), 2),
                "signal": signal.signal_type.value,
                "position": side,
                "equity": rec.get("equity_after", equity),
                "action": rec.get("action"),
                "reason": signal.reason,
            }
            if hasattr(self.executor, "snapshot"):
                self.executor.snapshot(strat, sym, price,
                                       rec.get("equity_after", equity),
                                       signal.signal_type.value, _ts_iso(ts_ms))
        except Exception as e:
            logger.exception(f"{sym} tick 异常: {e}")

    def _fetch_candles_bar(self, symbol: str, bar: str, limit: int) -> List[tuple]:
        """按指定周期抓 K 线（日线节奏用 1D）。"""
        raw = self.okx.get_candles(symbol, bar=bar, limit=limit)
        return _parse_candles(raw.get("data", []) if isinstance(raw, dict) else [])

    def _tick_symbol_daily(self, sym: str, strat, bias, report: Dict):
        """日线节奏（与回测语义对齐）：
        - 每小时：对账 + 权益快照 + ATR 止损止盈巡检（风控粒度不变）
        - 每天一次：检测到「新已收盘日K」才喂给策略并 generate_signal 决策
          （复刻回测引擎逐日收盘决策；决策价=当日价格，比日K收盘略滞后≤1小时）
        """
        # 1) 行情价格（用 1H 最新价）
        rows_1h = self._fetch_candles(sym, max(self.warmup_bars, 60))
        if not rows_1h:
            logger.warning(f"{sym} 取行情失败，跳过本周期")
            return
        price = rows_1h[-1][4]
        ts_ms_1h = rows_1h[-1][0]

        # 2) 对账 + 权益
        if hasattr(self.executor, "reconcile"):
            self.executor.reconcile(strat, sym, price)
        equity = self.executor.get_equity(sym, strat, price)
        strat.account_balance = equity
        side = strat.position.side.value if strat.position else "none"

        # 3) 引擎层保护：ATR 止损/止盈每小时巡检（ATR 来自日K序列，与回测一致）
        exit_reason = self._check_atr_exit(strat, price)
        if exit_reason and strat.position is not None:
            side_now = strat.position.side
            close_type = (SignalType.CLOSE_LONG if side_now == PositionSide.LONG
                          else SignalType.CLOSE_SHORT)
            close_sig = Signal(close_type, sym, price, strat.position.amount,
                               _ts_iso(ts_ms_1h), f"引擎保护触发: {exit_reason}")
            rec = self.executor.execute(strat, sym, close_sig, price,
                                        _ts_iso(ts_ms_1h))
            eq = rec.get("equity_after", equity)
            side_after = strat.position.side.value if strat.position else "none"
            logger.warning(f"[{sym}] {exit_reason.upper()} 触发强制离场 "
                           f"@{price:.2f} equity={eq:.2f} "
                           f"executed={rec.get('executed')}")
            report["symbols"][sym] = {
                "price": price, "bias": bias, "signal": close_type.value,
                "position": side_after, "equity": eq, "action": exit_reason,
                "reason": f"引擎保护: {exit_reason}",
            }
            if hasattr(self.executor, "snapshot"):
                self.executor.snapshot(strat, sym, price, eq,
                                       close_type.value, _ts_iso(ts_ms_1h))
            return

        # 4) 日线决策：有新的「已收盘日K」才决策（每天最多一次）
        daily_rows = self._fetch_candles_bar(sym, "1D", max(self.warmup_bars, 60))
        if not daily_rows:
            logger.warning(f"{sym} 取日K失败，本周期仅风控巡检")
            report["symbols"][sym] = {
                "price": price, "bias": bias, "signal": "daily_no_data",
                "position": side, "equity": equity, "action": "skip",
            }
            return
        now_ms = time.time() * 1000.0
        completed = [r for r in daily_rows if r[0] + 86400000 <= now_ms]
        if not completed:
            report["symbols"][sym] = {
                "price": price, "bias": bias, "signal": "daily_wait",
                "position": side, "equity": equity, "action": "skip",
            }
            return
        prev = self._last_ts.get(sym)
        new_bars = [r for r in completed if prev is None or r[0] > prev]
        if not new_bars:
            # 日K未收盘：只做风控巡检 + 快照（这就是与 1H 节奏的本质区别）
            logger.info(f"[{sym}] price={price:.2f} 日K未收盘，仅风控巡检 "
                        f"side={side} equity={equity:.2f}")
            report["symbols"][sym] = {
                "price": price, "bias": bias, "signal": "daily_wait",
                "position": side, "equity": equity, "action": "skip",
            }
            if hasattr(self.executor, "snapshot"):
                self.executor.snapshot(strat, sym, price, equity,
                                       "daily_wait", _ts_iso(ts_ms_1h))
            return

        # 喂入全部新收盘的日K（重启/停机后的补喂），决策用最新一根
        # ADX：按回测引擎语义，用「截至该日K」的日K序列计算
        adx_series = compute_adx(completed, self.adx_period)
        for r in new_bars:
            self._feed_bar(strat, r)
        self._last_ts[sym] = new_bars[-1][0]
        last_bar = new_bars[-1]
        adx_val = float(adx_series[len(completed) - 1])  # 最新已收盘日K的 ADX

        if bias is not None:
            strat.set_market_bias(bias)

        data = {"price": last_bar[4], "high": last_bar[2], "low": last_bar[3],
                "timestamp": _ts_iso(last_bar[0]), "adx": adx_val}
        signal = strat.generate_signal(data)
        rec = self.executor.execute(strat, sym, signal, price, _ts_iso(ts_ms_1h))
        side = strat.position.side.value if strat.position else "none"
        logger.info(f"[{sym}] 日线决策 price={price:.2f} adx={adx_val:.1f} bias={bias} "
                    f"signal={signal.signal_type.value} side={side} "
                    f"equity={rec.get('equity_after', equity):.2f} | {signal.reason}")
        report["symbols"][sym] = {
            "price": price, "bias": bias, "adx": round(adx_val, 2),
            "signal": signal.signal_type.value,
            "position": side,
            "equity": rec.get("equity_after", equity),
            "action": rec.get("action"),
            "reason": signal.reason,
        }
        if hasattr(self.executor, "snapshot"):
            self.executor.snapshot(strat, sym, price,
                                   rec.get("equity_after", equity),
                                   signal.signal_type.value, _ts_iso(ts_ms_1h))

    # ---------- 运行模式 ----------
    def run_once(self) -> Dict:
        """单次运行（测试用）。"""
        return self.tick()

    def run_forever(self):
        """小时级无限循环。Ctrl+C 退出。

        全局异常保护：单个周期出错（网络抖动、接口异常等）只记日志继续，
        不让守护进程退出——否则一次意外异常就把模拟盘杀掉。
        """
        logger.info(f"启动小时级循环，间隔 {self.interval}s（{self.bar}）")
        try:
            while True:
                try:
                    report = self.tick()
                    self._dump(report)
                except Exception:
                    logger.exception("本周期全局异常，跳过并继续下一周期")
                logger.info(f"休眠 {self.interval}s 至下一周期…")
                time.sleep(self.interval)
        except KeyboardInterrupt:
            logger.info("收到中断，停止调度器。")

    def _dump(self, report: Dict):
        """把周期报告落盘（覆盖式，供外部看板/告警读取）。"""
        if not self.report_path:
            return
        try:
            d = os.path.dirname(self.report_path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"报告落盘失败: {e}")


def setup_logging(log_file: str = "logs/live_monitor.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
    )
