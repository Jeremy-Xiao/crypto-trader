"""
动态策略切换元策略（MetaStrategy）

设计目标：在震荡市与趋势市中都能取得较好收益 —— 通过「在线集成（ensemble）+
市场状态门控（regime gating）+ 表现加权（performance weighting）」实时切换专家策略。

核心机制
--------
1. 并行运行 4 个专家（DoubleMA / MACDCross / Breakout 为趋势类，RSIBollinger 为均值回归类）。
2. 每个专家维护自己的「虚拟仓位」（vdir: -1/0/+1），通过状态机独立判定其当前想要的方向，
   与元策略的真实仓位解耦（专家不拥有真实仓位，只表达观点）。
3. 每个专家维护「虚拟净值」：每当其 vdir 变化，结算上一笔虚拟交易的收益率，
   用 EWMA 累积成表现评分 perf_i（在线学习，谁最近管用就给更高权重）。
4. 市场状态（trending / ranging）由 ADX + 波动率分位判定：
   - 趋势市：趋势类专家门控=1，均值回归类门控=gate_off（如 0.1）
   - 震荡市：反过来
5. 加权投票：
   - long_score  = Σ w_i · max(vdir_i, 0)
   - short_score = Σ w_i · max(-vdir_i, 0)
   - net = long_score - short_score，超过阈值→做多/做空，否则平仓（死区防抖）
6. 多种模式可切换，用于对比研究：
   - "regime"   : 纯市场状态路由（权重=门控，无表现倾斜）
   - "ensemble" : 市场状态门控 + 表现加权（exp(perf_temp·perf_i)）
   - "perf"     : 纯表现加权（无状态门控，完全靠在线学习挑赢家）
   - "adaptive" : 状态路由下取「表现最好的那一个专家」为王者（winner-take-all，最低抖动）

安全：元策略只输出 OPEN_LONG / OPEN_SHORT / CLOSE_LONG / CLOSE_SHORT / HOLD，
真实仓位与风控（ATR止损止盈、强平、回撤熔断）全部由引擎层负责，与单策略完全一致。
"""

from typing import Dict, Optional, List, Tuple
import os
import math
import logging
import numpy as np

from .base import BaseStrategy, Signal, SignalType, PositionSide, MarketRegime
from src.utils.indicators import EMA, MACD, RSI, BollingerBands

logger = logging.getLogger("strategies.meta")


# 默认专家池：(名字, 类别)。类别用于市场状态门控。
#   trend    : 趋势类，趋势市门控=1、震荡市=gate_off
#   meanrev  : 均值回归类，震荡市门控=1、趋势市=gate_off
#   devmom   : 偏离动量类（强势动量：涨得猛继续多/跌得狠继续空）——因子研究第15/16章验证为真 alpha
#   volstate : 波动率状态调节器（平静做多/动荡做空）——因子研究第16章验证 BTC/ETH 稳健
DEFAULT_EXPERTS = [
    ("DoubleMA", "trend"),
    ("MACDCross", "trend"),
    ("Breakout", "trend"),
    ("RSIBollinger", "meanrev"),
    ("DevMomentum", "devmom"),
    ("VolState", "volstate"),
]


class MetaStrategy(BaseStrategy):
    """动态策略切换元策略"""

    applicable_regime = "both"

    def __init__(
        self,
        instId: str,
        experts: Optional[List[Tuple[str, str]]] = None,
        mode: str = "adaptive",
        params: Optional[Dict] = None,
        allow_short: bool = True,
        market_bias: Optional[str] = None,  # 市场状态过滤器：'risk_off'/'risk_on'/'neutral'/None（接 monitor 的 trading_bias）
        # —— 元策略可调参数（用于多轮回测寻优）——
        gate_off: float = 0.3,        # 非当前状态专家的门控权重（0.3=轻度倾斜，不彻底关闭）
        perf_decay: float = 0.92,     # 表现 EWMA 衰减（0.92≈半衰期 8 根）
        perf_temp: float = 25.0,      # 表现→权重温度（越大越偏向近期赢家）
        vote_threshold: float = 0.25, # 投票死区（占权重总和的比例）
        adx_trend: float = 20.0,     # ADX 趋势阈值（扫描最优:20.0 比 25.0 提升最弱币 SOL 稳健性）
        adx_range: float = 15.0,     # ADX 震荡阈值（低于此才视为震荡，避免误杀温和趋势）
        perf_floor: float = -0.15,    # 表现评分钳制下限（防权重爆炸为负）
        perf_cap: float = 0.20,       # 表现评分钳制上限
        breakout_filtered: bool = True,  # Breakout 专家是否做趋势过滤（False=原始突破，更激进）
        # —— 波动率目标化仓位（vol-managed exposure, Moreira & Muir 2017；2025 Finance Research Letters 加密实证夏普↑）——
        # 根据近期已实现波动年率，把仓位缩放至目标波动：波动越高→自动减仓→降回撤、提夏普。
        vol_target_ann: Optional[float] = None,  # 目标年化波动（如 0.6=60%）；None=关闭
        vol_scale_min: float = 0.3,    # 最低仓位比例（再怎么减仓也不低于此，避免空仓踏空）
        # —— 暴跌后回补（rebound）：针对「暴涨暴跌连体」结构的对症机制 ——
        # 现象见 analyze_concentration.py：暴跌后往往暴力反弹，而 risk_on 过滤器只劝退做空、不回补做多，
        # 等于把反弹也放弃。本开关在「恐慌区(risk_on) + 已暴跌 + 止跌」时主动加多，吃反弹。
        rebound_enabled: bool = False,
        rebound_lookback: int = 5,      # 回看几日判断「已暴跌」
        rebound_drop: float = -0.12,    # 累计跌幅 <= 此值（如5日跌超12%）才判定「跌够了」
        rebound_recent_up: bool = True, # 要求最近1根收涨（止跌）才回补，避免接下跌中继的刀
        # —— 减弱 risk_off 对做多的压制（针对『跑输躺平』主病灶：高位清仓踏空主升浪）——
        # risk_off = 人群贪婪/高位。原逻辑一刀切把 LONG 压成 NONE（清仓）。
        # 但币圈动量强，高位之后常更高，清仓=在主升浪最猛时下车、踏空整段。
        # 改为：'flatten'=原行为(清仓) / 'hold'=保留多头不缩放 / 'half'=减仓至 riskoff_scale。
        riskoff_long_mode: str = "half",     # 2026-08-23 起默认减半：A方案实证比 flatten(清仓) 收益/回撤/夏普全面更优
        riskoff_scale: float = 0.5,     # 'half' 模式下的仓位比例（0.5=减半）
        # risk_on = 人群恐惧。第25章熊市复核发现：熊市利润主要来自做空，而压空恰在
        # 最该做空时自废武功（no_filter 熊市双优的原因）。对称处理：
        # 'flatten'=原行为(清空单) / 'half'=空单减半 / 'hold'=保留空单。
        riskon_short_mode: str = "flatten",
    ):
        all_params = {
            "mode": mode,
            "gate_off": gate_off,
            "perf_decay": perf_decay,
            "perf_temp": perf_temp,
            "vote_threshold": vote_threshold,
            "adx_trend": adx_trend,
            "adx_range": adx_range,
            "perf_floor": perf_floor,
            "perf_cap": perf_cap,
            "vol_target_ann": vol_target_ann,
            "vol_scale_min": vol_scale_min,
            "rebound_enabled": rebound_enabled,
            "rebound_lookback": rebound_lookback,
            "rebound_drop": rebound_drop,
            "rebound_recent_up": rebound_recent_up,
            "riskoff_long_mode": riskoff_long_mode,
            "riskoff_scale": riskoff_scale,
            "riskon_short_mode": riskon_short_mode,
        }
        if params:
            all_params.update(params)
        super().__init__(name="Meta-" + mode, instId=instId, params=all_params)

        self.allow_short = allow_short
        self.market_bias = self._parse_bias(market_bias)  # 反向校准护栏（monitor 注入）
        self.mode = mode
        self.experts = experts or DEFAULT_EXPERTS
        self.expert_names = [e[0] for e in self.experts]

        # 各专家基础权重（Breakout 噪声大，刻意压低；趋势类三者权重相当，共识越强越稳）
        self._base_weight = {
            "DoubleMA": 1.0, "MACDCross": 1.0, "Breakout": 0.35, "RSIBollinger": 0.6,
            "DevMomentum": 0.8, "VolState": 0.6,
        }

        # DevMomentum 专家参数（强势动量：偏离均线的方向与强度）
        self._devmom_ema_period = 20
        self._devmom_gap_thr = 0.015   # 价格偏离 20日EMA 超过 ±1.5% 视为动量信号触发
        # VolState 专家参数（波动率状态：vol-of-vol 的分位）
        self._volstate_win = 120       # 分位排名用的 vol-of-vol 历史窗口（天）
        self._volstate_top = 0.75      # 波动率动荡分位阈值 → 做空
        self._volstate_bot = 0.25      # 波动率平静分位阈值 → 做多

        # 每个专家的虚拟方向 / 虚拟开仓价 / 表现评分 / 交易记录
        self.vdir: Dict[str, int] = {n: 0 for n, _ in self.experts}
        self.ventry: Dict[str, Optional[float]] = {n: None for n, _ in self.experts}
        # 多/空表现分开跟踪：很多专家「做多赚、做空亏」（如 MACD 在牛市），
        # 分开后才能「跟它的多、不跟它的空」，这是元策略跑赢单策略的关键。
        self.perf_long: Dict[str, float] = {n: 0.0 for n, _ in self.experts}
        self.perf_short: Dict[str, float] = {n: 0.0 for n, _ in self.experts}
        self.perf: Dict[str, float] = {n: 0.0 for n, _ in self.experts}  # 净表现（用于展示）
        self.expert_trades: Dict[str, int] = {n: 0 for n, _ in self.experts}
        self.expert_wins: Dict[str, int] = {n: 0 for n, _ in self.experts}

        # 各专家内部状态机（用于判定方向，避免依赖真实仓位）
        self._state: Dict[str, Dict] = {n: {} for n, _ in self.experts}

        # 市场状态检测用的波动率缓冲（增量计算，避免 O(n²)）
        self._ret_buf: List[float] = []     # 近期日收益
        self._vol_buf: List[float] = []     # 近期滚动波动率
        self._regime = MarketRegime.UNKNOWN

        # RSI 参数（均值回归专家）
        self._rsi_period = 14
        self._boll_period = 20
        self._boll_std = 2.0
        self._rsi_low = 30
        self._rsi_high = 70

        # Breakout 参数
        self._breakout_lookback = 20
        self._breakout_filtered = bool(self.params.get("breakout_filtered", True))

        # 指标缓冲（用于交叉判定需要前一根值）
        self._ema_fast_prev = None
        self._ema_slow_prev = None
        self._macd_prev = None
        self._macd_sig_prev = None

        # 统计
        self.switch_count = 0
        self.last_target = PositionSide.NONE
        self._dbg_targets = []  # 调试用：记录每根K线的目标方向

        # 波动率目标化仓位：每根K线动态计算的仓位缩放（0~1），默认满仓
        self._vol_scale = 1.0
        self._realized_vol_ann = None  # 调试/展示用

        # 新专家状态缓冲
        self._devmom_buf: List[float] = []   # 历史 ema_gap_20 值（用于分位判定，避免硬编码阈值）
        self._vov_history: List[float] = []  # 历史 vol_of_vol 值（用于分位判定）
        self._bb_width_buf: List[float] = []  # 布林带宽历史（BollBreak 专家用，判波动释放）

    # ==================== 指标计算 ====================

    def _ema(self, period: int) -> Optional[float]:
        if len(self.price_history) < period + 1:
            return None
        return float(EMA(self.price_history, period).iloc[-1])

    def _macd(self) -> Optional[Tuple[float, float]]:
        if len(self.price_history) < 26 + 9 + 2:
            return None
        m = MACD(self.price_history, 12, 26, 9)
        return float(m["macd"].iloc[-1]), float(m["signal"].iloc[-1])

    def _rsi_boll(self) -> Optional[Tuple[float, float, float, float]]:
        if len(self.price_history) < max(self._rsi_period, self._boll_period) + 2:
            return None
        rsi = float(RSI(self.price_history, self._rsi_period).iloc[-1])
        bb = BollingerBands(self.price_history, self._boll_period, self._boll_std)
        upper = float(bb["upper"].iloc[-1])
        middle = float(bb["middle"].iloc[-1])
        lower = float(bb["lower"].iloc[-1])
        return rsi, upper, middle, lower

    def _breakout_levels(self) -> Tuple[Optional[float], Optional[float]]:
        lb = self._breakout_lookback
        if len(self.price_history) < lb + 2:
            return None, None
        window = self.price_history[-lb - 1:-1]
        return max(window), min(window)

    # ==================== 市场状态 ====================

    def _detect_regime(self, adx: float) -> MarketRegime:
        """趋势/震荡判定：ADX 阈值 + 波动率分位（增量计算）"""
        prices = self.price_history
        n = len(prices)
        if n >= 2:
            ret = prices[-1] / prices[-2] - 1.0
            self._ret_buf.append(ret)
            if len(self._ret_buf) > 60:
                self._ret_buf.pop(0)
        if len(self._ret_buf) >= 20:
            vol = float(__import__("numpy").std(self._ret_buf[-20:]))
            self._vol_buf.append(vol)
            if len(self._vol_buf) > 60:
                self._vol_buf.pop(0)

        vol_pct = 0.5
        if len(self._vol_buf) >= 2:
            recent = self._vol_buf[-1]
            hist = [v for v in self._vol_buf[:-1] if v == v]
            if hist:
                vol_pct = float(sum(1 for v in hist if v <= recent)) / len(hist)

        if adx >= self.params["adx_trend"]:
            regime = MarketRegime.TRENDING
        elif adx <= self.params["adx_range"]:
            regime = MarketRegime.RANGING
        else:
            regime = MarketRegime.TRENDING if vol_pct > 0.6 else MarketRegime.RANGING
        self._regime = regime
        return regime

    # ==================== 各专家方向 ====================

    def _expert_side(self, name: str, kind: str, price: float) -> int:
        """返回某专家当前想要的方向（-1/0/+1），由其内部状态机决定"""
        if kind == "trend":
            if name == "DoubleMA":
                return self._side_double_ma(price)
            if name == "MACDCross":
                return self._side_macd(price)
            if name == "Breakout":
                return self._side_breakout(price)
        elif kind == "meanrev":
            if name == "RSIBollinger":
                return self._side_rsi_boll(price)
        elif kind == "devmom":
            if name == "DevMomentum":
                return self._side_dev_momentum(price)
        elif kind == "volstate":
            if name == "VolState":
                return self._side_vol_state(price)
        elif kind == "bollbreak":
            if name == "BollBreak":
                return self._side_boll_break(price)
        return 0

    def _side_double_ma(self, price: float) -> int:
        fast = self._ema(10)
        slow = self._ema(30)
        if fast is None or slow is None:
            return 0
        prev_f = self._ema_fast_prev
        prev_s = self._ema_slow_prev
        self._ema_fast_prev = fast
        self._ema_slow_prev = slow
        if prev_f is None or prev_s is None:
            return self.vdir["DoubleMA"]  # 预热期保持原方向
        golden = prev_f <= prev_s and fast > slow
        death = prev_f >= prev_s and fast < slow
        if golden:
            return 1
        if death:
            return -1
        return self.vdir["DoubleMA"]  # 两交叉之间保持方向

    def _side_macd(self, price: float) -> int:
        m = self._macd()
        if m is None:
            return 0
        macd_line, signal_line = m
        prev_m = self._macd_prev
        prev_s = self._macd_sig_prev
        self._macd_prev = macd_line
        self._macd_sig_prev = signal_line
        if prev_m is None or prev_s is None:
            return self.vdir["MACDCross"]
        golden = prev_m <= prev_s and macd_line > signal_line
        death = prev_m >= prev_s and macd_line < signal_line
        if golden:
            return 1
        if death:
            return -1
        return self.vdir["MACDCross"]

    def _side_breakout(self, price: float) -> int:
        hi, lo = self._breakout_levels()
        if hi is None or lo is None:
            return 0
        if not self._breakout_filtered:
            # 原始突破：只要突破 N 日高低就确认方向（更激进，可能在区间内被扇耳光）
            if price > hi:
                return 1
            if price < lo:
                return -1
            return self.vdir["Breakout"]
        # 趋势过滤：只有突破方向与长期趋势一致时才确认，避免在区间内被反向突破噪音反复扇耳光
        trend_ema = self._ema(60)
        if trend_ema is not None:
            if price > hi and price > trend_ema:
                return 1
            if price < lo and price < trend_ema:
                return -1
        else:
            if price > hi:
                return 1
            if price < lo:
                return -1
        return self.vdir["Breakout"]

    def _side_rsi_boll(self, price: float) -> int:
        r = self._rsi_boll()
        if r is None:
            return 0
        rsi, upper, middle, lower = r
        st = self._state["RSIBollinger"]
        pos = st.get("pos", 0)
        if pos == 0:
            if rsi < self._rsi_low and price <= lower:
                pos = 1
            elif self.allow_short and rsi > self._rsi_high and price >= upper:
                pos = -1
        else:
            if pos == 1 and (rsi > self._rsi_high or price >= upper or price >= middle):
                pos = 0
            elif pos == -1 and (rsi < self._rsi_low or price <= lower or price <= middle):
                pos = 0
        st["pos"] = pos
        return pos

    def _side_dev_momentum(self, price: float) -> int:
        """强势动量专家（因子研究第15/16章验证的「真 alpha」）。

        关键修正：旧 RSIBollinger 用「超卖抄底」(均值回归) 亏损 -22%；
        因子研究证明 ema_gap_20 / rsi_14 的赚钱规则是**追强势**
        （价格显著高于均线 / RSI 高 → 继续多；显著低于 / RSI 低 → 继续空），
        属动量延续，与双均线/MACD 同源但用「偏离度+RSI」表达，更干净。
        """
        ema = self._ema(self._devmom_ema_period)
        if ema is None or ema <= 0:
            return 0
        gap = price / ema - 1.0  # ema_gap_20
        rsi = self._rsi_level()
        if rsi is None:
            return 0
        # 记录历史偏离，用于分位判定（避免固定阈值在特定行情失效）
        self._devmom_buf.append(gap)
        if len(self._devmom_buf) > self._volstate_win:
            self._devmom_buf.pop(0)
        st = self._state.setdefault("DevMomentum", {"pos": 0})
        pos = st.get("pos", 0)
        if pos == 0:
            # 追强：偏离为正且 RSI 偏强 → 多；偏离为负且 RSI 偏弱 → 空
            if gap > self._devmom_gap_thr and rsi > 50:
                pos = 1
            elif gap < -self._devmom_gap_thr and rsi < 50:
                pos = -1
        else:
            # 动量熄火即退出：多单在价格回落至均线下方时平，空单在回升至均线上方时平
            if pos == 1 and gap <= 0:
                pos = 0
            elif pos == -1 and gap >= 0:
                pos = 0
        st["pos"] = pos
        return pos

    def _side_boll_break(self, price: float) -> int:
        """布林带波动率突破专家（BollBreak）。

        Gate 研究院(2025)回测结论：在「区间震荡+波动扩张」阶段，布林带突破型动量
        明显优于 MACD/RSI（后者在弱趋势中大量假信号）。本专家捕捉「波动收敛后释放」
        的动量延续：
          - 仅当带宽处于释放状态（当前带宽 ≥ 近 20 日中位，排除死水区假突破）才参与；
          - 价格放量突破上轨 + RSI>50 → 做多；跌破下轨 + RSI<50 → 做空；
          - 价格回落至中轨即退出（动量熄火）。
        与现有 Donchian 突破(Breakout)互补：Breakout 看 N 日高低点，BollBreak 看
        波动率释放，触发条件不同、错位盈利。
        """
        r = self._rsi_boll()
        if r is None:
            return 0
        rsi, upper, middle, lower = r
        width = (upper - lower) / middle if middle > 0 else 0.0
        self._bb_width_buf.append(width)
        if len(self._bb_width_buf) > 60:
            self._bb_width_buf.pop(0)
        st = self._state.setdefault("BollBreak", {"pos": 0})
        pos = st.get("pos", 0)
        # 波动释放判定：带宽需高于自身近期中位，避免平静死水区的无意义突破
        if len(self._bb_width_buf) >= 10:
            med = float(np.median(self._bb_width_buf[-20:]))
            vol_release = width >= med
        else:
            vol_release = True
        if pos == 0:
            if vol_release and price > upper and rsi > 50:
                pos = 1
            elif vol_release and price < lower and rsi < 50:
                pos = -1
        else:
            if pos == 1 and price < middle:
                pos = 0
            elif pos == -1 and price > middle:
                pos = 0
        st["pos"] = pos
        return pos

    def _rsi_level(self) -> Optional[float]:
        if len(self.price_history) < self._rsi_period + 2:
            return None
        return float(RSI(self.price_history, self._rsi_period).iloc[-1])

    def _side_vol_state(self, price: float) -> int:
        """波动率状态调节器（因子研究第16章：vol_of_vol_60）。

        逻辑：波动率自身「平静」→ 做多；波动率「动荡」(vol-of-vol 高) → 做空。
        加密币规律：平静期阴涨、动荡期易暴跌，做多平静/做空动荡天然赚钱。
        BTC/ETH 稳健（前后半段都赚）；SOL 后半段失效，靠元策略表现加权自动降权。
        复用 _vol_buf（已是 20日滚动波动率序列）算 vol_of_vol，零额外计算。
        """
        if len(self._vol_buf) < 30:
            return 0
        vov = float(np.std(self._vol_buf[-60:]))  # vol_of_vol_60
        if not (vov == vov) or vov <= 0:
            return 0
        self._vov_history.append(vov)
        if len(self._vov_history) > self._volstate_win:
            self._vov_history.pop(0)
        st = self._state.setdefault("VolState", {"pos": 0})
        pos = st.get("pos", 0)
        if len(self._vov_history) < 20:
            return pos  # 预热：维持原方向但不新开仓
        # 当前 vol_of_vol 在历史窗口中的分位
        hist = np.array(self._vov_history[:-1]) if len(self._vov_history) > 1 else np.array(self._vov_history)
        pct = float((hist <= vov).mean())
        if pos == 0:
            if pct < self._volstate_bot:
                pos = 1   # 平静 → 做多
            elif pct > self._volstate_top:
                pos = -1  # 动荡 → 做空
        else:
            if pos == 1 and pct > 0.5:
                pos = 0
            elif pos == -1 and pct < 0.5:
                pos = 0
        st["pos"] = pos
        return pos

    # ==================== 表现跟踪 ====================

    def _update_perf(self, name: str, new_dir: int, price: float):
        """虚拟仓位变化时结算上一笔虚拟交易，分别更新「做多表现」与「做空表现」EWMA"""
        old = self.vdir[name]
        if old != 0 and self.ventry[name] is not None and old != new_dir:
            entry = self.ventry[name]
            # 虚拟收益率：多头 (price/entry-1)，空头 (entry/price-1)
            if old == 1:
                r = price / entry - 1.0
            else:
                r = entry / price - 1.0
            self.expert_trades[name] += 1
            if r > 0:
                self.expert_wins[name] += 1
            decay = self.params["perf_decay"]
            floor, cap = self.params["perf_floor"], self.params["perf_cap"]
            if old == 1:
                self.perf_long[name] = decay * self.perf_long[name] + (1 - decay) * r
                self.perf_long[name] = max(floor, min(cap, self.perf_long[name]))
            else:
                self.perf_short[name] = decay * self.perf_short[name] + (1 - decay) * r
                self.perf_short[name] = max(floor, min(cap, self.perf_short[name]))
            # 净表现（展示用）
            self.perf[name] = decay * self.perf[name] + (1 - decay) * r
            self.perf[name] = max(floor, min(cap, self.perf[name]))
        if new_dir != 0 and old != new_dir:
            self.ventry[name] = price  # 新开虚拟仓
        elif new_dir == 0:
            self.ventry[name] = None
        self.vdir[name] = new_dir

    # ==================== 聚合 ====================

    def generate_signal(self, data: Dict) -> Signal:
        price = data.get("price", 0)
        timestamp = data.get("timestamp", "")
        adx = float(data.get("adx", 0) or 0)

        if len(self.price_history) < 35:
            return Signal(SignalType.HOLD, self.instId, price, 0, timestamp, "预热")

        regime = self._detect_regime(adx)
        trending = (regime == MarketRegime.TRENDING)
        ranging = (regime == MarketRegime.RANGING)

        # 1) 更新各专家方向与表现
        long_raw = 0.0
        short_raw = 0.0
        for name, kind in self.experts:
            new_dir = self._expert_side(name, kind, price)
            self._update_perf(name, new_dir, price)

            # 门控
            if kind in ("trend", "devmom", "bollbreak"):
                # 趋势类 & 偏离动量类 & 布林突破：趋势市全开，震荡市降权（动量在趋势中更有效）
                gate = 1.0 if (trending or regime == MarketRegime.UNKNOWN) else self.params["gate_off"]
            elif kind == "volstate":
                # 波动率状态调节器：与市况正交，始终参与投票
                gate = 1.0
            else:  # meanrev
                gate = 1.0 if (ranging or regime == MarketRegime.UNKNOWN) else self.params["gate_off"]

            perf_long = self.perf_long[name]
            perf_short = self.perf_short[name]
            bw = self._base_weight.get(name, 1.0)
            if self.mode == "regime":
                w_long = bw * gate
                w_short = bw * gate
            elif self.mode == "perf":
                w_long = bw * math.exp(self.params["perf_temp"] * perf_long)
                w_short = bw * math.exp(self.params["perf_temp"] * perf_short)
            elif self.mode in ("ensemble", "adaptive"):
                w_long = bw * gate * math.exp(self.params["perf_temp"] * perf_long)
                w_short = bw * gate * math.exp(self.params["perf_temp"] * perf_short)
            else:
                w_long = bw * gate
                w_short = bw * gate

            if new_dir > 0:
                long_raw += w_long
            elif new_dir < 0:
                short_raw += w_short

        # 2) 决策（max-wins + 滞后，避免「多空抵消→频繁平仓」的抖动）
        #    long_raw / short_raw 分别是多方/空方的加权意愿之和（不相减，避免相互抵消）。
        #    持仓时用更高的「退出阈值」(exit_thr) 才反向，未持仓时用较低的「入场阈值」(enter_thr)，
        #    形成滞后：一旦上车就持仓，直到反方明确占优才下车，大幅降低抖动。
        current = self.position.side if self.position else PositionSide.NONE

        if self.mode == "adaptive":
            target = self._decide_adaptive()
        else:
            total_w = long_raw + short_raw + 1e-9
            enter_thr = self.params["vote_threshold"] * total_w
            exit_thr = self.params["vote_threshold"] * total_w * 2.0

            if current == PositionSide.LONG:
                # 持多：仅当空方明确占优且超过退出阈值才翻空，否则继续持有
                if short_raw > long_raw and short_raw > exit_thr:
                    target = PositionSide.SHORT
                else:
                    target = PositionSide.LONG
            elif current == PositionSide.SHORT:
                if long_raw > short_raw and long_raw > exit_thr:
                    target = PositionSide.LONG
                else:
                    target = PositionSide.SHORT
            else:  # 空仓：多方占优且过阈值→做多，空方占优且过阈值→做空
                if long_raw > short_raw and long_raw > enter_thr:
                    target = PositionSide.LONG
                elif short_raw > long_raw and short_raw > enter_thr:
                    target = PositionSide.SHORT
                else:
                    target = PositionSide.NONE

        # 2.5) 市场状态过滤（接 src/monitor 的 trading_bias，反向校准护栏）
        self._bias_scale = 1.0   # 每根重置；仅 risk_off+做多时按需下调
        bias = self._resolve_bias(data)
        target = self._apply_market_filter(target, current, bias)

        # 2.55) 暴跌后回补：恐慌区 + 已暴跌 + 止跌 → 主动加多，吃反弹
        target = self._maybe_rebound(target, bias)

        # 2.58) 仅做多模式（allow_short=False）
        target = self._apply_allow_short(target, current)

        # 2.6) 波动率目标化仓位缩放（vol-managed exposure）
        self._update_vol_scale()

        # 3) 映射为目标信号
        signal = self._map_target(target, current, price, timestamp,
                                  f"regime={regime.value} long={long_raw:.2f} short={short_raw:.2f}")
        if target != self.last_target and target != current:
            self.switch_count += 1
        self.last_target = target

        if self.position:
            self.position.update_price(price)
        if os.environ.get("META_DEBUG"):
            self._dbg_targets.append(target.value)
        return signal

    # ==================== 市场状态过滤器（接 monitor.trading_bias） ====================

    def _parse_bias(self, bias):
        """把各类输入（None / 字符串 / RegimeBias 枚举）归一为内部字符串。

        返回 'risk_off' / 'risk_on' / 'neutral' / None。与 src.monitor.base.RegimeBias
        的 .value（'risk_off'/'risk_on'/'neutral'）同源，可无缝对接。
        """
        if bias is None:
            return None
        if isinstance(bias, str):
            b = bias.strip().lower()
            if b in ("risk_off", "riskoff", "off"):
                return "risk_off"
            if b in ("risk_on", "rison", "on"):
                return "risk_on"
            return "neutral"
        # 传入 RegimeBias 枚举：取 .value 或 str 再判定
        name = getattr(bias, "value", None) or str(bias)
        name = str(name).lower()
        if "off" in name:
            return "risk_off"
        if "on" in name:
            return "risk_on"
        return "neutral"

    def set_market_bias(self, bias):
        """实盘接口：每根 K 拉取 monitor 快照后调用，注入当前市场状态。"""
        self.market_bias = self._parse_bias(bias)

    def _resolve_bias(self, data: Dict) -> str:
        """优先用回测回放的 data['market_bias']（逐根序列），否则用实例级 self.market_bias。"""
        if isinstance(data, dict):
            b = data.get("market_bias")
            if b is not None and not (isinstance(b, str) and b.strip() == ""):
                return self._parse_bias(b)
        return self.market_bias

    def _apply_allow_short(self, target, current):
        """仅做多模式（allow_short=False）的方向拦截。

        此前该开关只在个别专家内部生效，元决策层没有拦截，
        导致「仅做多」对照回测实际上仍在做空、结论失真（2026-09-28 修复）。

        语义：不建立空头；若已持空则平掉；若持多则维持不动。
        """
        if self.allow_short or target != PositionSide.SHORT:
            return target
        return current if current == PositionSide.LONG else PositionSide.NONE

    def _apply_market_filter(self, target, current, bias):
        """反向校准护栏（呼应『情绪用来校准风险，不是精准择时』）：

        - RISK_OFF（人群贪婪/多头拥挤）→ 抑制做多：把 LONG 目标降为 NONE（平仓/不开多），
                  保留 SHORT（做空正是对冲拥挤多头）。
        - RISK_ON（人群恐惧/空头拥挤）→ 抑制做空：把 SHORT 目标降为 NONE，保留 LONG（机会区可偏多）。
        - neutral / None → 不动。
        """
        if bias is None or bias == "neutral":
            return target
        if bias == "risk_off":
            if target == PositionSide.LONG:
                mode = self.params.get("riskoff_long_mode", "flatten")
                if mode == "flatten":
                    return PositionSide.NONE          # 原行为：清仓
                if mode == "half":
                    self._bias_scale = self.params.get("riskoff_scale", 0.5)
                    return PositionSide.LONG           # 减仓至 riskoff_scale，保留多头
                return PositionSide.LONG               # 'hold'：保留多头不缩放
            return target
        if bias == "risk_on":
            if target == PositionSide.SHORT:
                mode = self.params.get("riskon_short_mode", "flatten")
                if mode == "half":
                    self._bias_scale = self.params.get("riskoff_scale", 0.5)
                    return PositionSide.SHORT           # 空单减半（恐惧时留对冲）
                if mode == "hold":
                    return PositionSide.SHORT
                return PositionSide.NONE                # 原行为：清空单
            return target
        return target

    def _recent_cum_return(self, lb: int) -> float:
        """最近 lb 根 K 线的累计收益率（用 price_history 收盘价）。"""
        if len(self.price_history) < lb + 1:
            return 0.0
        p0 = self.price_history[-lb - 1]
        p1 = self.price_history[-1]
        if p0 <= 0:
            return 0.0
        return p1 / p0 - 1.0

    def _maybe_rebound(self, target, bias):
        """暴跌后回补：针对『暴涨暴跌连体』的对症机制。

        - 只在恐慌区（bias=='risk_on'，即人群恐惧/空头拥挤，通常对应刚经历暴跌）考虑。
        - 确认「已暴跌」：近期累计跌幅 <= rebound_drop（跌够了，反弹概率上升）。
        - 止跌确认（rebound_recent_up）：最近 1 根收涨，才回补；避免在下跌中继接刀。
        - 触发后把目标改为 LONG（吃反弹），无论之前是被压成 NONE 还是本就空仓。
        """
        if not self.params.get("rebound_enabled"):
            return target
        if bias != "risk_on":
            return target
        if target == PositionSide.LONG:
            return target  # 已有多头，不重复
        # 确认已暴跌
        drop = self._recent_cum_return(int(self.params.get("rebound_lookback", 5)))
        if drop > self.params.get("rebound_drop", -0.12):
            return target  # 还没跌够，不急
        # 止跌确认：最近一根收涨
        if self.params.get("rebound_recent_up", True):
            if len(self.price_history) >= 2 and self.price_history[-1] <= self.price_history[-2]:
                return target  # 仍在新低，不接刀
        return PositionSide.LONG

    def _map_target(self, target, current, price, timestamp, info="") -> Signal:
        """将目标方向映射为 engine 可执行的 Signal。

        - 目标做多 & 空仓 → 开多；& 持空 → 翻转开多
        - 目标做空 & 空仓 → 开空；& 持多 → 翻转开空
        - 目标平仓 & 持多/持空 → 对应平仓信号
        """
        signal = Signal(SignalType.HOLD, self.instId, price, 0, timestamp,
                        f"元策略 {info} target={target.value}")
        if target == PositionSide.LONG:
            if current == PositionSide.NONE:
                signal.signal_type = SignalType.OPEN_LONG
                signal.amount = self._atr_sized_amount(price)
                signal.reason = "元策略: 共识做多"
            elif current == PositionSide.SHORT:
                # 平空开多。必须发 OPEN_LONG：引擎对「信号方向 == 当前持仓方向」的
                # 处理是「已持有同向 → 忽略」，若这里误发 OPEN_SHORT，target 会被解析成
                # SHORT 并命中忽略分支，导致**空翻多永远不执行**。
                # （2026-09-28 修复：此前误发 OPEN_SHORT，回测与实盘的空翻多全部被吞掉。）
                signal.signal_type = SignalType.OPEN_LONG
                signal.amount = self._atr_sized_amount(price)
                signal.reason = "元策略: 翻转做多"
        elif target == PositionSide.SHORT:
            if current == PositionSide.NONE:
                signal.signal_type = SignalType.OPEN_SHORT
                signal.amount = self._atr_sized_amount(price)
                signal.reason = "元策略: 共识做空"
            elif current == PositionSide.LONG:
                signal.signal_type = SignalType.OPEN_SHORT  # 引擎自动翻转
                signal.amount = self._atr_sized_amount(price)
                signal.reason = "元策略: 翻转做空"
        else:  # FLAT
            if current == PositionSide.LONG:
                signal.signal_type = SignalType.CLOSE_LONG
                signal.amount = self.position.amount
                signal.reason = "元策略: 平仓观望"
            elif current == PositionSide.SHORT:
                signal.signal_type = SignalType.CLOSE_SHORT
                signal.amount = self.position.amount
                signal.reason = "元策略: 平仓观望"
        return signal

    def _decide_adaptive(self) -> PositionSide:
        """winner-take-all：在「当前状态允许」的专家里挑「该方向表现最好且为正」的那个"""
        best_name = None
        best_score = -1e9
        best_dir = 0
        for name, kind in self.experts:
            if self._regime == MarketRegime.TRENDING and kind == "meanrev":
                continue
            if self._regime == MarketRegime.RANGING and kind == "trend":
                continue
            # devmom 视为动量类，震荡市同样降权（此处仅作路由，门控已在 generate_signal 处理）
            # volstate 不受状态路由限制，两种市况都参与王者评选
            d = self.vdir[name]
            if d == 0:
                continue
            # 按「该方向」的表现评分：做多选 perf_long，做空选 perf_short
            score = self.perf_long[name] if d > 0 else self.perf_short[name]
            if score > best_score:
                best_score = score
                best_name = name
                best_dir = d
        if best_name is None or best_dir == 0:
            return PositionSide.NONE
        # 只跟随该方向表现为正的专家；否则空仓观望（保全本金）
        if best_score <= 0.0:
            return PositionSide.NONE
        return PositionSide.LONG if best_dir > 0 else PositionSide.SHORT

    def _atr_sized_amount(self, price: float) -> float:
        """复刻回测引擎 use_atr_risk=True 的开仓量（engine.py:379-391）。

        回测公式：仓位 = min( (equity×2%)/(ATR×2.0), equity×max_position_pct×leverage )
        其中 max_position_pct×leverage = 1.0×1.0 = 1.0（build_engine_config 实际值）。
        ATR 公式本身通常给出 20%~60% 权益的仓位；1.0 只是极端兜底。
        实盘调度器没有引擎这一层，直接用 signal.amount 会变成无条件 100% 满仓。
        MetaStrategy.calculate_atr_position_size 已自动乘 _vol_scale/_bias_scale。
        """
        atr = self.get_atr(14)
        if atr <= 0:
            logger.warning("ATR 不可用（预热不足），本次开仓退回全仓计算——实盘不应出现，请检查预热")
            return self.calculate_position_size(self.account_balance, price)
        amt = self.calculate_atr_position_size(
            self.account_balance, price, atr,
            risk_pct=0.02, atr_multiplier=2.0, max_pct=1.0)
        return min(amt, self.account_balance / price)  # 名义约束（回测约束2，1x）

    def calculate_position_size(self, account_balance: float, price: float) -> float:
        position_pct = self.params.get("position_pct", 1.0)
        # 波动率目标化 + 情绪调节：把基础仓位比例再乘上动态缩放（高波动→降仓；risk_off+做多→减仓）
        position_pct = position_pct * self._vol_scale * self._bias_scale
        return (account_balance * position_pct) / price

    def calculate_atr_position_size(self, account_balance, price, atr_val,
                                   risk_pct=0.02, atr_multiplier=2.0, max_pct=0.5) -> float:
        """覆盖基类：在 ATR 动态仓位基础上叠加波动率目标化缩放 + 情绪调节缩放。

        关键修复：引擎开启 use_atr_risk 时，开仓金额由本方法重算并『丢弃』signal.amount，
        若不在此乘上 _vol_scale / _bias_scale，波动率目标化/情绪减仓就永远是个空操作。
        """
        base_amt = super().calculate_atr_position_size(
            account_balance, price, atr_val, risk_pct, atr_multiplier, max_pct)
        return base_amt * self._vol_scale * self._bias_scale

    def _update_vol_scale(self) -> None:
        """波动率目标化仓位（vol-managed exposure）。

        用近期 20 日滚动日收益 std（已在 _vol_buf 维护）年化，缩放 = 目标波动/已实现波动，
        并钳制到 [vol_scale_min, 1.0]（只减仓不爆仓、最低保留底仓）。关闭时(self._vol_scale=1)。
        依据：Moreira & Muir (2017) Volatility-Managed Portfolios；2025 Finance Research Letters
        加密实证——逆波动加权把夏普 1.12→1.42。
        """
        vt = self.params.get("vol_target_ann", None)
        if vt is None or vt <= 0 or len(self._vol_buf) < 20:
            self._vol_scale = 1.0
            return
        daily_vol = float(self._vol_buf[-1])
        if daily_vol <= 0 or not (daily_vol == daily_vol):
            self._vol_scale = 1.0
            return
        realized_ann = daily_vol * math.sqrt(365.0)
        self._realized_vol_ann = realized_ann
        scale = vt / realized_ann
        scale = min(1.0, max(self.params.get("vol_scale_min", 0.3), scale))
        self._vol_scale = scale

    def describe(self) -> str:
        d = super().describe()
        d += f"\n元策略模式: {self.mode}\n专家池: {self.expert_names}\n"
        return d
