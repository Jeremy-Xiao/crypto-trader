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

from .base import BaseStrategy, Signal, SignalType, PositionSide, MarketRegime
from src.utils.indicators import EMA, MACD, RSI, BollingerBands


# 默认专家池：(名字, 类别)。类别用于市场状态门控。
DEFAULT_EXPERTS = [
    ("DoubleMA", "trend"),
    ("MACDCross", "trend"),
    ("Breakout", "trend"),
    ("RSIBollinger", "meanrev"),
]


class MetaStrategy(BaseStrategy):
    """动态策略切换元策略"""

    applicable_regime = "both"

    def __init__(
        self,
        instId: str,
        experts: Optional[List[Tuple[str, str]]] = None,
        mode: str = "ensemble",
        params: Optional[Dict] = None,
        allow_short: bool = True,
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
        }
        if params:
            all_params.update(params)
        super().__init__(name="Meta-" + mode, instId=instId, params=all_params)

        self.allow_short = allow_short
        self.mode = mode
        self.experts = experts or DEFAULT_EXPERTS
        self.expert_names = [e[0] for e in self.experts]

        # 各专家基础权重（Breakout 噪声大，刻意压低；趋势类三者权重相当，共识越强越稳）
        self._base_weight = {
            "DoubleMA": 1.0, "MACDCross": 1.0, "Breakout": 0.35, "RSIBollinger": 0.6
        }

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
            if kind == "trend":
                gate = 1.0 if (trending or regime == MarketRegime.UNKNOWN) else self.params["gate_off"]
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
                signal.amount = self.calculate_position_size(self.account_balance, price)
                signal.reason = "元策略: 共识做多"
            elif current == PositionSide.SHORT:
                signal.signal_type = SignalType.OPEN_SHORT  # 引擎自动翻转
                signal.amount = self.calculate_position_size(self.account_balance, price)
                signal.reason = "元策略: 翻转做多"
        elif target == PositionSide.SHORT:
            if current == PositionSide.NONE:
                signal.signal_type = SignalType.OPEN_SHORT
                signal.amount = self.calculate_position_size(self.account_balance, price)
                signal.reason = "元策略: 共识做空"
            elif current == PositionSide.LONG:
                signal.signal_type = SignalType.OPEN_SHORT  # 引擎自动翻转
                signal.amount = self.calculate_position_size(self.account_balance, price)
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

    def calculate_position_size(self, account_balance: float, price: float) -> float:
        position_pct = self.params.get("position_pct", 1.0)
        return (account_balance * position_pct) / price

    def describe(self) -> str:
        d = super().describe()
        d += f"\n元策略模式: {self.mode}\n专家池: {self.expert_names}\n"
        return d
