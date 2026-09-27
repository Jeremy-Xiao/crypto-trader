"""元策略信号映射回归测试（2026-09-28 两个 bug 的守卫）。

背景（两个都是真 bug，均在生产中生效）：

Bug 1 —— 「空翻多」信号类型错误
    `_map_target` 在 target=LONG 且 current=SHORT 时错误地发出 `OPEN_SHORT`，
    而引擎对「信号方向 == 当前持仓方向」的处理是「已持有同向 → 忽略」，
    于是 target 被解析成 SHORT 并命中忽略分支 → **空翻多永远不会执行**。
    实盘证据（2026-09-28 00:31）：
        元策略日志 "翻转做多" → 发出 signal=open_short
        → okx_paper 打印「交易所已有short仓，跳过重复开仓」，什么都没做。

Bug 2 —— `allow_short=False` 在元决策层失效
    该开关只被个别专家内部检查，元策略最终仍会输出 OPEN_SHORT，
    导致所有「仅做多」对照回测实际上仍在做空，结论失真。

运行：venv/bin/python -m pytest tests/test_meta_signal_mapping.py -v
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.strategies.base import BaseStrategy, PositionSide, Signal, SignalType
from src.strategies.meta import MetaStrategy


def make_meta(allow_short=True, current=PositionSide.NONE, **params):
    """构造已预热、且（可选）已持有指定方向仓位的元策略。"""
    s = MetaStrategy(instId="BTC-USDT", mode="adaptive",
                     params={"riskoff_long_mode": "half", **params},
                     allow_short=allow_short)
    s.price_history = [100.0] * 300
    s.high_history = [101.0] * 300
    s.low_history = [99.0] * 300
    s._bias_scale = 1.0
    if current != PositionSide.NONE:
        s.open_position(100.0, 1.0, "t", current)   # 平仓分支读 self.position.amount
    return s


# ============ Bug 1：方向映射 ============

def test_flat_to_long_emits_open_long():
    s = make_meta()
    sig = s._map_target(PositionSide.LONG, PositionSide.NONE, 100.0, "t")
    assert sig.signal_type == SignalType.OPEN_LONG


def test_short_to_long_must_emit_open_long():
    """空翻多必须发 OPEN_LONG —— 这是 bug 的核心。

    若发 OPEN_SHORT，引擎会解析 target=SHORT 并因「已持有同向」直接忽略，
    空单既不平、多单也不开。
    """
    s = make_meta(current=PositionSide.SHORT)
    sig = s._map_target(PositionSide.LONG, PositionSide.SHORT, 100.0, "t")
    assert sig.signal_type == SignalType.OPEN_LONG, (
        "空翻多发成了 OPEN_SHORT，会被引擎当作『已持有同向』忽略")
    assert "翻转做多" in sig.reason


def test_flat_to_short_emits_open_short():
    s = make_meta()
    sig = s._map_target(PositionSide.SHORT, PositionSide.NONE, 100.0, "t")
    assert sig.signal_type == SignalType.OPEN_SHORT


def test_long_to_short_emits_open_short():
    """多翻空发 OPEN_SHORT，引擎自动翻转（这条本来就是对的）"""
    s = make_meta(current=PositionSide.LONG)
    sig = s._map_target(PositionSide.SHORT, PositionSide.LONG, 100.0, "t")
    assert sig.signal_type == SignalType.OPEN_SHORT
    assert "翻转做空" in sig.reason


def test_flat_target_closes_both_directions():
    s_long = make_meta(current=PositionSide.LONG)
    assert s_long._map_target(PositionSide.NONE, PositionSide.LONG, 100.0, "t").signal_type \
        == SignalType.CLOSE_LONG
    s_short = make_meta(current=PositionSide.SHORT)
    assert s_short._map_target(PositionSide.NONE, PositionSide.SHORT, 100.0, "t").signal_type \
        == SignalType.CLOSE_SHORT


# ============ 集成：引擎确实能完成空翻多 ============

class _ScriptedStrategy(BaseStrategy):
    """按脚本发信号的测试策略：第 n 根执行指定信号。"""

    def __init__(self, script):
        super().__init__(name="scripted", instId="BTC-USDT")
        self.script = script          # {bar_index: (SignalType, coins)}
        self._n = 0

    def generate_signal(self, data) -> Signal:
        n = self._n
        self._n += 1
        st, amt = self.script.get(n, (SignalType.HOLD, 0.0))
        return Signal(signal_type=st, instId=self.instId, price=data["price"],
                      amount=amt, timestamp=str(data["timestamp"]))

    def calculate_position_size(self, account_balance, price):
        return 1.0


def _df(prices):
    ts = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({"timestamp": ts, "open": prices,
                         "high": [p * 1.001 for p in prices],
                         "low": [p * 0.999 for p in prices],
                         "close": prices, "volume": [1.0] * len(prices)})


def test_engine_flips_short_to_long_on_open_long():
    """引擎收到 OPEN_LONG 且当前持空 → 必须先平空再开多（reverse）"""
    script = {0: (SignalType.OPEN_SHORT, 1.0), 3: (SignalType.OPEN_LONG, 1.0)}
    e = BacktestEngine(_ScriptedStrategy(script), BacktestConfig(
        initial_balance=10000, fee_rate=0.0, slippage=0.0,
        use_atr_risk=False, use_trailing=False, use_mtf=False, use_regime=False,
        leverage=1.0, max_position_pct=1.0, min_adx_for_entry=0.0))
    e.load_data(_df([100.0] * 6))
    e.run()
    # 注：回测结束会强制平仓，故最后 position_side 必为 NONE，改看成交序列
    assert e.stop_stats["reverse"] >= 1, "应记录一次反向翻转"
    sides = [t["side"] for t in e.trades if t.get("side")]
    assert "short_open" in sides, "应先建立空头"
    assert "long_open" in sides, "空翻多后应真的建立多头（修复前这一条会失败）"


# ============ Bug 2：allow_short 开关 ============

def test_allow_short_false_blocks_short_from_flat():
    """仅做多模式：空仓时目标为空 → 不建立空头（走真实实现 _apply_allow_short）"""
    s = make_meta(allow_short=False)
    target = s._apply_allow_short(PositionSide.SHORT, PositionSide.NONE)
    assert target == PositionSide.NONE
    assert s._map_target(target, PositionSide.NONE, 100.0, "t").signal_type \
        == SignalType.HOLD


def test_allow_short_false_closes_existing_short():
    """仅做多模式：已持空且目标是空 → 平掉空头"""
    s = make_meta(allow_short=False, current=PositionSide.SHORT)
    target = s._apply_allow_short(PositionSide.SHORT, PositionSide.SHORT)
    assert target == PositionSide.NONE
    assert s._map_target(target, PositionSide.SHORT, 100.0, "t").signal_type \
        == SignalType.CLOSE_SHORT


def test_allow_short_false_keeps_long():
    """仅做多模式：持多且目标是空 → 维持多头不动（不误平多头）"""
    s = make_meta(allow_short=False, current=PositionSide.LONG)
    target = s._apply_allow_short(PositionSide.SHORT, PositionSide.LONG)
    assert target == PositionSide.LONG
    assert s._map_target(target, PositionSide.LONG, 100.0, "t").signal_type \
        == SignalType.HOLD


def test_allow_short_false_passes_through_long_target():
    """仅做多模式：目标本就是多 → 原样通过"""
    s = make_meta(allow_short=False)
    assert s._apply_allow_short(PositionSide.LONG, PositionSide.NONE) \
        == PositionSide.LONG


def test_allow_short_true_still_allows_short():
    """双向模式不受影响：空目标原样通过"""
    s = make_meta(allow_short=True)
    assert s.allow_short is True
    assert s._apply_allow_short(PositionSide.SHORT, PositionSide.NONE) \
        == PositionSide.SHORT
    assert s._map_target(PositionSide.SHORT, PositionSide.NONE, 100.0, "t").signal_type \
        == SignalType.OPEN_SHORT


# ============ 默认配置守卫 ============

def test_default_riskoff_long_mode_is_flatten():
    """默认 riskoff_long_mode 必须是 flatten（实盘 orchestrator 不显式传该参数）。

    变更史：2026-08-23 基于「A方案实证更优」改为 half；2026-09-28 修复空翻多 bug 后
    重跑同一实验，half/hold 优势消失、flatten 风险调整最优，改回 flatten。
    详见 STRATEGIES.md 第 34 章。此测试防止被误改回。
    """
    s = MetaStrategy(instId="BTC-USDT", mode="adaptive", params=dict(min_adx=0.0))
    assert s.params["riskoff_long_mode"] == "flatten", (
        "默认 riskoff_long_mode 被改动——若是有意为之，请同步更新 STRATEGIES 第34章与本测试")


def test_riskoff_flatten_clears_long_target():
    """flatten 语义：risk_off 时把多头目标压成 NONE（清仓）"""
    s = make_meta(riskoff_long_mode="flatten")
    assert s._apply_market_filter(PositionSide.LONG, PositionSide.LONG, "risk_off") \
        == PositionSide.NONE


def test_riskoff_half_keeps_long_with_scale():
    """half 语义：risk_off 时保留多头但按 riskoff_scale 缩放"""
    s = make_meta(riskoff_long_mode="half")
    assert s._apply_market_filter(PositionSide.LONG, PositionSide.LONG, "risk_off") \
        == PositionSide.LONG
    assert s._bias_scale == 0.5
