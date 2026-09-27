"""现货杠杆（OKX 单币种保证金模式 / 币币杠杆）强平模型测试。

被验证的引擎实现：BacktestConfig(margin_mode="spot_margin") 下的
  - _setup_spot_margin_position()  开仓负债/资产结构
  - _accrue_spot_margin_interest() 按实际负债计息
  - _spot_margin_liq_price()       官方强平价公式

官方公式（OKX 帮助中心「怎么计算杠杆强制平仓价格」，单币种保证金模式，
维持保证金率 ≤100% 时触发强平）：
    多仓 强平价 = (负债 + 利息) × (1 + 档位MMR) × (1 + 吃单费率) / 仓位资产
    空仓 强平价 = 仓位资产 / [(负债 + 利息) × (1 + 档位MMR) × (1 + 吃单费率)]

数学基准（P₀=100, E=10000, mmr=0.02, fee=0.001 → f = 1.02 × 1.001 = 1.02102）：
    做多 notional = 2E ：负债 10000 USDT，资产 200 币
                        liq = 10000 × 1.02102 / 200        = 51.0510
    做多 notional = 1.0E：负债 0（自有资金就够）→ 无强平
    做多 notional = 0.46E：负债 0 → 无强平
    做空 notional = 1.0E：资产 20000 USDT，负债 100 币
                        liq = 20000 / (100 × 1.02102)      = 195.8820
    做空 notional = 0.46E：资产 14600 USDT，负债 46 币
                        liq = 14600 / (46 × 1.02102)       = 310.8301

运行：venv/bin/python -m pytest tests/test_spot_margin.py -v
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.strategies.base import BaseStrategy, PositionSide, Signal, SignalType

E = 10000.0
P0 = 100.0
F = (1.0 + 0.02) * (1.0 + 0.001)   # 1.02102


class _AlwaysStrategy(BaseStrategy):
    """测试用策略：第 open_at 根开仓，之后一直持有（不产生任何退出信号）。"""

    def __init__(self, side: PositionSide, coins: float, open_at: int = 0):
        super().__init__(name="always_test", instId="BTC-USDT")
        self._side = side
        self._coins = coins
        self._open_at = open_at
        self._n = 0

    def generate_signal(self, data) -> Signal:
        n = self._n
        self._n += 1
        st, amt = SignalType.HOLD, 0.0
        if n == self._open_at:
            st = (SignalType.OPEN_LONG if self._side == PositionSide.LONG
                  else SignalType.OPEN_SHORT)
            amt = self._coins
        return Signal(signal_type=st, instId=self.instId, price=data["price"],
                      amount=amt, timestamp=str(data["timestamp"]))

    def calculate_position_size(self, account_balance: float, price: float) -> float:
        return self._coins


def make_engine(side=PositionSide.LONG, coins=200.0, **cfg) -> BacktestEngine:
    """构造一个不依赖 ATR/regime 的干净引擎，便于精确验证。"""
    base = dict(
        initial_balance=E, fee_rate=0.001, slippage=0.0,
        use_atr_risk=False, use_trailing=False, use_mtf=False, use_regime=False,
        tf_confirm=None, leverage=1.0, max_leverage=3.0,
        max_position_pct=10.0,          # 放开名义上限，让测试完全控制仓位
        margin_mode="spot_margin",
        spot_margin_mmr=0.02, spot_margin_fee=0.001,
        borrow_rate_daily=0.0,          # 默认关息，利息单独测
        spot_borrow_rate_base_daily=0.0,
    )
    base.update(cfg)
    strat = _AlwaysStrategy(side, coins)
    return BacktestEngine(strat, BacktestConfig(**base))


def arm(e: BacktestEngine, side: PositionSide, notional: float,
        equity: float = E, price: float = P0) -> BacktestEngine:
    """手动装配一个现货杠杆持仓（绕过信号流程，用于精确对拍公式）。"""
    e.position_side = side
    e.position_amount = notional / price
    e._setup_spot_margin_position(side, price, notional, equity)
    return e


def make_df(prices, start="2024-01-01"):
    ts = pd.date_range(start, periods=len(prices), freq="D")
    return pd.DataFrame({
        "timestamp": ts,
        "open": prices, "high": [p * 1.001 for p in prices],
        "low": [p * 0.999 for p in prices], "close": prices,
        "volume": [100.0] * len(prices),
    })


# ============ Part A：开仓结构 + 强平价公式（精确对拍） ============

def test_long_2x_liq_price():
    """做多 notional=2E：负债 E USDT，资产 2E/P0 币 → liq ≈ 0.51051·P0"""
    e = make_engine()
    arm(e, PositionSide.LONG, 2 * E)
    assert e.spot_liab_qty == pytest.approx(E, rel=1e-9)
    assert e.spot_liab_is_base is False
    assert e.spot_asset_qty == pytest.approx(2 * E / P0, rel=1e-9)
    liq = e._spot_margin_liq_price()
    assert liq == pytest.approx(E * F / (2 * E / P0), rel=1e-9)
    assert liq == pytest.approx(51.0510, rel=1e-4)


def test_long_1x_no_liq():
    """做多 notional=E：自有资金刚好够 → 负债 0 → 不会被强平（纯现货持有）"""
    e = make_engine()
    arm(e, PositionSide.LONG, E)
    assert e.spot_liab_qty == 0.0
    assert e._spot_margin_liq_price() is None


def test_long_partial_position_no_liq():
    """做多 notional=0.46E（策略实际的平均仓位水平）→ 同样无负债、无强平"""
    e = make_engine()
    arm(e, PositionSide.LONG, 0.46 * E)
    assert e.spot_liab_qty == 0.0
    assert e._spot_margin_liq_price() is None


def test_short_1x_liq_price():
    """做空 notional=E：资产 2E USDT，负债 E/P0 币 → liq ≈ 1.9588·P0"""
    e = make_engine()
    arm(e, PositionSide.SHORT, E)
    assert e.spot_liab_is_base is True
    assert e.spot_liab_qty == pytest.approx(E / P0, rel=1e-9)
    assert e.spot_asset_qty == pytest.approx(2 * E, rel=1e-9)
    liq = e._spot_margin_liq_price()
    assert liq == pytest.approx(2 * E / ((E / P0) * F), rel=1e-9)
    assert liq == pytest.approx(195.8820, rel=1e-4)


def test_short_partial_position_liq_price():
    """做空 notional=0.46E（策略实际仓位水平）：强平线被推得很远 ≈ 3.108·P0"""
    e = make_engine()
    arm(e, PositionSide.SHORT, 0.46 * E)
    liq = e._spot_margin_liq_price()
    assert liq == pytest.approx((E + 0.46 * E) / ((0.46 * E / P0) * F), rel=1e-9)
    assert liq == pytest.approx(310.8301, rel=1e-4)
    assert liq / P0 > 3.0, "低仓位做空的强平线应远在 3 倍之外"


def test_interest_raises_liq_price_for_short():
    """做空负债是币：利息累积让负债变多，强平价下降（更危险）"""
    e = make_engine()
    arm(e, PositionSide.SHORT, E)
    liq0 = e._spot_margin_liq_price()
    e.spot_interest_qty = 0.25          # 累计 0.25 个币的利息
    liq1 = e._spot_margin_liq_price()
    assert liq1 < liq0, "空头利息增加负债 → 强平价应下调"
    assert liq1 == pytest.approx(2 * E / ((E / P0 + 0.25) * F), rel=1e-9)


def test_interest_raises_liq_price_for_long():
    """做多负债是 USDT：利息累积让负债变大，强平价上升（更危险）"""
    e = make_engine()
    arm(e, PositionSide.LONG, 2 * E)
    liq0 = e._spot_margin_liq_price()
    e.spot_interest_qty = 100.0
    liq1 = e._spot_margin_liq_price()
    assert liq1 > liq0, "多头利息增加负债 → 强平价应上调"
    assert liq1 == pytest.approx((E + 100.0) * F / (2 * E / P0), rel=1e-9)


def test_mmr_and_fee_monotonic():
    """MMR 越高强平价越保守：多头更高、空头更低"""
    long_lo = make_engine(spot_margin_mmr=0.02); arm(long_lo, PositionSide.LONG, 2 * E)
    long_hi = make_engine(spot_margin_mmr=0.05); arm(long_hi, PositionSide.LONG, 2 * E)
    assert long_hi._spot_margin_liq_price() > long_lo._spot_margin_liq_price()

    short_lo = make_engine(spot_margin_mmr=0.02); arm(short_lo, PositionSide.SHORT, E)
    short_hi = make_engine(spot_margin_mmr=0.05); arm(short_hi, PositionSide.SHORT, E)
    assert short_hi._spot_margin_liq_price() < short_lo._spot_margin_liq_price()

    fee_lo = make_engine(spot_margin_fee=0.0001); arm(fee_lo, PositionSide.SHORT, E)
    fee_hi = make_engine(spot_margin_fee=0.002); arm(fee_hi, PositionSide.SHORT, E)
    assert fee_hi._spot_margin_liq_price() < fee_lo._spot_margin_liq_price()


def test_clear_position_resets_state():
    e = make_engine()
    arm(e, PositionSide.SHORT, E)
    e._clear_spot_margin_position()
    assert (e.spot_liab_qty, e.spot_asset_qty, e.spot_interest_qty) == (0.0, 0.0, 0.0)
    assert e.spot_borrowed_notional == 0.0


def test_no_position_has_no_liq_price():
    """空仓时不应返回强平价"""
    e = make_engine()
    assert e._spot_margin_liq_price() is None


# ============ Part B：端到端强平触发 ============

def test_e2e_long_levered_gets_liquidated():
    """做多 2x（借 E）：价格从 100 跌到 45（低于强平价 ~51）→ 必须强平"""
    prices = [P0] * 5 + [45.0] * 5
    e = make_engine(side=PositionSide.LONG, coins=2 * E / P0, leverage=2.0)
    e.load_data(make_df(prices))
    e.run()
    assert e.liquidations >= 1, f"2x 做多应被强平（实际 {e.liquidations}）"


def test_e2e_long_unlevered_survives_crash():
    """做多 1x（无负债）：同样暴跌到 45 也不该强平"""
    prices = [P0] * 5 + [45.0] * 5
    e = make_engine(side=PositionSide.LONG, coins=E / P0, leverage=1.0)
    e.load_data(make_df(prices))
    e.run()
    assert e.liquidations == 0, "无负债的现货多头不应被强平"


def test_e2e_short_gets_liquidated_on_spike():
    """做空 notional=E：价格从 100 涨到 210（高于强平价 195.88）→ 触发强平"""
    prices = [P0] * 5 + [210.0] * 5
    e = make_engine(side=PositionSide.SHORT, coins=E / P0, leverage=1.0)
    e.load_data(make_df(prices))
    e.run()
    assert e.liquidations >= 1, f"做空应在价格越过强平价时被强平（实际 {e.liquidations}）"


def test_e2e_short_low_position_survives_spike():
    """做空 notional=0.46E：涨到 210 也不该强平（强平线在 310.8）"""
    prices = [P0] * 5 + [210.0] * 5
    e = make_engine(side=PositionSide.SHORT, coins=0.46 * E / P0, leverage=1.0)
    e.load_data(make_df(prices))
    e.run()
    assert e.liquidations == 0, "低仓位做空的强平线远在 3 倍之外，不应被强平"


def test_e2e_short_survives_before_threshold():
    """做空 notional=E：涨到 190（未过 195.88）不该强平，涨到 210 才强平"""
    prices = [P0] * 5 + [190.0] * 5
    e = make_engine(side=PositionSide.SHORT, coins=E / P0, leverage=1.0)
    e.load_data(make_df(prices))
    e.run()
    assert e.liquidations == 0, "未越过强平价不应强平"


# ============ Part C：计息与回归 ============

def test_short_interest_accrues_into_liability_qty():
    """空头借币计息：利息以币计，累加到 spot_interest_qty 并推高强平风险"""
    e = make_engine(spot_borrow_rate_base_daily=0.01)   # 1%/天
    arm(e, PositionSide.SHORT, E)
    liq_before = e._spot_margin_liq_price()
    e._accrue_spot_margin_interest(P0)
    assert e.spot_interest_qty == pytest.approx(E / P0 * 0.01, rel=1e-9)
    assert e.spot_interest_paid == pytest.approx(E / P0 * 0.01 * P0, rel=1e-9)
    assert e.total_borrow_cost == pytest.approx(e.spot_interest_paid, rel=1e-9)
    assert e._spot_margin_liq_price() < liq_before, "计息后强平线应更近"


def test_long_interest_accrues_in_usdt():
    """多头借 USDT 计息：利息以 USDT 计（qty 与 paid 同值）"""
    e = make_engine(borrow_rate_daily=0.01)
    arm(e, PositionSide.LONG, 2 * E)
    liq_before = e._spot_margin_liq_price()
    e._accrue_spot_margin_interest(P0)
    expected = E * 0.01          # 借入 E USDT，1%/天
    assert e.spot_interest_qty == pytest.approx(expected, rel=1e-9)
    assert e.spot_interest_paid == pytest.approx(expected, rel=1e-9)
    assert e._spot_margin_liq_price() > liq_before


def test_no_interest_when_no_liability():
    """做多且 notional ≤ 权益（无负债）→ 不计息、且不产生强平价"""
    e = make_engine(borrow_rate_daily=0.01)
    arm(e, PositionSide.LONG, 0.5 * E)
    e._accrue_spot_margin_interest(P0)
    assert e.spot_interest_paid == 0.0
    assert e._spot_margin_liq_price() is None


def test_swap_mode_is_default_and_inert():
    """默认 margin_mode='swap'：不产生现货杠杆计息，输出字段为 0"""
    prices = [P0] * 5 + [70.0, 120.0, 90.0, 60.0, 130.0]
    e = make_engine(side=PositionSide.LONG, coins=E / P0, margin_mode="swap")
    e.load_data(make_df(prices))
    r = e.run()
    assert r["margin_mode"] == "swap"
    assert r["spot_interest_paid"] == 0.0
    assert e.spot_liab_qty == 0.0


def test_base_rate_falls_back_to_borrow_rate():
    """spot_borrow_rate_base_daily < 0 时应回落到 borrow_rate_daily"""
    e = make_engine(borrow_rate_daily=0.007, spot_borrow_rate_base_daily=-1.0)
    assert e._spot_base_borrow_rate() == pytest.approx(0.007)
    e2 = make_engine(borrow_rate_daily=0.007, spot_borrow_rate_base_daily=0.02)
    assert e2._spot_base_borrow_rate() == pytest.approx(0.02)
