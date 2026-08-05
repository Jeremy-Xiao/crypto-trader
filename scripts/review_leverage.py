"""
杠杆机制单元测试（review 用）

验证：保证金记账、1倍向后兼容、杠杆硬上限、单笔风险不随杠杆放大、
      名义价值上限、强平触发、借币计息、回撤熔断。

运行：cd /Users/hello/Documents/workspace/crypto-trader && venv/bin/python scripts/review_leverage.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.strategies.base import BaseStrategy, Signal, SignalType, PositionSide


def approx(a, b, tol=0.05):
    return abs(a - b) <= tol


class DummyStrategy(BaseStrategy):
    def generate_signal(self, data):
        return Signal(SignalType.HOLD, self.instId, data.get('price', 0), 0, data.get('timestamp', ''))

    def calculate_position_size(self, b, p):
        return 1.0


def make_engine(leverage=1.0, risk_pct=0.02, max_pct=0.5, atr=5.0, **kw):
    cfg = BacktestConfig(
        initial_balance=10000, fee_rate=0.001, slippage=0.0005,
        use_atr_risk=True, atr_period=14, atr_multiplier=2.0,
        atr_sl_multiplier=2.0, atr_tp_multiplier=4.0,
        use_trailing=False, min_adx_for_entry=0.0,
        risk_pct=risk_pct, max_position_pct=max_pct,
        leverage=leverage, **kw
    )
    strat = DummyStrategy("D", "BTC-USDT")
    e = BacktestEngine(strat, cfg)
    strat.get_atr = lambda p: atr
    return e, strat


ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        ok = False


print("=" * 60)
print("杠杆机制单元测试")
print("=" * 60)

# ---- Test 1: 杠杆硬上限 ----
e, _ = make_engine(leverage=10.0)
check("cap: leverage 10 clamped to max_leverage 3", approx(e.leverage, 3.0, 0.001))
e, _ = make_engine(leverage=0.2)
check("cap: leverage 0.2 clamped up to 1", approx(e.leverage, 1.0, 0.001))
e, _ = make_engine(leverage=3.0)
check("cap: leverage 3 accepted", approx(e.leverage, 3.0, 0.001))

# ---- Test 2: 1倍做多与现货口径等价 ----
# ATR仓位: 10000*0.02/(5*2)=20 币; 成交价100.05; notional=2001; margin=notional; fee=2.001
e, _ = make_engine(leverage=1.0)
e._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
check("1x long: amount == 20 (ATR sizing)", approx(e.position_amount, 20.0, 0.02))
check("1x long: margin == notional", approx(e.margin_used, 20.0 * 100.05, 0.5))
check("1x long: balance = 10000 - notional - fee",
      approx(e.balance, 10000 - 2001 - 2.001, 0.5))
check("1x long: equity only loses the fee",
      approx(e._current_equity(100.05), 10000 - 2.001, 0.5))
e._close_current_position(120.0, "t1", "close")
# 平仓价 119.94; pnl=(119.94-100.05)*20=397.8; exit fee=2398.8*0.001=2.3988
check("1x long: equity after +20% move",
      approx(e.balance, 10000 - 2.001 + 397.8 - 2.3988, 1.0))
check("1x long: margin released", approx(e.margin_used, 0.0, 1e-9))

# ---- Test 3: 3倍保证金记账 ----
e, _ = make_engine(leverage=3.0)
e._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
notional = e.position_amount * e.position_price
check("3x: margin == notional / 3", approx(e.margin_used, notional / 3.0, 0.5))
check("3x: balance = 10000 - margin - fee",
      approx(e.balance, 10000 - notional / 3.0 - notional * 0.001, 0.5))
check("3x: equity only loses the fee",
      approx(e._current_equity(e.position_price), 10000 - notional * 0.001, 0.5))

# ---- Test 4: 单笔风险不随杠杆放大（核心安全属性）----
e1, _ = make_engine(leverage=1.0)
e1._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
e3, _ = make_engine(leverage=3.0)
e3._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
check("risk: same position size at 1x and 3x when cash is not binding",
      approx(e1.position_amount, e3.position_amount, 0.01))
# 止损同为 2×ATR=10 → 两者单笔最大亏损相同
loss_1x = 10.0 * e1.position_amount
loss_3x = 10.0 * e3.position_amount
check("risk: identical max loss per trade at 1x vs 3x", approx(loss_1x, loss_3x, 0.5))
check("risk: max loss stays ~2% of equity at 3x", approx(loss_3x / 10000, 0.02, 0.002))

# ---- Test 5: 杠杆只放开资金约束 ----
# risk_pct=0.2 → 目标 200 币 = 20000 名义，远超 1x 的现金能力
e1, _ = make_engine(leverage=1.0, risk_pct=0.2, max_pct=1.0)
e1._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
n1 = e1.position_amount * e1.position_price
e3, _ = make_engine(leverage=3.0, risk_pct=0.2, max_pct=1.0)
e3._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
n3 = e3.position_amount * e3.position_price
check("cash: 1x notional capped near equity (~10000)", n1 <= 10000.5)
check("cash: 3x notional larger than 1x", n3 > n1 * 1.5)
check("cash: 3x notional never exceeds equity x max_pct x 3", n3 <= 10000 * 1.0 * 3 + 1)
check("cash: balance never goes negative at 3x", e3.balance >= -1e-6)

# ---- Test 6: 名义价值上限（max_position_pct × leverage）----
e, _ = make_engine(leverage=3.0, risk_pct=0.5, max_pct=0.5)
e._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
n = e.position_amount * e.position_price
check("cap: notional <= equity x 0.5 x 3 = 15000", n <= 15000 + 1)
check("cap: max_gross_leverage tracked and <= 3", e.max_gross_leverage <= 3.01)

# ---- Test 7: 强平（3倍多头暴跌）----
e, _ = make_engine(leverage=3.0, risk_pct=0.2, max_pct=1.0)
e._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
triggered = e._check_liquidation(high=101.0, low=99.0, timestamp="t1")
check("liq: no liquidation on small dip", not triggered)
triggered = e._check_liquidation(high=55.0, low=50.0, timestamp="t2")
check("liq: liquidated on -50% crash at 3x", triggered)
check("liq: position flat after liquidation", e.position_side == PositionSide.NONE)
check("liq: counter incremented", e.liquidations == 1)
check("liq: recorded in stop_stats", e.stop_stats['liquidation'] == 1)
check("liq: equity stays positive (buffer worked)", e.balance > 0)

# ---- Test 8: 1倍现货多头不会被强平 ----
e, _ = make_engine(leverage=1.0, risk_pct=0.2, max_pct=1.0)
e._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
triggered = e._check_liquidation(high=2.0, low=1.0, timestamp="t1")
check("liq: 1x spot long never liquidated", not triggered)
check("liq: 1x position intact", e.position_side == PositionSide.LONG)

# ---- Test 9: 空头强平（3倍暴涨）----
e, _ = make_engine(leverage=3.0, risk_pct=0.2, max_pct=1.0)
e._open_position(PositionSide.SHORT, 100.0, 1.0, "t0", "open")
check("liq: short opened at 3x", e.position_side == PositionSide.SHORT)
triggered = e._check_liquidation(high=200.0, low=190.0, timestamp="t1")
check("liq: short liquidated on +100% spike", triggered)
check("liq: short flat after liquidation", e.position_side == PositionSide.NONE)

# ---- Test 10: 借币利息 ----
e, _ = make_engine(leverage=3.0, risk_pct=0.2, max_pct=1.0)
e._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
notional = e.position_amount * 100.0
e._accrue_borrow_cost(100.0)
expected = notional * (1 - 1 / 3.0) * 0.0003
check("borrow: interest charged on borrowed portion only",
      approx(e.total_borrow_cost, expected, 0.05))
check("borrow: deducted from balance", e.total_borrow_cost > 0)

e1, _ = make_engine(leverage=1.0, risk_pct=0.2, max_pct=1.0)
e1._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
e1._accrue_borrow_cost(100.0)
check("borrow: zero interest at 1x (nothing borrowed)", approx(e1.total_borrow_cost, 0.0, 1e-9))

# ---- Test 11: 回撤熔断 ----
e, _ = make_engine(leverage=3.0, max_drawdown_halt=0.2)
e.peak_equity = 10000.0
e.balance = 7500.0        # 权益回撤 25% > 20% 阈值
e._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
check("halt: new entry blocked when drawdown exceeds threshold",
      e.position_side == PositionSide.NONE)
check("halt: dd_halt_blocked incremented", e.dd_halt_blocked == 1)

e, _ = make_engine(leverage=3.0, max_drawdown_halt=0.2)
e.peak_equity = 10000.0
e.balance = 9000.0        # 回撤 10% < 20%
e._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
check("halt: entry allowed when drawdown within threshold",
      e.position_side == PositionSide.LONG)
check("halt: no block counted", e.dd_halt_blocked == 0)

# ---- Test 12: 权益守恒（开仓只损失手续费）----
for lev in (1.0, 2.0, 3.0):
    e, _ = make_engine(leverage=lev, risk_pct=0.1, max_pct=1.0)
    before = e._current_equity(100.0)
    e._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
    entry = e.position_price
    after = e._current_equity(entry)
    fee = e.trades[-1]['fee']
    check(f"equity: conserved minus fee at {lev}x", approx(after, before - fee, 0.5))

# ---- Test 13: 多空保证金对称 ----
el, _ = make_engine(leverage=3.0, risk_pct=0.1, max_pct=1.0)
el._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
es, _ = make_engine(leverage=3.0, risk_pct=0.1, max_pct=1.0)
es._open_position(PositionSide.SHORT, 100.0, 1.0, "t0", "open")
# 多空成交价因滑点方向相反，名义价值有 ~0.1% 差异，属预期
check("symmetry: long/short margin nearly identical",
      approx(el.margin_used, es.margin_used, 5.0))
check("symmetry: long/short size nearly identical",
      approx(el.position_amount, es.position_amount, 0.1))

# ---- Test 14: 强平成交价与剩余权益 ----
e, _ = make_engine(leverage=3.0, risk_pct=0.2, max_pct=1.0)
e._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
margin0 = e.margin_used
liq_p = e._liquidation_price()
check("liqprice: liquidation price below entry for long", liq_p < e.position_price)
e._check_liquidation(high=liq_p + 1, low=liq_p - 20, timestamp="t1")
# 应在 liq_p 成交而非最低点 liq_p-20，剩余权益≈缓冲线 = margin×0.25
check("liqprice: filled at liq price, not at bar low",
      approx(e.balance, margin0 * 0.25, margin0 * 0.05))
check("liqprice: account survives with positive equity", e.balance > 0)

# ---- Test 15: 跳空穿价按开盘价成交（更差）----
e, _ = make_engine(leverage=3.0, risk_pct=0.2, max_pct=1.0)
e._open_position(PositionSide.LONG, 100.0, 1.0, "t0", "open")
liq_p = e._liquidation_price()
gap_open = liq_p - 10
e._check_liquidation(high=liq_p, low=gap_open - 5, timestamp="t1", open_price=gap_open)
check("gap: liquidated on gap-down", e.position_side == PositionSide.NONE)
check("gap: gap loss is worse than clean liquidation", e.balance < margin0 * 0.25)

# ---- Test 16: 空头强平价在开仓价之上 ----
e, _ = make_engine(leverage=3.0, risk_pct=0.2, max_pct=1.0)
e._open_position(PositionSide.SHORT, 100.0, 1.0, "t0", "open")
liq_s = e._liquidation_price()
check("liqprice: short liquidation price above entry", liq_s > e.position_price)
check("liqprice: short not liquidated below liq price",
      not e._check_liquidation(high=liq_s - 1, low=95.0, timestamp="t1"))

print("\n==== " + ("ALL LEVERAGE TESTS PASSED" if ok else "SOME TESTS FAILED") + " ====")
sys.exit(0 if ok else 1)
