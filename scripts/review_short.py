"""
做空机制单元测试（review 用）
验证：开空/平空的 balance 与 PnL 计算、空头止损/止盈方向、空头反向移动止损。
运行：cd /Users/hello/Documents/workspace/crypto-trader && venv/bin/python scripts/review_short.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.strategies.base import BaseStrategy, Signal, SignalType, PositionSide


def approx(a, b, tol=0.05):
    return abs(a - b) <= tol


class DummyStrategy(BaseStrategy):
    """不做信号，仅用于引擎内部方法测试"""
    def generate_signal(self, data):
        return Signal(SignalType.HOLD, self.instId, data.get('price', 0), 0, data.get('timestamp', ''))
    def calculate_position_size(self, b, p):
        return 1.0


def make_engine():
    cfg = BacktestConfig(
        initial_balance=10000, fee_rate=0.001, slippage=0.0005,
        use_atr_risk=True, atr_period=14, atr_sl_multiplier=2.0, atr_tp_multiplier=4.0,
        use_trailing=True, trailing_pct=0.02, min_adx_for_entry=0.0
    )
    strat = DummyStrategy("D", "BTC-USDT")
    e = BacktestEngine(strat, cfg)
    strat.get_atr = lambda p: 5.0  # 固定 ATR = 5，便于精确验算
    return e, strat


ok = True
def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        ok = False


# ---- Test 1: 开空 balance / 平空 PnL ----
e, strat = make_engine()
e._open_position(PositionSide.SHORT, 100.0, 10.0, "t0", "open")
# 开空成交价 = 100*(1-0.0005)=99.95; sell_value=999.5; fee=0.9995; balance=10000+999.5-0.9995=10998.5005
check("short: position_side==SHORT", e.position_side == PositionSide.SHORT)
check("short: balance += cash on open", approx(e.balance, 10998.5005))
e._close_current_position(90.0, "t1", "close")
# 平空成交价 = 90*(1+0.0005)=90.045; buy_value=900.45; fee=0.90045; pnl=(99.95-90.045)*10=99.05
# balance = 10998.5005 - 900.45 - 0.90045 = 10097.15005
check("short: position_side reset on close", e.position_side == PositionSide.NONE)
check("short: balance after close ~10097.15", approx(e.balance, 10097.15005, 0.1))
check("short: recorded pnl ~99.05", abs(e.trades[-1]['pnl'] - 99.05) < 0.5)
check("short: close action is short_close", e.trades[-1]['side'] == 'short_close')

# ---- Test 2: 空头止盈（价格下跌触发）----
e, strat = make_engine()
e._open_position(PositionSide.SHORT, 100.0, 10.0, "t0", "open")
# position_price=99.95; tp = 99.95 - 5*4 = 79.95
r = e._check_exit_conditions(79.9, 80.0, 79.8, "t1")
check("short: take_profit triggers on price drop", r == 'take_profit')

# ---- Test 3: 空头止损（价格上涨触发）----
e, strat = make_engine()
e._open_position(PositionSide.SHORT, 100.0, 10.0, "t0", "open")
# position_price=99.95; stop = 99.95 + 5*2 = 109.95
r = e._check_exit_conditions(110.0, 110.1, 109.9, "t1")
check("short: stop_loss triggers on price rise", r == 'stop_loss')

# ---- Test 4: 空头反向移动止损 ----
e, strat = make_engine()
e._open_position(PositionSide.SHORT, 100.0, 10.0, "t0", "open")
# price=95: 盈利5%>2%激活; lowest=95; trail=95+5*2=105
r1 = e._check_exit_conditions(95.0, 95.1, 94.9, "t1")
check("short: trailing not triggered at 95", r1 is None)
check("short: trailing_active set", e.strategy.position.trailing_active)
r2 = e._check_exit_conditions(106.0, 106.1, 105.9, "t2")
check("short: trailing_stop triggers when price rises past trail", r2 == 'trailing_stop')

# ---- Test 5: 多头逻辑回归（确保没破坏）----
e, strat = make_engine()
e._open_position(PositionSide.LONG, 100.0, 10.0, "t0", "open")
# position_price=100.05; tp=100.05+5*4=120.05
r = e._check_exit_conditions(121.0, 121.1, 120.9, "t1")
check("long: take_profit still works", r == 'take_profit')
e, strat = make_engine()
e._open_position(PositionSide.LONG, 100.0, 10.0, "t0", "open")
e._check_exit_conditions(105.0, 105.1, 104.9, "t1")  # activate highest=105 trail=105-10=95
r = e._check_exit_conditions(94.0, 94.1, 93.9, "t2")
check("long: trailing_stop still works", r == 'trailing_stop')

# ---- Test 6: 翻转（reverse，模拟 run() 主循环：先平后开）----
e, strat = make_engine()
e._open_position(PositionSide.LONG, 100.0, 10.0, "t0", "open")
e._close_current_position(95.0, "t1", "reverse")   # run() 主循环先平多
e._open_position(PositionSide.SHORT, 95.0, 10.0, "t1", "reverse")  # 再开空
check("reverse: side flips to SHORT", e.position_side == PositionSide.SHORT)
check("reverse: entry updated to 95 area", abs(e.position_price - 95.0 * (1 - 0.0005)) < 0.1)

# ---- Test 7: equity 记录（空头市值记为负）----
e, strat = make_engine()
e._open_position(PositionSide.SHORT, 100.0, 10.0, "t0", "open")
e._record_equity(90.0, "t0")
last = e.equity_curve[-1]
# balance≈10998.5; pos_value = -10*90 = -900; equity≈10098.5
check("short: equity reflects negative position value", approx(last['equity'], 10098.5, 1.0))
check("short: position_value negative", last['position_value'] < 0)

# ---- Test 8: 端到端 run() 主循环信号路由（含翻转）----
import pandas as pd
from src.strategies.base import BaseStrategy as _BS


class ScriptedStrategy(_BS):
    """按预定义信号序列发信号，用于验证 run() 的信号路由"""
    def __init__(self, signals):
        super().__init__("S", "BTC-USDT")
        self._signals = signals
        self._i = 0
    def generate_signal(self, data):
        sig = self._signals[self._i] if self._i < len(self._signals) else \
            Signal(SignalType.HOLD, self.instId, data['price'], 0, data['timestamp'])
        self._i += 1
        return sig
    def calculate_position_size(self, b, p):
        return 10.0


df = pd.DataFrame({
    'timestamp': [f't{i}' for i in range(20)],
    'open': [100.0] * 20, 'high': [101.0] * 20, 'low': [99.0] * 20,
    'close': [100.0] * 20, 'volume': [1.0] * 20,
})
sig_list = ([Signal(SignalType.OPEN_LONG, 'BTC-USDT', 100, 10, 't0')]
            + [Signal(SignalType.HOLD, 'BTC-USDT', 100, 0, f't{i}') for i in range(1, 10)]
            + [Signal(SignalType.OPEN_SHORT, 'BTC-USDT', 100, 10, 't10')]
            + [Signal(SignalType.HOLD, 'BTC-USDT', 100, 0, f't{i}') for i in range(11, 20)])
scfg = BacktestConfig(use_atr_risk=False, min_adx_for_entry=0.0)  # 关ATR，恒定价不触发退出
se = BacktestEngine(ScriptedStrategy(sig_list), scfg)
se.load_data(df)
se.run()
check("integ: has short_open trade (reverse happened)", any(t.get('side') == 'short_open' for t in se.trades))
check("integ: has long close (sell, no side)", any(t['action'] == 'sell' and t.get('side') is None for t in se.trades))
check("integ: ends flat after forced close at end", se.position_side == PositionSide.NONE)
last_trade = se.trades[-1]
check("integ: forced close of short at end (buy short_close)",
      last_trade['action'] == 'buy' and last_trade.get('side') == 'short_close')

print("\n==== " + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED") + " ====")
sys.exit(0 if ok else 1)
