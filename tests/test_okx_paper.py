"""OKXPaperExecutor 核心路径的离线单测（mock 交易所，不下真单）。

覆盖第 27 章修复的关键安全逻辑：
  1. 翻转信号：先平交易所反向仓，再开新仓
  2. 平仓：按交易所实际持仓量平（不信本地 amount）
  3. 交易所无仓时的 CLOSE：只清本地，不下单
  4. 同向已有仓：拒绝重复开仓
  5. reconcile：本地与交易所不符时以交易所为准
  6. _to_sz 浮点精度
运行：venv/bin/python -m pytest tests/test_okx_paper.py -v
"""
import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.live.okx_paper import OKXPaperExecutor  # noqa: E402
from src.strategies.base import Signal, SignalType, PositionSide, Position  # noqa: E402


class FakeStrategy:
    """最小策略桩：记录 open/close 调用。"""

    def __init__(self):
        self.position = None
        self.calls = []

    def open_position(self, price, amount, ts, side):
        from src.strategies.base import Position
        self.calls.append(("open", side.value, amount))
        self.position = Position(instId="SOL-USDT", side=side, amount=amount,
                                 entry_price=price, current_price=price,
                                 unrealized_pnl=0.0, unrealized_pnl_pct=0.0)

    def close_position(self, price, ts, reason=""):
        self.calls.append(("close", reason))
        self.position = None


def make_executor(pos_map=None):
    """构造带 Fake API 的执行器（不触网）。"""
    with patch.object(OKXPaperExecutor, "__init__", lambda self: None):
        ex = OKXPaperExecutor()
    ex._ct_val = {"SOL-USDT-SWAP": 1.0}
    ex.realized_pnl = {}
    ex.budget = 10000.0
    ex.trades_path = "/tmp/test_okx_trades.csv"
    ex.pnl_path = "/tmp/test_okx_pnl.json"
    ex._pos_map = pos_map or {}       # symbol -> position dict 或 None

    ex.api = type("FakeAPI", (), {})()
    ex.api.get_positions = lambda instType=None, instId=None: {
        "data": ([ex._pos_map[instId.replace("-SWAP", "")]]
                 if instId and instId.replace("-SWAP", "") in ex._pos_map else [])}
    ex.api.set_leverage = lambda *a, **k: {"code": "0"}

    ex._orders = []

    def fake_request(method, path, params=None, body=None):
        if method == "POST" and path == "/api/v5/trade/order":
            oid = f"ORD{len(ex._orders) + 1}"
            ex._orders.append({**body, "ordId": oid})
            return {"code": "0", "data": [{"ordId": oid}]}
        if method == "GET" and path == "/api/v5/trade/order":
            ord_id = params.get("ordId")
            for o in ex._orders:
                if o["ordId"] == ord_id:
                    return {"code": "0", "data": [{
                        "state": "filled", "accFillSz": o["sz"],
                        "avgPx": "105.0", "fee": "-0.5"}]}
            return {"code": "0", "data": []}
        return {"code": "0", "data": []}

    ex.api._request = fake_request
    return ex


def short_pos(sz="20"):
    return {"instId": "SOL-USDT-SWAP", "posSide": "short", "pos": sz,
            "avgPx": "103.0", "upl": "-5", "mgnMode": "cross", "last": "105.0"}


def long_pos(sz="48"):
    return {"instId": "SOL-USDT-SWAP", "posSide": "long", "pos": sz,
            "avgPx": "104.0", "upl": "+20", "mgnMode": "cross", "last": "105.0"}


class TestExecute(unittest.TestCase):
    def setUp(self):
        if os.path.exists("/tmp/test_okx_trades.csv"):
            os.remove("/tmp/test_okx_trades.csv")

    def test_flip_closes_opposite_first(self):
        """持空仓时收到开多信号：必须先平空，再开多。"""
        ex = make_executor({"SOL-USDT": short_pos()})
        strat = FakeStrategy()
        strat.open_position(103.0, 20.0, "t", PositionSide.SHORT)
        sig = Signal(SignalType.OPEN_LONG, "SOL-USDT", 105.0, 10.0,
                     "t", "翻转做多")
        rec = ex.execute(strat, "SOL-USDT", sig, 105.0, "t")
        self.assertTrue(rec["executed"])
        actions = [o["side"] + "/" + o["posSide"] for o in ex._orders]
        self.assertEqual(actions, ["buy/short", "buy/long"],
                         "必须先买平空单(buy/short)再开多单(buy/long)")
        self.assertEqual(strat.position.side, PositionSide.LONG)

    def test_close_uses_exchange_size(self):
        """平仓量以交易所为准：本地以为 48 币，交易所只有 20 张 → 只平 20。"""
        ex = make_executor({"SOL-USDT": long_pos("20")})
        strat = FakeStrategy()
        strat.open_position(104.0, 48.0, "t", PositionSide.LONG)  # 本地记 48（失真）
        sig = Signal(SignalType.CLOSE_LONG, "SOL-USDT", 105.0, 48.0, "t", "平多")
        rec = ex.execute(strat, "SOL-USDT", sig, 105.0, "t")
        self.assertEqual(float(ex._orders[0]["sz"]), 20.0, "平仓量应=交易所20张")
        self.assertIn("pnl", rec)
        # pnl = (105-104)*20 = +20，fee 0.5 → 账本净 +19.5
        self.assertAlmostEqual(ex.realized_pnl["SOL-USDT"], 20.0 - 0.5, places=2)

    def test_close_no_exchange_pos_clears_local_only(self):
        """交易所已无仓：只清本地状态，不下单。"""
        ex = make_executor({})  # 交易所空
        strat = FakeStrategy()
        strat.open_position(104.0, 48.0, "t", PositionSide.LONG)
        sig = Signal(SignalType.CLOSE_LONG, "SOL-USDT", 105.0, 48.0, "t", "平多")
        rec = ex.execute(strat, "SOL-USDT", sig, 105.0, "t")
        self.assertFalse(rec["executed"])
        self.assertEqual(rec["reason"], "no_exchange_pos")
        self.assertIsNone(strat.position)
        self.assertEqual(len(ex._orders), 0, "不应有任何下单")

    def test_skip_duplicate_open(self):
        """同向已有仓：拒绝重复开仓。"""
        ex = make_executor({"SOL-USDT": long_pos()})
        strat = FakeStrategy()
        sig = Signal(SignalType.OPEN_LONG, "SOL-USDT", 105.0, 48.0, "t", "做多")
        rec = ex.execute(strat, "SOL-USDT", sig, 105.0, "t")
        self.assertFalse(rec["executed"])
        self.assertEqual(rec["reason"], "already_in_position")
        self.assertEqual(len(ex._orders), 0)


class TestReconcile(unittest.TestCase):
    def test_exchange_none_clears_local(self):
        ex = make_executor({})
        strat = FakeStrategy()
        strat.open_position(104.0, 48.0, "t", PositionSide.LONG)
        ex.reconcile(strat, "SOL-USDT", 105.0)
        self.assertIsNone(strat.position)

    def test_exchange_diff_overrides_local(self):
        ex = make_executor({"SOL-USDT": long_pos("20")})  # 交易所 20 张
        strat = FakeStrategy()
        strat.open_position(104.0, 48.0, "t", PositionSide.LONG)  # 本地 48
        ex.reconcile(strat, "SOL-USDT", 105.0)
        self.assertEqual(strat.position.amount, 20.0, "应以交易所 20 张为准")

    def test_match_noop(self):
        ex = make_executor({"SOL-USDT": long_pos("48")})
        strat = FakeStrategy()
        strat.open_position(104.0, 48.0, "t", PositionSide.LONG)
        ex.reconcile(strat, "SOL-USDT", 105.0)
        self.assertEqual(strat.position.amount, 48.0)  # 一致则不动

    def test_negative_pos_handled(self):
        """pos 为负数（net 模式特征）时不应误判方向。"""
        neg = long_pos("20")
        neg["pos"] = "-20"
        ex = make_executor({"SOL-USDT": neg})
        strat = FakeStrategy()
        strat.open_position(104.0, 20.0, "t", PositionSide.LONG)
        ex.reconcile(strat, "SOL-USDT", 105.0)
        # posSide=long 仍应正确对齐而不是清仓
        self.assertIsNotNone(strat.position)
        self.assertEqual(strat.position.side, PositionSide.LONG)


class TestSzPrecision(unittest.TestCase):
    def test_float_trap(self):
        ex = make_executor()
        self.assertEqual(ex._to_sz("SOL-USDT", 0.07), 0.07)
        self.assertEqual(ex._to_sz("SOL-USDT", 0.999), 0.99)
        self.assertEqual(ex._to_sz("SOL-USDT", 96.5), 96.5)

    def test_negative_amount_zero(self):
        ex = make_executor()
        self.assertEqual(ex._to_sz("SOL-USDT", -5), 0.0)


if __name__ == "__main__":
    unittest.main()
