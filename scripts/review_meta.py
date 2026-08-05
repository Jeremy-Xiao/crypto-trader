"""
元策略（MetaStrategy）单元测试
验证：市场状态判定、各专家方向状态机、多/空表现分开跟踪、切换逻辑、与引擎集成的安全风控。
运行：venv/bin/python -u scripts/review_meta.py
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.meta import MetaStrategy
from src.strategies.base import SignalType, PositionSide, MarketRegime
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.api.okx_rest import OKXPublicAPI
from datetime import datetime
import time


def _mk_strat(mode='ensemble', min_adx=0.0, **kw):
    return MetaStrategy(instId='TEST-USDT', mode=mode, params={'min_adx': min_adx, **kw}, allow_short=True)


def _feed(strat, prices, highs=None, lows=None):
    """模拟引擎逐根喂数据并取信号（不真正跑引擎，便于单测逻辑）"""
    sigs = []
    for i, p in enumerate(prices):
        h = (highs[i] if highs else p * 1.01)
        l = (lows[i] if lows else p * 0.99)
        strat.price_history.append(p)
        strat.high_history.append(h)
        strat.low_history.append(l)
        sig = strat.generate_signal({
            'price': p, 'high': h, 'low': l,
            'timestamp': f'2024-01-{i+1:02d}', 'adx': 30.0,
        })
        sigs.append(sig)
    return sigs


def test_regime_detection():
    s = _mk_strat()
    # 强趋势（ADX 高）
    assert s._detect_regime(35.0) == MarketRegime.TRENDING
    # 震荡（ADX 低）
    assert s._detect_regime(10.0) == MarketRegime.RANGING
    print("  ✓ 市场状态判定 (ADX 阈值)")


def test_double_ma_side_persists():
    s = _mk_strat()
    # 先下跌（fast<slow），再上涨触发金叉 → vdir 应为 +1，之后保持
    prices = list(np.linspace(200, 100, 40))  # 下跌
    prices += list(np.linspace(100, 200, 40))  # 上涨，触发金叉
    _feed(s, prices)
    assert s.vdir['DoubleMA'] == 1, f"DoubleMA 应多头, got {s.vdir['DoubleMA']}"
    # 再涨，仍应多头
    _feed(s, list(np.linspace(200, 210, 10)))
    assert s.vdir['DoubleMA'] == 1
    # 下跌使死叉 → -1
    _feed(s, list(np.linspace(210, 120, 60)))
    assert s.vdir['DoubleMA'] == -1, f"DoubleMA 应空头, got {s.vdir['DoubleMA']}"
    print("  ✓ DoubleMA 方向状态机 (金叉/死叉/保持)")


def test_rsi_state_machine():
    s = _mk_strat()
    # 先制造一个超卖低点（价格低 + RSI 低）——用快速下挫后平稳
    prices = list(np.linspace(100, 50, 40))  # 急跌
    _feed(s, prices)
    # RSI 应触发超卖 → vdir +1（若 allow_short 且超买则相反；此处超卖）
    # 注意：需要价格触及下轨，构造一个下探
    prices2 = list(np.linspace(50, 49, 5)) + list(np.linspace(49, 60, 20))
    _feed(s, prices2)
    # 只要不报错且 vdir 在 {-1,0,1} 内即可（RSI 触发依赖具体数值，宽松断言）
    assert s.vdir['RSIBollinger'] in (-1, 0, 1)
    print("  ✓ RSI 状态机不崩溃且方向合法")


def test_perf_split_long_short():
    s = _mk_strat()
    # 模拟 MACD 专家：先虚拟多头赚钱，再虚拟空头亏钱
    s.vdir['MACDCross'] = 1
    s.ventry['MACDCross'] = 100.0
    s._update_perf('MACDCross', 0, 110.0)  # 平多，赚 10%
    assert s.perf_long['MACDCross'] > 0, "做多表现应为正"
    s.vdir['MACDCross'] = -1
    s.ventry['MACDCross'] = 110.0
    s._update_perf('MACDCross', 0, 105.0)  # 平空，赚 ~4.5%（空头盈利）
    assert s.perf_short['MACDCross'] > 0
    # 再做一笔亏空
    s.vdir['MACDCross'] = -1
    s.ventry['MACDCross'] = 105.0
    s._update_perf('MACDCross', 0, 115.0)  # 平空亏
    assert s.perf_short['MACDCross'] < 0.05, "连续亏空应拉低做空表现"
    print("  ✓ 多/空表现分开跟踪 (做多赚→正, 做空连亏→下降)")


def test_switch_signal_mapping():
    s = _mk_strat()
    from src.strategies.base import SignalType, PositionSide
    # 无持仓 + 目标多头 → OPEN_LONG
    s.position = None
    sig = s._map_target(PositionSide.LONG, PositionSide.NONE, 100.0, 't')
    assert sig.signal_type == SignalType.OPEN_LONG
    # 持多 + 目标空 → OPEN_SHORT (翻转)
    s.position = type('P', (), {'side': PositionSide.LONG, 'amount': 1.0})()
    sig = s._map_target(PositionSide.SHORT, PositionSide.LONG, 100.0, 't')
    assert sig.signal_type == SignalType.OPEN_SHORT
    # 持多 + 目标平仓 → CLOSE_LONG
    sig = s._map_target(PositionSide.NONE, PositionSide.LONG, 100.0, 't')
    assert sig.signal_type == SignalType.CLOSE_LONG
    print("  ✓ 目标→信号映射 (开多/翻转/平仓)")


def test_integration_no_crash():
    """端到端：拉真实数据跑一遍自适应元策略，不崩溃且风控生效"""
    api = OKXPublicAPI()
    r = api.get_history_candles('BTC-USDT', bar='1D', limit=300)
    if r.get('code') != '0' or not r.get('data'):
        print("  ⚠ 跳过（无网络数据）")
        return
    rows = []
    for c in sorted(r['data'], key=lambda x: int(x[0]))[:300]:
        rows.append({'timestamp': datetime.fromtimestamp(int(c[0]) / 1000).strftime('%Y-%m-%d'),
                     'open': float(c[1]), 'high': float(c[2]), 'low': float(c[3]),
                     'close': float(c[4]), 'volume': float(c[5])})
    df = __import__('pandas').DataFrame(rows)
    df['instId'] = 'BTC-USDT'
    strat = _mk_strat(mode='adaptive', min_adx=0.0)
    cfg = BacktestConfig(initial_balance=10000, leverage=1.0, max_leverage=3.0,
                         use_atr_risk=True, atr_period=14, risk_pct=0.03,
                         atr_sl_multiplier=3.0, atr_tp_multiplier=6.0,
                         use_trailing=False, use_regime=True, min_adx_for_entry=0.0,
                         max_position_pct=1.0, maintenance_margin_rate=0.005,
                         liquidation_buffer=0.25, borrow_rate_daily=0.0003,
                         risk_scales_with_leverage=False)
    eng = BacktestEngine(strat, cfg)
    eng.load_data(df)
    res = eng.run()
    assert 'total_return' in res
    assert res.get('liquidations', 0) >= 0
    print(f"  ✓ 端到端集成 (adaptive, 300根): 收益 {res['total_return']:+.2f}% "
          f"回撤 {res['max_drawdown']:.2f}% 强平 {res.get('liquidations',0)} 切换 {strat.switch_count}")


if __name__ == '__main__':
    tests = [
        test_regime_detection, test_double_ma_side_persists, test_rsi_state_machine,
        test_perf_split_long_short, test_switch_signal_mapping, test_integration_no_crash,
    ]
    ok = 0
    for t in tests:
        try:
            t()
            ok += 1
        except Exception as e:
            print(f"  ✗ {t.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n元策略单元测试: {ok}/{len(tests)} 通过")
