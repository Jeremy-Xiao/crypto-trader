"""
诊断：SOL DoubleMA 双向模式为何亏损 -2.38%
- 复现 shorting_backtest 中的 SOL-USDT / DoubleMA / allow_short=True
- 打印每一笔 trade 的完整明细（方向、价格、数量、手续费、pnl、原因）
- 打印权益曲线关键节点
- 同时输出仅多模式做对照

运行：cd /Users/hello/Documents/workspace/crypto-trader && venv/bin/python -u scripts/diagnose_sol_short.py
"""
import os
import sys
import copy
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.shorting_backtest import fetch_candles_long, BASE_CONFIGS, ENGINE_CFG, \
    INITIAL_BALANCE, FEE_RATE, SLIPPAGE
from src.backtest.engine import BacktestEngine, BacktestConfig

SYMBOL = 'SOL-USDT'
STRAT = 'DoubleMA'


def build_engine(params, data, allow_short):
    full = copy.deepcopy(params['params'])
    full['allow_short'] = allow_short
    strategy = params['class'](instId=SYMBOL, params=full)
    cfg = BacktestConfig(
        initial_balance=INITIAL_BALANCE, fee_rate=FEE_RATE, slippage=SLIPPAGE,
        use_atr_risk=True, atr_period=14, risk_pct=full.get('risk_pct', 0.02),
        atr_multiplier=full.get('atr_multiplier', 2.0),
        atr_sl_multiplier=full.get('atr_sl_multiplier', 3.0),
        atr_tp_multiplier=full.get('atr_tp_multiplier', 6.0),
        use_trailing=False, trailing_pct=0.02,
        use_mtf=False, tf_confirm=None,
        use_regime=True, min_adx_for_entry=ENGINE_CFG['min_adx'],
        max_position_pct=ENGINE_CFG['max_pos_pct'],
    )
    eng = BacktestEngine(strategy, cfg)
    eng.load_data(data)
    return eng


def dump(eng, label, data):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"交易笔数(raw trades记录): {len(eng.trades)}")
    for i, t in enumerate(eng.trades, 1):
        print(f"  [{i}] {t['timestamp']} action={t['action']:<5} "
              f"side={t.get('side', 'long'):<12} price={t['price']:.4f} "
              f"amount={t['amount']:.4f} fee={t['fee']:.2f} "
              f"pnl={t['pnl']:.2f} reason={t['reason']}")
    print(f"止损止盈统计: {eng.stop_stats}")
    print(f"ADX拦截次数: {eng.adx_blocked}")
    eq = eng.equity_curve
    if eq:
        print(f"起始权益: {eq[0]['equity']:.2f}  结束权益: {eq[-1]['equity']:.2f}")
        lows = min(eq, key=lambda x: x['equity'])
        print(f"最低权益: {lows['equity']:.2f} @ {lows['timestamp']} (price={lows['price']:.4f})")
    print(f"最终 balance={eng.balance:.2f} position_amount={eng.position_amount:.4f} "
          f"position_side={eng.position_side}")


def main():
    print(f"拉取 {SYMBOL} 1年日K ...")
    data = fetch_candles_long(SYMBOL, '1D', 365)
    print(f"数据: {len(data)} 条  {data.iloc[0]['timestamp']} ~ {data.iloc[-1]['timestamp']}")
    print(f"起始价 {data.iloc[0]['close']:.4f}  结束价 {data.iloc[-1]['close']:.4f}  "
          f"买入持有 {(data.iloc[-1]['close']/data.iloc[0]['close']-1)*100:.2f}%")

    params = BASE_CONFIGS[STRAT]

    for allow_short in (False, True):
        eng = build_engine(params, data, allow_short)
        res = eng.run()
        dump(eng, f"{SYMBOL} {STRAT} allow_short={allow_short}", data)
        print(f"引擎汇报: total_return={res['total_return']:.2f}% "
              f"total_trades={res['total_trades']} win_rate={res['win_rate']:.1f}% "
              f"total_pnl={res['total_pnl']:.2f} total_fee={res['total_fee']:.2f}")


if __name__ == '__main__':
    main()
