"""杠杆 × 工具口径 评估（2026-09-28）。

回答两个问题：
1) 资金量不大时，实盘该走「USDT 永续」还是「现货杠杆」？成本差多少？
2) 如果上杠杆，策略会变成什么样？安全吗？

关键背景：
- 实盘目前硬编码 1x（src/live/okx_paper.py:234），从未真正用过杠杆。
- 第13章（2026-08-05）做过 3x 杠杆回测，但用的是当时的老配置（D9 单策略 + ADX30），
  与当前元策略（MetaStrategy adaptive + riskoff half）不可比，需重跑。
- 两种工具的成本模型完全不同：
    * 永续：funding 对【全部名义】计费（第32章已实现）
    * 现货杠杆：借币利息只对【借入部分 notional×(1-1/L)】计费
  故必须分别用各自口径跑，不能混用（引擎里同时开会双重计费）。

OKX 借币日利率（2026-09-28 实测，基础档，正确单位见第31章更正）：
    BTC 0.001392%/天 / ETH 0.00276%/天 / SOL 0.010968%/天

运行：
    venv/bin/python -u scripts/review_leverage_plan.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.monitor.history import bias_for_dates
from src.strategies.meta import MetaStrategy
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.funding import load_funding_series
from scripts.meta_sweep import load_all

ENGINE_COMMON = dict(
    initial_balance=10000, fee_rate=0.001, slippage=0.0005,
    use_atr_risk=True, atr_period=14, use_trailing=False, use_mtf=False,
    tf_confirm=None, use_regime=True, max_leverage=3.0,
    maintenance_margin_rate=0.005, liquidation_buffer=0.25,
    max_position_pct=1.0,
)
SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
MODE = 'flatten'      # 2026-09-28 起为新默认（第34章 bug 修复后重估，见第36章）

# OKX 现货杠杆借币日利率（基础档）
OKX_DAILY_BORROW = {
    'BTC-USDT': 0.00001392,   # 0.51%/年
    'ETH-USDT': 0.0000276,    # 1.01%/年
    'SOL-USDT': 0.00010968,   # 4.00%/年
}

# (标签, 杠杆, 风险随杠杆放大?, 工具, 回撤熔断)
SCENARIOS = [
    ("1x 永续(当前实盘)",   1.0, False, 'perp', 0.0),
    ("2x 永续",             2.0, False, 'perp', 0.0),
    ("3x 永续",             3.0, False, 'perp', 0.0),
    ("1x 现货杠杆",         1.0, False, 'spot', 0.0),
    ("2x 现货杠杆",         2.0, False, 'spot', 0.0),
    ("3x 现货杠杆",         3.0, False, 'spot', 0.0),
    ("2x 永续 风险放大",    2.0, True,  'perp', 0.0),
    ("3x 永续 风险放大",    3.0, True,  'perp', 0.0),
    ("3x 永续 放大+熔断25%", 3.0, True,  'perp', 0.25),
    ("2x 现货 风险放大",    2.0, True,  'spot', 0.0),
]


def build_config(sym, lev, risk_scales, tool, halt, funding_series):
    common = dict(ENGINE_COMMON)
    common['leverage'] = lev
    common['risk_scales_with_leverage'] = risk_scales
    common['max_drawdown_halt'] = halt
    if tool == 'perp':
        common['funding_series'] = funding_series
        common['borrow_rate_daily'] = 0.0        # 永续口径：不计借币利息
    else:
        common['funding_series'] = None          # 现货杠杆口径：不计 funding
        common['borrow_rate_daily'] = OKX_DAILY_BORROW[sym]
    return BacktestConfig(
        **common,
        risk_pct=0.02, atr_multiplier=2.0,
        atr_sl_multiplier=3.0, atr_tp_multiplier=6.0,
        min_adx_for_entry=0.0,
    )


def run(df, sym, lev, risk_scales, tool, halt, funding_series):
    d = df.copy()
    d['market_bias'] = bias_for_dates(d['timestamp'].tolist())
    strat = MetaStrategy(instId=sym, mode='adaptive',
                         params=dict(min_adx=0.0, riskoff_long_mode=MODE),
                         allow_short=True)
    eng = BacktestEngine(strat, build_config(sym, lev, risk_scales, tool, halt,
                                             funding_series))
    eng.load_data(d)
    r = eng.run()
    return {
        'ret': r['total_return'],
        'mdd': r['max_drawdown'],
        'sharpe': r['sharpe_ratio'],
        'trades': r.get('total_trades', 0),
        'liq': r.get('liquidations', 0),
        'gross': r.get('max_gross_leverage', 0.0),
        'funding': r.get('funding_cost', 0.0),
        'borrow': r.get('borrow_cost', 0.0),
        'halt': r.get('dd_halt_blocked', 0),
    }


def main():
    print("=" * 112)
    print("  杠杆 × 工具口径评估 | 基准=第23章 half | 3币×3年日K | 永续用币安实测funding，现货杠杆用OKX实测借币利率")
    print("=" * 112)

    all_data = load_all()
    funding = {s: load_funding_series(s, all_data[s]['timestamp'].tolist())
               for s in SYMBOLS}

    results = {}
    for label, lev, rs, tool, halt in SCENARIOS:
        row = {}
        for sym in SYMBOLS:
            row[sym] = run(all_data[sym], sym, lev, rs, tool, halt, funding[sym])
        results[label] = row

    hdr = (f"{'场景':<22}{'币种':<8}{'收益':>10}{'回撤':>10}{'夏普':>7}"
           f"{'强平':>5}{'名义杠杆':>9}{'funding':>10}{'借币息':>9}{'熔断':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for label, _, _, _, _ in SCENARIOS:
        for sym in SYMBOLS:
            r = results[label][sym]
            print(f"{label:<22}{sym.split('-')[0]:<8}{r['ret']:>+9.2f}%{r['mdd']:>+9.2f}%"
                  f"{r['sharpe']:>7.2f}{r['liq']:>5d}{r['gross']:>8.2f}x"
                  f"{r['funding']:>+10.2f}{r['borrow']:>9.2f}{r['halt']:>6d}")
        print("-" * len(hdr))

    print("\n【平均汇总】")
    h2 = (f"{'场景':<22}{'平均收益':>10}{'平均回撤':>10}{'平均夏普':>9}"
          f"{'最差回撤':>10}{'强平合计':>9}{'成本合计':>10}")
    print(h2)
    print("-" * len(h2))
    for label, _, _, _, _ in SCENARIOS:
        rs = [results[label][s] for s in SYMBOLS]
        cost = sum(r['funding'] + r['borrow'] for r in rs)
        print(f"{label:<22}{np.mean([r['ret'] for r in rs]):>+9.2f}%"
              f"{np.mean([r['mdd'] for r in rs]):>+9.2f}%"
              f"{np.mean([r['sharpe'] for r in rs]):>9.2f}"
              f"{min(r['mdd'] for r in rs):>+9.2f}%"
              f"{sum(r['liq'] for r in rs):>9d}{cost:>10.2f}")

    print("\n【对照：1x 永续 基准】")
    base = results["1x 永续(当前实盘)"]
    print(f"  平均收益 {np.mean([base[s]['ret'] for s in SYMBOLS]):+.2f}%  "
          f"平均回撤 {np.mean([base[s]['mdd'] for s in SYMBOLS]):+.2f}%")


if __name__ == '__main__':
    main()
