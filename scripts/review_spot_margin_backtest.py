"""现货杠杆（OKX 币币杠杆）回测分析：模拟盘要不要从永续切到现货交易？

用第 23 章 half 基准（MetaStrategy adaptive + riskoff half + trading_bias），
3 币 × 3 年日 K，对比两个工具口径：

  永续（swap）    ：funding 对全部名义计费（币安实测费率）
  现货杠杆（spot_margin）：做多「自有先花、不够才借 USDT」，做空「全额借币」；
                          利息按实际负债计；强平走 OKX 官方公式（MMR=2%，≤100% 触发）

⚠️ 成本口径互斥：永续模式不计借币利息，现货模式不计 funding，不会双重计费。

OKX 实测参数（2026-09-28，基础档，单位见第31章更正）：
  借 USDT 日利率 0.000096（3.50%/年）
  借币日利率：BTC 0.00001392(0.51%/年) / ETH 0.0000276(1.01%/年) / SOL 0.00010968(4.00%/年)

运行：
    venv/bin/python -u scripts/review_spot_margin_backtest.py
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

USDT_DAILY = 0.000096       # 借 USDT 日利率（3.50%/年）
BASE_DAILY = {              # 借币日利率
    'BTC-USDT': 0.00001392,   # 0.51%/年
    'ETH-USDT': 0.0000276,    # 1.01%/年
    'SOL-USDT': 0.00010968,   # 4.00%/年
}

# (标签, 工具, 杠杆, 风险随杠杆放大)
SCENARIOS = [
    ("永续 1x（当前实盘）",   'perp', 1.0, False),
    ("永续 2x 风险放大",      'perp', 2.0, True),
    ("现货杠杆 1x",          'spot', 1.0, False),
    ("现货杠杆 2x",          'spot', 2.0, False),
    ("现货杠杆 2x 风险放大",  'spot', 2.0, True),
    ("现货杠杆 3x 风险放大",  'spot', 3.0, True),
]


def build_config(sym, tool, lev, risk_scales, funding_series):
    common = dict(ENGINE_COMMON)
    common['leverage'] = lev
    common['risk_scales_with_leverage'] = risk_scales
    if tool == 'perp':
        common['funding_series'] = funding_series
        common['borrow_rate_daily'] = 0.0
        common['margin_mode'] = 'swap'
    else:
        common['funding_series'] = None
        common['margin_mode'] = 'spot_margin'
        common['borrow_rate_daily'] = USDT_DAILY          # 做多借 USDT
        common['spot_borrow_rate_base_daily'] = BASE_DAILY[sym]   # 做空借币
        common['spot_margin_mmr'] = 0.02
        common['spot_margin_fee'] = 0.001
    return BacktestConfig(
        **common,
        risk_pct=0.02, atr_multiplier=2.0,
        atr_sl_multiplier=3.0, atr_tp_multiplier=6.0,
        min_adx_for_entry=0.0,
    )


def run(df, sym, tool, lev, risk_scales, funding_series):
    d = df.copy()
    d['market_bias'] = bias_for_dates(d['timestamp'].tolist())
    strat = MetaStrategy(instId=sym, mode='adaptive',
                         params=dict(min_adx=0.0, riskoff_long_mode=MODE),
                         allow_short=True)
    eng = BacktestEngine(strat, build_config(sym, tool, lev, risk_scales, funding_series))
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
        'long_stats': r.get('long_stats', {}),
        'short_stats': r.get('short_stats', {}),
    }


def main():
    print("=" * 116)
    print("  现货杠杆 vs 永续 | 基准=第23章 half | 3币×3年 | 现货走 OKX 官方强平价公式(MMR2%, ≤100%触发)")
    print("=" * 116)

    all_data = load_all()
    funding = {s: load_funding_series(s, all_data[s]['timestamp'].tolist()) for s in SYMBOLS}

    res = {}
    for label, tool, lev, rs in SCENARIOS:
        res[label] = {s: run(all_data[s], s, tool, lev, rs, funding[s]) for s in SYMBOLS}

    hdr = (f"{'场景':<22}{'币种':<7}{'收益':>10}{'回撤':>10}{'夏普':>7}"
           f"{'强平':>5}{'名义':>7}{'funding':>10}{'借币息':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for label, _, _, _ in SCENARIOS:
        for sym in SYMBOLS:
            r = res[label][sym]
            print(f"{label:<22}{sym.split('-')[0]:<7}{r['ret']:>+9.2f}%{r['mdd']:>+9.2f}%"
                  f"{r['sharpe']:>7.2f}{r['liq']:>5d}{r['gross']:>6.2f}x"
                  f"{r['funding']:>+10.2f}{r['borrow']:>10.2f}")
        print("-" * len(hdr))

    print("\n【平均汇总】")
    h2 = (f"{'场景':<22}{'平均收益':>10}{'平均回撤':>10}{'平均夏普':>9}"
          f"{'最差回撤':>10}{'强平合计':>9}{'成本合计':>11}")
    print(h2)
    print("-" * len(h2))
    for label, _, _, _ in SCENARIOS:
        rs = [res[label][s] for s in SYMBOLS]
        cost = sum(r['funding'] + r['borrow'] for r in rs)
        print(f"{label:<22}{np.mean([r['ret'] for r in rs]):>+9.2f}%"
              f"{np.mean([r['mdd'] for r in rs]):>+9.2f}%"
              f"{np.mean([r['sharpe'] for r in rs]):>9.2f}"
              f"{min(r['mdd'] for r in rs):>+9.2f}%"
              f"{sum(r['liq'] for r in rs):>9d}{cost:>11.2f}")

    base = res["永续 1x（当前实盘）"]
    spot1 = res["现货杠杆 1x"]
    print("\n【核心对比：永续 1x → 现货杠杆 1x】")
    for sym in SYMBOLS:
        b, s = base[sym], spot1[sym]
        print(f"  {sym:<10} 收益 {b['ret']:+.2f}% → {s['ret']:+.2f}% "
              f"({s['ret']-b['ret']:+.2f}pp) | 回撤 {b['mdd']:+.2f}% → {s['mdd']:+.2f}% | "
              f"夏普 {b['sharpe']:.2f} → {s['sharpe']:.2f} | 成本 {b['funding']+b['borrow']:.0f} → {s['funding']+s['borrow']:.0f}")
    dr = np.mean([spot1[s]['ret'] - base[s]['ret'] for s in SYMBOLS])
    print(f"  → 3币平均：{dr:+.2f}pp")

    print("\n【多空分解（现货杠杆 1x）】")
    for sym in SYMBOLS:
        L, S = spot1[sym]['long_stats'], spot1[sym]['short_stats']
        print(f"  {sym:<10} 多头 {L.get('count', L.get('trades', '?'))}笔 PnL={L.get('pnl', 0):>9.2f} | "
              f"空头 {S.get('count', S.get('trades', '?'))}笔 PnL={S.get('pnl', 0):>9.2f}")


if __name__ == '__main__':
    main()
