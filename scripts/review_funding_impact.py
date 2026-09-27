"""验证：给回测补上「永续资金费率（funding）」后的影响。

背景：回测引擎原本没有任何 funding 模型——引擎里的 borrow_rate_daily 只在 leverage>1
时生效（现货杠杆口径），而实盘跑的是 USDT 永续 1x 全仓，每 8 小时结算一次资金费率。
这是实盘真实存在、回测完全缺失的成本项（第 31 章）。

基准 = 第 23 章的 A 方案 half 配置（MetaStrategy adaptive + riskoff half + trading_bias），
3 年日 K（2023-08 ~ 2026-08），3 币。
对比 = funding 关（旧口径） vs funding 开（币安公开仓库历史实测费率）。

运行：
    cd /Users/hello/Documents/workspace/crypto-trader
    venv/bin/python -u scripts/review_funding_impact.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.monitor.history import bias_for_dates
from src.strategies.meta import MetaStrategy
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.funding import load_funding_series, summarize
from scripts.meta_sweep import load_all

ENGINE_COMMON = dict(
    initial_balance=10000, fee_rate=0.001, slippage=0.0005,
    use_atr_risk=True, atr_period=14, use_trailing=False, use_mtf=False,
    tf_confirm=None, use_regime=True, leverage=1.0, max_leverage=3.0,
    maintenance_margin_rate=0.005, liquidation_buffer=0.25,
    borrow_rate_daily=0.0003, risk_scales_with_leverage=False,
    max_position_pct=1.0,
)
SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
MODE = 'half'          # 第 23 章的默认方案


def build_config(p, funding_series=None, fee_rate=0.001):
    common = dict(ENGINE_COMMON)
    common['fee_rate'] = fee_rate
    return BacktestConfig(
        **common,
        risk_pct=p.get('risk_pct', 0.02),
        atr_multiplier=p.get('atr_multiplier', 2.0),
        atr_sl_multiplier=p.get('atr_sl_multiplier', 3.0),
        atr_tp_multiplier=p.get('atr_tp_multiplier', 6.0),
        min_adx_for_entry=p.get('min_adx', 0.0),
        funding_series=funding_series,
    )


def run_once(df, funding_series, fee_rate=0.001):
    p = dict(min_adx=0.0, riskoff_long_mode=MODE)
    d = df.copy()
    d['market_bias'] = bias_for_dates(d['timestamp'].tolist())
    strat = MetaStrategy(instId=d['instId'].iloc[0], mode='adaptive',
                         params=p, allow_short=True)
    eng = BacktestEngine(strat, build_config(p, funding_series, fee_rate))
    eng.load_data(d)
    r = eng.run()
    return {
        'ret': r['total_return'],
        'mdd': r['max_drawdown'],
        'sharpe': r['sharpe_ratio'],
        'trades': r.get('total_trades', 0),
        'funding': r.get('funding_cost', 0.0),
        'fee': r.get('total_fee', 0.0),
        'final': 10000 * (1 + r['total_return'] / 100),
    }


def main():
    print("=" * 104)
    print("  funding 影响验证 | 基准=第23章 A方案 half | 3年日K | 费率数据源=币安公开仓库")
    print("=" * 104)

    all_data = load_all()
    results = {}

    for sym in SYMBOLS:
        df = all_data[sym]
        series = load_funding_series(sym, df['timestamp'].tolist())
        print(f"\n[{sym}] {summarize(series, '费率')}")
        results[sym] = {
            'old':      run_once(df, None, 0.001),    # 旧口径：无 funding，费率 0.1%
            'fund':     run_once(df, series, 0.001),  # 补 funding，费率仍 0.1%
            'real':     run_once(df, series, 0.0005), # 实盘口径：funding + 永续 taker 0.05%
        }

    SCENARIOS = [
        ("无 funding(旧口径)", "old"),
        ("含 funding", "fund"),
        ("funding+费率0.05%(实盘口径)", "real"),
    ]

    print("\n" + "=" * 104)
    hdr = (f"{'币种':<10}{'口径':<26}{'收益':>10}{'回撤':>10}{'夏普':>8}"
           f"{'交易':>6}{'手续费':>10}{'funding':>11}{'期末余额':>11}")
    print(hdr)
    print("-" * len(hdr))

    for sym in SYMBOLS:
        for tag, key in SCENARIOS:
            r = results[sym][key]
            print(f"{sym:<10}{tag:<26}{r['ret']:>+9.2f}%{r['mdd']:>+9.2f}%"
                  f"{r['sharpe']:>8.2f}{r['trades']:>6d}{r['fee']:>10.2f}"
                  f"{r['funding']:>+11.2f}{r['final']:>11.2f}")
        d1 = results[sym]['fund']['ret'] - results[sym]['old']['ret']
        d2 = results[sym]['real']['ret'] - results[sym]['old']['ret']
        print(f"{'':<10}{'Δ 仅加funding':<26}{d1:>+9.2f}pp")
        print(f"{'':<10}{'Δ 实盘口径-旧口径':<26}{d2:>+9.2f}pp")
        print("-" * len(hdr))

    print("\n平均：")
    for tag, key in SCENARIOS:
        rets = [results[s][key]['ret'] for s in SYMBOLS]
        mdds = [results[s][key]['mdd'] for s in SYMBOLS]
        shs = [results[s][key]['sharpe'] for s in SYMBOLS]
        fund = [results[s][key]['funding'] for s in SYMBOLS]
        print(f"  {tag:<28} 收益均={np.mean(rets):>+7.2f}%  "
              f"回撤均={np.mean(mdds):>+7.2f}%  夏普均={np.mean(shs):>5.2f}  "
              f"funding合计={sum(fund):>+9.2f}")

    for tag, key in (("仅加 funding", 'fund'), ("实盘口径(含0.05%费率)", 'real')):
        d = np.mean([results[s][key]['ret'] - results[s]['old']['ret'] for s in SYMBOLS])
        print(f"\n  → {tag} 使 3 币平均收益变化：{d:+.2f}pp")


if __name__ == '__main__':
    main()
