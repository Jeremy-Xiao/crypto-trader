"""
动态策略切换元策略回测（第五轮）| 3年日K
对比：4个单策略 vs 元策略（4种模式） vs 买入持有 / 货币基金

设计：
- 公平比较：单策略与元策略用同一套引擎配置（ATR风控、做空、满仓、1x）
- 主实验 min_adx_for_entry=0：让元策略的「状态路由」能在震荡市启用均值回归策略，
  真正检验「震荡市+趋势市都能赚」；同时跑 min_adx=30 与上一轮 1x 结果可比。
- 元策略 4 模式：regime(纯状态路由) / ensemble(状态+表现加权) / perf(纯表现加权) / adaptive(王者通吃)

运行：cd /Users/hello/Documents/workspace/crypto-trader && venv/bin/python -u scripts/meta_backtest.py
"""
import os
import sys
import json
import copy
import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.okx_rest import OKXPublicAPI
from src.strategies.double_ma import DoubleMAStrategy
from src.strategies.rsi_bollinger import RSIBollingerStrategy
from src.strategies.macd_cross import MACDCrossStrategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.meta import MetaStrategy
from src.backtest.engine import BacktestEngine, BacktestConfig

SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
INITIAL_BALANCE = 10000
FEE_RATE = 0.001
SLIPPAGE = 0.0005
MMF_ANNUAL = 0.045
YEARS = 3
TOTAL_DAYS = 365 * YEARS
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def fetch_history(symbol, bar='1D', total=TOTAL_DAYS, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{symbol}_{bar}_{total}d.csv")
    if use_cache and os.path.exists(cache):
        df = pd.read_csv(cache)
        if len(df) >= total * 0.9:
            return df
    api = OKXPublicAPI()
    bars, after, seen = [], None, set()
    while len(seen) < total:
        r = api.get_history_candles(symbol, bar=bar, limit=300, after=after)
        if r.get('code') != '0' or not r.get('data'):
            break
        data = sorted(r['data'], key=lambda x: int(x[0]))
        new = [c for c in data if c[0] not in seen]
        if not new:
            break
        for c in new:
            seen.add(c[0])
        bars.extend(new)
        after = data[0][0]
        if len(data) < 300:
            break
        time.sleep(0.12)
    rows = []
    for c in sorted(bars, key=lambda x: int(x[0])):
        rows.append({
            'timestamp': datetime.fromtimestamp(int(c[0]) / 1000).strftime('%Y-%m-%d'),
            'open': float(c[1]), 'high': float(c[2]),
            'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5]),
        })
    df = pd.DataFrame(rows)
    if len(df) > total:
        df = df.iloc[-total:].reset_index(drop=True)
    if len(df) > 0:
        df.to_csv(cache, index=False)
    return df


BASE_CONFIGS = {
    'DoubleMA': dict(fast_period=10, slow_period=30, trend_period=60,
                     position_pct=0.2, risk_pct=0.03,
                     atr_multiplier=2.5, atr_sl_multiplier=3.0, atr_tp_multiplier=6.0),
    'MACD': dict(fast_period=12, slow_period=26, signal_period=9,
                 position_pct=0.2, risk_pct=0.03,
                 atr_multiplier=2.5, atr_sl_multiplier=3.0, atr_tp_multiplier=6.0),
    'Breakout': dict(lookback=20, position_pct=0.2, risk_pct=0.03,
                     atr_multiplier=2.0, atr_sl_multiplier=3.0, atr_tp_multiplier=6.0),
    'RSI_Boll': dict(rsi_period=14, boll_period=20, boll_std=2.0,
                     rsi_low=30, rsi_high=70, position_pct=0.2, risk_pct=0.02,
                     atr_multiplier=2.0, atr_sl_multiplier=3.0, atr_tp_multiplier=6.0),
}
ENGINE_COMMON = dict(
    initial_balance=INITIAL_BALANCE, fee_rate=FEE_RATE, slippage=SLIPPAGE,
    use_atr_risk=True, atr_period=14,
    use_trailing=False, use_mtf=False, tf_confirm=None, use_regime=True,
    leverage=1.0, max_leverage=3.0,
    maintenance_margin_rate=0.005, liquidation_buffer=0.25,
    borrow_rate_daily=0.0003, risk_scales_with_leverage=False,
    max_position_pct=1.0,
)


def build_engine_config(strat_params):
    return BacktestConfig(
        **ENGINE_COMMON,
        risk_pct=strat_params.get('risk_pct', 0.02),
        atr_multiplier=strat_params.get('atr_multiplier', 2.0),
        atr_sl_multiplier=strat_params.get('atr_sl_multiplier', 3.0),
        atr_tp_multiplier=strat_params.get('atr_tp_multiplier', 6.0),
        min_adx_for_entry=strat_params.get('min_adx', 0.0),
    )


def run_single(name, params, data, min_adx):
    full = copy.deepcopy(params)
    full['allow_short'] = True
    full['min_adx'] = min_adx
    cfg = BASE_CONFIGS[name]
    strategy = {
        'DoubleMA': DoubleMAStrategy, 'MACD': MACDCrossStrategy,
        'Breakout': BreakoutStrategy, 'RSI_Boll': RSIBollingerStrategy,
    }[name](instId=data.iloc[0].get('instId', 'UNKNOWN'), params=full)
    config = build_engine_config(full)
    engine = BacktestEngine(strategy, config)
    engine.load_data(data)
    res = engine.run()
    res['equity_curve'] = [e['equity'] for e in engine.equity_curve]
    res['switch_count'] = 0
    return res


def run_meta(mode, data, min_adx, meta_params=None):
    full = dict(min_adx=min_adx)
    if meta_params:
        full.update(meta_params)
    strategy = MetaStrategy(instId=data.iloc[0].get('instId', 'UNKNOWN'),
                            mode=mode, params=full, allow_short=True)
    config = build_engine_config(full)
    engine = BacktestEngine(strategy, config)
    engine.load_data(data)
    res = engine.run()
    res['equity_curve'] = [e['equity'] for e in engine.equity_curve]
    res['switch_count'] = strategy.switch_count
    res['expert_perf'] = {k: round(v, 4) for k, v in strategy.perf.items()}
    res['expert_perf_long'] = {k: round(v, 4) for k, v in strategy.perf_long.items()}
    res['expert_perf_short'] = {k: round(v, 4) for k, v in strategy.perf_short.items()}
    if os.environ.get('META_DEBUG'):
        from collections import Counter
        print(f"   [DEBUG {mode}] target分布: {dict(Counter(strategy._dbg_targets))}")
    return res


def buy_and_hold(all_data):
    bh = {}
    for s in SYMBOLS:
        if s in all_data:
            df = all_data[s]['df']
            bh[s] = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    if bh:
        bh['PORTFOLIO_EW'] = float(np.mean(list(bh.values())))
    return bh


def summarize(rows):
    if not rows:
        return {}
    rets = [r['total_return'] for r in rows]
    return {
        'avg_return': float(np.mean(rets)),
        'median_return': float(np.median(rets)),
        'best': float(np.max(rets)),
        'worst': float(np.min(rets)),
        'profitable': int(sum(1 for x in rets if x > 0)),
        'count': len(rows),
        'avg_max_drawdown': float(np.mean([r['max_drawdown'] for r in rows])),
        'avg_sharpe': float(np.mean([r['sharpe_ratio'] for r in rows])),
        'avg_trades': float(np.mean([r['total_trades'] for r in rows])),
        'total_liquidations': int(sum(r.get('liquidations', 0) for r in rows)),
    }


def main():
    print("=" * 104)
    print(f"  动态策略切换元策略回测 | {YEARS}年日K | 单策略 vs 元策略(4模式)")
    print("=" * 104)

    all_data = {}
    print(f"\n加载 {YEARS} 年日K数据（history-candles 缓存）...")
    for symbol in SYMBOLS:
        df = fetch_history(symbol, '1D', TOTAL_DAYS)
        if df is not None and len(df) > 0:
            df = df.copy()
            df['instId'] = symbol
            all_data[symbol] = {'df': df}
            print(f"  {symbol}: {len(df)} bars ({df.iloc[0]['timestamp']} ~ {df.iloc[-1]['timestamp']})")
    if not all_data:
        print("无数据，退出")
        return

    days = len(all_data[SYMBOLS[0]]['df'])
    bh = buy_and_hold(all_data)
    mfm = (pow(1 + MMF_ANNUAL, days / 365.0) - 1) * 100
    print(f"\n基准（{days}天）：买入持有组合 {bh['PORTFOLIO_EW']:+.2f}%  |  货币基金 {mfm:+.2f}%")

    # 实验组定义： (label, kind, key, min_adx, meta_params)
    SINGLES = [(n, 'single', n, adx, None) for n in BASE_CONFIGS for adx in (0.0, 30.0)]
    METAS = [(f"meta_{m}_adx{int(adx)}", 'meta', m, adx, None)
             for m in ('regime', 'ensemble', 'perf', 'adaptive') for adx in (0.0, 30.0)]
    EXPS = SINGLES + METAS

    results = []
    by_label = {}
    for label, kind, key, min_adx, mp in EXPS:
        print("\n" + "=" * 104)
        print(f"  {label}  (min_adx={min_adx})")
        print("=" * 104)
        for symbol in SYMBOLS:
            data = all_data[symbol]['df']
            if kind == 'single':
                r = run_single(key, BASE_CONFIGS[key], data, min_adx)
            else:
                r = run_meta(key, data, min_adx, mp)
            r.update({'symbol': symbol, 'label': label, 'kind': kind, 'min_adx': min_adx})
            results.append(r)
            by_label.setdefault(label, []).append(r)
            print(f"  {symbol:<10} 收益={r['total_return']:>+9.2f}%  回撤={r['max_drawdown']:>8.2f}%  "
                  f"夏普={r['sharpe_ratio']:>5.2f}  交易={r['total_trades']:>3}  "
                  f"强平={r.get('liquidations',0)}  切换={r.get('switch_count',0)}")

    # ==================== 汇总 ====================
    print("\n" + "=" * 104)
    print("  汇总对比（按标签）")
    print("=" * 104)
    print(f"{'标签':<22}{'平均收益':>11}{'中位数':>10}{'最好':>10}{'最差':>10}"
          f"{'盈利组':>9}{'平均回撤':>10}{'夏普':>8}{'交易':>7}")
    summaries = {}
    for label, rows in by_label.items():
        s = summarize(rows)
        summaries[label] = s
        print(f"{label:<22}{s['avg_return']:>+10.2f}%{s['median_return']:>+9.2f}%"
              f"{s['best']:>+9.2f}%{s['worst']:>+9.2f}%{s['profitable']:>5}/{s['count']:<3}"
              f"{s['avg_max_drawdown']:>+9.2f}%{s['avg_sharpe']:>8.2f}{s['avg_trades']:>7.1f}")

    print(f"\n  基准: 买入持有组合 {bh['PORTFOLIO_EW']:+.2f}%  |  货币基金 {mfm:+.2f}%")

    # 单策略(ADX0) 最佳 vs 元策略(ADX0) 最佳
    print("\n聚焦 min_adx=0（公平对比，允许震荡市交易）：")
    single0 = [r for r in results if r['kind'] == 'single' and r['min_adx'] == 0.0]
    meta0 = [r for r in results if r['kind'] == 'meta' and r['min_adx'] == 0.0]
    s0 = summarize(single0)
    m0 = summarize(meta0)
    best_single = max(single0, key=lambda x: x['total_return'])
    best_meta = max(meta0, key=lambda x: x['total_return'])
    print(f"  单策略平均 {s0['avg_return']:+.2f}% (最好 {best_single['label']}/{best_single['symbol']} "
          f"{best_single['total_return']:+.2f}%)")
    print(f"  元策略平均 {m0['avg_return']:+.2f}% (最好 {best_meta['label']}/{best_meta['symbol']} "
          f"{best_meta['total_return']:+.2f}%)")
    for label, rows in by_label.items():
        if label.startswith('meta_') and 'adx0' in label:
            s = summarize(rows)
            print(f"    {label:<22} 平均 {s['avg_return']:>+8.2f}%  中位 {s['median_return']:>+8.2f}%  "
                  f"盈利 {s['profitable']}/{s['count']}  回撤 {s['avg_max_drawdown']:>+7.2f}%")

    # 元策略内部模式对比（adx0）
    print("\n元策略 4 模式对比（min_adx=0，最高收益单组）：")
    for m in ('regime', 'ensemble', 'perf', 'adaptive'):
        label = f"meta_{m}_adx0"
        rows = by_label.get(label, [])
        if rows:
            s = summarize(rows)
            best = max(rows, key=lambda x: x['total_return'])
            print(f"  {m:<10} 平均 {s['avg_return']:>+8.2f}%  中位 {s['median_return']:>+8.2f}%  "
                  f"盈利 {s['profitable']}/{s['count']}  最好 {best['symbol']} {best['total_return']:+.2f}%")

    print("\nTop 8 单组表现：")
    for r in sorted(results, key=lambda x: -x['total_return'])[:8]:
        print(f"  {r['label']:<22} {r['symbol']:<10} 收益 {r['total_return']:>+9.2f}%  "
              f"回撤 {r['max_drawdown']:>8.2f}%  夏普 {r['sharpe_ratio']:>5.2f}")

    print("\n安全性核查：")
    total_liq = sum(r.get('liquidations', 0) for r in results)
    max_gross = max((r.get('max_gross_leverage', 0) for r in results), default=0)
    print(f"  全部 {len(results)} 组累计强平: {total_liq}  最大名义杠杆: {max_gross:.2f}x")

    out = {
        'generated_at': datetime.now().isoformat(),
        'period_days': days,
        'period': f"{all_data[SYMBOLS[0]]['df'].iloc[0]['timestamp']} ~ "
                  f"{all_data[SYMBOLS[0]]['df'].iloc[-1]['timestamp']}",
        'symbols': SYMBOLS,
        'benchmarks': {'buy_hold': bh, 'money_market': round(mfm, 2)},
        'summaries': summaries,
        'results': [{k: v for k, v in r.items() if k != 'equity_curve'} for r in results],
        'equity_curves': {
            f"{r['label']}|{r['symbol']}": r['equity_curve'] for r in results
        },
    }
    os.makedirs('backtest_reports', exist_ok=True)
    path = 'backtest_reports/meta_backtest.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)
    print(f"\n报告已保存: {path}")


if __name__ == '__main__':
    main()
