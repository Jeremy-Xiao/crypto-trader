"""
杠杆回测（第四轮）| 3年日K | 杠杆 1x / 2x / 3x 对照

设计要点：
- 拉 3 年日K（history-candles 接口，深度远大于 /market/candles），本地 CSV 缓存
- 唯一变量是 leverage，其余参数（策略、ATR风控、ADX过滤、做空开关）完全一致
- 风控前提：单笔风险不随杠杆放大（risk_scales_with_leverage=False），
  杠杆只放开资金约束；叠加维持保证金强平 + 借币利息 + 硬上限3x
- 额外跑一组「3x + 回撤熔断25%」，验证熔断能否改善尾部风险

运行：cd /Users/hello/Documents/workspace/crypto-trader && venv/bin/python -u scripts/leverage_backtest.py
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
from src.backtest.engine import BacktestEngine, BacktestConfig

SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
INITIAL_BALANCE = 10000
FEE_RATE = 0.001
SLIPPAGE = 0.0005
MMF_ANNUAL = 0.045
YEARS = 3
TOTAL_DAYS = 365 * YEARS
LEVERAGES = [1.0, 2.0, 3.0]
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def fetch_history(symbol, bar='1D', total=TOTAL_DAYS, use_cache=True):
    """分页拉取长周期历史K线（history-candles），带本地缓存"""
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


# 统一最优参数（沿用 D9：ADX30 + 满仓 + 无移动止损 + 宽止损3x/止盈6x）
BASE_CONFIGS = {
    'DoubleMA': {
        'class': DoubleMAStrategy,
        'params': dict(fast_period=10, slow_period=30, trend_period=60,
                       position_pct=0.2, risk_pct=0.03,
                       atr_multiplier=2.5, atr_sl_multiplier=3.0, atr_tp_multiplier=6.0),
    },
    'MACD': {
        'class': MACDCrossStrategy,
        'params': dict(fast_period=12, slow_period=26, signal_period=9,
                       position_pct=0.2, risk_pct=0.03,
                       atr_multiplier=2.5, atr_sl_multiplier=3.0, atr_tp_multiplier=6.0),
    },
    'Breakout': {
        'class': BreakoutStrategy,
        'params': dict(lookback=20, position_pct=0.2, risk_pct=0.03,
                       atr_multiplier=2.0, atr_sl_multiplier=3.0, atr_tp_multiplier=6.0),
    },
    'RSI_Boll': {
        'class': RSIBollingerStrategy,
        'params': dict(rsi_period=14, boll_period=20, boll_std=2.0,
                       rsi_low=30, rsi_high=70, position_pct=0.2, risk_pct=0.02,
                       atr_multiplier=2.0, atr_sl_multiplier=3.0, atr_tp_multiplier=6.0),
    },
}

ENGINE_CFG = dict(min_adx=30.0, max_pos_pct=1.0)


def run_one(strategy_class, params, data, leverage, dd_halt=0.0, risk_scales=False):
    full = copy.deepcopy(params)
    full['allow_short'] = True
    strategy = strategy_class(instId=data.iloc[0].get('instId', 'UNKNOWN'), params=full)
    config = BacktestConfig(
        initial_balance=INITIAL_BALANCE, fee_rate=FEE_RATE, slippage=SLIPPAGE,
        use_atr_risk=True, atr_period=14, risk_pct=full.get('risk_pct', 0.02),
        atr_multiplier=full.get('atr_multiplier', 2.0),
        atr_sl_multiplier=full.get('atr_sl_multiplier', 3.0),
        atr_tp_multiplier=full.get('atr_tp_multiplier', 6.0),
        use_trailing=False, use_mtf=False, tf_confirm=None, use_regime=True,
        min_adx_for_entry=ENGINE_CFG['min_adx'],
        max_position_pct=ENGINE_CFG['max_pos_pct'],
        # ===== 杠杆与风控 =====
        leverage=leverage,
        max_leverage=3.0,
        maintenance_margin_rate=0.005,
        liquidation_buffer=0.25,
        borrow_rate_daily=0.0003,
        risk_scales_with_leverage=risk_scales,
        max_drawdown_halt=dd_halt,
    )
    engine = BacktestEngine(strategy, config)
    engine.load_data(data)
    res = engine.run()
    res['equity_curve'] = [e['equity'] for e in engine.equity_curve]
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
        'count': len(rets),
        'avg_max_drawdown': float(np.mean([r['max_drawdown'] for r in rows])),
        'worst_max_drawdown': float(np.min([r['max_drawdown'] for r in rows])),
        'avg_sharpe': float(np.mean([r['sharpe_ratio'] for r in rows])),
        'total_liquidations': int(sum(r.get('liquidations', 0) for r in rows)),
        'total_borrow_cost': float(sum(r.get('borrow_cost', 0.0) for r in rows)),
        'avg_trades': float(np.mean([r['total_trades'] for r in rows])),
    }


def main():
    print("=" * 104)
    print(f"  杠杆回测 | {YEARS}年日K | 杠杆 1x / 2x / 3x 对照（唯一变量=leverage）")
    print("=" * 104)

    all_data = {}
    print(f"\n拉取 OKX {YEARS} 年日K数据（history-candles）...")
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
    print(f"\n基准（{days}天 ≈ {days/365:.2f}年）：")
    print(f"  货币基金(4.5%年化复利): {mfm:+.2f}%")
    print("  买入持有: " + "  ".join(f"{k}={v:+.2f}%" for k, v in bh.items()))

    # 三种模式：
    #   A 风险恒定（risk_scales=False）：杠杆只放开资金约束，单笔风险仍是 risk_pct
    #   B 风险放大（risk_scales=True）：杠杆同时放大仓位与单笔风险，才是「真正用上杠杆」
    #   C 风险放大 + 回撤熔断25%：给激进模式加安全阀
    MODES = [
        ('1x',            1.0, False, 0.0),
        ('2x_riskparity', 2.0, False, 0.0),
        ('3x_riskparity', 3.0, False, 0.0),
        ('2x_scaled',     2.0, True,  0.0),
        ('3x_scaled',     3.0, True,  0.0),
        ('3x_scaled_halt', 3.0, True, 0.25),
    ]
    ORDER = [m[0] for m in MODES]

    lev_results = {k: [] for k in ORDER}
    grid = []

    for key, lev, scales, halt in MODES:
        print("\n" + "=" * 104)
        label = ('风险恒定' if not scales else '风险随杠杆放大')
        print(f"  {key}  (杠杆{lev}x | {label}"
              f"{' | 回撤熔断25%' if halt else ''})")
        print("=" * 104)
        for symbol in SYMBOLS:
            data = all_data[symbol]['df']
            for name, cfg in BASE_CONFIGS.items():
                r = run_one(cfg['class'], cfg['params'], data, lev,
                            dd_halt=halt, risk_scales=scales)
                r.update({'symbol': symbol, 'strategy': name, 'lev': key})
                lev_results[key].append(r)
                grid.append(r)
                print(f"  {symbol:<10} {name:<10} 收益={r['total_return']:>+9.2f}%  "
                      f"回撤={r['max_drawdown']:>8.2f}%  夏普={r['sharpe_ratio']:>5.2f}  "
                      f"交易={r['total_trades']:>3}  强平={r.get('liquidations',0)}  "
                      f"利息={r.get('borrow_cost',0):>5.0f}  "
                      f"实际杠杆={r.get('max_gross_leverage',0):.2f}x  "
                      f"熔断={r.get('dd_halt_blocked',0)}")

    # ==================== 汇总 ====================
    print("\n" + "=" * 104)
    print("  汇总对比")
    print("=" * 104)
    print(f"{'模式':<17}{'平均收益':>11}{'中位数':>10}{'最好':>10}{'最差':>10}"
          f"{'盈利组':>8}{'平均回撤':>10}{'最差回撤':>10}{'夏普':>8}{'强平':>6}{'利息':>8}")
    summaries = {}
    for key in ORDER:
        s = summarize(lev_results[key])
        summaries[key] = s
        print(f"{key:<17}{s['avg_return']:>+10.2f}%{s['median_return']:>+9.2f}%"
              f"{s['best']:>+9.2f}%{s['worst']:>+9.2f}%{s['profitable']:>5}/{s['count']:<3}"
              f"{s['avg_max_drawdown']:>+9.2f}%{s['worst_max_drawdown']:>+9.2f}%"
              f"{s['avg_sharpe']:>8.2f}{s['total_liquidations']:>6}{s['total_borrow_cost']:>8.0f}")

    print(f"\n  基准: 买入持有组合 {bh['PORTFOLIO_EW']:+.2f}%  |  货币基金 {mfm:+.2f}%")

    print("\n杠杆放大效率（该模式平均收益 / 1倍平均收益）：")
    base = summaries['1x']['avg_return']
    for key in ORDER[1:]:
        ratio = summaries[key]['avg_return'] / base if abs(base) > 1e-6 else float('nan')
        print(f"  {key:<17} {summaries[key]['avg_return']:+7.2f}% / {base:+.2f}% = {ratio:5.2f}x")

    print("\nTop 8 单组表现：")
    for r in sorted(grid, key=lambda x: -x['total_return'])[:8]:
        print(f"  {r['lev']:<17} {r['symbol']:<10} {r['strategy']:<10} "
              f"{r['total_return']:>+9.2f}%  回撤{r['max_drawdown']:>8.2f}%  夏普{r['sharpe_ratio']:>5.2f}")
    print("\nWorst 5 单组表现：")
    for r in sorted(grid, key=lambda x: x['total_return'])[:5]:
        print(f"  {r['lev']:<17} {r['symbol']:<10} {r['strategy']:<10} "
              f"{r['total_return']:>+9.2f}%  回撤{r['max_drawdown']:>8.2f}%  强平{r.get('liquidations',0)}次")

    print("\n风险调整（平均收益 / |平均回撤|，越高越好）：")
    for key in ORDER:
        s = summaries[key]
        calmar = s['avg_return'] / abs(s['avg_max_drawdown']) if s['avg_max_drawdown'] != 0 else 0
        print(f"  {key:<17} {calmar:>6.2f}")

    print("\n安全性核查：")
    total_liq = sum(summaries[k]['total_liquidations'] for k in ORDER)
    max_gross = max((r.get('max_gross_leverage', 0) for r in grid), default=0)
    print(f"  全部 {len(grid)} 组回测累计强平次数: {total_liq}")
    print(f"  实际达到的最大名义杠杆: {max_gross:.2f}x（硬上限 3.00x）")
    print(f"  杠杆越界: {'否' if max_gross <= 3.01 else '是！'}")

    out = {
        'generated_at': datetime.now().isoformat(),
        'period_days': days,
        'period': f"{all_data[SYMBOLS[0]]['df'].iloc[0]['timestamp']} ~ "
                  f"{all_data[SYMBOLS[0]]['df'].iloc[-1]['timestamp']}",
        'symbols': SYMBOLS,
        'leverages': ['1x', '2x', '3x', '3x_ddhalt'],
        'risk_controls': {
            'max_leverage': 3.0,
            'maintenance_margin_rate': 0.005,
            'liquidation_buffer': 0.25,
            'borrow_rate_daily': 0.0003,
            'risk_scales_with_leverage': False,
            'min_adx_for_entry': ENGINE_CFG['min_adx'],
            'max_position_pct': ENGINE_CFG['max_pos_pct'],
            'atr_sl': 3.0, 'atr_tp': 6.0, 'allow_short': True,
        },
        'benchmarks': {'buy_hold': bh, 'money_market': round(mfm, 2)},
        'summaries': summaries,
        'results': [{k: v for k, v in r.items() if k != 'equity_curve'} for r in grid],
        'equity_curves': {
            f"{r['lev']}|{r['symbol']}|{r['strategy']}": r['equity_curve'] for r in grid
        },
    }
    os.makedirs('backtest_reports', exist_ok=True)
    path = 'backtest_reports/leverage_backtest.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)
    print(f"\n报告已保存: {path}")


if __name__ == '__main__':
    main()
