"""
做空机制回测（第三轮）
- 拉取1年日K（分页）
- 核心对比：相同参数下 allow_short=False(仅多) vs allow_short=True(双向含做空)
- 隔离「做空」这一个变量的增量贡献
- 买入持有 + 货币基金基准

运行：cd /Users/hello/Documents/workspace/crypto-trader && venv/bin/python -u scripts/shorting_backtest.py
"""
import os
import sys
import json
import copy
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


def fetch_candles_long(symbol, bar='1D', total=365):
    """分页拉取较长历史K线"""
    api = OKXPublicAPI()
    bars = []
    after = None
    while len(bars) < total:
        params = {'instId': symbol, 'bar': bar, 'limit': 300}
        if after is not None:
            params['after'] = after
        result = api._public_request("/api/v5/market/candles", params)
        if result.get('code') != '0' or not result.get('data'):
            break
        data = sorted(result['data'], key=lambda x: x[0])
        if not data:
            break
        bars.extend(data)
        after = data[0][0]
        if len(data) < 300:
            break
    rows = []
    for c in sorted(bars, key=lambda x: x[0]):
        rows.append({
            'timestamp': datetime.fromtimestamp(int(c[0]) / 1000).strftime('%Y-%m-%d'),
            'open': float(c[1]), 'high': float(c[2]),
            'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5]),
        })
    df = pd.DataFrame(rows)
    if len(df) > total:
        df = df.iloc[-total:]
    return df


# 统一最优参数（参考 D9：ADX30 + 满仓 + 无移动止损 + 宽止损3x/止盈6x）
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

# 引擎统一配置（趋势过滤 + 满仓 + 无移动止损）
ENGINE_CFG = dict(min_adx=30.0, max_pos_pct=1.0, use_trailing=False, use_atr=True)


def run_one(strategy_class, params, data, allow_short):
    full_params = copy.deepcopy(params)
    full_params['allow_short'] = allow_short
    strategy = strategy_class(instId=data.iloc[0].get('instId', 'UNKNOWN'), params=full_params)
    config = BacktestConfig(
        initial_balance=INITIAL_BALANCE, fee_rate=FEE_RATE, slippage=SLIPPAGE,
        use_atr_risk=True, atr_period=14, risk_pct=full_params.get('risk_pct', 0.02),
        atr_multiplier=full_params.get('atr_multiplier', 2.0),
        atr_sl_multiplier=full_params.get('atr_sl_multiplier', 3.0),
        atr_tp_multiplier=full_params.get('atr_tp_multiplier', 6.0),
        use_trailing=False, trailing_pct=0.02,
        use_mtf=False, tf_confirm=None,
        use_regime=True, min_adx_for_entry=ENGINE_CFG['min_adx'],
        max_position_pct=ENGINE_CFG['max_pos_pct'],
    )
    engine = BacktestEngine(strategy, config)
    engine.load_data(data)
    res = engine.run()
    res['stop_stats'] = engine.stop_stats
    res['adx_blocked'] = engine.adx_blocked
    res['allow_short'] = allow_short
    return res


def buy_and_hold(all_data):
    bh = {}
    for s in SYMBOLS:
        if s not in all_data:
            continue
        df = all_data[s]['df']
        bh[s] = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    if bh:
        bh['PORTFOLIO_EW'] = float(np.mean(list(bh.values())))
    return bh


def main():
    print("=" * 100)
    print("  做空机制回测 | 1年日K | 仅做多 vs 双向(含做空) 对照")
    print("=" * 100)

    all_data = {}
    print("\n拉取OKX 1年日K数据...")
    for symbol in SYMBOLS:
        df = fetch_candles_long(symbol, '1D', 365)
        if df is not None and len(df) > 0:
            df['instId'] = symbol
            all_data[symbol] = {'df': df}
            print(f"  {symbol}: {len(df)} bars ({df.iloc[0]['timestamp']} ~ {df.iloc[-1]['timestamp']})")

    if not all_data:
        print("无数据，退出")
        return

    bh = buy_and_hold(all_data)
    days = len(all_data[SYMBOLS[0]]['df'])
    mfm = (pow(1 + MMF_ANNUAL, days / 365.0) - 1) * 100
    print(f"\n基准: 货币基金 {mfm:+.2f}% | 买入持有: " + "  ".join(f"{k}={v:+.2f}%" for k, v in bh.items()))

    mode_results = {'long_only': [], 'bidirectional': []}
    pair_comparison = []

    for symbol in SYMBOLS:
        df = all_data[symbol]['df']
        for strat_name, bc in BASE_CONFIGS.items():
            lo = run_one(bc['class'], bc['params'], df, allow_short=False)
            bi = run_one(bc['class'], bc['params'], df, allow_short=True)
            lo['symbol'] = bi['symbol'] = symbol
            lo['strategy'] = bi['strategy'] = strat_name
            mode_results['long_only'].append(lo)
            mode_results['bidirectional'].append(bi)
            delta = bi['total_return'] - lo['total_return']
            pair_comparison.append({
                'symbol': symbol, 'strategy': strat_name,
                'long_only': lo['total_return'], 'bidirectional': bi['total_return'],
                'delta': delta,
                'lo_trades': lo['total_trades'], 'bi_trades': bi['total_trades'],
                # 真实做空笔数来自引擎的 short_stats（旧版误用 stop_loss+take_profit，与做空无关）
                'bi_short_trades': bi['short_stats']['trades'],
                'bi_short_pnl': round(bi['short_stats']['pnl'], 2),
                'bi_short_wins': bi['short_stats']['wins'],
                'bi_long_trades': bi['long_stats']['trades'],
                'bi_long_pnl': round(bi['long_stats']['pnl'], 2),
            })
            print(f"  {symbol:<11} {strat_name:<10} 仅多={lo['total_return']:>+7.2f}%  双向={bi['total_return']:>+7.2f}%  "
                  f"Δ={delta:>+6.2f}%  (多:{lo['total_trades']}笔 / 双向:{bi['total_trades']}笔"
                  f" [空{bi['short_stats']['trades']}笔 PnL{bi['short_stats']['pnl']:+.0f}])")

    # 汇总
    lo_rets = [r['total_return'] for r in mode_results['long_only']]
    bi_rets = [r['total_return'] for r in mode_results['bidirectional']]
    lo_avg = np.mean(lo_rets)
    bi_avg = np.mean(bi_rets)
    lo_win = sum(1 for r in lo_rets if r > 0)
    bi_win = sum(1 for r in bi_rets if r > 0)
    bh_port = bh.get('PORTFOLIO_EW', 0)

    print(f"\n{'='*100}")
    print(f"  汇总对比 (窗口 {days} 天, 12组/模式):")
    print(f"  仅做多    平均={lo_avg:+.2f}%  盈利组={lo_win}/12  最佳={max(lo_rets):+.2f}%  最差={min(lo_rets):+.2f}%")
    print(f"  双向(做空)平均={bi_avg:+.2f}%  盈利组={bi_win}/12  最佳={max(bi_rets):+.2f}%  最差={min(bi_rets):+.2f}%")
    print(f"  做空增量 Δ = {bi_avg - lo_avg:+.2f}% (平均每组)")

    # 做空真实成绩单（修复统计 bug 后首次可信）
    tot_short = sum(r['short_stats']['trades'] for r in mode_results['bidirectional'])
    tot_short_win = sum(r['short_stats']['wins'] for r in mode_results['bidirectional'])
    tot_short_pnl = sum(r['short_stats']['pnl'] for r in mode_results['bidirectional'])
    tot_long = sum(r['long_stats']['trades'] for r in mode_results['bidirectional'])
    tot_long_pnl = sum(r['long_stats']['pnl'] for r in mode_results['bidirectional'])
    sw = tot_short_win / tot_short * 100 if tot_short else 0
    print(f"  [双向模式明细] 做空 {tot_short}笔 胜{tot_short_win}笔({sw:.0f}%) PnL={tot_short_pnl:+.0f} | "
          f"做多 {tot_long}笔 PnL={tot_long_pnl:+.0f}")
    print(f"  买入持有组合={bh_port:+.2f}% | 货币基金={mfm:+.2f}%")
    print(f"  双向是否跑赢买入持有: {'YES' if bi_avg > bh_port else 'no'}  | 仅多是否跑赢: {'YES' if lo_avg > bh_port else 'no'}")

    best_bi = max(mode_results['bidirectional'], key=lambda x: x['total_return'])
    best_lo = max(mode_results['long_only'], key=lambda x: x['total_return'])

    report = {
        'date': datetime.now().isoformat(),
        'window_days': days,
        'engine_config': ENGINE_CFG,
        'benchmarks': {'money_market': round(mfm, 2), 'buy_and_hold': bh},
        'summary': {
            'long_only_avg': round(lo_avg, 2), 'long_only_profitable': lo_win,
            'bidirectional_avg': round(bi_avg, 2), 'bidirectional_profitable': bi_win,
            'short_delta_avg': round(bi_avg - lo_avg, 2),
            'buy_hold_portfolio': round(bh_port, 2), 'money_market': round(mfm, 2),
            'short_total_trades': tot_short, 'short_total_wins': tot_short_win,
            'short_win_rate': round(sw, 1), 'short_total_pnl': round(tot_short_pnl, 2),
            'long_total_trades': tot_long, 'long_total_pnl': round(tot_long_pnl, 2),
        },
        'pair_comparison': pair_comparison,
        'all_results': {
            'long_only': [{'symbol': r['symbol'], 'strategy': r['strategy'],
                           'total_return': r['total_return'], 'max_drawdown': r['max_drawdown'],
                           'sharpe_ratio': r['sharpe_ratio'], 'win_rate': r['win_rate'],
                           'total_trades': r['total_trades'], 'final_equity': r['final_equity'],
                           'stop_stats': r['stop_stats']} for r in mode_results['long_only']],
            'bidirectional': [{'symbol': r['symbol'], 'strategy': r['strategy'],
                               'total_return': r['total_return'], 'max_drawdown': r['max_drawdown'],
                               'sharpe_ratio': r['sharpe_ratio'], 'win_rate': r['win_rate'],
                               'total_trades': r['total_trades'], 'final_equity': r['final_equity'],
                               'stop_stats': r['stop_stats'], 'adx_blocked': r['adx_blocked'],
                               'long_stats': r['long_stats'], 'short_stats': r['short_stats']}
                              for r in mode_results['bidirectional']],
        },
        'best_bidirectional': {'symbol': best_bi['symbol'], 'strategy': best_bi['strategy'],
                               'total_return': best_bi['total_return'], 'max_drawdown': best_bi['max_drawdown'],
                               'sharpe_ratio': best_bi['sharpe_ratio'], 'total_trades': best_bi['total_trades']},
        'best_long_only': {'symbol': best_lo['symbol'], 'strategy': best_lo['strategy'],
                           'total_return': best_lo['total_return']},
    }

    os.makedirs('backtest_reports', exist_ok=True)
    path = 'backtest_reports/shorting_backtest.json'
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {path}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
