"""
深度优化（第二轮）：针对低收益根因
- 拉取1年日K（分页），覆盖趋势+震荡完整周期
- 计算买入持有(buy&hold)基准 + 货币基金基准
- 测试针对根因的优化：ADX入场过滤(震荡市不交易)、提高资金部署上限、组合
- 对比 策略 vs 买入持有 vs 货币基金
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
from src.strategies.base import detect_market_regime, MarketRegime

SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
INITIAL_BALANCE = 10000
FEE_RATE = 0.001
SLIPPAGE = 0.0005
MMF_ANNUAL = 0.045  # 货币基金年化 ~4.5%


def fetch_candles_long(symbol, bar='1D', total=365):
    """分页拉取较长历史K线"""
    api = OKXPublicAPI()
    bars = []
    after = None
    while len(bars) < total:
        params = {'instId': symbol, 'bar': bar, 'limit': 300}
        if after is not None:
            params['after'] = after
        # 直接调用底层请求
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


def fetch_4h_candles(symbol, limit=2200):
    api = OKXPublicAPI()
    result = api.get_candles(symbol, '4H', limit)
    if result.get('code') != '0':
        return None
    closes, timestamps = [], []
    for c in sorted(result['data'], key=lambda x: x[0]):
        timestamps.append(int(c[0]))
        closes.append(float(c[4]))
    return closes, timestamps


def align_4h_trend(daily_ts_list, tf_closes, tf_ts, fast=20, slow=50):
    from src.utils.indicators import EMA
    if not tf_closes or len(tf_closes) < max(fast, slow) + 5:
        return [True] * len(daily_ts_list)
    daily_ts_ms = [int(datetime.strptime(t, '%Y-%m-%d').timestamp() * 1000) for t in daily_ts_list]
    confirm = []
    tf_idx = 0
    for day_ms in daily_ts_ms:
        while tf_idx < len(tf_ts) and tf_ts[tf_idx] <= day_ms:
            tf_idx += 1
        available = tf_closes[:tf_idx]
        if len(available) < max(fast, slow) + 1:
            confirm.append(True)
            continue
        ema_fast = EMA(pd.Series(available), fast)
        ema_slow = EMA(pd.Series(available), slow)
        if pd.isna(ema_fast.iloc[-1]) or pd.isna(ema_slow.iloc[-1]):
            confirm.append(True)
        else:
            confirm.append(bool(ema_fast.iloc[-1] > ema_slow.iloc[-1]))
    return confirm


BASE_CONFIGS = {
    'DoubleMA': {
        'class': DoubleMAStrategy,
        'params': dict(fast_period=10, slow_period=30, trend_period=60,
                       position_pct=0.2, risk_pct=0.02,
                       atr_multiplier=2.5, atr_sl_multiplier=2.5, atr_tp_multiplier=5.0),
        'regime': 'trending',
    },
    'MACD': {
        'class': MACDCrossStrategy,
        'params': dict(fast_period=12, slow_period=26, signal_period=9,
                       position_pct=0.2, risk_pct=0.02,
                       atr_multiplier=2.5, atr_sl_multiplier=2.5, atr_tp_multiplier=5.0),
        'regime': 'trending',
    },
    'Breakout': {
        'class': BreakoutStrategy,
        'params': dict(lookback=20, position_pct=0.2, risk_pct=0.02,
                       atr_multiplier=2.0, atr_sl_multiplier=2.0, atr_tp_multiplier=4.0),
        'regime': 'trending',
    },
    'RSI_Boll': {
        'class': RSIBollingerStrategy,
        'params': dict(rsi_period=14, boll_period=20, boll_std=2.0,
                       rsi_low=30, rsi_high=70, position_pct=0.2, risk_pct=0.015,
                       atr_multiplier=2.0, atr_sl_multiplier=2.0, atr_tp_multiplier=4.0),
        'regime': 'ranging',
    },
}


def run_one_backtest(strategy_class, params, data, tf_confirm=None,
                     use_atr=True, use_trailing=True, trailing_pct=0.02,
                     use_mtf=True, min_adx=0.0, max_pos_pct=0.5):
    strategy = strategy_class(instId=data.iloc[0].get('instId', 'UNKNOWN'), **params)
    config = BacktestConfig(
        initial_balance=INITIAL_BALANCE, fee_rate=FEE_RATE, slippage=SLIPPAGE,
        use_atr_risk=use_atr, atr_period=14, risk_pct=params.get('risk_pct', 0.02),
        atr_multiplier=params.get('atr_multiplier', 2.0),
        atr_sl_multiplier=params.get('atr_sl_multiplier', 2.0),
        atr_tp_multiplier=params.get('atr_tp_multiplier', 4.0),
        use_trailing=use_trailing, trailing_pct=trailing_pct,
        use_mtf=(tf_confirm is not None and use_mtf), tf_confirm=tf_confirm,
        use_regime=True, min_adx_for_entry=min_adx, max_position_pct=max_pos_pct,
    )
    engine = BacktestEngine(strategy, config)
    engine.load_data(data)
    results = engine.run()
    results['stop_stats'] = engine.stop_stats
    results['adx_blocked'] = engine.adx_blocked
    return results


def run_round(round_name, rc, all_data, silent=False):
    results = []
    engine_ov = rc.get('engine_overrides', {})
    strat_ov = rc.get('strategy_overrides', {})
    only = rc.get('only_strategies', None)

    for symbol in SYMBOLS:
        if symbol not in all_data:
            continue
        df = all_data[symbol]['df']
        tf_confirm = all_data[symbol].get('tf')

        for strat_name, base_cfg in BASE_CONFIGS.items():
            if only and strat_name not in only:
                continue
            params = copy.deepcopy(base_cfg['params'])
            if strat_name in strat_ov:
                params.update(strat_ov[strat_name])
            try:
                result = run_one_backtest(
                    base_cfg['class'], params, df, tf_confirm,
                    use_atr=engine_ov.get('use_atr', True),
                    use_trailing=engine_ov.get('use_trailing', True),
                    trailing_pct=engine_ov.get('trailing_pct', 0.02),
                    use_mtf=engine_ov.get('use_mtf', True),
                    min_adx=engine_ov.get('min_adx', 0.0),
                    max_pos_pct=engine_ov.get('max_pos_pct', 0.5),
                )
                result['symbol'] = symbol
                result['strategy'] = strat_name
                result['round'] = round_name
                results.append(result)
                if not silent:
                    print(f"  {symbol:<11} {strat_name:<11} ret={result['total_return']:>+7.2f}%  "
                          f"dd={result['max_drawdown']:>6.2f}%  sharpe={result['sharpe_ratio']:>5.2f}  "
                          f"wr={result['win_rate']:>5.1f}%  trades={result['total_trades']}  "
                          f"adxblk={result.get('adx_blocked',0)}")
            except Exception as e:
                if not silent:
                    print(f"  {symbol} x {strat_name} ERROR: {e}")
                results.append({'symbol': symbol, 'strategy': strat_name, 'round': round_name,
                                'total_return': 0, 'max_drawdown': 0, 'sharpe_ratio': 0,
                                'win_rate': 0, 'total_trades': 0, 'final_equity': INITIAL_BALANCE,
                                'error': str(e)})
    return results


def get_round_configs():
    rounds = {}

    # D1: 基线（1年窗口，对照买入持有）
    rounds['D1_Baseline_1yr'] = {
        'description': 'Baseline on 1yr window (compare vs buy&hold)',
        'engine_overrides': {},
    }

    # D2: ADX>20 才入场（震荡市空仓，核心修复）
    rounds['D2_TrendingOnly_ADX20'] = {
        'description': 'Only enter when ADX>20 (sit flat in chop)',
        'engine_overrides': {'min_adx': 20.0},
    }

    # D3: ADX>25 才入场（更严格）
    rounds['D3_TrendingOnly_ADX25'] = {
        'description': 'Only enter when ADX>25 (stricter trend filter)',
        'engine_overrides': {'min_adx': 25.0},
    }

    # D4: 高资金部署（上限90%）
    rounds['D4_HighDeploy_90'] = {
        'description': 'Higher capital deployment: max_position_pct=0.9',
        'engine_overrides': {'max_pos_pct': 0.9},
    }

    # D5: 趋势过滤 + 高部署 组合
    rounds['D5_Trend_HighDeploy'] = {
        'description': 'ADX>20 + max_position_pct=0.9 (trend filter + capital efficiency)',
        'engine_overrides': {'min_adx': 20.0, 'max_pos_pct': 0.9},
    }

    # D6: 只跑MACD（唯一正收益策略）+ 趋势过滤
    rounds['D6_MACD_TrendOnly'] = {
        'description': 'MACD only + ADX>20 (focus on only positive strategy)',
        'engine_overrides': {'min_adx': 20.0, 'max_pos_pct': 0.9},
        'only_strategies': ['MACD'],
    }

    # D7: 宽止损无移动止损（R9最优单次配置）跑1年
    rounds['D7_WideSL_NoTrail_1yr'] = {
        'description': 'No trailing + wide TP 7x (best single config) on 1yr',
        'strategy_overrides': {
            'DoubleMA': dict(atr_tp_multiplier=7.0), 'MACD': dict(atr_tp_multiplier=7.0),
            'Breakout': dict(atr_tp_multiplier=7.0), 'RSI_Boll': dict(atr_tp_multiplier=7.0),
        },
        'engine_overrides': {'use_trailing': False},
    }

    # D8: 最优组合（MACD + ADX25 + 90%部署 + 宽止损无移动止损）
    rounds['D8_BestCombo'] = {
        'description': 'MACD + ADX25 + 90% deploy + wide SL 3x + no trailing + TP 6x',
        'strategy_overrides': {
            'MACD': dict(atr_sl_multiplier=3.0, atr_tp_multiplier=6.0, risk_pct=0.025),
        },
        'engine_overrides': {'min_adx': 25.0, 'max_pos_pct': 0.9, 'use_trailing': False},
        'only_strategies': ['MACD'],
    }

    # D9: 满仓趋势（MACD + ADX30 + 100%部署，最激进验证）
    rounds['D9_AllIn_Trend_ADX30'] = {
        'description': 'MACD + ADX30 + 100% deploy (most aggressive, trend only)',
        'strategy_overrides': {
            'MACD': dict(atr_sl_multiplier=3.0, atr_tp_multiplier=6.0, risk_pct=0.03),
        },
        'engine_overrides': {'min_adx': 30.0, 'max_pos_pct': 1.0, 'use_trailing': False},
        'only_strategies': ['MACD'],
    }

    return rounds


def buy_and_hold_benchmark(all_data):
    """计算每个币的买入持有收益 + 等权组合"""
    bh = {}
    for symbol in SYMBOLS:
        if symbol not in all_data:
            continue
        df = all_data[symbol]['df']
        ret = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        bh[symbol] = ret
    # 等权组合：初始资金三等分，各自买入持有
    if bh:
        portfolio = np.mean(list(bh.values()))
        bh['PORTFOLIO_EW'] = portfolio
    return bh


def main():
    print("=" * 100)
    print("  深度优化（第二轮）：针对低收益根因 | 1年日K | 策略 vs 买入持有 vs 货币基金")
    print("=" * 100)

    all_data = {}
    print("\n拉取OKX 1年日K数据（分页）...")
    for symbol in SYMBOLS:
        df = fetch_candles_long(symbol, '1D', 365)
        if df is not None and len(df) > 0:
            df['instId'] = symbol
            all_data[symbol] = {'df': df}
            print(f"  {symbol}: {len(df)} bars ({df.iloc[0]['timestamp']} ~ {df.iloc[-1]['timestamp']})")
            tf_result = fetch_4h_candles(symbol, 2200)
            if tf_result:
                tf_closes, tf_ts = tf_result
                all_data[symbol]['tf'] = align_4h_trend(df['timestamp'].tolist(), tf_closes, tf_ts)
            else:
                all_data[symbol]['tf'] = None
            regime, info = detect_market_regime(df['high'].tolist(), df['low'].tolist(), df['close'].tolist())
            all_data[symbol]['regime'] = (regime, info)
            print(f"    regime={regime.value} (ADX={info['adx']}, vol_pct={info['vol_pct']})")

    if not all_data:
        print("无数据，退出")
        return

    # 基准
    bh = buy_and_hold_benchmark(all_data)
    days = len(all_data[SYMBOLS[0]]['df'])
    mfm_total = (pow(1 + MMF_ANNUAL, days / 365.0) - 1) * 100
    print(f"\n{'='*100}")
    print(f"  基准对比 (窗口 {days} 天):")
    print(f"  货币基金(年化{MMF_ANNUAL*100:.1f}%): {mfm_total:+.2f}%")
    for k, v in bh.items():
        print(f"  买入持有 {k:<14}: {v:+.2f}%")

    round_configs = get_round_configs()
    all_round_results = {}

    for round_name, rc in round_configs.items():
        print(f"\n{'='*100}")
        print(f"  {round_name}: {rc['description']}")
        print(f"{'='*100}")
        results = run_round(round_name, rc, all_data)
        all_round_results[round_name] = results
        valid = [r for r in results if 'error' not in r]
        if valid:
            avg_ret = np.mean([r['total_return'] for r in valid])
            avg_dd = np.mean([r['max_drawdown'] for r in valid])
            profitable = [r for r in valid if r['total_return'] > 0]
            best = max(valid, key=lambda x: x['total_return'])
            worst = min(valid, key=lambda x: x['total_return'])
            print(f"\n  --- {round_name} Summary ---")
            print(f"  Avg Return: {avg_ret:+.2f}% | Avg DD: {avg_dd:.2f}% | "
                  f"Profitable: {len(profitable)}/{len(valid)}")
            print(f"  Best:  {best['symbol']} x {best['strategy']} = {best['total_return']:+.2f}%")
            print(f"  Worst: {worst['symbol']} x {worst['strategy']} = {worst['total_return']:+.2f}%")
            print(f"  vs 买入持有组合: {bh.get('PORTFOLIO_EW',0):+.2f}% | vs 货币基金: {mfm_total:+.2f}%")

    # 全局汇总
    print(f"\n{'='*100}")
    print(f"  GLOBAL SUMMARY — 深度优化 (窗口 {days} 天)")
    print(f"{'='*100}")
    print(f"  基准: 买入持有组合={bh.get('PORTFOLIO_EW',0):+.2f}% | 货币基金={mfm_total:+.2f}%")
    print(f"\n  {'Round':<24} {'AvgRet%':<10} {'AvgDD%':<9} {'Prof/Total':<11} {'Best%':<9} {'BeatB&H?':<9}")
    print(f"  {'-'*80}")

    round_summaries = []
    for round_name, results in all_round_results.items():
        valid = [r for r in results if 'error' not in r]
        if not valid:
            continue
        avg_ret = np.mean([r['total_return'] for r in valid])
        avg_dd = np.mean([r['max_drawdown'] for r in valid])
        profitable = len([r for r in valid if r['total_return'] > 0])
        best_ret = max(r['total_return'] for r in valid)
        beat = 'YES' if avg_ret > bh.get('PORTFOLIO_EW', 0) else 'no'
        round_summaries.append({'round': round_name, 'avg_return': avg_ret, 'avg_drawdown': avg_dd,
                                'profitable': profitable, 'total_tests': len(valid),
                                'best_return': best_ret, 'beat_bh': beat})
        print(f"  {round_name:<24} {avg_ret:>+8.2f}% {avg_dd:>8.2f}% {profitable:>3}/{len(valid):<3}       "
              f"{best_ret:>+8.2f}% {beat:<9}")

    all_valid = []
    for results in all_round_results.values():
        for r in results:
            if 'error' not in r:
                all_valid.append(r)
    all_valid.sort(key=lambda x: x['total_return'], reverse=True)

    print(f"\n  TOP 10 Individual Results:")
    print(f"  {'Rank':<5} {'Round':<24} {'Symbol':<11} {'Strategy':<11} {'Return%':<9} {'DD%':<8} {'Sharpe':<7} {'Trades':<6}")
    print(f"  {'-'*90}")
    for i, r in enumerate(all_valid[:10]):
        print(f"  {i+1:<5} {r['round']:<24} {r['symbol']:<11} {r['strategy']:<11} "
              f"{r['total_return']:>+8.2f}% {r['max_drawdown']:>6.2f}% {r['sharpe_ratio']:>6.2f}  {r['total_trades']:>4}")

    report = {
        'date': datetime.now().isoformat(),
        'window_days': days,
        'benchmarks': {'money_market': round(mfm_total, 2), 'buy_and_hold': bh},
        'round_summaries': round_summaries,
        'all_results': [{
            'round': r['round'], 'symbol': r['symbol'], 'strategy': r['strategy'],
            'total_return': r['total_return'], 'max_drawdown': r['max_drawdown'],
            'sharpe_ratio': r['sharpe_ratio'], 'win_rate': r['win_rate'],
            'total_trades': r['total_trades'], 'final_equity': r['final_equity'],
            'adx_blocked': r.get('adx_blocked', 0),
        } for r in all_valid],
        'top10': [{
            'rank': i+1, 'round': r['round'], 'symbol': r['symbol'], 'strategy': r['strategy'],
            'total_return': r['total_return'], 'max_drawdown': r['max_drawdown'],
            'sharpe_ratio': r['sharpe_ratio'], 'win_rate': r['win_rate'], 'total_trades': r['total_trades'],
        } for i, r in enumerate(all_valid[:10])],
    }

    report_path = 'backtest_reports/deep_optimize.json'
    os.makedirs('backtest_reports', exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
