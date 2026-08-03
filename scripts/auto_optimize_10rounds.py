"""
10轮自动优化回测
只拉一次OKX数据，测试10种参数配置，挑出最优策略。
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


def fetch_candles(symbol, bar='1D', limit=500):
    api = OKXPublicAPI()
    result = api.get_candles(symbol, bar, limit)
    if result.get('code') != '0':
        return None
    rows = []
    for c in sorted(result['data'], key=lambda x: x[0]):
        rows.append({
            'timestamp': datetime.fromtimestamp(int(c[0]) / 1000).strftime('%Y-%m-%d'),
            'open': float(c[1]),
            'high': float(c[2]),
            'low': float(c[3]),
            'close': float(c[4]),
            'volume': float(c[5]),
        })
    return pd.DataFrame(rows)


def fetch_4h_candles(symbol, limit=1500):
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


# ==================== 基础策略配置 ====================
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
        'params': dict(lookback=20,
                       position_pct=0.2, risk_pct=0.02,
                       atr_multiplier=2.0, atr_sl_multiplier=2.0, atr_tp_multiplier=4.0),
        'regime': 'trending',
    },
    'RSI_Boll': {
        'class': RSIBollingerStrategy,
        'params': dict(rsi_period=14, boll_period=20, boll_std=2.0,
                       rsi_low=30, rsi_high=70,
                       position_pct=0.2, risk_pct=0.015,
                       atr_multiplier=2.0, atr_sl_multiplier=2.0, atr_tp_multiplier=4.0),
        'regime': 'ranging',
    },
}


def run_one_backtest(strategy_class, params, data, tf_confirm=None,
                     use_atr=True, use_trailing=True, trailing_pct=0.02,
                     use_mtf=True, use_regime=True,
                     regime_filter=None):
    """运行一次回测，返回结果dict"""
    strategy = strategy_class(instId=data.iloc[0].get('instId', 'UNKNOWN'), **params)

    config = BacktestConfig(
        initial_balance=INITIAL_BALANCE,
        fee_rate=FEE_RATE,
        slippage=SLIPPAGE,
        use_atr_risk=use_atr,
        atr_period=14,
        risk_pct=params.get('risk_pct', 0.02),
        atr_multiplier=params.get('atr_multiplier', 2.0),
        atr_sl_multiplier=params.get('atr_sl_multiplier', 2.0),
        atr_tp_multiplier=params.get('atr_tp_multiplier', 4.0),
        use_trailing=use_trailing,
        trailing_pct=trailing_pct,
        use_mtf=(tf_confirm is not None and use_mtf),
        tf_confirm=tf_confirm,
        use_regime=use_regime,
    )

    engine = BacktestEngine(strategy, config)
    engine.load_data(data)
    results = engine.run()
    results['stop_stats'] = engine.stop_stats
    results['mtf_blocked'] = engine.mtf_blocked
    results['regime_history'] = engine.regime_history
    return results


def run_round(round_name, round_config, all_data, silent=False):
    """
    运行一轮优化（4策略 x 3币种）
    round_config: dict with keys:
        - strategy_overrides: {strategy_name: {param: value, ...}}
        - engine_overrides: {use_atr, use_trailing, trailing_pct, use_mtf, use_regime}
        - regime_filter: True/False (only run strategy in matching regime)
    """
    results = []
    engine_overrides = round_config.get('engine_overrides', {})
    strategy_overrides = round_config.get('strategy_overrides', {})
    regime_filter = round_config.get('regime_filter', False)

    for symbol in SYMBOLS:
        if symbol not in all_data:
            continue
        df = all_data[symbol]['df']
        tf_confirm = all_data[symbol].get('tf')
        regime, regime_info = all_data[symbol].get('regime', (MarketRegime.UNKNOWN, {}))

        for strat_name, base_cfg in BASE_CONFIGS.items():
            # 市场状态过滤
            if regime_filter:
                strat_regime = base_cfg['regime']
                if strat_regime == 'trending' and regime == MarketRegime.RANGING:
                    if not silent:
                        print(f"  SKIP {symbol} x {strat_name} (ranging, trend strategy)")
                    continue
                if strat_regime == 'ranging' and regime == MarketRegime.TRENDING:
                    if not silent:
                        print(f"  SKIP {symbol} x {strat_name} (trending, range strategy)")
                    continue

            # 应用参数覆盖
            params = copy.deepcopy(base_cfg['params'])
            if strat_name in strategy_overrides:
                params.update(strategy_overrides[strat_name])

            try:
                result = run_one_backtest(
                    base_cfg['class'], params, df, tf_confirm,
                    use_atr=engine_overrides.get('use_atr', True),
                    use_trailing=engine_overrides.get('use_trailing', True),
                    trailing_pct=engine_overrides.get('trailing_pct', 0.02),
                    use_mtf=engine_overrides.get('use_mtf', True),
                    use_regime=engine_overrides.get('use_regime', True),
                )
                result['symbol'] = symbol
                result['strategy'] = strat_name
                result['round'] = round_name
                results.append(result)

                if not silent:
                    print(f"  {symbol:<12} {strat_name:<12} ret={result['total_return']:>+7.2f}%  "
                          f"dd={result['max_drawdown']:>6.2f}%  sharpe={result['sharpe_ratio']:>5.2f}  "
                          f"wr={result['win_rate']:>5.1f}%  trades={result['total_trades']}")
            except Exception as e:
                if not silent:
                    print(f"  {symbol} x {strat_name} ERROR: {e}")
                results.append({
                    'symbol': symbol, 'strategy': strat_name, 'round': round_name,
                    'total_return': 0, 'max_drawdown': 0, 'sharpe_ratio': 0,
                    'win_rate': 0, 'total_trades': 0, 'final_equity': INITIAL_BALANCE,
                    'error': str(e),
                })

    return results


# ==================== 10轮优化配置 ====================
def get_round_configs():
    """返回10轮优化的配置"""
    rounds = {}

    # Round 1: 基线（当前参数）
    rounds['R1_Baseline'] = {
        'description': 'Baseline: current params (ATR 2.5x SL / 5x TP, trailing 2%, pos 20%)',
        'strategy_overrides': {},
        'engine_overrides': {},
    }

    # Round 2: 紧止损 + 宽止盈（截断亏损，让利润奔跑）
    rounds['R2_TightSL_WideTP'] = {
        'description': 'Tight SL 1.5x / Wide TP 6.0x (cut losses fast, let profits run)',
        'strategy_overrides': {
            'DoubleMA': dict(atr_sl_multiplier=1.5, atr_tp_multiplier=6.0),
            'MACD': dict(atr_sl_multiplier=1.5, atr_tp_multiplier=6.0),
            'Breakout': dict(atr_sl_multiplier=1.5, atr_tp_multiplier=6.0),
            'RSI_Boll': dict(atr_sl_multiplier=1.5, atr_tp_multiplier=6.0),
        },
        'engine_overrides': {},
    }

    # Round 3: 宽止损 + 缓止盈（给趋势更多空间）
    rounds['R3_WideSL_SlowTP'] = {
        'description': 'Wide SL 3.0x / Moderate TP 4.5x (give trends more room)',
        'strategy_overrides': {
            'DoubleMA': dict(atr_sl_multiplier=3.0, atr_tp_multiplier=4.5),
            'MACD': dict(atr_sl_multiplier=3.0, atr_tp_multiplier=4.5),
            'Breakout': dict(atr_sl_multiplier=3.0, atr_tp_multiplier=4.5),
            'RSI_Boll': dict(atr_sl_multiplier=3.0, atr_tp_multiplier=4.5),
        },
        'engine_overrides': {},
    }

    # Round 4: 市场状态过滤（趋势市只跑趋势策略，震荡市只跑震荡策略）
    rounds['R4_RegimeFilter'] = {
        'description': 'Regime filter: only run trend strategies in trending, range strategies in ranging',
        'strategy_overrides': {},
        'engine_overrides': {},
        'regime_filter': True,
    }

    # Round 5: 快MA周期（更灵敏）
    rounds['R5_FastMA'] = {
        'description': 'Faster MA periods: DoubleMA 5/15/30, Breakout lookback=10',
        'strategy_overrides': {
            'DoubleMA': dict(fast_period=5, slow_period=15, trend_period=30),
            'Breakout': dict(lookback=10),
        },
        'engine_overrides': {},
    }

    # Round 6: 慢MA周期 + 高移动止损激活（过滤噪音）
    rounds['R6_SlowMA_HighTrail'] = {
        'description': 'Slower MA: DoubleMA 20/50/100, Breakout lookback=30, trailing activation 4%',
        'strategy_overrides': {
            'DoubleMA': dict(fast_period=20, slow_period=50, trend_period=100),
            'Breakout': dict(lookback=30),
        },
        'engine_overrides': {'trailing_pct': 0.04},
    }

    # Round 7: 保守仓位（10%仓位 / 1%风险）
    rounds['R7_Conservative'] = {
        'description': 'Conservative sizing: pos 10%, risk 1%',
        'strategy_overrides': {
            'DoubleMA': dict(position_pct=0.10, risk_pct=0.01),
            'MACD': dict(position_pct=0.10, risk_pct=0.01),
            'Breakout': dict(position_pct=0.10, risk_pct=0.01),
            'RSI_Boll': dict(position_pct=0.10, risk_pct=0.008),
        },
        'engine_overrides': {},
    }

    # Round 8: 激进仓位（30%仓位 / 3%风险）
    rounds['R8_Aggressive'] = {
        'description': 'Aggressive sizing: pos 30%, risk 3%',
        'strategy_overrides': {
            'DoubleMA': dict(position_pct=0.30, risk_pct=0.03),
            'MACD': dict(position_pct=0.30, risk_pct=0.03),
            'Breakout': dict(position_pct=0.30, risk_pct=0.03),
            'RSI_Boll': dict(position_pct=0.30, risk_pct=0.025),
        },
        'engine_overrides': {},
    }

    # Round 9: 无移动止损 + 宽止盈（让趋势充分发展）
    rounds['R9_NoTrailing_WideTP'] = {
        'description': 'No trailing stop + wider TP 7x (let trends fully develop)',
        'strategy_overrides': {
            'DoubleMA': dict(atr_tp_multiplier=7.0),
            'MACD': dict(atr_tp_multiplier=7.0),
            'Breakout': dict(atr_tp_multiplier=7.0),
            'RSI_Boll': dict(atr_tp_multiplier=7.0),
        },
        'engine_overrides': {'use_trailing': False},
    }

    # Round 10: 最优组合（根据前9轮结果组合）
    # 先用一个合理的默认组合，后面根据结果调整
    rounds['R10_BestCombo'] = {
        'description': 'Best combo: tight SL 1.5x, wide TP 6x, regime filter, trailing 3%',
        'strategy_overrides': {
            'DoubleMA': dict(atr_sl_multiplier=1.5, atr_tp_multiplier=6.0, position_pct=0.15, risk_pct=0.015),
            'MACD': dict(atr_sl_multiplier=1.5, atr_tp_multiplier=6.0, position_pct=0.15, risk_pct=0.015),
            'Breakout': dict(atr_sl_multiplier=1.5, atr_tp_multiplier=6.0, position_pct=0.15, risk_pct=0.015),
            'RSI_Boll': dict(atr_sl_multiplier=1.5, atr_tp_multiplier=6.0, position_pct=0.15, risk_pct=0.012),
        },
        'engine_overrides': {'trailing_pct': 0.03},
        'regime_filter': True,
    }

    return rounds


def main():
    print("=" * 100)
    print("  10轮自动优化回测")
    print("  4策略 x 3币种 x 10种参数配置 = ~120组测试")
    print("=" * 100)

    # ====== 拉取数据（只拉一次） ======
    all_data = {}
    print("\n拉取OKX数据...")
    for symbol in SYMBOLS:
        df = fetch_candles(symbol, '1D', 500)
        if df is not None and len(df) > 0:
            df['instId'] = symbol
            all_data[symbol] = {'df': df}
            print(f"  {symbol}: {len(df)} bars ({df.iloc[0]['timestamp']} ~ {df.iloc[-1]['timestamp']})")

            # 4H数据
            tf_result = fetch_4h_candles(symbol, 1500)
            if tf_result:
                tf_closes, tf_ts = tf_result
                tf_confirm = align_4h_trend(df['timestamp'].tolist(), tf_closes, tf_ts)
                all_data[symbol]['tf'] = tf_confirm
            else:
                all_data[symbol]['tf'] = None

            # 市场状态
            regime, info = detect_market_regime(
                df['high'].tolist(), df['low'].tolist(), df['close'].tolist()
            )
            all_data[symbol]['regime'] = (regime, info)
            print(f"    regime={regime.value} (ADX={info['adx']}, vol_pct={info['vol_pct']})")

    if not all_data:
        print("无数据，退出")
        return

    # ====== 运行10轮 ======
    round_configs = get_round_configs()
    all_round_results = {}

    for round_name, round_cfg in round_configs.items():
        print(f"\n{'='*100}")
        print(f"  {round_name}: {round_cfg['description']}")
        print(f"{'='*100}")

        results = run_round(round_name, round_cfg, all_data)
        all_round_results[round_name] = results

        # 本轮汇总
        valid = [r for r in results if 'error' not in r]
        if valid:
            avg_ret = np.mean([r['total_return'] for r in valid])
            avg_dd = np.mean([r['max_drawdown'] for r in valid])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in valid])
            profitable = [r for r in valid if r['total_return'] > 0]
            best = max(valid, key=lambda x: x['total_return']) if valid else None
            worst = min(valid, key=lambda x: x['total_return']) if valid else None

            print(f"\n  --- {round_name} Summary ---")
            print(f"  Avg Return: {avg_ret:+.2f}% | Avg Drawdown: {avg_dd:.2f}% | Avg Sharpe: {avg_sharpe:.2f}")
            print(f"  Profitable: {len(profitable)}/{len(valid)}")
            if best:
                print(f"  Best: {best['symbol']} x {best['strategy']} = {best['total_return']:+.2f}%")
            if worst:
                print(f"  Worst: {worst['symbol']} x {worst['strategy']} = {worst['total_return']:+.2f}%")

    # ====== 全局汇总 ======
    print(f"\n{'='*100}")
    print(f"  GLOBAL SUMMARY — 10 Rounds")
    print(f"{'='*100}")

    # 每轮平均
    print(f"\n  {'Round':<25} {'AvgRet%':<10} {'AvgDD%':<10} {'AvgSharpe':<10} {'Profitable':<12} {'BestRet%':<10}")
    print(f"  {'-'*90}")

    round_summaries = []
    for round_name, results in all_round_results.items():
        valid = [r for r in results if 'error' not in r]
        if not valid:
            continue
        avg_ret = np.mean([r['total_return'] for r in valid])
        avg_dd = np.mean([r['max_drawdown'] for r in valid])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in valid])
        profitable_count = len([r for r in valid if r['total_return'] > 0])
        best_ret = max(r['total_return'] for r in valid)

        round_summaries.append({
            'round': round_name,
            'avg_return': avg_ret,
            'avg_drawdown': avg_dd,
            'avg_sharpe': avg_sharpe,
            'profitable': profitable_count,
            'total_tests': len(valid),
            'best_return': best_ret,
        })

        print(f"  {round_name:<25} {avg_ret:>+8.2f}% {avg_dd:>8.2f}% {avg_sharpe:>9.2f}  "
              f"{profitable_count:>2}/{len(valid):<2}       {best_ret:>+8.2f}%")

    # 全部测试中 Top 10
    all_valid = []
    for round_name, results in all_round_results.items():
        for r in results:
            if 'error' not in r:
                all_valid.append(r)

    all_valid.sort(key=lambda x: x['total_return'], reverse=True)

    print(f"\n  TOP 10 Individual Results (across all rounds):")
    print(f"  {'Rank':<5} {'Round':<25} {'Symbol':<12} {'Strategy':<12} {'Return%':<10} {'DD%':<8} {'Sharpe':<8} {'WinRate%':<8} {'Trades':<6}")
    print(f"  {'-'*100}")

    for i, r in enumerate(all_valid[:10]):
        print(f"  {i+1:<5} {r['round']:<25} {r['symbol']:<12} {r['strategy']:<12} "
              f"{r['total_return']:>+8.2f}% {r['max_drawdown']:>6.2f}% {r['sharpe_ratio']:>6.2f}  "
              f"{r['win_rate']:>6.1f}%  {r['total_trades']:>4}")

    # 最差5个
    print(f"\n  WORST 5:")
    for i, r in enumerate(all_valid[-5:]):
        print(f"  {i+1:<5} {r['round']:<25} {r['symbol']:<12} {r['strategy']:<12} "
              f"{r['total_return']:>+8.2f}% {r['max_drawdown']:>6.2f}%")

    # 按策略类型汇总（跨所有轮）
    print(f"\n  Strategy Performance (averaged across all rounds):")
    print(f"  {'Strategy':<12} {'AvgRet%':<10} {'AvgDD%':<10} {'AvgSharpe':<10} {'BestRet%':<10} {'WorstRet%':<10}")
    print(f"  {'-'*70}")
    for strat_name in BASE_CONFIGS.keys():
        strat_results = [r for r in all_valid if r['strategy'] == strat_name]
        if strat_results:
            avg_ret = np.mean([r['total_return'] for r in strat_results])
            avg_dd = np.mean([r['max_drawdown'] for r in strat_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in strat_results])
            best_ret = max(r['total_return'] for r in strat_results)
            worst_ret = min(r['total_return'] for r in strat_results)
            print(f"  {strat_name:<12} {avg_ret:>+8.2f}% {avg_dd:>8.2f}% {avg_sharpe:>9.2f}  {best_ret:>+8.2f}% {worst_ret:>+8.2f}%")

    # 按币种汇总
    print(f"\n  Symbol Performance (averaged across all rounds):")
    print(f"  {'Symbol':<12} {'AvgRet%':<10} {'BestRet%':<10} {'WorstRet%':<10}")
    print(f"  {'-'*50}")
    for symbol in SYMBOLS:
        sym_results = [r for r in all_valid if r['symbol'] == symbol]
        if sym_results:
            avg_ret = np.mean([r['total_return'] for r in sym_results])
            best_ret = max(r['total_return'] for r in sym_results)
            worst_ret = min(r['total_return'] for r in sym_results)
            print(f"  {symbol:<12} {avg_ret:>+8.2f}% {best_ret:>+8.2f}% {worst_ret:>+8.2f}%")

    # ====== 保存JSON报告 ======
    report = {
        'date': datetime.now().isoformat(),
        'config': {
            'initial_balance': INITIAL_BALANCE,
            'fee_rate': FEE_RATE,
            'slippage': SLIPPAGE,
            'symbols': SYMBOLS,
            'num_rounds': 10,
        },
        'round_summaries': round_summaries,
        'all_results': [{
            'round': r['round'],
            'symbol': r['symbol'],
            'strategy': r['strategy'],
            'total_return': r['total_return'],
            'max_drawdown': r['max_drawdown'],
            'sharpe_ratio': r['sharpe_ratio'],
            'win_rate': r['win_rate'],
            'total_trades': r['total_trades'],
            'final_equity': r['final_equity'],
            'stop_stats': r.get('stop_stats', {}),
            'mtf_blocked': r.get('mtf_blocked', 0),
        } for r in all_valid],
        'top10': [{
            'rank': i+1,
            'round': r['round'],
            'symbol': r['symbol'],
            'strategy': r['strategy'],
            'total_return': r['total_return'],
            'max_drawdown': r['max_drawdown'],
            'sharpe_ratio': r['sharpe_ratio'],
            'win_rate': r['win_rate'],
            'total_trades': r['total_trades'],
        } for i, r in enumerate(all_valid[:10])],
    }

    report_path = 'backtest_reports/auto_optimize_10rounds.json'
    os.makedirs('backtest_reports', exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
