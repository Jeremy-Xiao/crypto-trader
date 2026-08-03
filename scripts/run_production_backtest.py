"""
生产框架回测脚本
使用升级后的 src/ 框架（ATR动态风控 + 移动止损 + 市场状态自适应）
拉取 OKX 真实数据，多币种多策略回测，看是赚钱还是亏钱。
"""

import os
import sys
import json
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
from src.strategies.base import detect_market_regime

SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
INITIAL_BALANCE = 10000
FEE_RATE = 0.001
SLIPPAGE = 0.0005


def fetch_candles(symbol, bar='1D', limit=500):
    """拉取OKX K线数据，返回DataFrame"""
    api = OKXPublicAPI()
    result = api.get_candles(symbol, bar, limit)
    if result.get('code') != '0':
        print(f"  获取{symbol}失败: {result.get('msg', 'unknown')}")
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

    df = pd.DataFrame(rows)
    return df


def fetch_4h_candles(symbol, limit=1500):
    """拉取4H K线用于多时间框架确认"""
    api = OKXPublicAPI()
    result = api.get_candles(symbol, '4H', limit)
    if result.get('code') != '0':
        return None

    closes = []
    timestamps = []
    for c in sorted(result['data'], key=lambda x: x[0]):
        timestamps.append(int(c[0]))
        closes.append(float(c[4]))
    return closes, timestamps


def align_4h_trend(daily_ts_str, tf_closes, tf_ts, fast=20, slow=50):
    """将4H趋势映射到日K线时间点"""
    if not tf_closes or len(tf_closes) < max(fast, slow) + 5:
        return [True] * len(daily_ts_str)

    from src.utils.indicators import EMA

    # 把日K时间戳转为毫秒
    daily_ts_ms = [int(datetime.strptime(t, '%Y-%m-%d').timestamp() * 1000) for t in daily_ts_str]

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


# ==================== 策略配置 ====================

STRATEGY_CONFIGS = [
    # 趋势策略
    {
        'class': DoubleMAStrategy,
        'name': 'DoubleMA_10_30_60',
        'params': dict(fast_period=10, slow_period=30, trend_period=60,
                       position_pct=0.2, risk_pct=0.02,
                       atr_multiplier=2.5, atr_sl_multiplier=2.5, atr_tp_multiplier=5.0),
    },
    {
        'class': MACDCrossStrategy,
        'name': 'MACD_12_26_9',
        'params': dict(fast_period=12, slow_period=26, signal_period=9,
                       position_pct=0.2, risk_pct=0.02,
                       atr_multiplier=2.5, atr_sl_multiplier=2.5, atr_tp_multiplier=5.0),
    },
    {
        'class': BreakoutStrategy,
        'name': 'Breakout_20',
        'params': dict(lookback=20,
                       position_pct=0.2, risk_pct=0.02,
                       atr_multiplier=2.0, atr_sl_multiplier=2.0, atr_tp_multiplier=4.0),
    },
    # 震荡策略
    {
        'class': RSIBollingerStrategy,
        'name': 'RSI_Boll_14_20_2',
        'params': dict(rsi_period=14, boll_period=20, boll_std=2.0,
                       rsi_low=30, rsi_high=70,
                       position_pct=0.2, risk_pct=0.015,
                       atr_multiplier=2.0, atr_sl_multiplier=2.0, atr_tp_multiplier=4.0),
    },
]


def run_single_backtest(strategy_class, strategy_name, params, data, tf_confirm=None):
    """运行单策略回测（ATR动态风控版）"""
    strategy = strategy_class(instId=data.iloc[0].get('instId', 'UNKNOWN'), **params)

    config = BacktestConfig(
        initial_balance=INITIAL_BALANCE,
        fee_rate=FEE_RATE,
        slippage=SLIPPAGE,
        use_atr_risk=True,
        atr_period=14,
        risk_pct=params.get('risk_pct', 0.02),
        atr_multiplier=params.get('atr_multiplier', 2.0),
        atr_sl_multiplier=params.get('atr_sl_multiplier', 2.0),
        atr_tp_multiplier=params.get('atr_tp_multiplier', 4.0),
        use_trailing=True,
        trailing_pct=0.02,
        use_mtf=tf_confirm is not None,
        tf_confirm=tf_confirm,
        use_regime=True,
    )

    engine = BacktestEngine(strategy, config)
    engine.load_data(data)
    results = engine.run()

    results['strategy'] = strategy_name
    results['trades_detail'] = engine.get_trades_df().to_dict('records')
    results['equity_curve'] = engine.get_equity_curve_df().to_dict('records')
    results['stop_stats'] = engine.stop_stats
    results['mtf_blocked'] = engine.mtf_blocked

    return results


def run_fixed_backtest(strategy_class, strategy_name, params, data):
    """运行单策略回测（固定止损止盈版，用于对比）"""
    strategy = strategy_class(instId=data.iloc[0].get('instId', 'UNKNOWN'), **params)

    config = BacktestConfig(
        initial_balance=INITIAL_BALANCE,
        fee_rate=FEE_RATE,
        slippage=SLIPPAGE,
        use_atr_risk=False,       # 关闭 ATR 动态风控
        use_trailing=False,       # 关闭移动止损
        use_mtf=False,
        use_regime=True,
    )
    # 固定止损止盈
    config_fixed = config
    strategy.params['stop_loss_pct'] = 0.05
    strategy.params['take_profit_pct'] = 0.10

    engine = BacktestEngine(strategy, config_fixed)
    engine.load_data(data)
    results = engine.run()

    results['strategy'] = strategy_name + '_fixed'
    return results


def main():
    print("=" * 90)
    print("  生产框架回测 — ATR动态风控 + 移动止损 + 市场状态自适应")
    print("=" * 90)
    print(f"  初始资金: ${INITIAL_BALANCE} | 手续费: {FEE_RATE*100}% | 滑点: {SLIPPAGE*100}%")
    print(f"  币种: {SYMBOLS}")
    print(f"  策略: {[c['name'] for c in STRATEGY_CONFIGS]}")
    print("=" * 90)

    # 拉取数据
    all_data = {}
    for symbol in SYMBOLS:
        df = fetch_candles(symbol, '1D', 500)
        if df is not None and len(df) > 0:
            df['instId'] = symbol
            all_data[symbol] = df
            print(f"  {symbol}: {len(df)} 条日K ({df.iloc[0]['timestamp']} ~ {df.iloc[-1]['timestamp']})")

            # 获取4H数据用于MTF确认
            tf_result = fetch_4h_candles(symbol, 1500)
            if tf_result:
                tf_closes, tf_ts = tf_result
                tf_confirm = align_4h_trend(df['timestamp'].tolist(), tf_closes, tf_ts)
                all_data[symbol + '_tf'] = tf_confirm
                print(f"         4H-K线: {len(tf_closes)} 条, 趋势确认: {sum(tf_confirm)}/{len(tf_confirm)} 根日K处于4H多头")

    if not all_data:
        print("无数据，退出")
        return

    # 市场状态检测
    print(f"\n  市场状态检测:")
    for symbol in SYMBOLS:
        if symbol not in all_data:
            continue
        df = all_data[symbol]
        from src.strategies.base import detect_market_regime, MarketRegime
        regime, info = detect_market_regime(
            df['high'].tolist(),
            df['low'].tolist(),
            df['close'].tolist()
        )
        print(f"    {symbol}: {regime.value} (ADX={info['adx']}, 波动率分位={info['vol_pct']})")

    # 运行所有策略
    all_results = []

    for symbol in SYMBOLS:
        if symbol not in all_data:
            continue
        df = all_data[symbol]
        tf_confirm = all_data.get(symbol + '_tf')

        for config in STRATEGY_CONFIGS:
            print(f"\n{'─'*70}")
            print(f"  {symbol} × {config['name']} (ATR动态风控)")
            print(f"{'─'*70}")

            # ATR 动态风控版
            result = run_single_backtest(
                config['class'], config['name'], config['params'],
                df, tf_confirm
            )
            result['symbol'] = symbol
            all_results.append(result)

            # 固定止损止盈版（对比）
            fixed_result = run_fixed_backtest(
                config['class'], config['name'], config['params'],
                df
            )
            fixed_result['symbol'] = symbol
            all_results.append(fixed_result)

    # ==================== 汇总报告 ====================
    print(f"\n{'='*90}")
    print(f"  回测汇总报告")
    print(f"{'='*90}")

    print(f"\n  {'币种':<12} {'策略':<25} {'模式':<8} {'收益%':<8} {'回撤%':<8} {'夏普':<6} {'胜率%':<6} {'交易':<4} {'盈亏比':<6}")
    print(f"  {'-'*95}")

    for r in all_results:
        is_fixed = '_fixed' in r.get('strategy', '')
        mode = '固定风控' if is_fixed else 'ATR动态'
        strat = r['strategy'].replace('_fixed', '')
        print(f"  {r['symbol']:<12} {strat:<25} {mode:<8} "
              f"{r['total_return']:>+7.2f}% {r['max_drawdown']:>7.2f}% "
              f"{r['sharpe_ratio']:>5.2f} {r['win_rate']:>5.1f}% "
              f"{r['total_trades']:>4} {r.get('profit_ratio', 0):>5.2f}")

    # ATR vs 固定 对比
    print(f"\n  ATR动态风控 vs 固定风控 对比:")
    print(f"  {'币种':<12} {'策略':<25} {'ATR收益%':<10} {'固定收益%':<10} {'差异':<8}")
    print(f"  {'-'*75}")

    for symbol in SYMBOLS:
        for config in STRATEGY_CONFIGS:
            atr_r = next((r for r in all_results if r['symbol'] == symbol and r['strategy'] == config['name']), None)
            fixed_r = next((r for r in all_results if r['symbol'] == symbol and r['strategy'] == config['name'] + '_fixed'), None)
            if atr_r and fixed_r:
                diff = atr_r['total_return'] - fixed_r['total_return']
                print(f"  {symbol:<12} {config['name']:<25} "
                      f"{atr_r['total_return']:>+8.2f}% {fixed_r['total_return']:>+8.2f}% "
                      f"{diff:>+7.2f}%")

    # 退出方式统计
    print(f"\n  退出方式统计 (ATR动态风控版):")
    print(f"  {'币种':<12} {'策略':<25} {'止盈':<6} {'止损':<6} {'移动止损':<8} {'信号卖出':<8} {'MTF拦截':<6}")
    print(f"  {'-'*80}")

    for r in all_results:
        if '_fixed' in r.get('strategy', ''):
            continue
        ss = r.get('stop_stats', {})
        print(f"  {r['symbol']:<12} {r['strategy']:<25} "
              f"{ss.get('take_profit', 0):>5} {ss.get('stop_loss', 0):>5} "
              f"{ss.get('trailing_stop', 0):>7} {ss.get('signal_sell', 0):>7} "
              f"{r.get('mtf_blocked', 0):>5}")

    # 按策略汇总（跨币种平均）
    print(f"\n  策略跨币种平均表现:")
    print(f"  {'策略':<25} {'模式':<8} {'平均收益%':<10} {'平均回撤%':<10} {'平均夏普':<8} {'平均胜率%':<8}")
    print(f"  {'-'*75}")

    for config in STRATEGY_CONFIGS:
        for mode_suffix in ['', '_fixed']:
            mode = '固定风控' if mode_suffix else 'ATR动态'
            name = config['name'] + mode_suffix
            matching = [r for r in all_results if r['strategy'] == name]
            if matching:
                avg_ret = np.mean([r['total_return'] for r in matching])
                avg_dd = np.mean([r['max_drawdown'] for r in matching])
                avg_sharpe = np.mean([r['sharpe_ratio'] for r in matching])
                avg_wr = np.mean([r['win_rate'] for r in matching])
                print(f"  {config['name']:<25} {mode:<8} "
                      f"{avg_ret:>+8.2f}% {avg_dd:>8.2f}% "
                      f"{avg_sharpe:>7.2f} {avg_wr:>7.1f}%")

    # 最终结论
    atr_results = [r for r in all_results if '_fixed' not in r.get('strategy', '')]
    profitable = [r for r in atr_results if r['total_return'] > 0]
    losing = [r for r in atr_results if r['total_return'] <= 0]

    print(f"\n{'='*90}")
    print(f"  最终结论")
    print(f"{'='*90}")
    print(f"  ATR动态风控版:")
    print(f"    总测试: {len(atr_results)} | 盈利: {len(profitable)} | 亏损: {len(losing)}")
    print(f"    盈利率: {len(profitable)/len(atr_results)*100:.1f}%")
    if profitable:
        print(f"    最佳: {max(profitable, key=lambda x: x['total_return'])['symbol']} × "
              f"{max(profitable, key=lambda x: x['total_return'])['strategy']} "
              f"收益 {max(r['total_return'] for r in profitable):.2f}%")
    if losing:
        print(f"    最差: {min(losing, key=lambda x: x['total_return'])['symbol']} × "
              f"{min(losing, key=lambda x: x['total_return'])['strategy']} "
              f"收益 {min(r['total_return'] for r in losing):.2f}%")

    avg_atr = np.mean([r['total_return'] for r in atr_results])
    print(f"    平均收益: {avg_atr:+.2f}%")

    fixed_results = [r for r in all_results if '_fixed' in r.get('strategy', '')]
    avg_fixed = np.mean([r['total_return'] for r in fixed_results])
    print(f"\n  固定风控版平均收益: {avg_fixed:+.2f}%")
    print(f"  ATR动态风控平均收益: {avg_atr:+.2f}%")
    print(f"  差异: {avg_atr - avg_fixed:+.2f}%")

    # 保存JSON报告
    report = {
        'date': datetime.now().isoformat(),
        'config': {
            'initial_balance': INITIAL_BALANCE,
            'fee_rate': FEE_RATE,
            'slippage': SLIPPAGE,
            'symbols': SYMBOLS,
        },
        'results': [{
            'symbol': r['symbol'],
            'strategy': r['strategy'],
            'total_return': r['total_return'],
            'max_drawdown': r['max_drawdown'],
            'sharpe_ratio': r['sharpe_ratio'],
            'win_rate': r['win_rate'],
            'total_trades': r['total_trades'],
            'profit_ratio': r.get('profit_ratio', 0),
            'final_equity': r['final_equity'],
            'stop_stats': r.get('stop_stats', {}),
            'mtf_blocked': r.get('mtf_blocked', 0),
            'regime_stats': r.get('regime_stats', {}),
        } for r in all_results]
    }

    report_path = 'backtest_report_production.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  报告已保存: {report_path}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
