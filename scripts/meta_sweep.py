"""
元策略稳健性参数扫描（第六轮延伸）| 3年日K
目标：在「震荡市+趋势市都能赚」的前提下，找出让 最弱币种 也尽量好 的参数组合。
优化目标：最大化 三币种中最小收益(worst-coin)，其次平均收益。
复用 meta_backtest 的 run_meta / fetch_history / 数据缓存。

运行：cd /Users/hello/Documents/workspace/crypto-trader && venv/bin/python -u scripts/meta_sweep.py
"""
import os
import sys
import json
import copy
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.meta_backtest import (
    SYMBOLS, TOTAL_DAYS, fetch_history, run_meta, buy_and_hold,
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def load_all():
    all_data = {}
    for symbol in SYMBOLS:
        df = fetch_history(symbol, '1D', TOTAL_DAYS)
        if df is not None and len(df) > 0:
            df = df.copy()
            df['instId'] = symbol
            all_data[symbol] = df
    return all_data


def main():
    print("=" * 90)
    print("  元策略稳健性扫描 | 目标: 三币种最小收益最大化 (worst-coin robust)")
    print("=" * 90)
    all_data = load_all()
    if not all_data:
        print("无数据退出")
        return

    # 扫描网格：聚焦 min_adx=30（已知让4模式全部币种盈利）
    grid = []
    for mode in ('regime', 'ensemble', 'adaptive'):
        for min_adx in (30.0,):
            for gate_off in (0.3, 0.5):
                for perf_temp in (25.0, 50.0):
                    for adx_trend in (20.0, 25.0):
                        grid.append(dict(
                            mode=mode, min_adx=min_adx,
                            gate_off=gate_off, perf_temp=perf_temp, adx_trend=adx_trend,
                        ))

    print(f"网格规模: {len(grid)} 配置 × {len(SYMBOLS)} 币种\n")
    ranked = []
    for i, mp in enumerate(grid, 1):
        tag = f"{mp['mode']}|g{mp['gate_off']}|t{mp['perf_temp']}|adx{mp['adx_trend']}"
        coin_rets = {}
        ok = True
        for sym in SYMBOLS:
            r = run_meta(mp['mode'], all_data[sym], mp['min_adx'], mp)
            if r is None:
                ok = False
                break
            coin_rets[sym] = r['total_return']
        if not ok:
            continue
        worst = min(coin_rets.values())
        avg = sum(coin_rets.values()) / len(coin_rets)
        ranked.append((worst, avg, tag, coin_rets, mp))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    print(f"\n{'排名':<4}{'worst(最弱币)':>14}{'平均':>10}   配置")
    print("-" * 90)
    for i, (worst, avg, tag, cr, mp) in enumerate(ranked[:15], 1):
        print(f"{i:<4}{worst:>+13.2f}%{avg:>+9.2f}%   {tag:<40} "
              f"BTC={cr['BTC-USDT']:+.1f} ETH={cr['ETH-USDT']:+.1f} SOL={cr['SOL-USDT']:+.1f}")

    # 保存
    out = {
        'generated_at': datetime.now().isoformat(),
        'objective': 'maximize worst-coin return across BTC/ETH/SOL',
        'grid_size': len(grid),
        'top': [{
            'tag': tag, 'worst': worst, 'avg': avg, 'coins': cr, 'params': mp
        } for worst, avg, tag, cr, mp in ranked[:15]],
    }
    os.makedirs('backtest_reports', exist_ok=True)
    path = 'backtest_reports/meta_sweep.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)
    print(f"\n扫描报告已保存: {path}")

    if ranked:
        best = ranked[0]
        print(f"\n★ 最稳健配置: {best[2]}")
        print(f"  最弱币种收益 {best[0]:+.2f}%  平均 {best[1]:+.2f}%")
        print(f"  BTC={best[3]['BTC-USDT']:+.2f}% ETH={best[3]['ETH-USDT']:+.2f}% SOL={best[3]['SOL-USDT']:+.2f}%")
        print(f"  参数: {best[4]}")


if __name__ == '__main__':
    main()
