"""诊断实验：1H 语义回测（2026-09-18）

背景：模拟盘 2026-09-06 上线后一周 -5.29%，与日线回测基准（第23章 A方案 half：
3币平均 +36.1%/3年，回撤 -12.9%）方向背离。主嫌疑=实盘用 1H K线决策，
而回测验证的是日线语义。本脚本把**完全相同**的元策略配置放到 1H K线上跑，
回答一个问题：1H 语义本身赚不赚钱？

- 数据：OKX history-candles bar='1H'，尽量拉满 3 年（26,280 根/币），本地缓存
- 策略：MetaStrategy(adaptive, riskoff_long_mode='half', min_adx=0)，与第23章一致
- 引擎：与 meta_riskoff.py 的 ENGINE_COMMON 完全一致（ATR风控3x/6x、满仓、1x、双向）
- 情绪过滤：trading_bias 按日期回放（小时K线取日期部分对齐）
- 注意：引擎夏普按 √365 年化，1H 数据需另乘 √24 才可比（脚本已输出修正值）

运行：cd /Users/hello/Documents/workspace/crypto-trader && venv/bin/python -u scripts/meta_1h_diagnosis.py
（需 export HTTP_PROXY/HTTPS_PROXY=http://127.0.0.1:7897）
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.okx_rest import OKXPublicAPI
from src.monitor.history import bias_for_dates
from src.strategies.meta import MetaStrategy
from src.backtest.engine import BacktestEngine, BacktestConfig

SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
YEARS = 3
TOTAL_BARS = 365 * YEARS * 24  # 26,280
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# 与 meta_riskoff.py 逐字段一致（第23章 A方案基准配置）
ENGINE_COMMON = dict(
    initial_balance=10000, fee_rate=0.001, slippage=0.0005,
    use_atr_risk=True, atr_period=14, use_trailing=False, use_mtf=False,
    tf_confirm=None, use_regime=True, leverage=1.0, max_leverage=3.0,
    maintenance_margin_rate=0.005, liquidation_buffer=0.25,
    borrow_rate_daily=0.0003, risk_scales_with_leverage=False,
    max_position_pct=1.0,
)


def fetch_1h(symbol, total=TOTAL_BARS, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{symbol}_1H_{total}h.csv")
    if use_cache and os.path.exists(cache):
        df = pd.read_csv(cache)
        if len(df) >= total * 0.9:
            print(f"[{symbol}] 使用缓存 {len(df)} 根 1H K线")
            return df
    api = OKXPublicAPI()
    bars, after, seen = [], None, set()
    t0 = time.time()
    while len(seen) < total:
        r = api.get_history_candles(symbol, bar='1H', limit=300, after=after)
        if r.get('code') != '0' or not r.get('data'):
            print(f"[{symbol}] 接口停止返回: code={r.get('code')}（已拉 {len(seen)} 根）")
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
            'timestamp': datetime.fromtimestamp(int(c[0]) / 1000).strftime('%Y-%m-%d %H:%M'),
            'open': float(c[1]), 'high': float(c[2]),
            'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5]),
        })
    df = pd.DataFrame(rows)
    if len(df) > total:
        df = df.iloc[-total:].reset_index(drop=True)
    if len(df) > 0:
        df.to_csv(cache, index=False)
    print(f"[{symbol}] 拉取完成 {len(df)} 根 1H K线，耗时 {time.time()-t0:.0f}s")
    return df


def run_1h(sym, df):
    df = df.copy()
    df['market_bias'] = bias_for_dates([ts[:10] for ts in df['timestamp'].tolist()])
    p = dict(min_adx=0.0, riskoff_long_mode='half')
    strat = MetaStrategy(instId=sym, mode='adaptive', params=p, allow_short=True)
    cfg = BacktestConfig(**ENGINE_COMMON, risk_pct=0.02,
                         atr_multiplier=2.0,
                         atr_sl_multiplier=3.0, atr_tp_multiplier=6.0,
                         min_adx_for_entry=0.0)
    eng = BacktestEngine(strat, cfg)
    eng.load_data(df)
    r = eng.run()
    return r


def main():
    print("=" * 100)
    print("  1H 语义诊断回测 | MetaStrategy adaptive + half | 与第23章日线基准同配置")
    print("=" * 100)
    results = {}
    for sym in SYMBOLS:
        df = fetch_1h(sym)
        if df is None or len(df) < 2000:
            print(f"[{sym}] 数据不足（{0 if df is None else len(df)} 根），跳过")
            continue
        print(f"[{sym}] 开始 1H 回测（{len(df)} 根，约 {len(df)/24:.0f} 天）…")
        t0 = time.time()
        r = run_1h(sym, df)
        results[sym] = r
        print(f"[{sym}] 完成，耗时 {time.time()-t0:.0f}s")

    print()
    print(f"{'币种':<10}{'收益':>10}{'回撤':>10}{'夏普(√365)':>11}{'夏普(√8760修正)':>16}{'交易数':>8}{'强平':>6}")
    print("-" * 100)
    trs, mdds, shs = [], [], []
    for sym, r in results.items():
        n = max(len(r.get('equity_curve', []) or []) - 1, 1)
        # 引擎夏普按 √365 年化；1H 每年 8760 根 → 修正 = 原值 * sqrt(24)
        sh_fix = r.get('sharpe_ratio', 0.0) * np.sqrt(24)
        liq = r.get('liquidations', r.get('total_liquidations', 0))
        print(f"{sym:<10}{r['total_return']:>+9.2f}%{r['max_drawdown']:>+9.2f}%"
              f"{r.get('sharpe_ratio', 0.0):>11.2f}{sh_fix:>16.2f}"
              f"{r.get('total_trades', 0):>8d}{liq:>6d}")
        trs.append(r['total_return']); mdds.append(r['max_drawdown']); shs.append(sh_fix)
    if trs:
        print("-" * 100)
        print(f"平均: 收益 {np.mean(trs):+.2f}% | 回撤 {np.mean(mdds):+.2f}% | 夏普(修正) {np.mean(shs):.2f}")
    print()
    print("日线基准（第23章 A方案 half，3年）: BTC +20.76% / ETH +34.77% / SOL +52.74%"
          " | 平均 +36.1% | 回撤均 -12.9% | 交易 45/49/42")
    print("=" * 100)


if __name__ == '__main__':
    main()
