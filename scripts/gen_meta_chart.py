"""
动态策略切换元策略 —— 对比图生成（WeChat 可见 PNG）

四面板：
A. 各策略/模式 3 币种平均收益（柱）
B. 风险-收益散点（平均回撤 vs 平均收益）
C. 归一化净值曲线（3 币种等权平均）：最佳单策略 vs 最佳元策略 vs 买入持有 vs 货币基金
D. 分币种对比：最佳单策略 vs 最佳元策略（分组柱）
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# —— CJK 字体 ——
FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]
FONT_PATH = None
for f in FONT_CANDIDATES:
    if os.path.exists(f):
        FONT_PATH = f
        break
if FONT_PATH:
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=FONT_PATH).get_name()]
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, 'backtest_reports', 'meta_backtest.json')
CACHE_DIR = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'backtest_reports', 'backtest_meta_chart.png')

SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
MMF_ANNUAL = 0.045


def load_buyhold_equity():
    """从 CSV 计算买入持有组合（3 币种等权）归一化净值（起点=100）"""
    norms = []
    days = None
    for s in SYMBOLS:
        p = os.path.join(CACHE_DIR, f"{s}_1D_1095d.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        n = df['close'].iloc[-1] / df['close'].iloc[0]
        norms.append(n)
        days = len(df)
    if not norms:
        return None
    port = np.mean(norms) * 100.0
    # 构造逐根曲线用于绘图（假设等权每日再平衡近似：用第一只的长度）
    df0 = pd.read_csv(os.path.join(CACHE_DIR, f"{SYMBOLS[0]}_1D_1095d.csv"))
    n = len(df0)
    eq = np.ones(n) * 100.0
    for s in SYMBOLS:
        df = pd.read_csv(os.path.join(CACHE_DIR, f"{s}_1D_1095d.csv"))
        eq = eq + (df['close'].values / df['close'].values[0]) * 100.0
    eq = eq / len(SYMBOLS)
    return eq


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)
    summaries = data['summaries']
    curves = data['equity_curves']
    bh = data.get('benchmarks', {}).get('buy_hold', {})
    mfm = data.get('benchmarks', {}).get('money_market', 0)

    # 仅看 min_adx=0 的公平对比
    single_labels = [l for l in summaries if l.startswith('DoubleMA') or l.startswith('MACD')
                     or l.startswith('Breakout') or l.startswith('RSI_Boll')]
    # 单策略 adx0 标签无后缀，adx30 为 *_adx30
    single0 = [l for l in single_labels if not l.endswith('_adx30')]
    meta0 = [l for l in summaries if l.startswith('meta_') and l.endswith('_adx0')]

    best_single = max(single0, key=lambda l: summaries[l]['avg_return']) if single0 else None
    best_meta = max(meta0, key=lambda l: summaries[l]['avg_return']) if meta0 else None

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('动态策略切换元策略 · 3年日K回测对比', fontsize=16, fontweight='bold')

    # —— Panel A: 平均收益柱 ——
    ax = axes[0, 0]
    labels_a, vals_a, colors_a = [], [], []
    for l in single0:
        labels_a.append(l.replace('_adx0', ''))
        vals_a.append(summaries[l]['avg_return'])
        colors_a.append('#7f7f7f')
    for l in meta0:
        labels_a.append(l.replace('meta_', '').replace('_adx0', ''))
        vals_a.append(summaries[l]['avg_return'])
        colors_a.append('#2ca02c')
    labels_a.append('买入持有')
    vals_a.append(bh.get('PORTFOLIO_EW', 0))
    colors_a.append('#d62728')
    labels_a.append('货币基金')
    vals_a.append(mfm)
    colors_a.append('#1f77b4')
    y = np.arange(len(labels_a))
    ax.barh(y, vals_a, color=colors_a)
    ax.set_yticks(y)
    ax.set_yticklabels(labels_a, fontsize=9)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('3币种平均收益 (%)')
    ax.set_title('A. 各策略/模式平均收益（绿=元策略）', fontsize=11)
    for i, v in enumerate(vals_a):
        ax.text(v + (1 if v >= 0 else -1) * 0.5, i, f'{v:+.1f}%', va='center',
                ha='left' if v >= 0 else 'right', fontsize=8)

    # —— Panel B: 风险-收益散点 ——
    ax = axes[0, 1]
    for l in meta0:
        s = summaries[l]
        ax.scatter(s['avg_max_drawdown'], s['avg_return'], s=90, color='#2ca02c',
                   label=l.replace('meta_', '').replace('_adx0', ''))
    if best_single:
        s = summaries[best_single]
        ax.scatter(s['avg_max_drawdown'], s['avg_return'], s=120, color='#7f7f7f',
                   marker='*', label='最佳单策略')
    ax.scatter(-0, bh.get('PORTFOLIO_EW', 0), s=140, color='#d62728', marker='P',
               label='买入持有')
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('平均最大回撤 (%)')
    ax.set_ylabel('平均收益 (%)')
    ax.set_title('B. 风险-收益（右/上更优）', fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)

    # —— Panel C: 归一化净值曲线（3币种等权平均）——
    ax = axes[1, 0]

    def avg_equity(label):
        eqs = [np.array(curves[f"{label}|{s}"]) for s in SYMBOLS if f"{label}|{s}" in curves]
        if not eqs:
            return None
        m = min(len(e) for e in eqs)
        mat = np.array([e[:m] for e in eqs])
        return mat.mean(axis=0) / mat.mean(axis=0)[0] * 100.0

    bh_eq = load_buyhold_equity()
    xlen = 1095
    if bh_eq is not None:
        xlen = len(bh_eq)
        ax.plot(range(len(bh_eq)), bh_eq, color='#d62728', lw=2, label='买入持有')
    ax.plot(range(xlen), np.ones(xlen) * (100 * pow(1 + MMF_ANNUAL, np.arange(xlen) / 365.0)),
            color='#1f77b4', lw=1.5, label='货币基金(4.5%)')
    if best_single:
        eq = avg_equity(best_single)
        if eq is not None:
            ax.plot(range(len(eq)), eq, color='#7f7f7f', lw=1.8, label=f'最佳单策略({best_single.replace("_adx0","")})')
    if best_meta:
        eq = avg_equity(best_meta)
        if eq is not None:
            ax.plot(range(len(eq)), eq, color='#2ca02c', lw=2.2, label=f'最佳元策略({best_meta.replace("meta_","").replace("_adx0","")})')
    ax.set_xlabel('交易日（3年≈1095根）')
    ax.set_ylabel('归一化净值（起点=100）')
    ax.set_title('C. 净值曲线（3币种等权平均）', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    # —— Panel D: 分币种 最佳单策略 vs 最佳元策略 ——
    ax = axes[1, 1]
    if best_single and best_meta:
        single_sym = [summaries[best_single]['avg_return']]
        meta_sym_ret = []
        single_sym_ret = []
        for s in SYMBOLS:
            r_s = next((r['total_return'] for r in data['results']
                        if r['label'] == best_single and r['symbol'] == s), None)
            r_m = next((r['total_return'] for r in data['results']
                        if r['label'] == best_meta and r['symbol'] == s), None)
            if r_s is not None:
                single_sym_ret.append(r_s)
            if r_m is not None:
                meta_sym_ret.append(r_m)
        x = np.arange(len(SYMBOLS))
        w = 0.38
        ax.bar(x - w/2, single_sym_ret, w, color='#7f7f7f', label=f'单策略({best_single.replace("_adx0","")})')
        ax.bar(x + w/2, meta_sym_ret, w, color='#2ca02c', label=f'元策略({best_meta.replace("meta_","").replace("_adx0","")})')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(SYMBOLS, fontsize=9)
        ax.set_ylabel('收益 (%)')
        ax.set_title('D. 分币种收益对比', fontsize=11)
        ax.legend(fontsize=8)
        for i, (a, b) in enumerate(zip(single_sym_ret, meta_sym_ret)):
            ax.text(i - w/2, a + (1 if a >= 0 else -1)*0.5, f'{a:+.0f}', ha='center',
                    va='bottom' if a >= 0 else 'top', fontsize=7)
            ax.text(i + w/2, b + (1 if b >= 0 else -1)*0.5, f'{b:+.0f}', ha='center',
                    va='bottom' if b >= 0 else 'top', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT, dpi=130)
    print(f"图表已保存: {OUT}")


if __name__ == '__main__':
    main()
