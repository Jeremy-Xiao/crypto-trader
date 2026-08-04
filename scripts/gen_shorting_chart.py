"""生成做空回测对比图（仅多 vs 双向）"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('backtest_reports/shorting_backtest.json') as f:
    rep = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Short-Selling Backtest | 1Y Daily K | Long-Only vs Bidirectional',
             fontsize=14, fontweight='bold', y=0.98)

# Panel 1: per-group comparison
pc = rep['pair_comparison']
labels = [f"{p['symbol'].replace('-USDT', '')} {p['strategy'][:6]}" for p in pc]
lo = [p['long_only'] for p in pc]
bi = [p['bidirectional'] for p in pc]
x = np.arange(len(labels))
w = 0.38
ax1 = axes[0]
b1 = ax1.bar(x - w / 2, lo, w, label='Long-Only', color='#9CA3AF')
b2 = ax1.bar(x + w / 2, bi, w, label='Bidirectional (+Short)', color='#D85A30')
for bars, vals in ((b1, lo), (b2, bi)):
    for bar, val in zip(bars, vals):
        off = 0.15 if val >= 0 else -0.15
        ax1.text(bar.get_x() + bar.get_width() / 2, val + off,
                 f'{val:+.1f}', ha='center', va='bottom' if val >= 0 else 'top',
                 fontsize=6, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
ax1.axhline(0, color='#333', lw=0.8)
ax1.set_ylabel('Return (%)')
ax1.set_title('Per-Group: Long-Only vs Bidirectional', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.2, linestyle='--')

# Panel 2: summary
ax2 = axes[1]
cats = ['Long-Only\navg', 'Bidirectional\navg', 'Buy&Hold\nportfolio', 'Money\nMarket']
vals = [rep['summary']['long_only_avg'], rep['summary']['bidirectional_avg'],
        rep['summary']['buy_hold_portfolio'], rep['summary']['money_market']]
colors = ['#9CA3AF', '#D85A30', '#5DCAA5', '#3366CC']
bars = ax2.bar(cats, vals, color=colors, width=0.55)
for bar, val in zip(bars, vals):
    off = 1.2 if val >= 0 else -1.2
    ax2.text(bar.get_x() + bar.get_width() / 2, val + off, f'{val:+.1f}%',
             ha='center', va='bottom' if val >= 0 else 'top', fontsize=11, fontweight='bold')
ax2.axhline(0, color='#333', lw=0.8)
ax2.set_ylabel('Return (%)')
ax2.set_title('Average Return: Shorting Lifts -2.9% to -0.09%', fontsize=11, fontweight='bold')
ax2.grid(axis='y', alpha=0.2, linestyle='--')
ax2.set_ylim(-55, 10)

fig.text(0.5, 0.01,
         f"Shorting adds +{rep['summary']['short_delta_avg']:.2f}% avg/group. "
         f"Bidirectional profitable {rep['summary']['bidirectional_profitable']}/12 "
         f"vs Long-Only {rep['summary']['long_only_profitable']}/12. "
         f"In a -49.7% bear market, shorting turned the strategy near-flat.",
         ha='center', fontsize=9, color='#666')
plt.tight_layout(rect=[0, 0.04, 1, 0.95])
plt.savefig('backtest_shorting_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
print('chart saved: backtest_shorting_chart.png')
