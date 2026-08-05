"""生成做空回测对比图（仅多 vs 双向）"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('backtest_reports/shorting_backtest.json') as f:
    rep = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle('Short-Selling Backtest (v2, stats bugs fixed) | 1Y Daily K | Long-Only vs Bidirectional',
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
s = rep['summary']
ax2.set_title(f"Average Return: Shorting Lifts {s['long_only_avg']:+.2f}% to {s['bidirectional_avg']:+.2f}%",
              fontsize=11, fontweight='bold')
ax2.grid(axis='y', alpha=0.2, linestyle='--')
ax2.set_ylim(-55, 10)

# Panel 3: PnL attribution -- long vs short legs (only trustworthy after stats fix)
ax3 = axes[2]
legs = ['Long leg', 'Short leg']
pnls = [s.get('long_total_pnl', 0), s.get('short_total_pnl', 0)]
lcolors = ['#5DCAA5' if v >= 0 else '#9CA3AF' for v in pnls]
lcolors[1] = '#D85A30' if pnls[1] >= 0 else '#9CA3AF'
bars3 = ax3.bar(legs, pnls, color=lcolors, width=0.5)
for bar, val in zip(bars3, pnls):
    off = 150 if val >= 0 else -150
    ax3.text(bar.get_x() + bar.get_width() / 2, val + off, f'{val:+,.0f}',
             ha='center', va='bottom' if val >= 0 else 'top', fontsize=12, fontweight='bold')
ax3.axhline(0, color='#333', lw=0.8)
ax3.set_ylabel('Cumulative PnL (USDT, 12 groups)')
ax3.set_title(f"PnL Attribution: Short {s.get('short_total_trades',0)} trades "
              f"({s.get('short_win_rate',0):.0f}% win) vs Long {s.get('long_total_trades',0)} trades",
              fontsize=11, fontweight='bold')
ax3.grid(axis='y', alpha=0.2, linestyle='--')

fig.text(0.5, 0.01,
         f"Shorting adds {s['short_delta_avg']:+.2f}% avg/group. "
         f"Bidirectional profitable {s['bidirectional_profitable']}/12 "
         f"vs Long-Only {s['long_only_profitable']}/12. "
         f"In a {s['buy_hold_portfolio']:.1f}% bear market, the short leg is the ONLY profitable side "
         f"({s.get('short_total_pnl',0):+,.0f} vs long {s.get('long_total_pnl',0):+,.0f}).",
         ha='center', fontsize=9, color='#666')
plt.tight_layout(rect=[0, 0.04, 1, 0.95])
plt.savefig('backtest_shorting_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
print('chart saved: backtest_shorting_chart.png')
