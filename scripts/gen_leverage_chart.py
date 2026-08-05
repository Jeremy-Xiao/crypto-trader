"""
杠杆回测图表生成 | 读取 backtest_reports/leverage_backtest.json
输出 backtest_reports/backtest_leverage_chart.png（微信小程序可见）

四面板：
  1) 各模式平均收益（红涨绿跌）+ 基准参考线（货币基金 / 买入持有组合）
  2) 风险调整后收益 Calmar（平均收益 / |平均回撤|）
  3) 3年净值曲线（6模式平均 + 货币基金 + 买入持有组合基准）
  4) 安全性核查（强平次数 + 借币利息 + 最大名义杠杆越界）
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm

# 中文显示：优先使用系统 CJK 字体（macOS）
for _cand in ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Songti SC',
              'Heiti SC', 'Microsoft YaHei', 'Noto Sans CJK SC']:
    try:
        if any(_cand.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [_cand]
            break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(HERE, 'backtest_reports', 'leverage_backtest.json')
OUT_PATH = os.path.join(HERE, 'backtest_reports', 'backtest_leverage_chart.png')

# 中文涨跌色：涨=红，跌=绿
RED = '#e23b3b'
GREEN = '#1faa59'
INK = '#222222'
GRID = '#dddddd'
BENCH = '#7a5cc9'     # 买入持有组合
MMF = '#2b7de9'       # 货币基金

MODE_LABELS = {
    '1x': '1x 基准',
    '2x_riskparity': '2x 风险恒定',
    '3x_riskparity': '3x 风险恒定',
    '2x_scaled': '2x 风险放大',
    '3x_scaled': '3x 风险放大',
    '3x_scaled_halt': '3x 放大+熔断',
}
MODE_ORDER = ['1x', '2x_riskparity', '3x_riskparity', '2x_scaled', '3x_scaled', '3x_scaled_halt']
MODE_COLORS = {
    '1x': '#555555',
    '2x_riskparity': '#3b7dd8',
    '3x_riskparity': '#1f5fb0',
    '2x_scaled': '#e08a2b',
    '3x_scaled': '#d2552b',
    '3x_scaled_halt': '#b02b7a',
}


def load():
    with open(JSON_PATH) as f:
        return json.load(f)


def pct_color(v):
    return RED if v >= 0 else GREEN


def main():
    data = load()
    summaries = data['summaries']
    bench = data['benchmarks']
    bh_port = bench['buy_hold'].get('PORTFOLIO_EW', 0.0)
    mmf = bench['money_market']
    days = data.get('period_days', 1095)
    period = data.get('period', '')

    equity = data.get('equity_curves', {})
    # 按模式分组平均净值曲线（每根曲线起点均为 10000）
    mode_curves = {m: [] for m in MODE_ORDER}
    for key, curve in equity.items():
        mode = key.split('|')[0]
        if mode in mode_curves:
            mode_curves[mode].append(np.array(curve, dtype=float))
    avg_curves = {}
    for m, arr_list in mode_curves.items():
        if arr_list:
            avg_curves[m] = np.mean(arr_list, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11))
    fig.suptitle(f'3年杠杆回测对比（{period}）\n唯一变量=杠杆；其余参数一致（ADX30+满仓+做空+ATR风控）',
                 fontsize=14, fontweight='bold', color=INK)
    plt.subplots_adjust(hspace=0.32, wspace=0.22, top=0.90, bottom=0.07, left=0.07, right=0.97)

    # ---------- 面板1：平均收益 ----------
    ax = axes[0, 0]
    vals = [summaries[m]['avg_return'] for m in MODE_ORDER]
    bars = ax.bar(range(len(MODE_ORDER)), vals, color=[pct_color(v) for v in vals],
                  edgecolor='white', linewidth=0.6, zorder=3)
    ax.axhline(0, color=INK, linewidth=0.8, zorder=4)
    ax.axhline(mmf, color=MMF, linewidth=1.3, linestyle='--', zorder=4, label=f'货币基金 {mmf:+.1f}%')
    ax.axhline(bh_port, color=BENCH, linewidth=1.3, linestyle='--', zorder=4,
               label=f'买入持有组合 {bh_port:+.1f}%')
    for i, v in enumerate(vals):
        ax.text(i, v + (6 if v >= 0 else -6), f'{v:+.1f}%', ha='center',
                va='bottom' if v >= 0 else 'top', fontsize=9, color=INK, fontweight='bold')
    ax.set_xticks(range(len(MODE_ORDER)))
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER], rotation=20, ha='right', fontsize=8.5)
    ax.set_ylabel('平均收益 (%)', fontsize=10)
    ax.set_title('① 各模式平均收益（红涨绿跌）', fontsize=11, fontweight='bold')
    ax.grid(axis='y', color=GRID, zorder=0)
    ax.legend(fontsize=8, loc='upper right')

    # ---------- 面板2：Calmar 风险调整 ----------
    ax = axes[0, 1]
    calmars = []
    for m in MODE_ORDER:
        s = summaries[m]
        dd = s['avg_max_drawdown']
        cal = s['avg_return'] / abs(dd) if dd != 0 else 0
        calmars.append(cal)
    bars = ax.bar(range(len(MODE_ORDER)), calmars,
                  color=[MODE_COLORS[m] for m in MODE_ORDER], edgecolor='white',
                  linewidth=0.6, zorder=3)
    for i, v in enumerate(calmars):
        ax.text(i, v + (0.01 if v >= 0 else -0.01), f'{v:.2f}', ha='center',
                va='bottom' if v >= 0 else 'top', fontsize=9, color=INK, fontweight='bold')
    ax.axhline(0, color=INK, linewidth=0.8, zorder=4)
    ax.set_xticks(range(len(MODE_ORDER)))
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER], rotation=20, ha='right', fontsize=8.5)
    ax.set_ylabel('Calmar = 收益 / |回撤|', fontsize=10)
    ax.set_title('② 风险调整后收益（越高越好）', fontsize=11, fontweight='bold')
    ax.grid(axis='y', color=GRID, zorder=0)

    # ---------- 面板3：净值曲线 ----------
    ax = axes[1, 0]
    x = np.arange(days)
    t_years = x / 365.0
    for m in MODE_ORDER:
        if m in avg_curves:
            ax.plot(t_years, avg_curves[m] / 100.0, color=MODE_COLORS[m], linewidth=1.4,
                    label=MODE_LABELS[m])
    # 基准：货币基金复利曲线
    mfm_curve = 10000 * (1 + 0.045) ** t_years
    ax.plot(t_years, mfm_curve / 100.0, color=MMF, linewidth=1.8, linestyle='--',
            label=f'货币基金 {mmf:+.1f}%')
    # 基准：买入持有组合（终点固定，线性近似）
    bh_curve = 10000 * (1 + bh_port / 100.0)
    ax.plot(t_years, np.full_like(t_years, bh_curve) / 100.0, color=BENCH, linewidth=1.8,
            linestyle='--', label=f'买入持有组合 {bh_port:+.1f}%')
    ax.axhline(100, color=INK, linewidth=0.8, linestyle=':', zorder=1)
    ax.set_xlabel('年', fontsize=10)
    ax.set_ylabel('净值（初始=100）', fontsize=10)
    ax.set_title('③ 3年净值曲线（各模式平均）', fontsize=11, fontweight='bold')
    ax.grid(color=GRID, zorder=0)
    ax.legend(fontsize=8, loc='upper left', ncol=2)

    # ---------- 面板4：安全性核查 ----------
    ax = axes[1, 1]
    liqs = [summaries[m].get('total_liquidations', 0) for m in MODE_ORDER]
    costs = [summaries[m].get('total_borrow_cost', 0.0) for m in MODE_ORDER]
    xidx = np.arange(len(MODE_ORDER))
    bars = ax.bar(xidx - 0.2, liqs, width=0.4, color='#c0392b', zorder=3,
                  label='强平次数(累计)')
    for i, v in enumerate(liqs):
        ax.text(i - 0.2, v + 0.15, str(int(v)), ha='center', va='bottom', fontsize=8.5,
                color=INK, fontweight='bold')
    ax.set_ylabel('强平次数', fontsize=10, color='#c0392b')
    ax.set_xticks(range(len(MODE_ORDER)))
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER], rotation=20, ha='right', fontsize=8.5)
    ax.set_title('④ 安全性核查（强平 / 利息 / 杠杆越界）', fontsize=11, fontweight='bold')
    ax.grid(axis='y', color=GRID, zorder=0)
    ax2 = ax.twinx()
    ax2.plot(xidx + 0.2, costs, color='#8e44ad', marker='o', linewidth=1.6, zorder=4,
             label='借币利息(累计, USDT)')
    for i, v in enumerate(costs):
        ax2.text(i + 0.2, v, f'{v:.0f}', ha='center', va='bottom', fontsize=8, color='#8e44ad')
    ax2.set_ylabel('借币利息 (USDT)', fontsize=10, color='#8e44ad')
    # 杠杆越界核查
    max_gross = max((r.get('max_gross_leverage', 0.0) for r in data['results']), default=0.0)
    over = '否' if max_gross <= 3.01 else '是！'
    ax2.text(0.5, -0.18,
             f'硬上限 3.00x ｜ 实际最大名义杠杆 {max_gross:.2f}x ｜ 越界: {over}',
             transform=ax2.transAxes, ha='center', fontsize=8.5, color=INK,
             bbox=dict(boxstyle='round', fc='#f3f3f3', ec=GRID))
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc='upper right')

    fig.text(0.5, 0.025,
             '风控前提：单笔风险不随杠杆放大（risk_scales=False 模式）；3x 放大模式(risk_scales=True)才真正放大仓位。'
             '叠加维持保证金强平 + 借币利息 + 硬上限3x。红色系=风险放大，蓝色系=风险恒定。',
             ha='center', fontsize=8.5, color='#555555')

    fig.savefig(OUT_PATH, dpi=120, bbox_inches='tight')
    print(f'图表已保存: {OUT_PATH}')


if __name__ == '__main__':
    main()
