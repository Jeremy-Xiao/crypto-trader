"""
策略择优实验
10轮测试，多币种多策略多参数组合
初始资金: $1000 USDT
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.okx_rest import OKXPublicAPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 实验配置
INITIAL_BALANCE = 1000
FEE_RATE = 0.001
REPORT_DIR = "backtest_reports"

# 币种列表
SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'XRP-USDT', 'BNB-USDT']

# 策略参数组合
STRATEGY_CONFIGS = [
    # 双均线策略
    {'name': 'DoubleMA_5_20', 'type': 'double_ma', 'fast': 5, 'slow': 20, 'position': 0.2, 'stop': 0.08, 'profit': 0.15},
    {'name': 'DoubleMA_10_30', 'type': 'double_ma', 'fast': 10, 'slow': 30, 'position': 0.15, 'stop': 0.06, 'profit': 0.12},
    {'name': 'DoubleMA_7_25', 'type': 'double_ma', 'fast': 7, 'slow': 25, 'position': 0.25, 'stop': 0.10, 'profit': 0.20},
    {'name': 'DoubleMA_3_15', 'type': 'double_ma', 'fast': 3, 'slow': 15, 'position': 0.1, 'stop': 0.05, 'profit': 0.10},
    {'name': 'DoubleMA_12_50', 'type': 'double_ma', 'fast': 12, 'slow': 50, 'position': 0.3, 'stop': 0.10, 'profit': 0.25},
]

class SimpleStrategy:
    """简化策略类，用于快速测试"""
    def __init__(self, instId, config):
        self.instId = instId
        self.config = config
        self.price_history = []
        self.ema_fast = []
        self.ema_slow = []
        self.position = None
        self.trades = []
        
    def calc_ema(self, prices, period):
        """计算EMA - 使用pandas ewm的完整实现"""
        if len(prices) < period:
            return 0
        s = pd.Series(prices)
        return float(s.ewm(span=period, adjust=False).mean().iloc[-1])
    
    def should_stop_loss(self, price):
        if not self.position:
            return False
        loss_pct = (self.position['price'] - price) / self.position['price']
        return loss_pct >= self.config['stop']
    
    def should_take_profit(self, price):
        if not self.position:
            return False
        profit_pct = (price - self.position['price']) / self.position['price']
        return profit_pct >= self.config['profit']
    
    def process(self, price, timestamp):
        self.price_history.append(price)
        
        if len(self.price_history) < self.config['slow'] + 1:
            return None
        
        fast = self.calc_ema(self.price_history, self.config['fast'])
        slow = self.calc_ema(self.price_history, self.config['slow'])
        
        self.ema_fast.append(fast)
        self.ema_slow.append(slow)
        
        # 检测交叉
        if len(self.ema_fast) >= 2:
            prev_fast = self.ema_fast[-2]
            prev_slow = self.ema_slow[-2]
            
            # 金叉买入
            if prev_fast <= prev_slow and fast > slow and not self.position:
                amount = (INITIAL_BALANCE * self.config['position']) / price
                cost = amount * price * (1 + FEE_RATE)
                self.position = {'price': price, 'amount': amount, 'cost': cost, 'time': timestamp}
                self.trades.append({'action': 'buy', 'price': price, 'amount': amount, 'time': timestamp})
                return 'buy'
            
            # 死叉卖出
            elif prev_fast >= prev_slow and fast < slow and self.position:
                revenue = self.position['amount'] * price * (1 - FEE_RATE)
                pnl = revenue - self.position['cost']
                self.trades.append({'action': 'sell', 'price': price, 'amount': self.position['amount'], 'pnl': pnl, 'time': timestamp})
                self.position = None
                return 'sell'
            
            # 止损
            elif self.position and self.should_stop_loss(price):
                revenue = self.position['amount'] * price * (1 - FEE_RATE)
                pnl = revenue - self.position['cost']
                self.trades.append({'action': 'sell', 'price': price, 'amount': self.position['amount'], 'pnl': pnl, 'time': timestamp, 'reason': 'stop_loss'})
                self.position = None
                return 'sell'
            
            # 止盈
            elif self.position and self.should_take_profit(price):
                revenue = self.position['amount'] * price * (1 - FEE_RATE)
                pnl = revenue - self.position['cost']
                self.trades.append({'action': 'sell', 'price': price, 'amount': self.position['amount'], 'pnl': pnl, 'time': timestamp, 'reason': 'take_profit'})
                self.position = None
                return 'sell'
        
        return 'hold'


def get_data(symbol):
    """获取历史数据"""
    api = OKXPublicAPI()
    result = api.get_candles(symbol, '1D', 200)
    
    if result.get('code') != '0':
        return None
    
    data = []
    for candle in sorted(result['data'], key=lambda x: x[0]):
        data.append({
            'timestamp': pd.to_datetime(int(candle[0]), unit='ms'),
            'close': float(candle[4])
        })
    
    return pd.DataFrame(data)


def run_backtest(symbol, config, round_num):
    """运行单次回测"""
    df = get_data(symbol)
    if df is None or len(df) < 50:
        return None
    
    strategy = SimpleStrategy(symbol, config)
    
    balance = INITIAL_BALANCE
    position_value = 0
    equity_curve = []
    
    for _, row in df.iterrows():
        price = row['close']
        ts = row['timestamp']
        
        action = strategy.process(price, ts)
        
        if action == 'buy' and strategy.position:
            cost = strategy.position['amount'] * price * (1 + FEE_RATE)  # 含手续费
            balance -= cost
            position_value = strategy.position['amount'] * price
        elif action == 'sell' and not strategy.position:
            revenue = strategy.trades[-1]['amount'] * price * (1 - FEE_RATE)  # 含手续费
            balance += revenue
            position_value = 0
        
        if strategy.position:
            position_value = strategy.position['amount'] * price
        
        equity = balance + position_value
        equity_curve.append({'time': ts, 'equity': equity, 'price': price})
    
    # 强制平仓
    if strategy.position:
        final_price = df['close'].iloc[-1]
        revenue = strategy.position['amount'] * final_price * (1 - FEE_RATE)
        pnl = revenue - strategy.position['cost']
        balance += revenue
        strategy.trades.append({
            'action': 'sell', 
            'price': final_price, 
            'amount': strategy.position['amount'], 
            'pnl': pnl, 
            'time': df['timestamp'].iloc[-1],
            'reason': 'final_close'
        })
    
    # 计算绩效
    equity_df = pd.DataFrame(equity_curve)
    total_return = (equity_df['equity'].iloc[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    
    peak = equity_df['equity'].cummax()
    drawdown = (equity_df['equity'] - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    sell_trades = [t for t in strategy.trades if t['action'] == 'sell']
    wins = [t for t in sell_trades if t.get('pnl', 0) > 0]
    win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0
    
    total_pnl = sum([t.get('pnl', 0) for t in sell_trades])
    
    return {
        'symbol': symbol,
        'config': config,
        'round': round_num,
        'start_date': df['timestamp'].min().strftime('%Y-%m-%d'),
        'end_date': df['timestamp'].max().strftime('%Y-%m-%d'),
        'days': len(df),
        'price_min': df['close'].min(),
        'price_max': df['close'].max(),
        'volatility': (df['close'].max() - df['close'].min()) / df['close'].min() * 100,
        'initial_balance': INITIAL_BALANCE,
        'final_equity': balance,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'total_trades': len(sell_trades),
        'winning_trades': len(wins),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'trades': strategy.trades,
        'equity_curve': equity_df.to_dict('records')
    }


def save_report(report, round_num, idx):
    """保存回测报告"""
    filename = f"round{round_num}_{idx}_{report['symbol'].replace('-', '')}_{report['config']['name']}.json"
    filepath = os.path.join(REPORT_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # 生成图表
    equity_df = pd.DataFrame(report['equity_curve'])
    plt.figure(figsize=(10, 4))
    plt.plot(equity_df['time'], equity_df['equity'], 'b-')
    plt.axhline(y=INITIAL_BALANCE, color='gray', linestyle='--', alpha=0.5)
    plt.title(f"{report['symbol']} - {report['config']['name']} - Return: {report['total_return']:.1f}%")
    plt.ylabel('Equity (USDT)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(REPORT_DIR, f"round{round_num}_{idx}_{report['symbol'].replace('-', '')}_chart.png")
    plt.savefig(chart_path, dpi=80)
    plt.close()
    
    return filepath


# 执行10轮测试
print("=" * 70)
print("  策略择优实验 - 10轮测试")
print("=" * 70)
print(f"初始资金: $1,000")
print(f"币种: {', '.join(SYMBOLS)}")
print(f"策略组合: {len(STRATEGY_CONFIGS)} 种")
print(f"总测试: {len(SYMBOLS) * len(STRATEGY_CONFIGS)} 次")
print("=" * 70)

all_results = []

for round_num in range(1, 11):
    print(f"\n【第 {round_num} 轮测试】")
    
    round_results = []
    idx = 0
    
    for symbol in SYMBOLS:
        for config in STRATEGY_CONFIGS:
            idx += 1
            print(f"  [{idx}] {symbol} | {config['name']}")
            
            report = run_backtest(symbol, config, round_num)
            
            if report:
                filepath = save_report(report, round_num, idx)
                round_results.append(report)
                
                print(f"      收益: {report['total_return']:.2f}% | 回撤: {report['max_drawdown']:.2f}% | 胜率: {report['win_rate']:.0f}%")
                all_results.append({
                    'round': round_num,
                    'symbol': symbol,
                    'strategy': config['name'],
                    'return': report['total_return'],
                    'drawdown': report['max_drawdown'],
                    'win_rate': report['win_rate'],
                    'trades': report['total_trades'],
                    'file': filepath
                })

# 汇总报告
print("\n" + "=" * 70)
print("  实验汇总")
print("=" * 70)

# 排序找最优
sorted_results = sorted(all_results, key=lambda x: x['return'], reverse=True)

print(f"\n总测试数: {len(all_results)}")

print("\n【TOP 10 最优策略】")
print(f"{'排名':<4} {'轮':<4} {'币种':<12} {'策略':<15} {'收益':<8} {'回撤':<8} {'胜率':<6}")
print("-" * 70)
for i, r in enumerate(sorted_results[:10], 1):
    print(f"{i:<4} {r['round']:<4} {r['symbol']:<12} {r['strategy']:<15} {r['return']:>6.1f}% {r['drawdown']:>6.1f}% {r['win_rate']:>5.0f}%")

print("\n【按币种最优】")
for symbol in SYMBOLS:
    symbol_best = [r for r in sorted_results if r['symbol'] == symbol]
    if symbol_best:
        best = symbol_best[0]
        print(f"  {symbol}: {best['strategy']} | 收益 {best['return']:.1f}%")

print("\n【按策略最优】")
for config in STRATEGY_CONFIGS[:5]:
    strategy_best = [r for r in sorted_results if r['strategy'] == config['name']]
    if strategy_best:
        avg_return = sum([r['return'] for r in strategy_best]) / len(strategy_best)
        print(f"  {config['name']}: 平均收益 {avg_return:.1f}%")

# 保存汇总报告
summary = {
    'experiment_date': datetime.now().isoformat(),
    'rounds': 10,
    'symbols': SYMBOLS,
    'strategies': [c['name'] for c in STRATEGY_CONFIGS],
    'total_tests': len(all_results),
    'top_10': sorted_results[:10],
    'all_results': all_results
}

with open(os.path.join(REPORT_DIR, 'experiment_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n汇总报告已保存: {REPORT_DIR}/experiment_summary.json")
print("=" * 70)