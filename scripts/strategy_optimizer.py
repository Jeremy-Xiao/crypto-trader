"""
策略择优实验框架
支持多策略、多参数、多币种组合测试
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.okx_rest import OKXPublicAPI
from src.utils.indicators import EMA as calc_ema_series, RSI as calc_rsi_series, BollingerBands as calc_boll_series

# 配置
INITIAL_BALANCE = 1000  # USDT
REPORT_DIR = "backtest_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# 交易币种
SYMBOLS = [
    'BTC-USDT',
    'ETH-USDT', 
    'SOL-USDT',
    'XRP-USDT',
    'BNB-USDT',
    'ADA-USDT',
    'DOGE-USDT',
    'AVAX-USDT',
    'MATIC-USDT',
    'LINK-USDT'
]


class BacktestResult:
    """回测结果"""
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.final_equity = INITIAL_BALANCE
        self.total_pnl = 0
        self.max_drawdown = 0
        self.sharpe_ratio = 0
        self.win_rate = 0
        self.profit_factor = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.avg_win = 0
        self.avg_loss = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
    
    def to_dict(self) -> Dict:
        return {
            'final_equity': self.final_equity,
            'total_return': (self.final_equity - INITIAL_BALANCE) / INITIAL_BALANCE * 100,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'trades': self.trades
        }


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, initial_balance: float = INITIAL_BALANCE, fee_rate: float = 0.001):
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.balance = initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = []
    
    def buy(self, price: float, amount: float, timestamp: str, reason: str = ""):
        """买入"""
        if self.position:
            return
        
        cost = amount * price * (1 + self.fee_rate)
        if cost > self.balance:
            amount = self.balance / (price * (1 + self.fee_rate))
            cost = amount * price * (1 + self.fee_rate)
        
        self.balance -= cost
        self.position = {
            'price': price,
            'amount': amount,
            'timestamp': timestamp,
            'cost': cost
        }
        self.trades.append({
            'action': 'buy',
            'price': price,
            'amount': amount,
            'timestamp': timestamp,
            'reason': reason,
            'cost': cost
        })
    
    def sell(self, price: float, timestamp: str, reason: str = ""):
        """卖出"""
        if not self.position:
            return
        
        revenue = self.position['amount'] * price * (1 - self.fee_rate)
        pnl = revenue - self.position['cost']
        
        self.balance += revenue
        self.trades.append({
            'action': 'sell',
            'price': price,
            'amount': self.position['amount'],
            'timestamp': timestamp,
            'reason': reason,
            'pnl': pnl,
            'revenue': revenue
        })
        self.position = None
    
    def get_position_value(self, price: float) -> float:
        """获取持仓市值"""
        if not self.position:
            return 0
        return self.position['amount'] * price
    
    def get_equity(self, price: float) -> float:
        """获取总权益"""
        return self.balance + self.get_position_value(price)
    
    def run(self, df: pd.DataFrame, strategy) -> BacktestResult:
        """运行回测"""
        result = BacktestResult()
        peak_equity = self.initial_balance
        max_dd = 0
        equity_values = []
        
        for _, row in df.iterrows():
            price = row['close']
            timestamp = row['timestamp']
            
            # 策略决策
            action = strategy.generate_signal(price, timestamp, self.position)
            
            if action == 'buy' and not self.position:
                amount = self.initial_balance * strategy.get_position_pct() / price
                self.buy(price, amount, timestamp, strategy.get_buy_reason())
            elif action == 'sell' and self.position:
                self.sell(price, timestamp, strategy.get_sell_reason())
            
            # 记录权益
            equity = self.get_equity(price)
            equity_values.append(equity)
            
            if equity > peak_equity:
                peak_equity = equity
            
            dd = (peak_equity - equity) / peak_equity * 100
            if dd > max_dd:
                max_dd = dd
            
            result.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'price': price
            })
        
        # 强制平仓
        if self.position:
            final_price = df['close'].iloc[-1]
            self.sell(final_price, df['timestamp'].iloc[-1], 'end_of_backtest')
        
        # 计算绩效
        result.trades = self.trades
        result.final_equity = self.balance
        
        sell_trades = [t for t in self.trades if t['action'] == 'sell']
        result.total_trades = len(sell_trades)
        
        wins = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losses = [t for t in sell_trades if t.get('pnl', 0) < 0]
        
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0
        
        result.total_pnl = sum([t.get('pnl', 0) for t in sell_trades])
        result.avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        result.avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
        
        gross_profit = sum([t['pnl'] for t in wins]) if wins else 0
        gross_loss = abs(sum([t['pnl'] for t in losses])) if losses else 0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        result.max_drawdown = max_dd
        
        # 计算夏普比率
        if len(equity_values) > 1:
            returns = pd.Series(equity_values).pct_change().dropna()
            if returns.std() > 0:
                result.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365)
        
        # 连续盈亏
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for t in sell_trades:
            if t.get('pnl', 0) > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            elif t.get('pnl', 0) < 0:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        result.max_consecutive_wins = max_consecutive_wins
        result.max_consecutive_losses = max_consecutive_losses
        
        return result


class Strategy:
    """策略基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.buy_reason = ""
        self.sell_reason = ""
    
    def generate_signal(self, price: float, timestamp: str, position: dict) -> str:
        """返回 'buy', 'sell', 或 'hold'"""
        raise NotImplementedError
    
    def get_position_pct(self) -> float:
        return 0.2
    
    def get_buy_reason(self) -> str:
        return self.buy_reason
    
    def get_sell_reason(self) -> str:
        return self.sell_reason


class DoubleMAStrategy(Strategy):
    """双均线策略"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30, position_pct: float = 0.2, stop_loss: float = 0.08, take_profit: float = 0.15):
        super().__init__(f"DoubleMA_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_pct = position_pct
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.prices = []
        self.fast_ema = []
        self.slow_ema = []
    
    def calc_ema(self, prices, period):
        """计算EMA - 使用pandas ewm的完整实现"""
        if len(prices) < period:
            return 0
        s = pd.Series(prices)
        return float(s.ewm(span=period, adjust=False).mean().iloc[-1])
    
    def generate_signal(self, price: float, timestamp: str, position: dict) -> str:
        self.prices.append(price)
        
        if len(self.prices) < self.slow_period + 1:
            return 'hold'
        
        fast = self.calc_ema(self.prices, self.fast_period)
        slow = self.calc_ema(self.prices, self.slow_period)
        
        self.fast_ema.append(fast)
        self.slow_ema.append(slow)
        
        # 检查交叉
        if len(self.fast_ema) >= 2:
            prev_fast = self.fast_ema[-2]
            prev_slow = self.slow_ema[-2]
            
            # 金叉
            if prev_fast <= prev_slow and fast > slow:
                if not position:
                    self.buy_reason = f"Golden cross: EMA{self.fast_period}({fast:.0f}) > EMA{self.slow_period}({slow:.0f})"
                    return 'buy'
            
            # 死叉
            elif prev_fast >= prev_slow and fast < slow:
                if position:
                    self.sell_reason = f"Death cross: EMA{self.fast_period}({fast:.0f}) < EMA{self.slow_period}({slow:.0f})"
                    return 'sell'
        
        # 止损止盈
        if position:
            pnl_pct = (price - position['price']) / position['price']
            
            if pnl_pct <= -self.stop_loss:
                self.sell_reason = f"Stop loss: {pnl_pct*100:.1f}%"
                return 'sell'
            
            if pnl_pct >= self.take_profit:
                self.sell_reason = f"Take profit: {pnl_pct*100:.1f}%"
                return 'sell'
        
        return 'hold'
    
    def get_position_pct(self) -> float:
        return self.position_pct


class RSIStrategy(Strategy):
    """RSI策略"""
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70, position_pct: float = 0.2):
        super().__init__(f"RSI_{period}_{int(oversold)}_{int(overbought)}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.position_pct = position_pct
        self.prices = []
        self.gains = []
        self.losses = []
    
    def calc_rsi(self):
        """计算RSI - 使用pandas实现"""
        if len(self.prices) < self.period + 1:
            return 50
        s = pd.Series(self.prices)
        delta = s.diff()
        gain = delta.where(delta > 0, 0)
        loss = delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        if pd.isna(avg_gain.iloc[-1]) or pd.isna(avg_loss.iloc[-1]):
            return 50
        if avg_loss.iloc[-1] == 0:
            return 100
        rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
        return float(100 - (100 / (1 + rs)))
    
    def generate_signal(self, price: float, timestamp: str, position: dict) -> str:
        self.prices.append(price)
        
        if len(self.prices) < self.period + 2:
            return 'hold'
        
        rsi = self.calc_rsi()
        
        # RSI超卖买入
        if rsi < self.oversold and not position:
            self.buy_reason = f"RSI oversold: {rsi:.1f} < {self.oversold}"
            return 'buy'
        
        # RSI超买卖出
        if rsi > self.overbought and position:
            self.sell_reason = f"RSI overbought: {rsi:.1f} > {self.overbought}"
            return 'sell'
        
        return 'hold'
    
    def get_position_pct(self) -> float:
        return self.position_pct


class BollingerStrategy(Strategy):
    """布林带策略"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, position_pct: float = 0.2):
        super().__init__(f"Bollinger_{period}_{std_dev}")
        self.period = period
        self.std_dev = std_dev
        self.position_pct = position_pct
        self.prices = []
    
    def generate_signal(self, price: float, timestamp: str, position: dict) -> str:
        self.prices.append(price)
        
        if len(self.prices) < self.period:
            return 'hold'
        
        s = pd.Series(self.prices)
        middle = s.rolling(window=self.period).mean()
        std = s.rolling(window=self.period).std()
        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std
        
        if pd.isna(upper.iloc[-1]):
            return 'hold'
        
        upper_val = float(upper.iloc[-1])
        middle_val = float(middle.iloc[-1])
        lower_val = float(lower.iloc[-1])
        
        # 价格触及下轨买入
        if price <= lower_val and not position:
            self.buy_reason = f"Price at lower band: {price:.0f} <= {lower_val:.0f}"
            return 'buy'
        
        # 价格触及上轨卖出
        if price >= upper_val and position:
            self.sell_reason = f"Price at upper band: {price:.0f} >= {upper_val:.0f}"
            return 'sell'
        
        # 止损止盈
        if position:
            pnl_pct = (price - position['price']) / position['price']
            if pnl_pct <= -0.05:
                self.sell_reason = f"Stop loss: {pnl_pct*100:.1f}%"
                return 'sell'
            if pnl_pct >= 0.15:
                self.sell_reason = f"Take profit: {pnl_pct*100:.1f}%"
                return 'sell'
        
        return 'hold'
    
    def get_position_pct(self) -> float:
        return self.position_pct


class MomentumStrategy(Strategy):
    """动量策略"""
    
    def __init__(self, lookback: int = 10, threshold: float = 0.05, position_pct: float = 0.2):
        super().__init__(f"Momentum_{lookback}_{threshold}")
        self.lookback = lookback
        self.threshold = threshold
        self.position_pct = position_pct
        self.prices = []
    
    def generate_signal(self, price: float, timestamp: str, position: dict) -> str:
        self.prices.append(price)
        
        if len(self.prices) < self.lookback + 1:
            return 'hold'
        
        old_price = self.prices[-self.lookback-1]
        momentum = (price - old_price) / old_price
        
        # 正向动量买入
        if momentum > self.threshold and not position:
            self.buy_reason = f"Momentum: {momentum*100:.1f}% > {self.threshold*100}%"
            return 'buy'
        
        # 负向动量卖出
        if momentum < -self.threshold and position:
            self.sell_reason = f"Momentum: {momentum*100:.1f}% < {-self.threshold*100}%"
            return 'sell'
        
        return 'hold'
    
    def get_position_pct(self) -> float:
        return self.position_pct


def get_data(symbol: str) -> pd.DataFrame:
    """获取历史数据"""
    api = OKXPublicAPI()
    result = api.get_candles(symbol, '1D', 200)
    
    if result.get('code') != '0':
        return None
    
    data = []
    for candle in sorted(result['data'], key=lambda x: x[0]):
        data.append({
            'timestamp': pd.to_datetime(int(candle[0]), unit='ms'),
            'open': float(candle[1]),
            'high': float(candle[2]),
            'low': float(candle[3]),
            'close': float(candle[4]),
            'volume': float(candle[5])
        })
    
    return pd.DataFrame(data)


def run_experiment():
    """运行策略实验"""
    print("=" * 70)
    print("  策略择优实验")
    print("=" * 70)
    print(f"初始资金: ${INITIAL_BALANCE}")
    print(f"币种数量: {len(SYMBOLS)}")
    print("=" * 70)
    
    all_results = []
    
    # 定义策略组合
    strategies = [
        DoubleMAStrategy(fast_period=5, slow_period=20, position_pct=0.2, stop_loss=0.08, take_profit=0.15),
        DoubleMAStrategy(fast_period=10, slow_period=30, position_pct=0.15, stop_loss=0.06, take_profit=0.12),
        DoubleMAStrategy(fast_period=7, slow_period=25, position_pct=0.25, stop_loss=0.10, take_profit=0.20),
        DoubleMAStrategy(fast_period=3, slow_period=15, position_pct=0.1, stop_loss=0.05, take_profit=0.10),
        DoubleMAStrategy(fast_period=12, slow_period=50, position_pct=0.3, stop_loss=0.10, take_profit=0.25),
        RSIStrategy(period=14, oversold=30, overbought=70, position_pct=0.2),
        RSIStrategy(period=14, oversold=25, overbought=75, position_pct=0.15),
        RSIStrategy(period=10, oversold=35, overbought=65, position_pct=0.2),
        BollingerStrategy(period=20, std_dev=2.0, position_pct=0.2),
        BollingerStrategy(period=15, std_dev=1.5, position_pct=0.25),
        BollingerStrategy(period=25, std_dev=2.5, position_pct=0.15),
        MomentumStrategy(lookback=10, threshold=0.05, position_pct=0.2),
        MomentumStrategy(lookback=20, threshold=0.08, position_pct=0.15),
        MomentumStrategy(lookback=5, threshold=0.03, position_pct=0.25),
    ]
    
    round_num = 1
    
    for symbol in SYMBOLS:
        print(f"\n【{symbol}】")
        
        df = get_data(symbol)
        if df is None or len(df) < 50:
            print(f"  数据不足，跳过")
            continue
        
        print(f"  数据: {df['timestamp'].min().strftime('%Y-%m-%d')} ~ {df['timestamp'].max().strftime('%Y-%m-%d')}")
        print(f"  价格: ${df['close'].min():,.0f} ~ ${df['close'].max():,.0f}")
        
        for strategy in strategies:
            engine = BacktestEngine()
            result = engine.run(df, strategy)
            
            result_dict = result.to_dict()
            result_dict['symbol'] = symbol
            result_dict['strategy'] = strategy.name
            result_dict['round'] = round_num
            result_dict['data_points'] = len(df)
            result_dict['price_range'] = f"${df['close'].min():,.0f}-${df['close'].max():,.0f}"
            
            all_results.append(result_dict)
            
            print(f"  {strategy.name:<30} | 收益: {result_dict['total_return']:>6.1f}% | 回撤: {result_dict['max_drawdown']:>6.1f}% | 胜率: {result_dict['win_rate']:>5.1f}% | 交易: {result_dict['total_trades']}")
        
        round_num += 1
    
    # 排序
    all_results.sort(key=lambda x: x['total_return'], reverse=True)
    
    # 保存报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(REPORT_DIR, f'round_{round_num-1}_report.json')
    
    with open(report_file, 'w') as f:
        json.dump({
            'experiment_date': datetime.now().isoformat(),
            'total_tests': len(all_results),
            'symbols': SYMBOLS,
            'strategies': [s.name for s in strategies],
            'results': all_results
        }, f, indent=2, default=str)
    
    # 打印TOP 10
    print("\n" + "=" * 70)
    print("  TOP 10 最优策略")
    print("=" * 70)
    print(f"{'排名':<4} {'币种':<12} {'策略':<30} {'收益':<8} {'回撤':<8} {'胜率':<6} {'交易'}")
    print("-" * 70)
    
    for i, r in enumerate(all_results[:10], 1):
        print(f"{i:<4} {r['symbol']:<12} {r['strategy']:<30} {r['total_return']:>6.1f}% {r['max_drawdown']:>6.1f}% {r['win_rate']:>5.1f}% {r['total_trades']}")
    
    # 按币种找最优
    print("\n" + "=" * 70)
    print("  各币种最优策略")
    print("=" * 70)
    
    for symbol in SYMBOLS:
        symbol_results = [r for r in all_results if r['symbol'] == symbol]
        if symbol_results:
            best = symbol_results[0]
            print(f"  {symbol:<12} | {best['strategy']:<30} | {best['total_return']:>6.1f}%")
    
    print(f"\n完整报告已保存: {report_file}")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    run_experiment()