"""
多币种轮动组合策略
主动选择币种、动态轮动、组合配置
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.okx_rest import OKXPublicAPI

INITIAL_BALANCE = 1000
REPORT_DIR = "backtest_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# 币种池
SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'XRP-USDT', 'BNB-USDT', 
           'ADA-USDT', 'DOGE-USDT', 'AVAX-USDT', 'LINK-USDT', 'MATIC-USDT']


class MultiAssetStrategy:
    """多资产轮动策略"""
    
    def __init__(self, config: dict):
        self.config = config
        self.name = config['name']
        
        # 策略参数
        self.lookback = config.get('lookback', 20)
        self.top_n = config.get('top_n', 3)  # 持有前N个最强币种
        self.position_pct = config.get('position_pct', 0.3)  # 单币种仓位
        self.rebalance_freq = config.get('rebalance_freq', 7)  # 调仓频率（天）
        self.stop_loss = config.get('stop_loss', 0.08)
        self.take_profit = config.get('take_profit', 0.15)
        self.trend_filter = config.get('trend_filter', True)  # 是否启用趋势过滤
        
        # 状态
        self.positions = {}  # {symbol: {price, amount, entry_day}}
        self.day_count = 0
        self.last_rebalance = 0
    
    def calc_momentum(self, prices: list, lookback: int) -> float:
        """计算动量（收益率）"""
        if len(prices) < lookback + 1:
            return 0
        return (prices[-1] - prices[-lookback]) / prices[-lookback]
    
    def calc_volatility(self, prices: list, lookback: int) -> float:
        """计算波动率"""
        if len(prices) < lookback:
            return 0
        recent = prices[-lookback:]
        mean = sum(recent) / len(recent)
        variance = sum([(p - mean)**2 for p in recent]) / len(recent)
        return variance ** 0.5
    
    def calc_rsi(self, prices: list, period: int = 14) -> float:
        """计算RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calc_trend(self, prices: list) -> str:
        """判断整体趋势"""
        if len(prices) < 50:
            return 'neutral'
        
        # 使用EMA判断趋势
        ema20 = self.calc_ema(prices[-30:], 20)
        ema50 = self.calc_ema(prices[-70:], 50)
        
        if ema20 > ema50 * 1.02:
            return 'bull'
        elif ema20 < ema50 * 0.98:
            return 'bear'
        return 'neutral'
    
    def calc_ema(self, prices: list, period: int) -> float:
        """计算EMA"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        k = 2 / (period + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = p * k + ema * (1 - k)
        return ema
    
    def rank_symbols(self, market_data: dict) -> list:
        """对所有币种进行评分排名"""
        scores = []
        
        for symbol, data in market_data.items():
            prices = data['prices']
            
            if len(prices) < self.lookback + 1:
                continue
            
            momentum = self.calc_momentum(prices, self.lookback)
            volatility = self.calc_volatility(prices, self.lookback)
            rsi = self.calc_rsi(prices)
            
            # 综合评分
            # 动量正向加分，波动率低加分，RSI不极端加分
            momentum_score = momentum * 100  # 动量贡献
            
            # RSI评分：接近50最好
            rsi_score = -abs(rsi - 50) / 50  # RSI偏离度
            
            # 波动率评分：适度波动加分，太低或太高减分
            vol_score = 0
            if volatility > 0.02 and volatility < 0.10:
                vol_score = 1  # 适度波动
            elif volatility > 0.15:
                vol_score = -1  # 高波动
            
            # 趋势过滤
            trend = self.calc_trend(prices)
            trend_score = 1 if trend == 'bull' else -1 if trend == 'bear' else 0
            
            # 总评分
            total_score = momentum_score + rsi_score + vol_score + trend_score
            
            scores.append({
                'symbol': symbol,
                'score': total_score,
                'momentum': momentum,
                'rsi': rsi,
                'volatility': volatility,
                'trend': trend
            })
        
        # 按评分排序
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores
    
    def should_rebalance(self) -> bool:
        """是否需要调仓"""
        if self.day_count - self.last_rebalance >= self.rebalance_freq:
            return True
        return False
    
    def should_exit_position(self, symbol: str, current_price: float, entry_price: float) -> bool:
        """检查止损止盈"""
        pnl_pct = (current_price - entry_price) / entry_price
        
        if pnl_pct <= -self.stop_loss:
            return 'stop_loss'
        if pnl_pct >= self.take_profit:
            return 'take_profit'
        return None


class PortfolioBacktest:
    """组合回测引擎"""
    
    def __init__(self, strategy: MultiAssetStrategy):
        self.strategy = strategy
        self.balance = INITIAL_BALANCE
        self.positions = {}  # {symbol: {price, amount, entry_day}}
        self.trades = []
        self.equity_curve = []
        self.fee_rate = 0.001
    
    def get_all_data(self) -> dict:
        """获取所有币种数据"""
        api = OKXPublicAPI()
        market_data = {}
        
        for symbol in SYMBOLS:
            result = api.get_candles(symbol, '1D', 200)
            if result.get('code') == '0':
                prices = []
                timestamps = []
                for candle in sorted(result['data'], key=lambda x: x[0]):
                    prices.append(float(candle[4]))
                    timestamps.append(pd.to_datetime(int(candle[0]), unit='ms'))
                market_data[symbol] = {
                    'prices': prices,
                    'timestamps': timestamps
                }
        
        return market_data
    
    def buy(self, symbol: str, price: float, amount: float, day: int, reason: str):
        """买入"""
        cost = amount * price * (1 + self.fee_rate)
        if cost > self.balance:
            amount = self.balance / (price * (1 + self.fee_rate))
            cost = amount * price * (1 + self.fee_rate)
        
        self.balance -= cost
        self.positions[symbol] = {
            'price': price,
            'amount': amount,
            'entry_day': day,
            'cost': cost
        }
        self.trades.append({
            'action': 'buy',
            'symbol': symbol,
            'price': price,
            'amount': amount,
            'day': day,
            'reason': reason,
            'cost': cost
        })
    
    def sell(self, symbol: str, price: float, day: int, reason: str):
        """卖出"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        revenue = pos['amount'] * price * (1 - self.fee_rate)
        pnl = revenue - pos['cost']
        
        self.balance += revenue
        self.trades.append({
            'action': 'sell',
            'symbol': symbol,
            'price': price,
            'amount': pos['amount'],
            'day': day,
            'reason': reason,
            'pnl': pnl,
            'revenue': revenue
        })
        del self.positions[symbol]
    
    def get_equity(self, market_data: dict, day: int) -> float:
        """计算总权益"""
        equity = self.balance
        
        for symbol, pos in self.positions.items():
            if symbol in market_data and day < len(market_data[symbol]['prices']):
                price = market_data[symbol]['prices'][day]
                equity += pos['amount'] * price
        
        return equity
    
    def run(self) -> dict:
        """运行回测"""
        market_data = self.get_all_data()
        
        if not market_data:
            return {'error': 'No data'}
        
        # 找最长数据长度
        max_days = max([len(d['prices']) for d in market_data.values()])
        
        peak_equity = INITIAL_BALANCE
        max_drawdown = 0
        
        for day in range(max_days):
            self.strategy.day_count = day
            
            # 当前价格
            current_prices = {}
            for symbol, data in market_data.items():
                if day < len(data['prices']):
                    current_prices[symbol] = data['prices'][day]
            
            # 检查止损止盈
            for symbol in list(self.positions.keys()):
                if symbol in current_prices:
                    pos = self.positions[symbol]
                    exit_reason = self.strategy.should_exit_position(
                        symbol, current_prices[symbol], pos['price']
                    )
                    if exit_reason:
                        self.sell(symbol, current_prices[symbol], day, exit_reason)
            
            # 调仓检查
            if self.strategy.should_rebalance():
                # 清仓
                for symbol in list(self.positions.keys()):
                    if symbol in current_prices:
                        self.sell(symbol, current_prices[symbol], day, 'rebalance')
                
                # 重新排名选币
                current_data = {}
                for symbol, data in market_data.items():
                    if day < len(data['prices']):
                        current_data[symbol] = {
                            'prices': data['prices'][:day+1]
                        }
                
                ranked = self.strategy.rank_symbols(current_data)
                
                # 趋势过滤
                if self.strategy.trend_filter:
                    # 检查BTC趋势作为大盘判断
                    if 'BTC-USDT' in current_data:
                        btc_trend = self.strategy.calc_trend(current_data['BTC-USDT']['prices'])
                        if btc_trend == 'bear':
                            # 空仓或减仓
                            ranked = ranked[:1]  # 只持有1个
                
                # 买入Top N
                for i, item in enumerate(ranked[:self.strategy.top_n]):
                    symbol = item['symbol']
                    if symbol in current_prices:
                        amount = INITIAL_BALANCE * self.strategy.position_pct / current_prices[symbol]
                        reason = f"rank#{i+1}, momentum={item['momentum']*100:.1f}%, rsi={item['rsi']:.0f}"
                        self.buy(symbol, current_prices[symbol], amount, day, reason)
                
                self.strategy.last_rebalance = day
            
            # 记录权益
            equity = self.get_equity(market_data, day)
            self.equity_curve.append({
                'day': day,
                'equity': equity
            })
            
            if equity > peak_equity:
                peak_equity = equity
            
            dd = (peak_equity - equity) / peak_equity * 100
            if dd > max_drawdown:
                max_drawdown = dd
        
        # 最后清仓
        last_day = max_days - 1
        for symbol in list(self.positions.keys()):
            if symbol in market_data and last_day < len(market_data[symbol]['prices']):
                self.sell(symbol, market_data[symbol]['prices'][last_day], last_day, 'end')
        
        # 统计
        sell_trades = [t for t in self.trades if t['action'] == 'sell']
        wins = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losses = [t for t in sell_trades if t.get('pnl', 0) < 0]
        
        total_pnl = sum([t.get('pnl', 0) for t in sell_trades])
        
        return {
            'strategy': self.strategy.name,
            'initial_balance': INITIAL_BALANCE,
            'final_equity': self.balance,
            'total_return': (self.balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'total_trades': len(sell_trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(sell_trades) * 100 if sell_trades else 0,
            'avg_win': np.mean([t['pnl'] for t in wins]) if wins else 0,
            'avg_loss': np.mean([t['pnl'] for t in losses]) if losses else 0,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }


def run_multi_asset_experiment():
    """运行多资产策略实验"""
    print("=" * 70)
    print("  多币种轮动组合策略实验")
    print("=" * 70)
    
    # 定义策略配置
    configs = [
        {
            'name': 'Momentum_Top3_7d',
            'lookback': 20,
            'top_n': 3,
            'position_pct': 0.30,
            'rebalance_freq': 7,
            'stop_loss': 0.08,
            'take_profit': 0.15,
            'trend_filter': True
        },
        {
            'name': 'Momentum_Top5_14d',
            'lookback': 30,
            'top_n': 5,
            'position_pct': 0.20,
            'rebalance_freq': 14,
            'stop_loss': 0.10,
            'take_profit': 0.20,
            'trend_filter': True
        },
        {
            'name': 'Momentum_Top2_5d',
            'lookback': 10,
            'top_n': 2,
            'position_pct': 0.40,
            'rebalance_freq': 5,
            'stop_loss': 0.06,
            'take_profit': 0.12,
            'trend_filter': True
        },
        {
            'name': 'RSI_Momentum_Top3',
            'lookback': 14,
            'top_n': 3,
            'position_pct': 0.30,
            'rebalance_freq': 7,
            'stop_loss': 0.08,
            'take_profit': 0.15,
            'trend_filter': False
        },
        {
            'name': 'Aggressive_Top5',
            'lookback': 7,
            'top_n': 5,
            'position_pct': 0.20,
            'rebalance_freq': 3,
            'stop_loss': 0.05,
            'take_profit': 0.10,
            'trend_filter': False
        },
        {
            'name': 'Conservative_Top2',
            'lookback': 50,
            'top_n': 2,
            'position_pct': 0.40,
            'rebalance_freq': 30,
            'stop_loss': 0.12,
            'take_profit': 0.25,
            'trend_filter': True
        },
        {
            'name': 'Dynamic_Vol_Top4',
            'lookback': 20,
            'top_n': 4,
            'position_pct': 0.25,
            'rebalance_freq': 10,
            'stop_loss': 0.08,
            'take_profit': 0.15,
            'trend_filter': True
        },
    ]
    
    results = []
    
    for config in configs:
        print(f"\n测试策略: {config['name']}")
        print(f"  配置: Top{config['top_n']}, {config['rebalance_freq']}天调仓, {config['position_pct']*100}%仓位")
        
        strategy = MultiAssetStrategy(config)
        engine = PortfolioBacktest(strategy)
        result = engine.run()
        
        if 'error' not in result:
            results.append(result)
            print(f"  结果: 收益={result['total_return']:.1f}%, 回撤={result['max_drawdown']:.1f}%, 胜率={result['win_rate']:.0f}%, 交易={result['total_trades']}")
    
    # 排序
    results.sort(key=lambda x: x['total_return'], reverse=True)
    
    # 保存报告
    report = {
        'experiment_date': datetime.now().isoformat(),
        'strategies': configs,
        'results': results,
        'top_strategy': results[0] if results else None
    }
    
    with open(os.path.join(REPORT_DIR, 'multi_asset_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # 输出
    print("\n" + "=" * 70)
    print("  多币种轮动策略结果")
    print("=" * 70)
    print(f"{'策略':<25} {'收益':<8} {'回撤':<8} {'胜率':<8} {'交易'}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['strategy']:<25} {r['total_return']:>6.1f}% {r['max_drawdown']:>6.1f}% {r['win_rate']:>6.0f}% {r['total_trades']}")
    
    print("\n最优策略:")
    if results:
        best = results[0]
        print(f"  {best['strategy']}")
        print(f"  收益: {best['total_return']:.1f}%")
        print(f"  回撤: {best['max_drawdown']:.1f}%")
        print(f"  胜率: {best['win_rate']:.0f}%")
        print(f"  交易: {best['total_trades']} 次")
        
        # 显示交易分布
        symbols_traded = {}
        for t in best['trades']:
            if t['action'] == 'buy':
                sym = t['symbol']
                symbols_traded[sym] = symbols_traded.get(sym, 0) + 1
        
        print(f"\n  交易币种分布:")
        for sym, count in sorted(symbols_traded.items(), key=lambda x: -x[1]):
            print(f"    {sym}: {count} 次")
    
    print(f"\n报告已保存: {REPORT_DIR}/multi_asset_report.json")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_multi_asset_experiment()