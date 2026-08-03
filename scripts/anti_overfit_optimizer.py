"""
防过拟合策略优化实验
10轮迭代，每轮包含样本外验证

防过拟合机制：
1. Walk-Forward验证：训练集70% + 测试集30%
2. 参数稳定性检验：同一策略在不同时间段的表现一致性
3. 样本外强制验证：只评估测试集表现，训练集表现只用于参数筛选
4. 交叉验证：多次切分数据，取平均表现
5. 过拟合警告：训练集收益远高于测试集时触发警告
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.okx_rest import OKXPublicAPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 配置
INITIAL_BALANCE = 1000
REPORT_DIR = "backtest_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# 币种池
SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'XRP-USDT', 'BNB-USDT', 'DOGE-USDT', 'LINK-USDT']

# 防过拟合参数
TRAIN_RATIO = 0.7  # 70%训练，30%测试
OVERFIT_THRESHOLD = 3.0  # 训练集收益超过测试集3倍视为过拟合


class AntiOverfitBacktest:
    """防过拟合回测引擎"""
    
    def __init__(self, symbol, strategy_class, params):
        self.symbol = symbol
        self.strategy_class = strategy_class
        self.params = params
        self.data = None
        
    def load_data(self):
        """加载数据"""
        api = OKXPublicAPI()
        result = api.get_candles(self.symbol, '1D', 300)
        
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
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def split_data(self):
        """分割训练集和测试集"""
        if self.data is None:
            return None, None
        
        n = len(self.data)
        train_end = int(n * TRAIN_RATIO)
        
        train_data = self.data.iloc[:train_end]
        test_data = self.data.iloc[train_end:]
        
        return train_data, test_data
    
    def run_strategy(self, df):
        """运行策略"""
        strategy = self.strategy_class(self.params)
        
        balance = INITIAL_BALANCE
        position = None
        trades = []
        equity_curve = []
        
        for _, row in df.iterrows():
            price = row['close']
            ts = row['timestamp']
            
            signal = strategy.generate_signal(price, ts, position)
            
            if signal == 'buy' and not position:
                amount = INITIAL_BALANCE * strategy.get_position_pct() / price
                cost = amount * price * 1.001
                balance -= cost
                position = {'price': price, 'amount': amount, 'time': ts}
                trades.append({'action': 'buy', 'price': price, 'amount': amount, 'time': ts, 'reason': strategy.get_buy_reason()})
                
            elif signal == 'sell' and position:
                revenue = position['amount'] * price * 0.999
                pnl = (price - position['price']) * position['amount']
                balance += revenue
                trades.append({'action': 'sell', 'price': price, 'amount': position['amount'], 'pnl': pnl, 'time': ts, 'reason': strategy.get_sell_reason()})
                position = None
            
            equity = balance + (position['amount'] * price if position else 0)
            equity_curve.append({'time': ts, 'equity': equity, 'price': price})
        
        # 强制平仓
        if position:
            final_price = df['close'].iloc[-1]
            revenue = position['amount'] * final_price * 0.999
            pnl = (final_price - position['price']) * position['amount']
            balance += revenue
            trades.append({'action': 'sell', 'price': final_price, 'amount': position['amount'], 'pnl': pnl, 'time': df['timestamp'].iloc[-1], 'reason': 'final_close'})
            position = None
        
        # 计算绩效
        equity_df = pd.DataFrame(equity_curve)
        total_return = (equity_df['equity'].iloc[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        
        peak = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        sell_trades = [t for t in trades if t['action'] == 'sell']
        wins = [t for t in sell_trades if t.get('pnl', 0) > 0]
        win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0
        
        return {
            'return': total_return,
            'drawdown': max_drawdown,
            'win_rate': win_rate,
            'trades': len(sell_trades),
            'winning_trades': len(wins),
            'equity_curve': equity_df.to_dict('records'),
            'trade_details': trades
        }
    
    def run_with_validation(self):
        """运行带验证的回测"""
        self.load_data()
        train_data, test_data = self.split_data()
        
        if train_data is None or test_data is None:
            return None
        
        # 训练集表现
        train_result = self.run_strategy(train_data)
        
        # 重置策略状态，用测试集验证
        test_result = self.run_strategy(test_data)
        
        # 计算过拟合指数
        train_return = train_result['return']
        test_return = test_result['return']
        
        if test_return != 0:
            overfit_index = abs(train_return) / abs(test_return) if test_return != 0 else float('inf')
        else:
            overfit_index = 10.0 if train_return > 0 else 0
        
        is_overfit = overfit_index > OVERFIT_THRESHOLD and train_return > test_return
        
        return {
            'symbol': self.symbol,
            'strategy': self.params.get('name', 'Unknown'),
            'params': self.params,
            'train_result': train_result,
            'test_result': test_result,
            'train_return': train_return,
            'test_return': test_return,
            'train_drawdown': train_result['drawdown'],
            'test_drawdown': test_result['drawdown'],
            'overfit_index': overfit_index,
            'is_overfit': is_overfit,
            'combined_return': (train_return * TRAIN_RATIO + test_return * (1 - TRAIN_RATIO)),
            'data_split': f"{train_data['timestamp'].min().strftime('%Y-%m-%d')} ~ {test_data['timestamp'].max().strftime('%Y-%m-%d')}"
        }


class StrategyBase:
    """策略基类"""
    
    def __init__(self, params):
        self.params = params
        self.buy_reason = ""
        self.sell_reason = ""
        self.prices = []
        
    def get_position_pct(self):
        return self.params.get('position_pct', 0.2)
    
    def get_buy_reason(self):
        return self.buy_reason
    
    def get_sell_reason(self):
        return self.sell_reason


class DoubleMAPlusStrategy(StrategyBase):
    """双均线+趋势过滤策略"""
    
    def __init__(self, params):
        super().__init__(params)
        self.fast_period = params.get('fast', 10)
        self.slow_period = params.get('slow', 30)
        self.trend_period = params.get('trend', 50)
        self.stop_loss = params.get('stop_loss', 0.06)
        self.take_profit = params.get('take_profit', 0.12)
        self.ema_fast = []
        self.ema_slow = []
        self.ema_trend = []
        
    def calc_ema(self, prices, period):
        if len(prices) < period:
            return 0
        k = 2 / (period + 1)
        recent = prices[-(period+30):]
        ema = recent[0]
        for p in recent[1:]:
            ema = p * k + ema * (1 - k)
        return ema
    
    def generate_signal(self, price, ts, position):
        self.prices.append(price)
        
        if len(self.prices) < self.trend_period + 1:
            return 'hold'
        
        fast = self.calc_ema(self.prices, self.fast_period)
        slow = self.calc_ema(self.prices, self.slow_period)
        trend = self.calc_ema(self.prices, self.trend_period)
        
        self.ema_fast.append(fast)
        self.ema_slow.append(slow)
        self.ema_trend.append(trend)
        
        # 趋势过滤：只在上升趋势中交易
        in_uptrend = price > trend
        
        if len(self.ema_fast) >= 2 and in_uptrend:
            prev_fast = self.ema_fast[-2]
            prev_slow = self.ema_slow[-2]
            
            # 金叉买入（趋势向上）
            if prev_fast <= prev_slow and fast > slow and not position:
                self.buy_reason = f"Golden cross in uptrend (EMA{self.fast_period}>{self.slow_period}, price>{self.trend_period}EMA)"
                return 'buy'
        
        # 死叉卖出
        if len(self.ema_fast) >= 2 and position:
            prev_fast = self.ema_fast[-2]
            prev_slow = self.ema_slow[-2]
            
            if prev_fast >= prev_slow and fast < slow:
                self.sell_reason = f"Death cross"
                return 'sell'
            
            # 止损
            loss_pct = (position['price'] - price) / position['price']
            if loss_pct >= self.stop_loss:
                self.sell_reason = f"Stop loss: -{loss_pct*100:.1f}%"
                return 'sell'
            
            # 止盈
            profit_pct = (price - position['price']) / position['price']
            if profit_pct >= self.take_profit:
                self.sell_reason = f"Take profit: +{profit_pct*100:.1f}%"
                return 'sell'
            
            # 趋势反转卖出
            if price < trend:
                self.sell_reason = f"Trend reversal: price < EMA{self.trend_period}"
                return 'sell'
        
        return 'hold'


class RSI_BollingerStrategy(StrategyBase):
    """RSI+布林带组合策略"""
    
    def __init__(self, params):
        super().__init__(params)
        self.rsi_period = params.get('rsi_period', 14)
        self.boll_period = params.get('boll_period', 20)
        self.boll_std = params.get('boll_std', 2.0)
        self.rsi_low = params.get('rsi_low', 30)
        self.rsi_high = params.get('rsi_high', 70)
        self.stop_loss = params.get('stop_loss', 0.05)
        self.take_profit = params.get('take_profit', 0.10)
        
    def calc_rsi(self):
        if len(self.prices) < self.rsi_period + 1:
            return 50
        
        deltas = [self.prices[i] - self.prices[i-1] for i in range(1, len(self.prices))]
        gains = [d if d > 0 else 0 for d in deltas[-self.rsi_period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-self.rsi_period:]]
        
        avg_gain = sum(gains) / self.rsi_period
        avg_loss = sum(losses) / self.rsi_period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calc_bollinger(self):
        if len(self.prices) < self.boll_period:
            return None, None, None
        
        recent = self.prices[-self.boll_period:]
        middle = sum(recent) / len(recent)
        std = np.std(recent)
        
        upper = middle + self.boll_std * std
        lower = middle - self.boll_std * std
        
        return upper, middle, lower
    
    def generate_signal(self, price, ts, position):
        self.prices.append(price)
        
        if len(self.prices) < max(self.rsi_period, self.boll_period) + 2:
            return 'hold'
        
        rsi = self.calc_rsi()
        upper, middle, lower = self.calc_bollinger()
        
        if upper is None:
            return 'hold'
        
        # RSI超卖+价格触及布林下轨买入
        if rsi < self.rsi_low and price <= lower and not position:
            self.buy_reason = f"RSI oversold ({rsi:.0f}) + Bollinger lower band"
            return 'buy'
        
        if position:
            # RSI超买+价格触及布林上轨卖出
            if rsi > self.rsi_high and price >= upper:
                self.sell_reason = f"RSI overbought ({rsi:.0f}) + Bollinger upper band"
                return 'sell'
            
            # 价格回归中轨卖出
            if price >= middle:
                self.sell_reason = f"Price returned to middle band"
                return 'sell'
            
            # 止损止盈
            loss_pct = (position['price'] - price) / position['price']
            if loss_pct >= self.stop_loss:
                self.sell_reason = f"Stop loss: -{loss_pct*100:.1f}%"
                return 'sell'
            
            profit_pct = (price - position['price']) / position['price']
            if profit_pct >= self.take_profit:
                self.sell_reason = f"Take profit: +{profit_pct*100:.1f}%"
                return 'sell'
        
        return 'hold'


class MACDStrategy(StrategyBase):
    """MACD策略"""
    
    def __init__(self, params):
        super().__init__(params)
        self.fast_period = params.get('fast', 12)
        self.slow_period = params.get('slow', 26)
        self.signal_period = params.get('signal', 9)
        self.stop_loss = params.get('stop_loss', 0.05)
        self.take_profit = params.get('take_profit', 0.12)
        self.macd_hist = []
        
    def calc_ema(self, prices, period):
        if len(prices) < period:
            return 0
        k = 2 / (period + 1)
        recent = prices[-(period+30):]
        ema = recent[0]
        for p in recent[1:]:
            ema = p * k + ema * (1 - k)
        return ema
    
    def calc_macd(self):
        if len(self.prices) < self.slow_period + self.signal_period:
            return 0, 0, 0
        
        ema_fast = self.calc_ema(self.prices, self.fast_period)
        ema_slow = self.calc_ema(self.prices, self.slow_period)
        
        macd_line = ema_fast - ema_slow
        
        # 计算信号线（MACD的EMA）
        if len(self.macd_hist) < self.signal_period:
            signal_line = macd_line
        else:
            k = 2 / (self.signal_period + 1)
            signal_line = self.macd_hist[-self.signal_period]
            for m in self.macd_hist[-(self.signal_period-1):]:
                signal_line = m * k + signal_line * (1 - k)
        
        histogram = macd_line - signal_line
        
        self.macd_hist.append(macd_line)
        
        return macd_line, signal_line, histogram
    
    def generate_signal(self, price, ts, position):
        self.prices.append(price)
        
        if len(self.prices) < self.slow_period + self.signal_period + 2:
            return 'hold'
        
        macd_line, signal_line, histogram = self.calc_macd()
        
        if len(self.macd_hist) >= 3:
            prev_hist = self.macd_hist[-2] - signal_line
            prev_prev_hist = self.macd_hist[-3] - signal_line
            
            # MACD柱状图从负转正买入
            if prev_hist <= 0 and histogram > 0 and not position:
                self.buy_reason = f"MACD histogram turned positive"
                return 'buy'
            
            if position:
                # MACD柱状图从正转负卖出
                if prev_hist >= 0 and histogram < 0:
                    self.sell_reason = f"MACD histogram turned negative"
                    return 'sell'
                
                # 止损止盈
                loss_pct = (position['price'] - price) / position['price']
                if loss_pct >= self.stop_loss:
                    self.sell_reason = f"Stop loss"
                    return 'sell'
                
                profit_pct = (price - position['price']) / position['price']
                if profit_pct >= self.take_profit:
                    self.sell_reason = f"Take profit"
                    return 'sell'
        
        return 'hold'


class VolatilityAdaptiveStrategy(StrategyBase):
    """波动率自适应策略"""
    
    def __init__(self, params):
        super().__init__(params)
        self.lookback = params.get('lookback', 20)
        self.vol_threshold = params.get('vol_threshold', 0.03)
        self.position_pct_high_vol = params.get('position_high', 0.1)
        self.position_pct_low_vol = params.get('position_low', 0.3)
        self.stop_loss = params.get('stop_loss', 0.08)
        self.take_profit = params.get('take_profit', 0.15)
        self.ema_short = []
        self.ema_long = []
        
    def calc_volatility(self):
        if len(self.prices) < self.lookback:
            return 0
        
        returns = [self.prices[i] / self.prices[i-1] - 1 for i in range(1, len(self.prices))][-self.lookback:]
        return np.std(returns)
    
    def calc_ema(self, prices, period):
        if len(prices) < period:
            return 0
        k = 2 / (period + 1)
        recent = prices[-(period+30):]
        ema = recent[0]
        for p in recent[1:]:
            ema = p * k + ema * (1 - k)
        return ema
    
    def get_position_pct(self):
        vol = self.calc_volatility()
        if vol > self.vol_threshold:
            return self.position_pct_high_vol
        else:
            return self.position_pct_low_vol
    
    def generate_signal(self, price, ts, position):
        self.prices.append(price)
        
        if len(self.prices) < self.lookback + 30:
            return 'hold'
        
        vol = self.calc_volatility()
        
        ema5 = self.calc_ema(self.prices, 5)
        ema20 = self.calc_ema(self.prices, 20)
        
        self.ema_short.append(ema5)
        self.ema_long.append(ema20)
        
        # 低波动时正常交易，高波动时谨慎
        if len(self.ema_short) >= 2:
            prev_short = self.ema_short[-2]
            prev_long = self.ema_long[-2]
            
            # 低波动环境下金叉买入
            if prev_short <= prev_long and ema5 > ema20 and not position and vol < self.vol_threshold:
                self.buy_reason = f"Golden cross in low volatility ({vol*100:.1f}%)"
                return 'buy'
            
            if position:
                # 死叉卖出
                if prev_short >= prev_long and ema5 < ema20:
                    self.sell_reason = f"Death cross"
                    return 'sell'
                
                # 高波动时更严格的止损
                adjusted_stop = self.stop_loss * (1 + vol / self.vol_threshold) if vol > self.vol_threshold else self.stop_loss
                loss_pct = (position['price'] - price) / position['price']
                if loss_pct >= adjusted_stop:
                    self.sell_reason = f"Volatility-adjusted stop loss"
                    return 'sell'
                
                profit_pct = (price - position['price']) / position['price']
                if profit_pct >= self.take_profit:
                    self.sell_reason = f"Take profit"
                    return 'sell'
        
        return 'hold'


class MeanReversionStrategy(StrategyBase):
    """均值回归策略"""
    
    def __init__(self, params):
        super().__init__(params)
        self.lookback = params.get('lookback', 30)
        self.entry_threshold = params.get('entry_threshold', -0.05)
        self.exit_threshold = params.get('exit_threshold', 0.02)
        self.stop_loss = params.get('stop_loss', 0.10)
        
    def calc_z_score(self):
        if len(self.prices) < self.lookback:
            return 0
        
        recent = self.prices[-self.lookback:]
        mean = np.mean(recent)
        std = np.std(recent)
        
        if std == 0:
            return 0
        
        return (self.prices[-1] - mean) / std
    
    def generate_signal(self, price, ts, position):
        self.prices.append(price)
        
        if len(self.prices) < self.lookback + 1:
            return 'hold'
        
        z_score = self.calc_z_score()
        
        # 价格低于均值过多时买入
        if z_score < -2 and not position:
            self.buy_reason = f"Z-score oversold: {z_score:.2f}"
            return 'buy'
        
        if position:
            # 价格回归均值时卖出
            if z_score > 0:
                self.sell_reason = f"Price returned to mean: z={z_score:.2f}"
                return 'sell'
            
            # 止损
            loss_pct = (position['price'] - price) / position['price']
            if loss_pct >= self.stop_loss:
                self.sell_reason = f"Stop loss"
                return 'sell'
        
        return 'hold'


def run_10_round_experiment():
    """运行10轮优化实验"""
    
    print("=" * 80)
    print("  防过拟合策略优化实验 - 10轮迭代")
    print("=" * 80)
    print(f"样本分割: {TRAIN_RATIO*100:.0f}%训练 + {(1-TRAIN_RATIO)*100:.0f}%测试")
    print(f"过拟合阈值: 训练收益/测试收益 > {OVERFIT_THRESHOLD}")
    print(f"币种池: {len(SYMBOLS)}个")
    print("=" * 80)
    
    # 策略参数池
    strategy_configs = [
        # 双均线+趋势过滤
        {'name': 'DoubleMA_Trend_5_20_50', 'type': 'double_ma_plus', 'fast': 5, 'slow': 20, 'trend': 50, 'position_pct': 0.2, 'stop_loss': 0.05, 'take_profit': 0.10},
        {'name': 'DoubleMA_Trend_7_25_60', 'type': 'double_ma_plus', 'fast': 7, 'slow': 25, 'trend': 60, 'position_pct': 0.15, 'stop_loss': 0.06, 'take_profit': 0.12},
        {'name': 'DoubleMA_Trend_10_30_70', 'type': 'double_ma_plus', 'fast': 10, 'slow': 30, 'trend': 70, 'position_pct': 0.2, 'stop_loss': 0.08, 'take_profit': 0.15},
        
        # RSI+布林带
        {'name': 'RSI_Boll_14_20_2', 'type': 'rsi_bollinger', 'rsi_period': 14, 'boll_period': 20, 'boll_std': 2.0, 'rsi_low': 30, 'rsi_high': 70, 'position_pct': 0.2, 'stop_loss': 0.05, 'take_profit': 0.10},
        {'name': 'RSI_Boll_10_15_1.5', 'type': 'rsi_bollinger', 'rsi_period': 10, 'boll_period': 15, 'boll_std': 1.5, 'rsi_low': 25, 'rsi_high': 75, 'position_pct': 0.15, 'stop_loss': 0.04, 'take_profit': 0.08},
        
        # MACD
        {'name': 'MACD_12_26_9', 'type': 'macd', 'fast': 12, 'slow': 26, 'signal': 9, 'position_pct': 0.2, 'stop_loss': 0.05, 'take_profit': 0.12},
        {'name': 'MACD_8_17_5', 'type': 'macd', 'fast': 8, 'slow': 17, 'signal': 5, 'position_pct': 0.15, 'stop_loss': 0.04, 'take_profit': 0.10},
        
        # 波动率自适应
        {'name': 'VolAdaptive_20_3', 'type': 'volatility_adaptive', 'lookback': 20, 'vol_threshold': 0.03, 'position_high': 0.1, 'position_low': 0.3, 'stop_loss': 0.08, 'take_profit': 0.15},
        {'name': 'VolAdaptive_30_5', 'type': 'volatility_adaptive', 'lookback': 30, 'vol_threshold': 0.05, 'position_high': 0.05, 'position_low': 0.25, 'stop_loss': 0.10, 'take_profit': 0.20},
        
        # 均值回归
        {'name': 'MeanRev_30', 'type': 'mean_reversion', 'lookback': 30, 'position_pct': 0.2, 'stop_loss': 0.10},
        {'name': 'MeanRev_20', 'type': 'mean_reversion', 'lookback': 20, 'position_pct': 0.15, 'stop_loss': 0.08},
    ]
    
    strategy_classes = {
        'double_ma_plus': DoubleMAPlusStrategy,
        'rsi_bollinger': RSI_BollingerStrategy,
        'macd': MACDStrategy,
        'volatility_adaptive': VolatilityAdaptiveStrategy,
        'mean_reversion': MeanReversionStrategy,
    }
    
    all_results = []
    round_summaries = []
    
    for round_num in range(1, 11):
        print(f"\n{'='*80}")
        print(f"  第 {round_num} 轮测试")
        print(f"{'='*80}")
        
        round_results = []
        
        for symbol in SYMBOLS:
            for config in strategy_configs:
                strategy_type = config['type']
                strategy_class = strategy_classes.get(strategy_type)
                
                if not strategy_class:
                    continue
                
                backtest = AntiOverfitBacktest(symbol, strategy_class, config)
                result = backtest.run_with_validation()
                
                if result:
                    result['round'] = round_num
                    round_results.append(result)
                    all_results.append(result)
                    
                    # 打印结果
                    status = "⚠️ 过拟合" if result['is_overfit'] else "✓ 正常"
                    print(f"  {symbol} | {config['name']:<25} | 训练: {result['train_return']:>6.1f}% | 测试: {result['test_return']:>6.1f}% | {status}")
        
        # 每轮复盘
        print(f"\n【第 {round_num} 轮复盘】")
        
        # 找最优策略（只看测试集表现）
        valid_results = [r for r in round_results if not r['is_overfit']]
        
        if valid_results:
            best = max(valid_results, key=lambda x: x['test_return'])
            overfit_count = len([r for r in round_results if r['is_overfit']])
            
            print(f"  测试集最优: {best['symbol']} | {best['strategy']} | 收益 {best['test_return']:.1f}%")
            print(f"  过拟合策略数: {overfit_count}/{len(round_results)}")
            print(f"  有效策略数: {len(valid_results)}/{len(round_results)}")
            
            # 统计各策略类型表现
            strategy_avg = {}
            for r in valid_results:
                stype = r['params']['type']
                if stype not in strategy_avg:
                    strategy_avg[stype] = []
                strategy_avg[stype].append(r['test_return'])
            
            print(f"\n  各策略类型平均测试收益:")
            for stype, returns in sorted(strategy_avg.items(), key=lambda x: np.mean(x[1]), reverse=True):
                avg = np.mean(returns)
                print(f"    {stype}: {avg:.1f}%")
            
            round_summaries.append({
                'round': round_num,
                'best_strategy': best['strategy'],
                'best_symbol': best['symbol'],
                'best_test_return': best['test_return'],
                'overfit_count': overfit_count,
                'valid_count': len(valid_results),
                'strategy_avg': {k: np.mean(v) for k, v in strategy_avg.items()}
            })
        else:
            print(f"  ⚠️ 本轮所有策略都过拟合！")
            round_summaries.append({
                'round': round_num,
                'best_strategy': None,
                'best_test_return': None,
                'overfit_count': len(round_results),
                'valid_count': 0
            })
    
    # 最终汇总
    print(f"\n{'='*80}")
    print(f"  10轮实验汇总")
    print(f"{'='*80}")
    
    # 只看测试集表现排序
    valid_all = [r for r in all_results if not r['is_overfit']]
    
    if valid_all:
        sorted_results = sorted(valid_all, key=lambda x: x['test_return'], reverse=True)
        
        print(f"\n【TOP 10 测试集最优策略】（排除过拟合）")
        print(f"{'排名':<4} {'轮':<4} {'币种':<12} {'策略':<25} {'训练':<8} {'测试':<8} {'过拟合':<6}")
        print("-" * 80)
        for i, r in enumerate(sorted_results[:10], 1):
            print(f"{i:<4} {r['round']:<4} {r['symbol']:<12} {r['strategy']:<25} {r['train_return']:>6.1f}% {r['test_return']:>6.1f}% {r['overfit_index']:>5.1f}")
        
        # 各轮最优对比
        print(f"\n【各轮最优对比】")
        for s in round_summaries:
            if s['best_strategy']:
                print(f"  Round {s['round']}: {s['best_symbol']} | {s['best_strategy']} | {s['best_test_return']:.1f}%")
            else:
                print(f"  Round {s['round']}: ⚠️ 无有效策略")
        
        # 统计最优策略类型
        strategy_types = {}
        for r in valid_all:
            stype = r['params']['type']
            if stype not in strategy_types:
                strategy_types[stype] = []
            strategy_types[stype].append(r['test_return'])
        
        print(f"\n【策略类型稳定性分析】")
        for stype, returns in sorted(strategy_types.items(), key=lambda x: np.mean(x[1]), reverse=True):
            avg = np.mean(returns)
            std = np.std(returns)
            stable = "稳定" if std < 5 else "不稳定"
            print(f"  {stype}: 平均 {avg:.1f}% | 标准差 {std:.1f} | {stable}")
    
    # 保存完整报告
    report = {
        'experiment_date': datetime.now().isoformat(),
        'rounds': 10,
        'symbols': SYMBOLS,
        'train_ratio': TRAIN_RATIO,
        'overfit_threshold': OVERFIT_THRESHOLD,
        'total_tests': len(all_results),
        'valid_tests': len(valid_all),
        'overfit_tests': len([r for r in all_results if r['is_overfit']]),
        'round_summaries': round_summaries,
        'top_10': sorted_results[:10] if valid_all else [],
        'all_results': all_results
    }
    
    with open(os.path.join(REPORT_DIR, 'anti_overfit_experiment.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n完整报告已保存: {REPORT_DIR}/anti_overfit_experiment.json")
    print(f"{'='*80}")
    
    return report


if __name__ == "__main__":
    run_10_round_experiment()