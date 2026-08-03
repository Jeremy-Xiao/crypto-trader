"""
防过拟合策略优化 - 10轮迭代实验
轻量版：无图表依赖，纯数据输出
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.okx_rest import OKXPublicAPI
from src.utils.indicators import EMA, RSI, BollingerBands

INITIAL_BALANCE = 1000
FEE_RATE = 0.001
REPORT_DIR = "backtest_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'XRP-USDT', 'BNB-USDT', 'DOGE-USDT', 'LINK-USDT']

TRAIN_RATIO = 0.7
OVERFIT_THRESHOLD = 3.0


def get_candles(symbol, bar='1D', limit=300):
    """获取K线数据"""
    api = OKXPublicAPI()
    result = api.get_candles(symbol, bar, limit)
    if result.get('code') != '0':
        return None
    closes = []
    for c in sorted(result['data'], key=lambda x: x[0]):
        closes.append(float(c[4]))
    return closes


# 统一使用 src/utils/indicators.py 的指标计算，消除重复代码
from src.utils.indicators import EMA as _ema_series, RSI as _rsi_series, BollingerBands as _bb_series


def calc_ema(prices, period):
    """计算EMA - 统一使用 src/utils/indicators.py"""
    if len(prices) < period:
        return 0
    return float(_ema_series(prices, period).iloc[-1])


def calc_rsi(prices, period=14):
    """计算RSI - 统一使用 src/utils/indicators.py"""
    if len(prices) < period + 1:
        return 50
    val = float(_rsi_series(prices, period).iloc[-1])
    return val if val == val else 50


def calc_bollinger(prices, period=20, std_dev=2.0):
    """计算布林带 - 统一使用 src/utils/indicators.py"""
    if len(prices) < period:
        return None
    bb = _bb_series(prices, period, std_dev)
    upper = float(bb["upper"].iloc[-1])
    middle = float(bb["middle"].iloc[-1])
    lower = float(bb["lower"].iloc[-1])
    if upper != upper:
        return None
    return upper, middle, lower


def calc_macd(prices, fast=12, slow=26, signal=9):
    """计算MACD"""
    if len(prices) < slow + signal:
        return None
    ema_fast = calc_ema(prices, fast)
    ema_slow = calc_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    return macd_line


def run_backtest(closes, strategy_func, params):
    """运行回测"""
    if not closes or len(closes) < 50:
        return None
    
    balance = INITIAL_BALANCE
    position = None
    trades = []
    equity = []
    
    for i, price in enumerate(closes):
        signal = strategy_func(closes[:i+1], position, params)
        
        if signal == 'buy' and not position:
            pct = params.get('position_pct', 0.2)
            amount = (balance * pct) / price
            cost = amount * price * (1 + FEE_RATE)
            if cost > balance:
                amount = balance / (price * (1 + FEE_RATE))
                cost = balance
            balance -= cost
            position = {'price': price, 'amount': amount, 'cost': cost}
            trades.append({'action': 'buy', 'price': price, 'amount': amount})
        
        elif signal == 'sell' and position:
            revenue = position['amount'] * price * (1 - FEE_RATE)
            pnl = revenue - position['cost']
            balance += revenue
            trades.append({'action': 'sell', 'price': price, 'amount': position['amount'], 'pnl': pnl})
            position = None
        
        eq = balance + (position['amount'] * price if position else 0)
        equity.append(eq)
    
    # 强制平仓
    if position:
        final_price = closes[-1]
        revenue = position['amount'] * final_price * (1 - FEE_RATE)
        pnl = revenue - position['cost']
        balance += revenue
        trades.append({'action': 'sell', 'price': final_price, 'amount': position['amount'], 'pnl': pnl})
        position = None
        equity.append(balance)
    
    if not equity:
        return None
    
    total_return = (equity[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    
    # 最大回撤
    peak = np.maximum.accumulate(equity)
    drawdown = (np.array(equity) - peak) / peak * 100
    max_drawdown = float(drawdown.min())
    
    # 胜率
    sell_trades = [t for t in trades if t['action'] == 'sell']
    wins = [t for t in sell_trades if t.get('pnl', 0) > 0]
    win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0
    
    # 夏普比率
    returns = np.diff(np.array(equity)) / np.array(equity)[:-1]
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(365)) if len(returns) > 1 and np.std(returns) > 0 else 0
    
    return {
        'return': total_return,
        'drawdown': max_drawdown,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'trades': len(sell_trades),
        'wins': len(wins),
        'final_equity': equity[-1]
    }


# ==================== 策略函数 ====================

def strategy_double_ma_trend(closes, position, params):
    """双均线+趋势过滤"""
    fast = params.get('fast', 10)
    slow = params.get('slow', 30)
    trend = params.get('trend', 60)
    stop = params.get('stop_loss', 0.06)
    profit = params.get('take_profit', 0.12)
    
    if len(closes) < trend + 2:
        return 'hold'
    
    fast_ema = calc_ema(closes, fast)
    slow_ema = calc_ema(closes, slow)
    trend_ema = calc_ema(closes, trend)
    price = closes[-1]
    
    # 趋势过滤
    in_uptrend = price > trend_ema
    
    # 需要历史EMA数据检测交叉
    if len(closes) >= trend + 3:
        prev_fast = calc_ema(closes[:-1], fast)
        prev_slow = calc_ema(closes[:-1], slow)
        
        if prev_fast <= prev_slow and fast_ema > slow_ema and not position and in_uptrend:
            return 'buy'
        
        if position and prev_fast >= prev_slow and fast_ema < slow_ema:
            return 'sell'
    
    if position:
        loss_pct = (position['price'] - price) / position['price']
        if loss_pct >= stop:
            return 'sell'
        profit_pct = (price - position['price']) / position['price']
        if profit_pct >= profit:
            return 'sell'
        if price < trend_ema:
            return 'sell'
    
    return 'hold'


def strategy_rsi_bollinger(closes, position, params):
    """RSI+布林带"""
    rsi_period = params.get('rsi_period', 14)
    boll_period = params.get('boll_period', 20)
    boll_std = params.get('boll_std', 2.0)
    rsi_low = params.get('rsi_low', 30)
    rsi_high = params.get('rsi_high', 70)
    stop = params.get('stop_loss', 0.05)
    profit = params.get('take_profit', 0.10)
    
    if len(closes) < max(rsi_period, boll_period) + 2:
        return 'hold'
    
    rsi = calc_rsi(closes, rsi_period)
    boll = calc_bollinger(closes, boll_period, boll_std)
    if not boll:
        return 'hold'
    upper, mid, lower = boll
    price = closes[-1]
    
    if rsi < rsi_low and price <= lower and not position:
        return 'buy'
    
    if position:
        if rsi > rsi_high and price >= upper:
            return 'sell'
        if price >= mid:
            return 'sell'
        loss_pct = (position['price'] - price) / position['price']
        if loss_pct >= stop:
            return 'sell'
        profit_pct = (price - position['price']) / position['price']
        if profit_pct >= profit:
            return 'sell'
    
    return 'hold'


def strategy_macd(closes, position, params):
    """MACD"""
    fast = params.get('fast', 12)
    slow = params.get('slow', 26)
    signal = params.get('signal', 9)
    stop = params.get('stop_loss', 0.05)
    profit = params.get('take_profit', 0.12)
    
    if len(closes) < slow + signal + 2:
        return 'hold'
    
    macd = calc_macd(closes, fast, slow, signal)
    prev_macd = calc_macd(closes[:-1], fast, slow, signal) if len(closes) > slow + signal + 1 else macd
    
    if macd is None or prev_macd is None:
        return 'hold'
    
    price = closes[-1]
    
    # MACD从负转正买入
    if prev_macd <= 0 and macd > 0 and not position:
        return 'buy'
    
    if position:
        # MACD从正转负卖出
        if prev_macd >= 0 and macd < 0:
            return 'sell'
        loss_pct = (position['price'] - price) / position['price']
        if loss_pct >= stop:
            return 'sell'
        profit_pct = (price - position['price']) / position['price']
        if profit_pct >= profit:
            return 'sell'
    
    return 'hold'


def strategy_vol_adaptive(closes, position, params):
    """波动率自适应"""
    lookback = params.get('lookback', 20)
    vol_threshold = params.get('vol_threshold', 0.03)
    stop = params.get('stop_loss', 0.08)
    profit = params.get('take_profit', 0.15)
    
    if len(closes) < lookback + 30:
        return 'hold'
    
    returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))][-lookback:]
    vol = np.std(returns)
    
    ema5 = calc_ema(closes, 5)
    ema20 = calc_ema(closes, 20)
    price = closes[-1]
    
    if len(closes) >= lookback + 31:
        prev_ema5 = calc_ema(closes[:-1], 5)
        prev_ema20 = calc_ema(closes[:-1], 20)
        
        if prev_ema5 <= prev_ema20 and ema5 > ema20 and not position and vol < vol_threshold:
            return 'buy'
        
        if position and prev_ema5 >= prev_ema20 and ema5 < ema20:
            return 'sell'
    
    if position:
        adjusted_stop = stop * (1 + vol / vol_threshold) if vol > vol_threshold else stop
        loss_pct = (position['price'] - price) / position['price']
        if loss_pct >= adjusted_stop:
            return 'sell'
        profit_pct = (price - position['price']) / position['price']
        if profit_pct >= profit:
            return 'sell'
    
    return 'hold'


def strategy_mean_reversion(closes, position, params):
    """均值回归"""
    lookback = params.get('lookback', 30)
    stop = params.get('stop_loss', 0.10)
    
    if len(closes) < lookback + 1:
        return 'hold'
    
    recent = closes[-lookback:]
    mean = np.mean(recent)
    std = np.std(recent)
    price = closes[-1]
    
    if std == 0:
        return 'hold'
    
    z_score = (price - mean) / std
    
    if z_score < -2 and not position:
        return 'buy'
    
    if position:
        if z_score > 0:
            return 'sell'
        loss_pct = (position['price'] - price) / position['price']
        if loss_pct >= stop:
            return 'sell'
    
    return 'hold'


def strategy_adaptive_dca(closes, position, params):
    """自适应定投策略"""
    lookback = params.get('lookback', 50)
    entry_pct = params.get('entry_pct', 0.02)
    stop = params.get('stop_loss', 0.15)
    profit = params.get('take_profit', 0.20)
    
    if len(closes) < lookback + 1:
        return 'hold'
    
    recent = closes[-lookback:]
    mean = np.mean(recent)
    price = closes[-1]
    
    # 价格低于均值一定比例时买入
    if price < mean * (1 - entry_pct) and not position:
        return 'buy'
    
    if position:
        # 价格回到均值以上卖出
        if price > mean * (1 + entry_pct * 0.5):
            return 'sell'
        loss_pct = (position['price'] - price) / position['price']
        if loss_pct >= stop:
            return 'sell'
        profit_pct = (price - position['price']) / position['price']
        if profit_pct >= profit:
            return 'sell'
    
    return 'hold'


# ==================== 策略参数池 ====================

STRATEGY_POOL = [
    {'name': 'DoubleMA_Trend_5_20_50', 'func': strategy_double_ma_trend, 'fast': 5, 'slow': 20, 'trend': 50, 'position_pct': 0.2, 'stop_loss': 0.05, 'take_profit': 0.10},
    {'name': 'DoubleMA_Trend_7_25_60', 'func': strategy_double_ma_trend, 'fast': 7, 'slow': 25, 'trend': 60, 'position_pct': 0.15, 'stop_loss': 0.06, 'take_profit': 0.12},
    {'name': 'DoubleMA_Trend_10_30_70', 'func': strategy_double_ma_trend, 'fast': 10, 'slow': 30, 'trend': 70, 'position_pct': 0.2, 'stop_loss': 0.08, 'take_profit': 0.15},
    {'name': 'DoubleMA_Trend_15_40_80', 'func': strategy_double_ma_trend, 'fast': 15, 'slow': 40, 'trend': 80, 'position_pct': 0.25, 'stop_loss': 0.10, 'take_profit': 0.20},
    {'name': 'RSI_Boll_14_20_2', 'func': strategy_rsi_bollinger, 'rsi_period': 14, 'boll_period': 20, 'boll_std': 2.0, 'rsi_low': 30, 'rsi_high': 70, 'position_pct': 0.2, 'stop_loss': 0.05, 'take_profit': 0.10},
    {'name': 'RSI_Boll_10_15_1.5', 'func': strategy_rsi_bollinger, 'rsi_period': 10, 'boll_period': 15, 'boll_std': 1.5, 'rsi_low': 25, 'rsi_high': 75, 'position_pct': 0.15, 'stop_loss': 0.04, 'take_profit': 0.08},
    {'name': 'RSI_Boll_7_10_1', 'func': strategy_rsi_bollinger, 'rsi_period': 7, 'boll_period': 10, 'boll_std': 1.0, 'rsi_low': 20, 'rsi_high': 80, 'position_pct': 0.1, 'stop_loss': 0.03, 'take_profit': 0.06},
    {'name': 'MACD_12_26_9', 'func': strategy_macd, 'fast': 12, 'slow': 26, 'signal': 9, 'position_pct': 0.2, 'stop_loss': 0.05, 'take_profit': 0.12},
    {'name': 'MACD_8_17_5', 'func': strategy_macd, 'fast': 8, 'slow': 17, 'signal': 5, 'position_pct': 0.15, 'stop_loss': 0.04, 'take_profit': 0.10},
    {'name': 'MACD_6_13_4', 'func': strategy_macd, 'fast': 6, 'slow': 13, 'signal': 4, 'position_pct': 0.1, 'stop_loss': 0.03, 'take_profit': 0.08},
    {'name': 'VolAdapt_20_3', 'func': strategy_vol_adaptive, 'lookback': 20, 'vol_threshold': 0.03, 'position_pct': 0.2, 'stop_loss': 0.08, 'take_profit': 0.15},
    {'name': 'VolAdapt_30_5', 'func': strategy_vol_adaptive, 'lookback': 30, 'vol_threshold': 0.05, 'position_pct': 0.15, 'stop_loss': 0.10, 'take_profit': 0.20},
    {'name': 'MeanRev_20', 'func': strategy_mean_reversion, 'lookback': 20, 'position_pct': 0.15, 'stop_loss': 0.08},
    {'name': 'MeanRev_30', 'func': strategy_mean_reversion, 'lookback': 30, 'position_pct': 0.2, 'stop_loss': 0.10},
    {'name': 'MeanRev_50', 'func': strategy_mean_reversion, 'lookback': 50, 'position_pct': 0.25, 'stop_loss': 0.12},
    {'name': 'DCA_2pct', 'func': strategy_adaptive_dca, 'lookback': 50, 'entry_pct': 0.02, 'position_pct': 0.2, 'stop_loss': 0.15, 'take_profit': 0.20},
    {'name': 'DCA_3pct', 'func': strategy_adaptive_dca, 'lookback': 50, 'entry_pct': 0.03, 'position_pct': 0.15, 'stop_loss': 0.12, 'take_profit': 0.15},
    {'name': 'DCA_5pct', 'func': strategy_adaptive_dca, 'lookback': 50, 'entry_pct': 0.05, 'position_pct': 0.1, 'stop_loss': 0.10, 'take_profit': 0.10},
]


def run_with_validation(closes, strategy_func, params):
    """运行带样本外验证的回测"""
    if not closes or len(closes) < 60:
        return None
    
    train_end = int(len(closes) * TRAIN_RATIO)
    train_closes = closes[:train_end]
    test_closes = closes[train_end:]
    
    train_result = run_backtest(train_closes, strategy_func, params)
    test_result = run_backtest(test_closes, strategy_func, params)
    
    if not train_result or not test_result:
        return None
    
    train_ret = train_result['return']
    test_ret = test_result['return']
    
    # 过拟合指数
    if abs(test_ret) > 0.1:
        overfit_idx = abs(train_ret) / abs(test_ret)
    else:
        overfit_idx = 10.0 if train_ret > 0 else 0
    
    is_overfit = overfit_idx > OVERFIT_THRESHOLD and train_ret > test_ret
    
    return {
        'train_return': train_ret,
        'test_return': test_ret,
        'train_drawdown': train_result['drawdown'],
        'test_drawdown': test_result['drawdown'],
        'train_sharpe': train_result['sharpe'],
        'test_sharpe': test_result['sharpe'],
        'train_trades': train_result['trades'],
        'test_trades': test_result['trades'],
        'train_win_rate': train_result['win_rate'],
        'test_win_rate': test_result['win_rate'],
        'overfit_index': round(overfit_idx, 2),
        'is_overfit': is_overfit,
        'combined_return': round(train_ret * TRAIN_RATIO + test_ret * (1 - TRAIN_RATIO), 2),
        'data_points': len(closes),
        'train_points': len(train_closes),
        'test_points': len(test_closes),
    }


def run_10_rounds():
    """运行10轮实验"""
    print("=" * 90)
    print("  防过拟合策略优化实验 - 10轮迭代")
    print("=" * 90)
    print(f"样本分割: {TRAIN_RATIO*100:.0f}%训练 + {(1-TRAIN_RATIO)*100:.0f}%测试")
    print(f"过拟合阈值: 训练收益/测试收益 > {OVERFIT_THRESHOLD}")
    print(f"币种: {len(SYMBOLS)} | 策略: {len(STRATEGY_POOL)} | 总测试: {len(SYMBOLS)*len(STRATEGY_POOL)*10}")
    print("=" * 90)
    
    all_results = []
    round_summaries = []
    
    for round_num in range(1, 11):
        print(f"\n{'='*90}")
        print(f"  Round {round_num}/10")
        print(f"{'='*90}")
        
        round_results = []
        
        for symbol in SYMBOLS:
            closes = get_candles(symbol)
            if not closes or len(closes) < 60:
                print(f"  {symbol}: 数据不足，跳过")
                continue
            
            for config in STRATEGY_POOL:
                result = run_with_validation(closes, config['func'], config)
                if result:
                    result['symbol'] = symbol
                    result['strategy'] = config['name']
                    result['round'] = round_num
                    result['params'] = {k: v for k, v in config.items() if k not in ['name', 'func']}
                    round_results.append(result)
                    all_results.append(result)
        
        # 复盘
        print(f"\n  Round {round_num} 复盘:")
        valid = [r for r in round_results if not r['is_overfit']]
        overfit = [r for r in round_results if r['is_overfit']]
        
        if valid:
            best = max(valid, key=lambda x: x['test_return'])
            avg_test = np.mean([r['test_return'] for r in valid])
            
            print(f"    有效策略: {len(valid)}/{len(round_results)}")
            print(f"    过拟合策略: {len(overfit)}/{len(round_results)}")
            print(f"    最优: {best['symbol']} | {best['strategy']} | 测试收益 {best['test_return']:.1f}% | 夏普 {best['test_sharpe']:.2f}")
            print(f"    平均测试收益: {avg_test:.1f}%")
            
            # 按策略类型统计
            type_stats = {}
            for r in valid:
                # 提取策略类型
                stype = r['strategy'].split('_')[0]
                if stype not in type_stats:
                    type_stats[stype] = []
                type_stats[stype].append(r['test_return'])
            
            print(f"    各类型平均测试收益:")
            for stype, returns in sorted(type_stats.items(), key=lambda x: np.mean(x[1]), reverse=True):
                print(f"      {stype}: {np.mean(returns):.1f}% ({len(returns)}个)")
            
            round_summaries.append({
                'round': round_num,
                'total': len(round_results),
                'valid': len(valid),
                'overfit': len(overfit),
                'best_strategy': best['strategy'],
                'best_symbol': best['symbol'],
                'best_test_return': best['test_return'],
                'best_sharpe': best['test_sharpe'],
                'avg_test_return': round(avg_test, 2),
                'type_stats': {k: round(np.mean(v), 2) for k, v in type_stats.items()}
            })
        else:
            print(f"    ⚠️ 全部过拟合！")
            round_summaries.append({
                'round': round_num,
                'total': len(round_results),
                'valid': 0,
                'overfit': len(round_results),
            })
    
    # 最终汇总
    print(f"\n{'='*90}")
    print(f"  10轮实验总汇")
    print(f"{'='*90}")
    
    valid_all = [r for r in all_results if not r['is_overfit']]
    
    if valid_all:
        sorted_results = sorted(valid_all, key=lambda x: x['test_return'], reverse=True)
        
        print(f"\n  总测试: {len(all_results)} | 有效: {len(valid_all)} | 过拟合: {len(all_results)-len(valid_all)}")
        
        print(f"\n  【TOP 15 测试集最优策略】")
        print(f"  {'排名':<4} {'轮':<4} {'币种':<12} {'策略':<22} {'训练':<8} {'测试':<8} {'夏普':<6} {'过拟合':<6}")
        print(f"  {'-'*80}")
        for i, r in enumerate(sorted_results[:15], 1):
            print(f"  {i:<4} {r['round']:<4} {r['symbol']:<12} {r['strategy']:<22} {r['train_return']:>6.1f}% {r['test_return']:>6.1f}% {r['test_sharpe']:>5.2f} {r['overfit_index']:>5.1f}")
        
        # 各轮最优
        print(f"\n  【各轮最优】")
        for s in round_summaries:
            if 'best_strategy' in s:
                print(f"  Round {s['round']}: {s['best_symbol']} | {s['best_strategy']} | 测试 {s['best_test_return']:.1f}% | 夏普 {s['best_sharpe']:.2f}")
        
        # 策略类型稳定性
        print(f"\n  【策略类型稳定性】")
        type_all = {}
        for r in valid_all:
            stype = r['strategy'].split('_')[0]
            if stype not in type_all:
                type_all[stype] = []
            type_all[stype].append(r['test_return'])
        
        for stype, returns in sorted(type_all.items(), key=lambda x: np.mean(x[1]), reverse=True):
            avg = np.mean(returns)
            std = np.std(returns)
            stable = "✅ 稳定" if std < 3 else "⚠️ 波动大"
            print(f"  {stype:<15} 平均 {avg:>5.1f}% | 标准差 {std:>4.1f} | {stable}")
        
        # 各币种最优
        print(f"\n  【各币种最优策略】")
        for symbol in SYMBOLS:
            symbol_results = [r for r in sorted_results if r['symbol'] == symbol]
            if symbol_results:
                best = symbol_results[0]
                print(f"  {symbol:<12} {best['strategy']:<22} 测试 {best['test_return']:>5.1f}% | 夏普 {best['test_sharpe']:>5.2f}")
    
    # 保存报告
    report = {
        'date': datetime.now().isoformat(),
        'rounds': 10,
        'symbols': SYMBOLS,
        'strategies': [s['name'] for s in STRATEGY_POOL],
        'train_ratio': TRAIN_RATIO,
        'overfit_threshold': OVERFIT_THRESHOLD,
        'total_tests': len(all_results),
        'valid_tests': len(valid_all),
        'overfit_tests': len(all_results) - len(valid_all),
        'round_summaries': round_summaries,
        'top_15': sorted_results[:15] if valid_all else [],
        'all_results': all_results
    }
    
    with open(os.path.join(REPORT_DIR, 'anti_overfit_10rounds.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n  报告已保存: {REPORT_DIR}/anti_overfit_10rounds.json")
    print(f"{'='*90}")
    
    return report


if __name__ == "__main__":
    run_10_rounds()
