"""
滚动窗口10轮实验
每轮使用不同的时间段，模拟真实的前向测试
"""

import os
import sys
import json
import itertools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.okx_rest import OKXPublicAPI
from src.utils.indicators import EMA, RSI, BollingerBands, ADX, ATR

INITIAL_BALANCE = 1000
FEE_RATE = 0.001  # 0.1% 手续费
REPORT_DIR = "backtest_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'XRP-USDT', 'BNB-USDT', 'LINK-USDT']

NUM_ROUNDS = 10
TRAIN_RATIO = 0.7
OVERFIT_THRESHOLD = 3.0

# 组合层面风控参数
MAX_POSITIONS = 3           # 最大同时持仓数
CORR_THRESHOLD = 0.9        # 相关性阈值，超过此值的两个币种不重复持仓

# 手续费感知参数
ROUND_TRIP_FEE = 2 * FEE_RATE   # 往返手续费 0.2%（买入+卖出）
MIN_MOVE_MULTIPLIER = 2.0       # 预期波动(ATR/price)至少为往返手续费的N倍才值得交易
MIN_TP_FEE_RATIO = 5.0          # 止盈目标至少为往返手续费的N倍（即至少1%）
MIN_SIGNAL_STRENGTH = 0.1       # 最低信号强度门槛，低于此值不值得支付手续费


def get_candles(symbol, bar='1D', limit=500):
    """获取K线数据"""
    api = OKXPublicAPI()
    result = api.get_candles(symbol, bar, limit)
    if result.get('code') != '0':
        return None
    closes = []
    highs = []
    lows = []
    timestamps = []
    for c in sorted(result['data'], key=lambda x: x[0]):
        timestamps.append(int(c[0]))
        lows.append(float(c[3]))
        closes.append(float(c[4]))
        highs.append(float(c[2]))
    return closes, timestamps, highs, lows


def get_candles_4h(symbol, limit=1500):
    """获取4H K线数据（用于多时间框架确认）"""
    api = OKXPublicAPI()
    result = api.get_candles(symbol, '4H', limit)
    if result.get('code') != '0':
        return None
    closes = []
    timestamps = []
    for c in sorted(result['data'], key=lambda x: x[0]):
        timestamps.append(int(c[0]))
        closes.append(float(c[4]))
    return closes, timestamps


def align_4h_trend_to_daily(daily_ts, tf_4h_closes, tf_4h_ts, fast=20, slow=50, require_bars=60):
    """
    将 4H K 线趋势映射到日 K 线的每个时间点。
    
    对每个日 K 索引 i，找出当日之前所有 4H K 线，计算 EMA(fast) vs EMA(slow)，
    返回 True（4H多头）或 False（4H空头）。
    
    Returns:
        list[bool]: 与 daily_ts 等长的趋势确认数组，True 表示 4H 处于上升趋势
    """
    if not tf_4h_closes or len(tf_4h_closes) < max(fast, slow) + 5:
        # 4H 数据不足，默认全部通过（不拦截信号）
        return [True] * len(daily_ts)
    
    confirm = []
    tf_idx = 0
    
    for i, day_ts in enumerate(daily_ts):
        # 找到所有时间戳 ≤ 当日时间戳的4H K线
        while tf_idx < len(tf_4h_ts) and tf_4h_ts[tf_idx] <= day_ts:
            tf_idx += 1
        
        # 使用截至当日的4H数据计算趋势
        available = tf_4h_closes[:tf_idx]
        if len(available) < max(fast, slow) + 1:
            confirm.append(True)  # 数据不足时放过，不误杀信号
            continue
        
        # 数据充足的判断：至少 require_bars 根 4H K 线形成有效趋势判断
        if len(available) < require_bars:
            confirm.append(True)
            continue
        
        ema_fast = EMA(pd.Series(available), fast)
        ema_slow = EMA(pd.Series(available), slow)
        
        if pd.isna(ema_fast.iloc[-1]) or pd.isna(ema_slow.iloc[-1]):
            confirm.append(True)
        else:
            confirm.append(bool(ema_fast.iloc[-1] > ema_slow.iloc[-1]))
    
    return confirm


# 统一使用 src/utils/indicators.py 的指标计算，消除重复代码
# calc_ema / calc_rsi / calc_bollinger 是对 src/utils 返回 pd.Series 的封装，取最后一个值
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
    return val if val == val else 50  # NaN check


def calc_bollinger(prices, period=20, std_dev=2.0):
    """计算布林带 - 统一使用 src/utils/indicators.py"""
    if len(prices) < period:
        return None
    bb = _bb_series(prices, period, std_dev)
    upper = float(bb["upper"].iloc[-1])
    middle = float(bb["middle"].iloc[-1])
    lower = float(bb["lower"].iloc[-1])
    if upper != upper:  # NaN check
        return None
    return upper, middle, lower


def detect_regime(closes, highs, lows, adx_period=14, lookback=50):
    """
    检测市场状态：trending（趋势）/ ranging（震荡）

    综合两个指标：
    1. ADX > 25 → 趋势市；ADX < 20 → 震荡市；20-25 → 中性
    2. 波动率分位数：近期波动率在历史中的位置，辅助判断

    Returns:
        ('trending' | 'ranging', {'adx': float, 'vol_pct': float})
    """
    n = len(closes)
    if n < adx_period + 5 or not highs or not lows:
        return 'ranging', {'adx': 0, 'vol_pct': 0.5}

    # ADX
    adx_series = ADX(highs, lows, closes, adx_period)
    adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0

    # 波动率分位数
    returns = pd.Series(closes).pct_change().dropna()
    recent_vol = returns.iloc[-20:].std() if len(returns) >= 20 else returns.std()
    if len(returns) >= lookback:
        hist_vols = [returns.iloc[i:i+20].std() for i in range(len(returns) - 20) if not pd.isna(returns.iloc[i:i+20].std())]
        if hist_vols:
            vol_pct = sum(1 for v in hist_vols if v <= recent_vol) / len(hist_vols)
        else:
            vol_pct = 0.5
    else:
        vol_pct = 0.5

    # 综合判断
    if adx_val >= 25:
        regime = 'trending'
    elif adx_val <= 20:
        regime = 'ranging'
    else:
        # ADX 20-25 中性区间，用波动率辅助
        regime = 'trending' if vol_pct > 0.6 else 'ranging'

    return regime, {'adx': round(adx_val, 2), 'vol_pct': round(vol_pct, 3)}


def run_backtest(closes, strategy_func, params, highs=None, lows=None, tf_confirm=None):
    """
    ATR 动态仓位管理 + 多时间框架确认 + 动态止盈止损 + 移动止损：
    - risk_pct: 每笔交易风险占账户的百分比（如 0.02 = 2%）
    - atr_multiplier: ATR 倍数作为仓位计算中的止损距离
    - atr_sl_multiplier: ATR 倍数作为实际止损距离（默认 2.0）
    - atr_tp_multiplier: ATR 倍数作为止盈距离（默认 4.0，盈亏比 2:1）
    - use_trailing: 启用移动止损（默认 True）
    - trailing_pct: 盈利达到此百分比后激活移动止损（默认 0.02）
    - 仓位 = (balance × risk_pct) / (ATR × atr_multiplier)
    - tf_confirm: 多时间框架确认数组，与 closes 等长，True=4H多头确认买入
    
    波动大 → ATR大 → 止损宽、仓位小；波动小 → 反之。
    移动止损：盈利后止损线自动上移，锁定浮盈。
    买入信号需要 4H EMA(20) > EMA(50) 确认（趋势过滤假信号）。
    """
    if not closes or len(closes) < 50:
        return None
    
    balance = INITIAL_BALANCE
    position = None
    trades = []
    equity = []
    
    risk_pct = params.get('risk_pct', 0.02)
    atr_multiplier = params.get('atr_multiplier', 2.0)
    atr_sl_multiplier = params.get('atr_sl_multiplier', 2.0)
    atr_tp_multiplier = params.get('atr_tp_multiplier', 4.0)
    use_trailing = params.get('use_trailing', True)
    trailing_pct = params.get('trailing_pct', 0.02)
    atr_period = 14
    has_ohlc = highs is not None and lows is not None and len(highs) == len(closes) and len(lows) == len(closes)
    has_mtf = tf_confirm is not None and len(tf_confirm) == len(closes)
    
    mtf_blocked = 0
    stop_stats = {'stop_loss': 0, 'take_profit': 0, 'trailing_stop': 0, 'signal_sell': 0}
    
    for i, price in enumerate(closes):
        signal = strategy_func(closes[:i+1], position, params)
        
        # ====== 引擎层动态止盈止损 + 移动止损（持仓时优先检查） ======
        if position:
            should_exit = False
            exit_reason = ''
            
            if has_ohlc and i >= atr_period:
                atr_val = float(ATR(
                    pd.Series(highs[:i+1]),
                    pd.Series(lows[:i+1]),
                    pd.Series(closes[:i+1]),
                    atr_period
                ).iloc[-1])
                
                if not pd.isna(atr_val) and atr_val > 0:
                    # 计算 ATR 动态止损和止盈价格
                    stop_price = position['entry_price'] - atr_val * atr_sl_multiplier
                    tp_price = position['entry_price'] + atr_val * atr_tp_multiplier
                    
                    # 移动止损：追踪最高价，止损线跟随上移
                    if use_trailing:
                        position['highest_price'] = max(position.get('highest_price', price), price)
                        
                        # 盈利达到 trailing_pct 后激活移动止损
                        if not position.get('trailing_active'):
                            if price >= position['entry_price'] * (1 + trailing_pct):
                                position['trailing_active'] = True
                                position['trailing_stop'] = position['highest_price'] - atr_val * atr_sl_multiplier
                        
                        # 已激活，持续更新移动止损线
                        if position.get('trailing_active'):
                            new_trail = position['highest_price'] - atr_val * atr_sl_multiplier
                            position['trailing_stop'] = max(position.get('trailing_stop', stop_price), new_trail)
                    
                    # 检查是否触发退出
                    effective_stop = position.get('trailing_stop', stop_price) if position.get('trailing_active') else stop_price
                    
                    if price <= effective_stop:
                        should_exit = True
                        exit_reason = 'trailing_stop' if position.get('trailing_active') else 'stop_loss'
                    elif price >= tp_price:
                        should_exit = True
                        exit_reason = 'take_profit'
            
            if should_exit:
                revenue = position['amount'] * price * (1 - FEE_RATE)
                pnl = revenue - position['cost']
                balance += revenue
                trades.append({'action': 'sell', 'price': price, 'pnl': pnl, 'reason': exit_reason})
                stop_stats[exit_reason] += 1
                position = None
                eq = balance
                equity.append(eq)
                continue
            
            # 策略层卖出信号（引擎未触发退出时）
            if signal == 'sell':
                revenue = position['amount'] * price * (1 - FEE_RATE)
                pnl = revenue - position['cost']
                balance += revenue
                trades.append({'action': 'sell', 'price': price, 'pnl': pnl, 'reason': 'signal_sell'})
                stop_stats['signal_sell'] += 1
                position = None
                eq = balance
                equity.append(eq)
                continue
        
        # ====== 买入逻辑 ======
        if signal == 'buy' and not position:
            if has_mtf and not tf_confirm[i]:
                mtf_blocked += 1
                eq = balance
                equity.append(eq)
                continue
            if has_ohlc and i >= atr_period:
                atr_val = float(ATR(
                    pd.Series(highs[:i+1]),
                    pd.Series(lows[:i+1]),
                    pd.Series(closes[:i+1]),
                    atr_period
                ).iloc[-1])
                if pd.isna(atr_val) or atr_val <= 0:
                    pct = params.get('position_pct', 0.2)
                    amount = (balance * pct) / price
                else:
                    risk_amount = balance * risk_pct
                    stop_distance = atr_val * atr_multiplier
                    amount = risk_amount / stop_distance
                    max_amount = balance * 0.5 / price
                    amount = min(amount, max_amount)
            else:
                pct = params.get('position_pct', 0.2)
                amount = (balance * pct) / price
            
            cost = amount * price * (1 + FEE_RATE)
            if cost > balance:
                amount = balance / (price * (1 + FEE_RATE))
                cost = balance
            balance -= cost
            position = {
                'price': price,
                'amount': amount,
                'cost': cost,
                'entry_price': price,
                'highest_price': price,
                'trailing_active': False,
                'trailing_stop': 0,
            }
            trades.append({'action': 'buy', 'price': price, 'amount': amount})
        
        eq = balance + (position['amount'] * price if position else 0)
        equity.append(eq)
    
    # 强制平仓
    if position:
        final_price = closes[-1]
        revenue = position['amount'] * final_price * (1 - FEE_RATE)
        pnl = revenue - position['cost']
        balance += revenue
        equity.append(balance)
        trades.append({'action': 'sell', 'price': final_price, 'pnl': pnl, 'reason': 'force_close'})
        position = None
    
    if not equity:
        return None
    
    total_return = (equity[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    peak = np.maximum.accumulate(equity)
    drawdown = (np.array(equity) - peak) / peak * 100
    max_drawdown = float(drawdown.min())
    
    sell_trades = [t for t in trades if t['action'] == 'sell']
    wins = [t for t in sell_trades if t.get('pnl', 0) > 0]
    win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0
    
    returns = np.diff(np.array(equity)) / np.array(equity)[:-1]
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(365)) if len(returns) > 1 and np.std(returns) > 0 else 0
    
    return {
        'return': total_return,
        'drawdown': max_drawdown,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'trades': len(sell_trades),
        'wins': len(wins),
        'mtf_blocked': mtf_blocked,
        'stop_stats': stop_stats,
    }


# ==================== 策略 ====================

def strategy_double_ma_trend(closes, position, params):
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
    
    in_uptrend = price > trend_ema
    
    if len(closes) >= trend + 3:
        prev_fast = calc_ema(closes[:-1], fast)
        prev_slow = calc_ema(closes[:-1], slow)
        
        if prev_fast <= prev_slow and fast_ema > slow_ema and not position and in_uptrend:
            return 'buy'
        
        if position and prev_fast >= prev_slow and fast_ema < slow_ema:
            return 'sell'
    
    if position:
        # 止损止盈由引擎 ATR 动态管理，策略层只负责趋势判断
        if price < trend_ema:
            return 'sell'
    
    return 'hold'


def strategy_rsi_bollinger(closes, position, params):
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
        # 止损止盈由引擎 ATR 动态管理，策略层只负责震荡信号
        if rsi > rsi_high and price >= upper:
            return 'sell'
        if price >= mid:
            return 'sell'
    
    return 'hold'


def strategy_macd(closes, position, params):
    fast = params.get('fast', 12)
    slow = params.get('slow', 26)
    signal = params.get('signal', 9)
    stop = params.get('stop_loss', 0.05)
    profit = params.get('take_profit', 0.12)
    
    if len(closes) < slow + signal + 2:
        return 'hold'
    
    macd = calc_ema(closes, fast) - calc_ema(closes, slow)
    prev_macd = calc_ema(closes[:-1], fast) - calc_ema(closes[:-1], slow) if len(closes) > slow + signal + 1 else macd
    
    price = closes[-1]
    
    if prev_macd <= 0 and macd > 0 and not position:
        return 'buy'
    
    if position:
        # 止损止盈由引擎 ATR 动态管理，策略层只负责 MACD 死叉
        if prev_macd >= 0 and macd < 0:
            return 'sell'
    
    return 'hold'


def strategy_vol_adaptive(closes, position, params):
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
        # 止损止盈由引擎 ATR 动态管理，策略层只负责 EMA 死叉
        if prev_ema5 >= prev_ema20 and ema5 < ema20:
            return 'sell'
    
    return 'hold'


def strategy_adaptive_dca(closes, position, params):
    lookback = params.get('lookback', 50)
    entry_pct = params.get('entry_pct', 0.02)
    stop = params.get('stop_loss', 0.15)
    profit = params.get('take_profit', 0.20)
    
    if len(closes) < lookback + 1:
        return 'hold'
    
    recent = closes[-lookback:]
    mean = np.mean(recent)
    price = closes[-1]
    
    if price < mean * (1 - entry_pct) and not position:
        return 'buy'
    
    if position:
        # 止损止盈由引擎 ATR 动态管理，策略层只负责均值回归退出
        if price > mean * (1 + entry_pct * 0.5):
            return 'sell'
    
    return 'hold'


def strategy_ema_pullback(closes, position, params):
    """EMA回踩策略 - 价格回踩EMA后反弹买入"""
    ema_period = params.get('ema_period', 20)
    pullback_pct = params.get('pullback_pct', 0.03)
    stop = params.get('stop_loss', 0.05)
    profit = params.get('take_profit', 0.10)
    
    if len(closes) < ema_period + 5:
        return 'hold'
    
    ema = calc_ema(closes, ema_period)
    price = closes[-1]
    
    # 价格在EMA下方一定比例时视为回踩
    if price < ema * (1 + pullback_pct) and price > ema * (1 - pullback_pct) and not position:
        # 需要前一根K线在EMA下方
        if len(closes) >= ema_period + 6:
            prev_price = closes[-2]
            prev_ema = calc_ema(closes[:-1], ema_period)
            if prev_price < prev_ema:
                return 'buy'
    
    if position:
        # 止损止盈由引擎 ATR 动态管理，策略层只负责跌破 EMA 退出
        if price < ema:
            return 'sell'
    
    return 'hold'


def strategy_breakout(closes, position, params):
    """突破策略 - 价格突破N日高点买入"""
    lookback = params.get('lookback', 20)
    stop = params.get('stop_loss', 0.05)
    profit = params.get('take_profit', 0.15)
    
    if len(closes) < lookback + 1:
        return 'hold'
    
    recent = closes[-lookback-1:-1]
    high = max(recent)
    price = closes[-1]
    
    if price > high and not position:
        return 'buy'
    
    if position:
        # 止损止盈由引擎 ATR 动态管理，突破策略无额外退出条件
        pass
    
    return 'hold'


# ==================== 策略池 ====================

STRATEGY_POOL = [
    # --- 趋势策略：趋势市表现好，震荡市容易反复打止损 ---
    # 趋势策略止损较宽（atr_sl=2.0~3.0），止盈=2x止损（盈亏比2:1），防止被趋势中途回调震出
    {'name': 'DoubleMA_5_20_50', 'func': strategy_double_ma_trend, 'regime': 'trending', 'fast': 5, 'slow': 20, 'trend': 50, 'position_pct': 0.2, 'risk_pct': 0.02, 'atr_multiplier': 3.0, 'atr_sl_multiplier': 3.0, 'atr_tp_multiplier': 6.0, 'use_trailing': True, 'trailing_pct': 0.02},
    {'name': 'DoubleMA_7_25_60', 'func': strategy_double_ma_trend, 'regime': 'trending', 'fast': 7, 'slow': 25, 'trend': 60, 'position_pct': 0.15, 'risk_pct': 0.015, 'atr_multiplier': 3.0, 'atr_sl_multiplier': 3.0, 'atr_tp_multiplier': 6.0, 'use_trailing': True, 'trailing_pct': 0.02},
    {'name': 'DoubleMA_10_30_70', 'func': strategy_double_ma_trend, 'regime': 'trending', 'fast': 10, 'slow': 30, 'trend': 70, 'position_pct': 0.2, 'risk_pct': 0.02, 'atr_multiplier': 2.5, 'atr_sl_multiplier': 2.5, 'atr_tp_multiplier': 5.0, 'use_trailing': True, 'trailing_pct': 0.02},
    {'name': 'DoubleMA_15_40_80', 'func': strategy_double_ma_trend, 'regime': 'trending', 'fast': 15, 'slow': 40, 'trend': 80, 'position_pct': 0.25, 'risk_pct': 0.025, 'atr_multiplier': 2.5, 'atr_sl_multiplier': 2.5, 'atr_tp_multiplier': 5.0, 'use_trailing': True, 'trailing_pct': 0.02},
    {'name': 'MACD_12_26_9', 'func': strategy_macd, 'regime': 'trending', 'fast': 12, 'slow': 26, 'signal': 9, 'position_pct': 0.2, 'risk_pct': 0.02, 'atr_multiplier': 2.5, 'atr_sl_multiplier': 2.5, 'atr_tp_multiplier': 5.0, 'use_trailing': True, 'trailing_pct': 0.02},
    {'name': 'MACD_8_17_5', 'func': strategy_macd, 'regime': 'trending', 'fast': 8, 'slow': 17, 'signal': 5, 'position_pct': 0.15, 'risk_pct': 0.015, 'atr_multiplier': 2.5, 'atr_sl_multiplier': 2.5, 'atr_tp_multiplier': 5.0, 'use_trailing': True, 'trailing_pct': 0.02},
    {'name': 'MACD_6_13_4', 'func': strategy_macd, 'regime': 'trending', 'fast': 6, 'slow': 13, 'signal': 4, 'position_pct': 0.1, 'risk_pct': 0.01, 'atr_multiplier': 2.0, 'atr_sl_multiplier': 2.0, 'atr_tp_multiplier': 4.0, 'use_trailing': True, 'trailing_pct': 0.02},
    {'name': 'EMAPullback_20_3', 'func': strategy_ema_pullback, 'regime': 'trending', 'ema_period': 20, 'pullback_pct': 0.03, 'position_pct': 0.2, 'risk_pct': 0.02, 'atr_multiplier': 2.5, 'atr_sl_multiplier': 2.5, 'atr_tp_multiplier': 5.0, 'use_trailing': True, 'trailing_pct': 0.02},
    {'name': 'EMAPullback_50_5', 'func': strategy_ema_pullback, 'regime': 'trending', 'ema_period': 50, 'pullback_pct': 0.05, 'position_pct': 0.15, 'risk_pct': 0.015, 'atr_multiplier': 3.0, 'atr_sl_multiplier': 3.0, 'atr_tp_multiplier': 6.0, 'use_trailing': True, 'trailing_pct': 0.02},
    {'name': 'Breakout_20', 'func': strategy_breakout, 'regime': 'trending', 'lookback': 20, 'position_pct': 0.2, 'risk_pct': 0.02, 'atr_multiplier': 2.0, 'atr_sl_multiplier': 2.0, 'atr_tp_multiplier': 4.0, 'use_trailing': True, 'trailing_pct': 0.02},
    {'name': 'Breakout_10', 'func': strategy_breakout, 'regime': 'trending', 'lookback': 10, 'position_pct': 0.15, 'risk_pct': 0.015, 'atr_multiplier': 2.0, 'atr_sl_multiplier': 2.0, 'atr_tp_multiplier': 4.0, 'use_trailing': True, 'trailing_pct': 0.02},
    # --- 震荡策略：震荡市低买高卖，趋势市容易逆势亏损 ---
    # 震荡策略止损较窄（atr_sl=1.5~2.0），快进快出
    {'name': 'RSI_Boll_14_20_2', 'func': strategy_rsi_bollinger, 'regime': 'ranging', 'rsi_period': 14, 'boll_period': 20, 'boll_std': 2.0, 'rsi_low': 30, 'rsi_high': 70, 'position_pct': 0.2, 'risk_pct': 0.015, 'atr_multiplier': 2.0, 'atr_sl_multiplier': 2.0, 'atr_tp_multiplier': 4.0, 'use_trailing': True, 'trailing_pct': 0.015},
    {'name': 'RSI_Boll_10_15_1.5', 'func': strategy_rsi_bollinger, 'regime': 'ranging', 'rsi_period': 10, 'boll_period': 15, 'boll_std': 1.5, 'rsi_low': 25, 'rsi_high': 75, 'position_pct': 0.15, 'risk_pct': 0.01, 'atr_multiplier': 1.5, 'atr_sl_multiplier': 1.5, 'atr_tp_multiplier': 3.0, 'use_trailing': True, 'trailing_pct': 0.015},
    {'name': 'RSI_Boll_7_10_1', 'func': strategy_rsi_bollinger, 'regime': 'ranging', 'rsi_period': 7, 'boll_period': 10, 'boll_std': 1.0, 'rsi_low': 20, 'rsi_high': 80, 'position_pct': 0.1, 'risk_pct': 0.01, 'atr_multiplier': 1.5, 'atr_sl_multiplier': 1.5, 'atr_tp_multiplier': 3.0, 'use_trailing': True, 'trailing_pct': 0.015},
    {'name': 'DCA_2pct', 'func': strategy_adaptive_dca, 'regime': 'ranging', 'lookback': 50, 'entry_pct': 0.02, 'position_pct': 0.2, 'risk_pct': 0.015, 'atr_multiplier': 2.0, 'atr_sl_multiplier': 2.0, 'atr_tp_multiplier': 4.0, 'use_trailing': True, 'trailing_pct': 0.015},
    {'name': 'DCA_3pct', 'func': strategy_adaptive_dca, 'regime': 'ranging', 'lookback': 50, 'entry_pct': 0.03, 'position_pct': 0.15, 'risk_pct': 0.01, 'atr_multiplier': 2.0, 'atr_sl_multiplier': 2.0, 'atr_tp_multiplier': 4.0, 'use_trailing': True, 'trailing_pct': 0.015},
    {'name': 'DCA_5pct', 'func': strategy_adaptive_dca, 'regime': 'ranging', 'lookback': 50, 'entry_pct': 0.05, 'position_pct': 0.1, 'risk_pct': 0.01, 'atr_multiplier': 2.0, 'atr_sl_multiplier': 2.0, 'atr_tp_multiplier': 4.0, 'use_trailing': True, 'trailing_pct': 0.015},
    # --- 自适应策略：两种市场都可运行 ---
    {'name': 'VolAdapt_20_3', 'func': strategy_vol_adaptive, 'regime': 'both', 'lookback': 20, 'vol_threshold': 0.03, 'position_pct': 0.2, 'risk_pct': 0.02, 'atr_multiplier': 2.0, 'atr_sl_multiplier': 2.0, 'atr_tp_multiplier': 4.0, 'use_trailing': True, 'trailing_pct': 0.02},
    {'name': 'VolAdapt_30_5', 'func': strategy_vol_adaptive, 'regime': 'both', 'lookback': 30, 'vol_threshold': 0.05, 'position_pct': 0.15, 'risk_pct': 0.015, 'atr_multiplier': 2.5, 'atr_sl_multiplier': 2.5, 'atr_tp_multiplier': 5.0, 'use_trailing': True, 'trailing_pct': 0.02},
]

# ==================== 滚动参数寻优 (WFO) ====================

# 参数搜索空间：对每个策略类型，在训练期网格搜索最优风险/仓位参数
# 策略特定参数（如 EMA 周期、RSI 参数）保持固定，只搜索通用的风险参数
PARAM_GRIDS = {
    'DoubleMA': {
        'risk_pct': [0.015, 0.025],
        'atr_sl_multiplier': [2.0, 3.0],
    },
    'MACD': {
        'risk_pct': [0.015, 0.025],
        'atr_sl_multiplier': [2.0, 3.0],
    },
    'EMAPullback': {
        'risk_pct': [0.015, 0.025],
        'atr_sl_multiplier': [2.0, 3.0],
    },
    'Breakout': {
        'risk_pct': [0.015, 0.02],
        'atr_sl_multiplier': [1.5, 2.5],
    },
    'RSI': {
        'risk_pct': [0.01, 0.02],
        'atr_sl_multiplier': [1.5, 2.0],
    },
    'DCA': {
        'risk_pct': [0.015, 0.025],
        'atr_sl_multiplier': [1.5, 2.5],
    },
    'VolAdapt': {
        'risk_pct': [0.015, 0.025],
        'atr_sl_multiplier': [2.0, 3.0],
    },
}


def optimize_strategy_params(window_data, config, verbose=False):
    """
    Walk-Forward Optimization: 在训练期搜索最优风险参数。

    只在1个代表币种上评估（第一个symbol），减少计算量。
    用训练期收益（非夏普，因为很多策略交易少时夏普不稳定）作为评分。

    Returns: optimized config dict
    """
    strategy_type = config['name'].split('_')[0]
    param_grid = PARAM_GRIDS.get(strategy_type, {})

    if not param_grid:
        return config

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    if verbose:
        print(f"      WFO: {config['name']} ({strategy_type}) 搜索 {len(combos)} 组参数...")

    # 只用第一个币种评估参数，减少计算量
    eval_symbol = next(iter(window_data))
    eval_data = window_data[eval_symbol]
    closes = eval_data['closes']
    train_end = int(len(closes) * TRAIN_RATIO)
    train_closes = closes[:train_end]
    train_highs = eval_data['highs'][:train_end] if eval_data.get('highs') else None
    train_lows = eval_data['lows'][:train_end] if eval_data.get('lows') else None
    train_tf = eval_data['tf_confirm'][:train_end] if eval_data.get('tf_confirm') else None

    best_params = None
    best_score = -float('inf')

    for combo in combos:
        test_params = config.copy()
        for k, v in zip(keys, combo):
            test_params[k] = v
        # atr_multiplier 跟随 atr_sl_multiplier
        test_params['atr_multiplier'] = test_params['atr_sl_multiplier']
        # 保持 2:1 盈亏比
        test_params['atr_tp_multiplier'] = test_params['atr_sl_multiplier'] * 2

        result = run_backtest(train_closes, config['func'], test_params,
                              train_highs, train_lows, train_tf)
        if result:
            # 评分 = 收益率 + 0.5×夏普（兼顾收益和风险调整收益）
            score = result['return'] + 0.5 * result['sharpe']
            if score > best_score:
                best_score = score
                best_params = test_params.copy()

    if best_params and verbose:
        orig_risk = config.get('risk_pct', 0.02)
        orig_sl = config.get('atr_sl_multiplier', 2.0)
        opt_risk = best_params.get('risk_pct', 0.02)
        opt_sl = best_params.get('atr_sl_multiplier', 2.0)
        print(f"        最优: risk={orig_risk}→{opt_risk}, sl={orig_sl}→{opt_sl}, 训练评分={best_score:.2f}")

    return best_params if best_params else config


def run_with_validation(closes, strategy_func, params, highs=None, lows=None, tf_confirm=None):
    if not closes or len(closes) < 60:
        return None
    
    train_end = int(len(closes) * TRAIN_RATIO)
    train_closes = closes[:train_end]
    test_closes = closes[train_end:]
    train_highs = highs[:train_end] if highs else None
    train_lows = lows[:train_end] if lows else None
    test_highs = highs[train_end:] if highs else None
    test_lows = lows[train_end:] if lows else None
    train_tf = tf_confirm[:train_end] if tf_confirm else None
    test_tf = tf_confirm[train_end:] if tf_confirm else None
    
    train_result = run_backtest(train_closes, strategy_func, params, train_highs, train_lows, train_tf)
    test_result = run_backtest(test_closes, strategy_func, params, test_highs, test_lows, test_tf)
    
    if not train_result or not test_result:
        return None
    
    train_ret = train_result['return']
    test_ret = test_result['return']
    
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
        'mtf_blocked_train': train_result.get('mtf_blocked', 0),
        'mtf_blocked_test': test_result.get('mtf_blocked', 0),
    }


# ==================== 组合层面风控 ====================

def calc_correlation_matrix(symbols_data):
    """
    计算币种间日收益率相关性矩阵。
    用于组合风控：相关性高的币种不重复持仓，避免集中风险。
    """
    returns_dict = {}
    for symbol, data in symbols_data.items():
        closes = pd.Series(data['closes'])
        returns_dict[symbol] = closes.pct_change().dropna()
    df = pd.DataFrame(returns_dict)
    return df.corr()


def calc_signal_strength(closes, params):
    """
    计算买入信号强度 (0.0~1.0, 越高越强)。
    综合 RSI 超卖程度 + 布林带位置 + 偏离均线程度。
    用于多币种同时出现买入信号时的优先级排序。
    """
    strength = 0.0
    price = closes[-1]

    # RSI 贡献 (0~0.4)：RSI越低越超卖，买入信号越强
    rsi = calc_rsi(closes, 14)
    if rsi < 30:
        strength += (30 - rsi) / 30 * 0.4
    elif rsi < 50:
        strength += (50 - rsi) / 20 * 0.15

    # 布林带位置贡献 (0~0.3)：价格在下轨附近更超卖
    boll = calc_bollinger(closes, 20, 2.0)
    if boll:
        upper, mid, lower = boll
        if price <= lower:
            strength += 0.3
        elif price < mid and mid > lower:
            strength += (mid - price) / (mid - lower) * 0.2

    # 偏离EMA20贡献 (0~0.3)：价格低于EMA20越多，反弹空间越大
    ema20 = calc_ema(closes, 20)
    if ema20 > 0:
        dev = (ema20 - price) / ema20
        if dev > 0:
            strength += min(dev * 5, 0.3)

    return min(strength, 1.0)


def check_fee_viable(price, data, i, params, strength, atr_period=14):
    """
    手续费可行性检查：评估当前交易是否值得支付手续费。

    三层过滤：
    1. 波动率过滤：ATR占价格的百分比必须 >= 往返手续费 * 倍数
       市场太安静时，价格波动连手续费都覆盖不了，不值得交易
    2. 止盈比过滤：ATR动态止盈(ATR% × atr_tp_multiplier) >= 往返手续费 * 倍数
       预期收益不够覆盖手续费时不值得交易
    3. 信号强度过滤：信号强度必须 >= 最低门槛
       弱信号成功率低，不值得支付手续费

    Returns: (is_viable: bool, reason: str)
    """
    atr_val = None
    atr_pct = None
    
    # 1. 波动率过滤 + 计算 ATR
    highs = data.get('highs', [])
    lows = data.get('lows', [])
    closes = data['closes']
    if highs and lows and len(highs) > i and i >= atr_period:
        atr_val = float(ATR(
            pd.Series(highs[:i+1]),
            pd.Series(lows[:i+1]),
            pd.Series(closes[:i+1]),
            atr_period
        ).iloc[-1])
        if not pd.isna(atr_val) and atr_val > 0 and price > 0:
            atr_pct = atr_val / price
            min_required_move = ROUND_TRIP_FEE * MIN_MOVE_MULTIPLIER
            if atr_pct < min_required_move:
                return False, 'low_volatility'

    # 2. 止盈比过滤：ATR动态止盈 >= 往返手续费 * 倍数
    if atr_val is not None and atr_pct is not None:
        atr_tp_multiplier = params.get('atr_tp_multiplier', 4.0)
        expected_tp_pct = atr_pct * atr_tp_multiplier
        min_required_tp = ROUND_TRIP_FEE * MIN_TP_FEE_RATIO
        if expected_tp_pct < min_required_tp:
            return False, 'tp_too_small'

    # 3. 信号强度过滤
    if strength < MIN_SIGNAL_STRENGTH:
        return False, 'weak_signal'

    return True, 'ok'


CONSENSUS_BOOST = 0.15  # 每个额外策略共识，信号强度提升15%
MOMENTUM_LOOKBACK = 20  # 动量计算回看天数
MOMENTUM_TOPK = 4       # 只在前K名动量币种上交易（共6个币种）

# 策略感知轮动：不同策略类型选不同动量方向的币种
# 趋势策略选高动量（追涨），均值回归策略选低动量（抄底），自适应策略不轮动
TREND_STRATS = {'DoubleMA', 'MACD', 'EMAPullback', 'Breakout'}
REVERT_STRATS = {'RSI', 'DCA'}
# VolAdapt = 'both'，不轮动，使用全部币种


def calc_momentum(closes, lookback=MOMENTUM_LOOKBACK):
    """
    计算动量分数：过去N天收益率。
    正动量 = 近期上涨趋势，负动量 = 近期下跌趋势。
    用于币种轮动：只在强势币种上开仓。
    """
    if len(closes) < lookback + 1:
        return 0.0
    return (closes[-1] - closes[-lookback]) / closes[-lookback] * 100


def precompute_consensus(symbols_data, all_configs, regime):
    """
    预计算多策略信号共振：对每个币种每个时间步，
    统计有多少个策略同时发出买入信号。

    Returns: {symbol: [consensus_count at each step]}
    """
    # 筛选当前市场状态下匹配的策略
    matching_configs = [
        c for c in all_configs
        if c.get('regime', 'both') == 'both' or c.get('regime') == regime
    ]

    consensus_map = {}
    for symbol, data in symbols_data.items():
        closes = data['closes']
        n = len(closes)
        counts = [0] * n
        for config in matching_configs:
            func = config['func']
            params = config
            for i in range(50, n):
                if closes[i] is None:
                    continue
                signal = func(closes[:i+1], None, params)
                if signal == 'buy':
                    counts[i] += 1
        consensus_map[symbol] = counts
    return consensus_map


def run_portfolio_backtest(symbols_data, strategy_func, params, max_positions=MAX_POSITIONS,
                           corr_threshold=CORR_THRESHOLD, corr_matrix=None,
                           consensus_map=None):
    """
    组合层面回测：多币种共享资金池，限制同时持仓数量，高相关币种择优。

    核心逻辑：
    1. 共享一个资金池（INITIAL_BALANCE），所有币种的交易从同一个余额扣款
    2. 限制同时持仓数 ≤ max_positions，避免过度分散或过度集中
    3. 每个时间步收集所有币种的买入信号，按信号强度排序
    4. 高相关币种（相关性 > corr_threshold）不重复持仓
    5. 仓位按 ATR 动态计算，单币种上限30%防止过度集中
    6. 多策略信号共振：多个策略同时喊buy时，信号强度加权提升
    """
    min_len = min(len(d['closes']) for d in symbols_data.values())
    if min_len < 50:
        return None

    # 计算相关性矩阵（如果未提供）
    if corr_matrix is None:
        corr_matrix = calc_correlation_matrix(symbols_data)

    balance = INITIAL_BALANCE
    positions = {}  # {symbol: {'price': ..., 'amount': ..., 'cost': ...}}
    trades = []
    equity = []
    concurrent_history = []

    risk_pct = params.get('risk_pct', 0.02)
    atr_multiplier = params.get('atr_multiplier', 2.0)
    atr_sl_multiplier = params.get('atr_sl_multiplier', 2.0)
    atr_tp_multiplier = params.get('atr_tp_multiplier', 4.0)
    use_trailing = params.get('use_trailing', True)
    trailing_pct = params.get('trailing_pct', 0.02)
    atr_period = 14
    position_pct = params.get('position_pct', 0.2)

    stats = {
        'max_concurrent': 0,
        'corr_blocked': 0,
        'slot_blocked': 0,
        'fee_blocked': 0,
        'total_buy_signals': 0,
        'executed_buys': 0,
        'consensus_boosted': 0,  # 被共识提升的信号数
        'total_consensus': 0,    # 总共识数（多策略同时喊buy的次数）
    }
    stop_stats = {'stop_loss': 0, 'take_profit': 0, 'trailing_stop': 0, 'signal_sell': 0}

    for i in range(min_len):
        # 1. 引擎层止盈止损 + 移动止损 + 策略卖出信号
        for symbol in list(positions.keys()):
            data = symbols_data[symbol]
            closes = data['closes'][:i + 1]
            signal = strategy_func(closes, positions[symbol], params)
            price = closes[-1]
            pos = positions[symbol]
            
            should_exit = False
            exit_reason = ''
            
            # 引擎层动态止盈止损 + 移动止损
            highs_sym = data.get('highs', [])
            lows_sym = data.get('lows', [])
            if highs_sym and lows_sym and len(highs_sym) > i and i >= atr_period:
                atr_val = float(ATR(
                    pd.Series(highs_sym[:i + 1]),
                    pd.Series(lows_sym[:i + 1]),
                    pd.Series(closes),
                    atr_period
                ).iloc[-1])
                
                if not pd.isna(atr_val) and atr_val > 0:
                    stop_price = pos.get('entry_price', pos['price']) - atr_val * atr_sl_multiplier
                    tp_price = pos.get('entry_price', pos['price']) + atr_val * atr_tp_multiplier
                    
                    if use_trailing:
                        pos['highest_price'] = max(pos.get('highest_price', price), price)
                        if not pos.get('trailing_active'):
                            if price >= pos.get('entry_price', pos['price']) * (1 + trailing_pct):
                                pos['trailing_active'] = True
                                pos['trailing_stop'] = pos['highest_price'] - atr_val * atr_sl_multiplier
                        if pos.get('trailing_active'):
                            new_trail = pos['highest_price'] - atr_val * atr_sl_multiplier
                            pos['trailing_stop'] = max(pos.get('trailing_stop', stop_price), new_trail)
                    
                    effective_stop = pos.get('trailing_stop', stop_price) if pos.get('trailing_active') else stop_price
                    
                    if price <= effective_stop:
                        should_exit = True
                        exit_reason = 'trailing_stop' if pos.get('trailing_active') else 'stop_loss'
                    elif price >= tp_price:
                        should_exit = True
                        exit_reason = 'take_profit'
            
            if should_exit:
                revenue = pos['amount'] * price * (1 - FEE_RATE)
                pnl = revenue - pos['cost']
                balance += revenue
                trades.append({
                    'symbol': symbol, 'action': 'sell', 'price': price,
                    'pnl': pnl, 'step': i, 'reason': exit_reason
                })
                stop_stats[exit_reason] += 1
                del positions[symbol]
                continue
            
            # 策略层卖出信号（引擎未触发退出时）
            if signal == 'sell':
                revenue = pos['amount'] * price * (1 - FEE_RATE)
                pnl = revenue - pos['cost']
                balance += revenue
                trades.append({
                    'symbol': symbol, 'action': 'sell', 'price': price,
                    'pnl': pnl, 'step': i, 'reason': 'signal_sell'
                })
                stop_stats['signal_sell'] += 1
                del positions[symbol]

        # 2. 收集所有未持仓币种的买入信号
        buy_candidates = []
        for symbol, data in symbols_data.items():
            if symbol in positions:
                continue
            closes = data['closes'][:i + 1]
            if len(closes) < 50:
                continue

            # 4H 多时间框架确认
            tf_confirm = data.get('tf_confirm')
            if tf_confirm and i < len(tf_confirm) and not tf_confirm[i]:
                continue

            signal = strategy_func(closes, None, params)
            if signal == 'buy':
                strength = calc_signal_strength(closes, params)

                # 多策略信号共振：检查其他策略是否也喊buy
                consensus_count = 0
                if consensus_map and symbol in consensus_map and i < len(consensus_map[symbol]):
                    consensus_count = consensus_map[symbol][i]
                    if consensus_count > 1:
                        # 共识加成：每多一个策略同意，信号强度提升15%
                        strength = min(strength * (1 + (consensus_count - 1) * CONSENSUS_BOOST), 1.0)
                        stats['consensus_boosted'] += 1
                        stats['total_consensus'] += consensus_count

                # 手续费可行性检查
                viable, fee_reason = check_fee_viable(closes[-1], data, i, params, strength)
                if not viable:
                    stats['fee_blocked'] += 1
                    continue

                buy_candidates.append((symbol, strength, closes[-1], data))
                stats['total_buy_signals'] += 1

        # 3. 按信号强度排序，择优执行买入
        available_slots = max_positions - len(positions)
        if available_slots <= 0 and buy_candidates:
            stats['slot_blocked'] += len(buy_candidates)

        if available_slots > 0 and buy_candidates:
            buy_candidates.sort(key=lambda x: x[1], reverse=True)

            for symbol, strength, price, data in buy_candidates:
                if available_slots <= 0:
                    stats['slot_blocked'] += 1
                    continue

                # 相关性检查：不与已持有的高相关币种重复持仓
                is_correlated = False
                for held in positions:
                    if held == symbol:
                        continue
                    try:
                        c = corr_matrix.loc[symbol, held]
                    except (KeyError, AttributeError):
                        c = 0
                    if abs(c) > corr_threshold:
                        is_correlated = True
                        break

                if is_correlated:
                    stats['corr_blocked'] += 1
                    continue

                # ATR 动态仓位计算
                highs = data.get('highs', [])
                lows = data.get('lows', [])

                if highs and lows and len(highs) > i and i >= atr_period:
                    atr_val = float(ATR(
                        pd.Series(highs[:i + 1]),
                        pd.Series(lows[:i + 1]),
                        pd.Series(data['closes'][:i + 1]),
                        atr_period
                    ).iloc[-1])
                    if not pd.isna(atr_val) and atr_val > 0:
                        risk_amount = balance * risk_pct
                        stop_distance = atr_val * atr_multiplier
                        amount = risk_amount / stop_distance
                        # 单币种仓位上限30%，防止过度集中
                        max_amount = balance * 0.3 / price
                        amount = min(amount, max_amount)
                    else:
                        amount = (balance * position_pct / max_positions) / price
                else:
                    amount = (balance * position_pct / max_positions) / price

                cost = amount * price * (1 + FEE_RATE)
                if cost > balance:
                    amount = balance / (price * (1 + FEE_RATE))
                    cost = balance

                balance -= cost
                positions[symbol] = {
                    'price': price, 'amount': amount, 'cost': cost,
                    'entry_price': price, 'highest_price': price,
                    'trailing_active': False, 'trailing_stop': 0,
                }
                trades.append({
                    'symbol': symbol, 'action': 'buy', 'price': price,
                    'amount': amount, 'strength': round(strength, 3), 'step': i
                })
                stats['executed_buys'] += 1
                available_slots -= 1

        stats['max_concurrent'] = max(stats['max_concurrent'], len(positions))
        concurrent_history.append(len(positions))

        # 4. 计算组合权益
        total_pos_value = 0
        for symbol, pos in positions.items():
            price = symbols_data[symbol]['closes'][i]
            total_pos_value += pos['amount'] * price
        equity.append(balance + total_pos_value)

    # 强制平仓
    for symbol in list(positions.keys()):
        price = symbols_data[symbol]['closes'][min_len - 1]
        revenue = positions[symbol]['amount'] * price * (1 - FEE_RATE)
        pnl = revenue - positions[symbol]['cost']
        balance += revenue
        trades.append({
            'symbol': symbol, 'action': 'sell', 'price': price,
            'pnl': pnl, 'step': min_len - 1, 'reason': 'force_close'
        })

    if not equity:
        return None

    total_return = (equity[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    peak = np.maximum.accumulate(equity)
    drawdown = (np.array(equity) - peak) / peak * 100
    max_drawdown = float(drawdown.min())

    sell_trades = [t for t in trades if t['action'] == 'sell']
    wins = [t for t in sell_trades if t.get('pnl', 0) > 0]
    win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0

    returns = np.diff(np.array(equity)) / np.array(equity)[:-1]
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(365)) if len(returns) > 1 and np.std(returns) > 0 else 0

    stats['avg_concurrent'] = round(float(np.mean(concurrent_history)), 2) if concurrent_history else 0

    # 各币种贡献分析
    symbol_pnl = {}
    for t in trades:
        s = t['symbol']
        if s not in symbol_pnl:
            symbol_pnl[s] = {'trades': 0, 'pnl': 0.0}
        if t['action'] == 'sell':
            symbol_pnl[s]['trades'] += 1
            symbol_pnl[s]['pnl'] += t.get('pnl', 0)

    return {
        'return': total_return,
        'drawdown': max_drawdown,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'trades': len(sell_trades),
        'wins': len(wins),
        'max_concurrent': stats['max_concurrent'],
        'avg_concurrent': stats['avg_concurrent'],
        'corr_blocked': stats['corr_blocked'],
        'slot_blocked': stats['slot_blocked'],
        'fee_blocked': stats['fee_blocked'],
        'total_buy_signals': stats['total_buy_signals'],
        'executed_buys': stats['executed_buys'],
        'consensus_boosted': stats['consensus_boosted'],
        'total_consensus': stats['total_consensus'],
        'symbol_pnl': symbol_pnl,
        'stop_stats': stop_stats,
    }


def run_portfolio_with_validation(symbols_data, strategy_func, params,
                                  max_positions=MAX_POSITIONS, corr_threshold=CORR_THRESHOLD,
                                  consensus_map=None):
    """
    组合层面回测 + 训练/测试验证。
    相关性矩阵从训练期计算，应用于训练和测试。
    """
    min_len = min(len(d['closes']) for d in symbols_data.values())
    if min_len < 60:
        return None

    train_end = int(min_len * TRAIN_RATIO)

    # 分割数据
    train_data = {}
    test_data = {}
    for symbol, data in symbols_data.items():
        train_data[symbol] = {
            'closes': data['closes'][:train_end],
            'highs': data['highs'][:train_end] if data.get('highs') else None,
            'lows': data['lows'][:train_end] if data.get('lows') else None,
            'tf_confirm': data['tf_confirm'][:train_end] if data.get('tf_confirm') else None,
        }
        test_data[symbol] = {
            'closes': data['closes'][train_end:],
            'highs': data['highs'][train_end:] if data.get('highs') else None,
            'lows': data['lows'][train_end:] if data.get('lows') else None,
            'tf_confirm': data['tf_confirm'][train_end:] if data.get('tf_confirm') else None,
        }

    # 从训练期计算相关性矩阵
    corr_matrix = calc_correlation_matrix(train_data)

    # 分割共识数据（训练/测试各取对应部分）
    train_consensus = {}
    test_consensus = {}
    if consensus_map:
        for symbol, counts in consensus_map.items():
            train_consensus[symbol] = counts[:train_end] if len(counts) > train_end else counts
            test_consensus[symbol] = counts[train_end:] if len(counts) > train_end else []

    train_result = run_portfolio_backtest(train_data, strategy_func, params,
                                          max_positions, corr_threshold, corr_matrix,
                                          train_consensus if train_consensus else None)
    test_result = run_portfolio_backtest(test_data, strategy_func, params,
                                         max_positions, corr_threshold, corr_matrix,
                                         test_consensus if test_consensus else None)

    if not train_result or not test_result:
        return None

    train_ret = train_result['return']
    test_ret = test_result['return']

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
        'max_concurrent': max(train_result['max_concurrent'], test_result['max_concurrent']),
        'avg_concurrent': round((train_result['avg_concurrent'] + test_result['avg_concurrent']) / 2, 2),
        'corr_blocked': train_result['corr_blocked'] + test_result['corr_blocked'],
        'slot_blocked': train_result['slot_blocked'] + test_result['slot_blocked'],
        'fee_blocked': train_result['fee_blocked'] + test_result['fee_blocked'],
        'total_buy_signals': train_result['total_buy_signals'] + test_result['total_buy_signals'],
        'executed_buys': train_result['executed_buys'] + test_result['executed_buys'],
        'consensus_boosted': train_result.get('consensus_boosted', 0) + test_result.get('consensus_boosted', 0),
        'total_consensus': train_result.get('total_consensus', 0) + test_result.get('total_consensus', 0),
        'test_symbol_pnl': test_result['symbol_pnl'],
        'stop_stats': {
            k: (train_result.get('stop_stats', {}).get(k, 0) + 
                test_result.get('stop_stats', {}).get(k, 0))
            for k in ['stop_loss', 'take_profit', 'trailing_stop', 'signal_sell']
        },
    }


def run_rolling_10rounds():
    """滚动窗口10轮实验 - 组合层面风控版"""
    print("=" * 90)
    print("  滚动窗口10轮实验 - WFO + 信号共振（无币种轮动）")
    print("=" * 90)
    print(f"  市场自适应 + WFO寻优 + 多策略共振 + ATR仓位 + 4H多时框 + 组合风控 + 手续费过滤 + ATR止盈止损")
    print(f"  WFO: 每轮训练期网格搜索最优risk_pct/atr_multiplier/atr_sl，用最优参数跑测试期")
    print(f"  信号共振: 多策略同时喊buy时信号强度加权提升(每额外策略+{int(CONSENSUS_BOOST*100)}%)")
    print(f"  组合风控: 最大同时持仓{MAX_POSITIONS}个 | 相关性>{CORR_THRESHOLD}不重复 | 信号强度排序择优")
    print(f"  手续费感知: 往返费{ROUND_TRIP_FEE*100:.1f}% | 最小波动{MIN_MOVE_MULTIPLIER}x费 | 动态止盈>={MIN_TP_FEE_RATIO}x费 | 信号强度>={MIN_SIGNAL_STRENGTH}")
    print(f"  止盈止损: ATR动态(盈亏比2:1) | 移动止损激活盈利{2.0}% | 趋势宽止损/震荡窄止损")
    print(f"  买入条件: 日K信号 + 4H EMA(20)>EMA(50) + 手续费可行性检查")
    print(f"  ATR参数: risk_pct=1~2.5%, sl=1.5~3.0xATR, tp=3.0~6.0xATR")
    print(f"样本分割: {TRAIN_RATIO*100:.0f}%训练 + {(1-TRAIN_RATIO)*100:.0f}%测试")
    print(f"过拟合阈值: 训练收益/测试收益 > {OVERFIT_THRESHOLD}")
    print(f"币种: {len(SYMBOLS)} | 策略: {len(STRATEGY_POOL)}")
    print("=" * 90)

    # 获取所有币种数据
    all_data = {}
    for symbol in SYMBOLS:
        result = get_candles(symbol, limit=500)
        if result:
            closes, timestamps, highs, lows = result
            all_data[symbol] = {'closes': closes, 'timestamps': timestamps, 'highs': highs, 'lows': lows}
            print(f"  {symbol}: {len(closes)} 条日K ({datetime.fromtimestamp(timestamps[0]/1000).strftime('%Y-%m-%d')} ~ {datetime.fromtimestamp(timestamps[-1]/1000).strftime('%Y-%m-%d')})")

        # 获取4H K线用于多时间框架确认
        tf_result = get_candles_4h(symbol, limit=1500)
        if tf_result and symbol in all_data:
            tf_closes, tf_ts = tf_result
            tf_confirm = align_4h_trend_to_daily(all_data[symbol]['timestamps'], tf_closes, tf_ts)
            all_data[symbol]['tf_confirm'] = tf_confirm
            print(f"         4H-K线: {len(tf_closes)} 条, 趋势确认: {sum(tf_confirm)}/{len(tf_confirm)} 根日K处于4H多头")

    if not all_data:
        print("无数据，退出")
        return

    # 打印相关性矩阵
    print(f"\n  组合相关性矩阵（日收益率）:")
    corr_full = calc_correlation_matrix(all_data)
    short_names = [s.split('-')[0] for s in SYMBOLS if s in all_data]
    header = "         " + "  ".join(f"{n:>6}" for n in short_names)
    print(header)
    for i, s1 in enumerate(SYMBOLS):
        if s1 not in all_data:
            continue
        row = f"  {short_names[i]:>6}"
        for j, s2 in enumerate(SYMBOLS):
            if s2 not in all_data:
                continue
            val = corr_full.loc[s1, s2] if s1 in corr_full.index and s2 in corr_full.columns else 0
            row += f"  {val:>5.2f}"
        print(row)

    all_results = []
    round_summaries = []

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'='*90}")
        print(f"  Round {round_num}/{NUM_ROUNDS}")
        print(f"{'='*90}")

        # 构建所有币种的窗口数据
        window_data = {}
        window_start = ""
        window_end = ""

        for symbol in SYMBOLS:
            if symbol not in all_data:
                continue
            data = all_data[symbol]
            closes = data['closes']
            timestamps = data['timestamps']
            highs = data['highs']
            lows = data['lows']

            total_len = len(closes)
            window_size = int(total_len * 0.6)
            shift = int(total_len * 0.04)

            start_idx = (round_num - 1) * shift
            end_idx = start_idx + window_size

            if end_idx > total_len:
                end_idx = total_len
                start_idx = end_idx - window_size
            if start_idx < 0:
                start_idx = 0

            window_closes = closes[start_idx:end_idx]
            window_ts = timestamps[start_idx:end_idx]
            window_highs = highs[start_idx:end_idx]
            window_lows = lows[start_idx:end_idx]
            window_tf = data.get('tf_confirm', [])
            window_tf = window_tf[start_idx:end_idx] if window_tf else None

            if len(window_closes) < 60:
                continue

            if not window_start:
                window_start = datetime.fromtimestamp(window_ts[0] / 1000).strftime('%Y-%m-%d')
                window_end = datetime.fromtimestamp(window_ts[-1] / 1000).strftime('%Y-%m-%d')

            window_data[symbol] = {
                'closes': window_closes,
                'highs': window_highs,
                'lows': window_lows,
                'tf_confirm': window_tf,
            }

        if not window_data:
            print("  无有效窗口数据")
            continue

        print(f"  时间窗口: {window_start} ~ {window_end} | 币种数: {len(window_data)}")

        # 检测组合市场状态（用所有币种平均ADX）
        adx_values = []
        for symbol, data in window_data.items():
            _, info = detect_regime(data['closes'], data['highs'], data['lows'])
            adx_values.append(info['adx'])
        avg_adx = np.mean(adx_values) if adx_values else 0
        if avg_adx >= 25:
            regime = 'trending'
        elif avg_adx <= 20:
            regime = 'ranging'
        else:
            regime = 'trending' if avg_adx > 22 else 'ranging'
        print(f"  组合市场状态: {regime} (平均ADX: {avg_adx:.1f})")

        # 预计算多策略信号共振
        consensus_map = precompute_consensus(window_data, STRATEGY_POOL, regime)
        total_consensus_buys = sum(1 for s, counts in consensus_map.items() for c in counts if c > 1)
        print(f"  信号共振: {total_consensus_buys}个时间步出现多策略共识买入")

        round_results = []

        for config in STRATEGY_POOL:
            strat_regime = config.get('regime', 'both')
            if strat_regime != 'both' and strat_regime != regime:
                continue

            # WFO: 在训练期优化风险参数
            opt_config = optimize_strategy_params(window_data, config, verbose=True)

            result = run_portfolio_with_validation(window_data, config['func'], opt_config,
                                                    consensus_map=consensus_map)
            if result:
                result['strategy'] = config['name']
                result['round'] = round_num
                result['window_start'] = window_start
                result['window_end'] = window_end
                result['regime'] = regime
                result['avg_adx'] = round(avg_adx, 1)
                result['params'] = {k: v for k, v in opt_config.items() if k not in ['name', 'func']}
                result['wfo_optimized'] = True
                round_results.append(result)
                all_results.append(result)

            consensus_str = f" | 共振 {result['consensus_boosted']}" if result.get('consensus_boosted', 0) > 0 else ""
            print(f"  {config['name']:<25} 训练 {result['train_return']:>6.1f}% | 测试 {result['test_return']:>6.1f}% | "
                  f"夏普 {result['test_sharpe']:>5.2f} | 交易 {result['test_trades']:>3} | "
                  f"并发 {result['avg_concurrent']:.1f} | 相关 {result['corr_blocked']} | 槽位 {result['slot_blocked']} | 费率 {result['fee_blocked']}{consensus_str}")
            
            # 退出方式统计
            ss = result.get('stop_stats', {})
            if ss:
                exit_parts = []
                if ss.get('trailing_stop'): exit_parts.append(f"移动止损{ss['trailing_stop']}")
                if ss.get('stop_loss'): exit_parts.append(f"止损{ss['stop_loss']}")
                if ss.get('take_profit'): exit_parts.append(f"止盈{ss['take_profit']}")
                if ss.get('signal_sell'): exit_parts.append(f"信号{ss['signal_sell']}")
                if exit_parts:
                    print(f"  {'':<25} 退出方式: {' | '.join(exit_parts)}")

        # 复盘
        print(f"\n  Round {round_num} 复盘:")
        valid = [r for r in round_results if not r['is_overfit']]
        overfit = [r for r in round_results if r['is_overfit']]

        if valid:
            best = max(valid, key=lambda x: x['test_return'])
            avg_test = np.mean([r['test_return'] for r in valid])
            avg_sharpe = np.mean([r['test_sharpe'] for r in valid])
            total_corr_blocked = sum(r['corr_blocked'] for r in valid)
            total_slot_blocked = sum(r['slot_blocked'] for r in valid)
            total_fee_blocked = sum(r['fee_blocked'] for r in valid)
            total_consensus_boosted = sum(r.get('consensus_boosted', 0) for r in valid)

            print(f"    有效策略: {len(valid)}/{len(round_results)} | 过拟合: {len(overfit)}")
            print(f"    最优: {best['strategy']} | 测试 {best['test_return']:.1f}% | 夏普 {best['test_sharpe']:.2f}")
            print(f"    平均测试收益: {avg_test:.1f}% | 平均夏普: {avg_sharpe:.2f}")
            print(f"    组合风控: 相关拦截 {total_corr_blocked}次 | 槽位拦截 {total_slot_blocked}次 | 手续费拦截 {total_fee_blocked}次 | 共振提升 {total_consensus_boosted}次 | 平均并发 {np.mean([r['avg_concurrent'] for r in valid]):.1f}")
            
            # 退出方式汇总
            total_ss = {'trailing_stop': 0, 'stop_loss': 0, 'take_profit': 0, 'signal_sell': 0}
            for r in valid:
                for k in total_ss:
                    total_ss[k] += r.get('stop_stats', {}).get(k, 0)
            if sum(total_ss.values()) > 0:
                print(f"    退出方式: 移动止损{total_ss['trailing_stop']} | 止损{total_ss['stop_loss']} | 止盈{total_ss['take_profit']} | 信号卖出{total_ss['signal_sell']}")

            # 测试期各币种贡献
            print(f"    最优策略各币种PnL贡献:")
            symbol_pnl = best.get('test_symbol_pnl', {})
            for symbol in SYMBOLS:
                if symbol in symbol_pnl:
                    sp = symbol_pnl[symbol]
                    print(f"      {symbol:<12} 交易{sp['trades']:>3}次 | PnL {sp['pnl']:>+8.2f}")

            round_summaries.append({
                'round': round_num,
                'window': f"{window_start} ~ {window_end}",
                'regime': regime,
                'avg_adx': round(avg_adx, 1),
                'total': len(round_results),
                'valid': len(valid),
                'overfit': len(overfit),
                'best_strategy': best['strategy'],
                'best_test_return': best['test_return'],
                'best_sharpe': best['test_sharpe'],
                'avg_test_return': round(avg_test, 2),
                'avg_sharpe': round(avg_sharpe, 2),
                'total_corr_blocked': total_corr_blocked,
                'total_slot_blocked': total_slot_blocked,
                'total_fee_blocked': total_fee_blocked,
                'total_consensus_boosted': total_consensus_boosted,
                'stop_stats': total_ss,
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
    print(f"  10轮滚动窗口实验总汇（WFO+信号共振+自适应+ATR+4H确认+组合风控+手续费+动态止盈止损）")
    print(f"{'='*90}")

    valid_all = [r for r in all_results if not r['is_overfit']]

    consistent = []
    if valid_all:
        sorted_results = sorted(valid_all, key=lambda x: x['test_return'], reverse=True)

        print(f"\n  总测试: {len(all_results)} | 有效: {len(valid_all)} | 过拟合: {len(all_results) - len(valid_all)}")
        print(f"  过拟合率: {(len(all_results) - len(valid_all)) / len(all_results) * 100:.1f}%")

        # 组合风控统计
        total_corr = sum(r['corr_blocked'] for r in valid_all)
        total_slot = sum(r['slot_blocked'] for r in valid_all)
        total_fee = sum(r['fee_blocked'] for r in valid_all)
        total_signals = sum(r['total_buy_signals'] for r in valid_all)
        total_executed = sum(r['executed_buys'] for r in valid_all)
        total_consensus = sum(r.get('consensus_boosted', 0) for r in valid_all)
        avg_concurrent = np.mean([r['avg_concurrent'] for r in valid_all])
        
        # 退出方式汇总
        total_ss_all = {'trailing_stop': 0, 'stop_loss': 0, 'take_profit': 0, 'signal_sell': 0}
        for r in valid_all:
            for k in total_ss_all:
                total_ss_all[k] += r.get('stop_stats', {}).get(k, 0)
        total_exits = sum(total_ss_all.values())
        
        print(f"  组合风控统计:")
        print(f"    买入信号总数: {total_signals} | 执行买入: {total_executed} | 执行率: {total_executed/total_signals*100:.1f}%" if total_signals > 0 else "    无买入信号")
        print(f"    相关性拦截: {total_corr}次 | 槽位拦截: {total_slot}次 | 手续费拦截: {total_fee}次 | 共振提升: {total_consensus}次 | 平均并发持仓: {avg_concurrent:.1f}")
        if total_exits > 0:
            print(f"    退出方式: 移动止损{total_ss_all['trailing_stop']}({total_ss_all['trailing_stop']/total_exits*100:.0f}%) | "
                  f"止损{total_ss_all['stop_loss']}({total_ss_all['stop_loss']/total_exits*100:.0f}%) | "
                  f"止盈{total_ss_all['take_profit']}({total_ss_all['take_profit']/total_exits*100:.0f}%) | "
                  f"信号{total_ss_all['signal_sell']}({total_ss_all['signal_sell']/total_exits*100:.0f}%)")

        print(f"\n  【TOP 20 测试集最优组合策略】")
        print(f"  {'排名':<4} {'轮':<4} {'策略':<25} {'训练':<8} {'测试':<8} {'夏普':<6} {'并发':<4} {'相关拦截':<6} {'过拟合':<6}")
        print(f"  {'-'*85}")
        for i, r in enumerate(sorted_results[:20], 1):
            print(f"  {i:<4} {r['round']:<4} {r['strategy']:<25} {r['train_return']:>6.1f}% {r['test_return']:>6.1f}% {r['test_sharpe']:>5.2f} {r['avg_concurrent']:>4.1f} {r['corr_blocked']:>6} {r['overfit_index']:>5.1f}")

        # 各轮最优
        print(f"\n  【各轮最优】")
        for s in round_summaries:
            if 'best_strategy' in s:
                print(f"  Round {s['round']}: {s['window']} | {s['regime']}(ADX {s['avg_adx']}) | {s['best_strategy']} | 测试 {s['best_test_return']:.1f}% | 夏普 {s['best_sharpe']:.2f}")

        # 策略类型稳定性
        print(f"\n  【策略类型稳定性分析】")
        type_all = {}
        for r in valid_all:
            stype = r['strategy'].split('_')[0]
            if stype not in type_all:
                type_all[stype] = []
            type_all[stype].append(r['test_return'])

        for stype, returns in sorted(type_all.items(), key=lambda x: np.mean(x[1]), reverse=True):
            avg = np.mean(returns)
            std = np.std(returns)
            positive_pct = len([r for r in returns if r > 0]) / len(returns) * 100
            stable = "✅ 稳定盈利" if avg > 0 and std < 3 else ("⚠️ 波动大" if std > 5 else "❌ 亏损")
            print(f"  {stype:<15} 平均 {avg:>5.1f}% | 标准差 {std:>4.1f} | 正收益 {positive_pct:>4.1f}% | {stable}")

        # 各币种PnL贡献汇总
        print(f"\n  【各币种PnL贡献（测试期汇总）】")
        symbol_total_pnl = {s: {'trades': 0, 'pnl': 0.0} for s in SYMBOLS}
        for r in valid_all:
            for s, sp in r.get('test_symbol_pnl', {}).items():
                if s in symbol_total_pnl:
                    symbol_total_pnl[s]['trades'] += sp['trades']
                    symbol_total_pnl[s]['pnl'] += sp['pnl']
        for symbol in SYMBOLS:
            sp = symbol_total_pnl[symbol]
            if sp['trades'] > 0:
                print(f"  {symbol:<12} 交易 {sp['trades']:>4}次 | 总PnL {sp['pnl']:>+10.2f} | 单次均值 {sp['pnl']/sp['trades']:>+8.2f}")

        # 跨轮次一致性分析
        print(f"\n  【跨轮次一致性分析】（哪些策略在多轮中持续盈利）")
        strategy_cross_round = {}
        for r in valid_all:
            key = r['strategy']
            if key not in strategy_cross_round:
                strategy_cross_round[key] = []
            strategy_cross_round[key].append(r['test_return'])

        for strategy, returns in strategy_cross_round.items():
            positive_count = len([r for r in returns if r > 0])
            total_count = len(returns)
            avg_ret = np.mean(returns)
            if positive_count / total_count > 0.5 and avg_ret > 0:
                consistent.append({
                    'strategy': strategy,
                    'avg_return': round(avg_ret, 2),
                    'positive_rounds': positive_count,
                    'total_rounds': total_count,
                    'positive_rate': round(positive_count / total_count * 100, 1)
                })

        consistent.sort(key=lambda x: x['avg_return'], reverse=True)

        if consistent:
            print(f"  {'策略':<25} {'平均收益':<8} {'盈利轮次':<8} {'盈利率':<6}")
            print(f"  {'-'*60}")
            for c in consistent[:10]:
                print(f"  {c['strategy']:<25} {c['avg_return']:>5.1f}% {c['positive_rounds']}/{c['total_rounds']:>3}轮 {c['positive_rate']:>5.1f}%")
        else:
            print(f"  ⚠️ 无策略在多轮中持续盈利")

    # 保存报告
    report = {
        'date': datetime.now().isoformat(),
        'experiment_type': 'rolling_window_10rounds_portfolio_risk',
        'rounds': NUM_ROUNDS,
        'symbols': SYMBOLS,
        'strategies': [s['name'] for s in STRATEGY_POOL],
        'strategy_regimes': {s['name']: s.get('regime', 'both') for s in STRATEGY_POOL},
        'train_ratio': TRAIN_RATIO,
        'overfit_threshold': OVERFIT_THRESHOLD,
        'regime_aware': True,
        'atr_sizing': True,
        'multi_timeframe': True,
        'portfolio_risk': True,
        'fee_aware': True,
        'dynamic_sl_tp': True,
        'wfo_optimization': True,
        'wfo_param_grids': {k: list(v.keys()) for k, v in PARAM_GRIDS.items()},
        'signal_consensus': True,
        'consensus_boost': CONSENSUS_BOOST,
        'momentum_rotation': False,
        'portfolio_config': {
            'max_positions': MAX_POSITIONS,
            'corr_threshold': CORR_THRESHOLD,
        },
        'fee_config': {
            'round_trip_fee': ROUND_TRIP_FEE,
            'min_move_multiplier': MIN_MOVE_MULTIPLIER,
            'min_tp_fee_ratio': MIN_TP_FEE_RATIO,
            'min_signal_strength': MIN_SIGNAL_STRENGTH,
        },
        'sl_tp_config': {
            'type': 'ATR_dynamic',
            'atr_sl_multiplier': '1.5~3.0',
            'atr_tp_multiplier': '3.0~6.0',
            'reward_risk_ratio': '2:1',
            'trailing_stop': True,
            'trailing_activation_pct': '1.5~2.0%',
        },
        'mtf_config': {'tf': '4H', 'fast_ema': 20, 'slow_ema': 50, 'min_bars': 60},
        'correlation_matrix': corr_full.to_dict() if 'corr_full' in dir() else {},
        'total_tests': len(all_results),
        'valid_tests': len(valid_all),
        'overfit_tests': len(all_results) - len(valid_all),
        'overfit_rate': f"{(len(all_results) - len(valid_all)) / len(all_results) * 100:.1f}%" if all_results else "0%",
        'portfolio_stats': {
            'total_buy_signals': sum(r['total_buy_signals'] for r in valid_all) if valid_all else 0,
            'executed_buys': sum(r['executed_buys'] for r in valid_all) if valid_all else 0,
            'corr_blocked': sum(r['corr_blocked'] for r in valid_all) if valid_all else 0,
            'slot_blocked': sum(r['slot_blocked'] for r in valid_all) if valid_all else 0,
            'fee_blocked': sum(r['fee_blocked'] for r in valid_all) if valid_all else 0,
            'consensus_boosted': sum(r.get('consensus_boosted', 0) for r in valid_all) if valid_all else 0,
            'avg_concurrent': round(float(np.mean([r['avg_concurrent'] for r in valid_all])), 2) if valid_all else 0,
            'stop_stats': total_ss_all,
        },
        'round_summaries': round_summaries,
        'top_20': sorted_results[:20] if valid_all else [],
        'consistent_strategies': consistent,
        'all_results': all_results,
    }

    with open(os.path.join(REPORT_DIR, 'rolling_window_portfolio.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  报告已保存: {REPORT_DIR}/rolling_window_portfolio.json")
    print(f"{'='*90}")

    return report


if __name__ == "__main__":
    run_rolling_10rounds()
