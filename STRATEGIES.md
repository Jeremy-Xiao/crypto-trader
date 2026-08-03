# 投资策略文档 (STRATEGIES.md)

> **最后更新**: 2026-08-04  
> **维护规则**: 每轮优化完成并提交代码后，同步更新本文档。

---

## 目录

1. [系统架构](#1-系统架构)
2. [生产层策略 (src/strategies/)](#2-生产层策略-srcstrategies)
3. [研究层策略池 (scripts/rolling_window_10rounds.py)](#3-研究层策略池-scriptsrolling_window_10roundspy)
4. [风控体系](#4-风控体系)
5. [组合层面风控](#5-组合层面风控)
6. [可用技术指标](#6-可用技术指标)
7. [回测结果汇总](#7-回测结果汇总)
8. [如何运行回测](#8-如何运行回测)

---

## 1. 系统架构

项目采用"研究-生产"双轨架构：

```
crypto-trader/
├── src/                    # 生产框架（OOP架构，用于实盘）
│   ├── strategies/         # 策略实现（继承 BaseStrategy）
│   │   ├── base.py         # 策略基类 + ATR风控 + 移动止损 + 市场状态检测
│   │   ├── double_ma.py    # 双均线交叉（趋势）
│   │   ├── macd_cross.py   # MACD金叉死叉（趋势）
│   │   ├── rsi_bollinger.py# RSI+布林带（震荡）
│   │   └── breakout.py     # N日突破（趋势）
│   ├── backtest/
│   │   └── engine.py       # 回测引擎（ATR动态止盈止损 + 移动止损 + MTF确认）
│   ├── api/                # OKX REST/WebSocket 封装
│   └── utils/
│       └── indicators.py   # 统一指标库（SMA/EMA/RSI/MACD/Bollinger/ATR/KDJ/ADX/VolumeProfile）
│
├── scripts/                # 研究实验层（策略丰富，用于回测验证）
│   ├── rolling_window_10rounds.py       # 第3代：滚动窗口+市场自适应+组合风控（19策略配置）
│   ├── anti_overfit_10rounds.py         # 第2代：防过拟合+样本外验证（6策略类型）
│   ├── strategy_optimizer.py            # 第1代：4策略横向对比
│   ├── multi_asset_strategy.py          # 多币种轮动组合
│   └── run_production_backtest.py       # 生产框架回测脚本
│
└── backtest_reports/       # 回测报告（JSON + PNG）
```

### 策略能力3代演进

| 代次 | 核心特征 | 代表脚本 |
|---|---|---|
| 第1代 | 固定规则横向对比，固定仓位%/止损止盈 | strategy_optimizer.py |
| 第2代 | 样本外验证 + 过拟合检测，参数稳定性 | anti_overfit_10rounds.py |
| 第3代 | 市场状态自适应 + ATR动态风控 + 移动止损 + MTF确认 + 组合风控 + 手续费感知 | rolling_window_10rounds.py |

---

## 2. 生产层策略 (src/strategies/)

所有生产层策略继承 `BaseStrategy`，统一使用 `src/utils/indicators.py` 计算指标。

### 2.1 DoubleMA - 双均线交叉（趋势）

| 属性 | 值 |
|---|---|
| 文件 | `src/strategies/double_ma.py` |
| 适用市场 | trending（趋势市） |
| 核心指标 | EMA（快线 + 慢线 + 趋势线） |

**买入条件**: 快线EMA上穿慢线EMA（金叉）+ 价格在趋势EMA上方  
**卖出条件**: 快线EMA下穿慢线EMA（死叉）或价格跌破趋势EMA

**默认参数**:
- fast_period: 10, slow_period: 30, trend_period: 60
- position_pct: 20%, risk_pct: 2%
- ATR止损: 2.5x, ATR止盈: 5.0x, 移动止损: 盈利2%后激活

---

### 2.2 MACDCross - MACD金叉死叉（趋势）

| 属性 | 值 |
|---|---|
| 文件 | `src/strategies/macd_cross.py` |
| 适用市场 | trending（趋势市） |
| 核心指标 | MACD (DIF + DEA + Histogram) |

**买入条件**: MACD线上穿信号线（金叉）  
**卖出条件**: MACD线下穿信号线（死叉）

**默认参数**:
- fast: 12, slow: 26, signal: 9
- position_pct: 20%, risk_pct: 2%
- ATR止损: 2.5x, ATR止盈: 5.0x, 移动止损: 盈利2%后激活

**回测表现**: 生产框架回测中唯一正收益策略，ETH固定风控 +3.33% 最佳。

---

### 2.3 RSIBollinger - RSI + 布林带均值回归（震荡）

| 属性 | 值 |
|---|---|
| 文件 | `src/strategies/rsi_bollinger.py` |
| 适用市场 | ranging（震荡市） |
| 核心指标 | RSI + Bollinger Bands |

**买入条件**: RSI超卖（<30）+ 价格触及布林带下轨  
**卖出条件**: RSI超买（>70）+ 价格触及布林带上轨，或价格回归中轨

**默认参数**:
- rsi_period: 14, rsi_low: 30, rsi_high: 70
- boll_period: 20, boll_std: 2.0
- position_pct: 20%, risk_pct: 1.5%
- ATR止损: 2.0x, ATR止盈: 4.0x, 移动止损: 盈利1.5%后激活

---

### 2.4 Breakout - N日突破（趋势）

| 属性 | 值 |
|---|---|
| 文件 | `src/strategies/breakout.py` |
| 适用市场 | trending（趋势市） |
| 核心指标 | N日最高价 |

**买入条件**: 价格突破过去N根K线最高价  
**卖出条件**: 无显式卖出信号，完全由引擎ATR止损/止盈/移动止损管理

**默认参数**:
- lookback: 20
- position_pct: 20%, risk_pct: 2%
- ATR止损: 2.0x, ATR止盈: 4.0x, 移动止损: 盈利2%后激活

---

## 3. 研究层策略池 (scripts/rolling_window_10rounds.py)

研究层共 7 种策略类型、19 个策略配置。每个配置有不同的参数组合，在滚动窗口中通过 WFO（Walk-Forward Optimization）寻找最优风控参数。

### 3.1 趋势策略（10个配置）

> 趋势市表现好，震荡市容易反复打止损。止损较宽（ATR 2.0~3.0x），止盈 = 2x止损（盈亏比2:1）。

| 策略名 | 类型 | 关键参数 | 风控参数 |
|---|---|---|---|
| DoubleMA_5_20_50 | 双均线 | fast=5, slow=20, trend=50 | risk=2%, sl=3.0xATR, tp=6.0xATR |
| DoubleMA_7_25_60 | 双均线 | fast=7, slow=25, trend=60 | risk=1.5%, sl=3.0xATR, tp=6.0xATR |
| DoubleMA_10_30_70 | 双均线 | fast=10, slow=30, trend=70 | risk=2%, sl=2.5xATR, tp=5.0xATR |
| DoubleMA_15_40_80 | 双均线 | fast=15, slow=40, trend=80 | risk=2.5%, sl=2.5xATR, tp=5.0xATR |
| MACD_12_26_9 | MACD | fast=12, slow=26, signal=9 | risk=2%, sl=2.5xATR, tp=5.0xATR |
| MACD_8_17_5 | MACD | fast=8, slow=17, signal=5 | risk=1.5%, sl=2.5xATR, tp=5.0xATR |
| MACD_6_13_4 | MACD | fast=6, slow=13, signal=4 | risk=1%, sl=2.0xATR, tp=4.0xATR |
| EMAPullback_20_3 | EMA回踩 | ema=20, pullback=3% | risk=2%, sl=2.5xATR, tp=5.0xATR |
| EMAPullback_50_5 | EMA回踩 | ema=50, pullback=5% | risk=1.5%, sl=3.0xATR, tp=6.0xATR |
| Breakout_20 | 突破 | lookback=20 | risk=2%, sl=2.0xATR, tp=4.0xATR |
| Breakout_10 | 突破 | lookback=10 | risk=1.5%, sl=2.0xATR, tp=4.0xATR |

**EMA回踩策略逻辑**: 价格在EMA上方且回踩EMA一定百分比后反弹时买入。适合趋势中途回调入场。

### 3.2 震荡策略（6个配置）

> 震荡市低买高卖，趋势市容易逆势亏损。止损较窄（ATR 1.5~2.0x），快进快出。

| 策略名 | 类型 | 关键参数 | 风控参数 |
|---|---|---|---|
| RSI_Boll_14_20_2 | RSI+布林 | RSI=14, BB=20(2std), 30/70 | risk=1.5%, sl=2.0xATR, tp=4.0xATR |
| RSI_Boll_10_15_1.5 | RSI+布林 | RSI=10, BB=15(1.5std), 25/75 | risk=1%, sl=1.5xATR, tp=3.0xATR |
| RSI_Boll_7_10_1 | RSI+布林 | RSI=7, BB=10(1std), 20/80 | risk=1%, sl=1.5xATR, tp=3.0xATR |
| DCA_2pct | 定投抄底 | lookback=50, entry=2% | risk=1.5%, sl=2.0xATR, tp=4.0xATR |
| DCA_3pct | 定投抄底 | lookback=50, entry=3% | risk=1%, sl=2.0xATR, tp=4.0xATR |
| DCA_5pct | 定投抄底 | lookback=50, entry=5% | risk=1%, sl=2.0xATR, tp=4.0xATR |

**DCA（定投抄底）策略逻辑**: 价格低于50日均线一定百分比时分批买入，适合震荡市抄底。回测中表现最稳定（平均 +0.4%，正收益率 55.6%）。

### 3.3 自适应策略（2个配置）

> 两种市场都可运行，根据波动率自动切换行为模式。

| 策略名 | 关键参数 | 风控参数 |
|---|---|---|
| VolAdapt_20_3 | lookback=20, vol_threshold=3% | risk=2%, sl=2.0xATR, tp=4.0xATR |
| VolAdapt_30_5 | lookback=30, vol_threshold=5% | risk=1.5%, sl=2.5xATR, tp=5.0xATR |

**波动率自适应策略逻辑**: 低波动时按均值回归逻辑操作，高波动时按趋势跟随逻辑操作。

### 3.4 第2代研究脚本额外策略 (anti_overfit_10rounds.py)

第2代脚本中还包含一个 MeanReversion（均值回归）策略，核心逻辑是价格偏离均线一定程度后回归买入/卖出。

---

## 4. 风控体系

### 4.1 ATR 动态风控

ATR（Average True Range）衡量市场波动程度，用于动态调整止损止盈和仓位。

| 风控项 | 固定风控 | ATR 动态风控 |
|---|---|---|
| 止损 | 固定 -5% | 买入价 - ATR x sl_multiplier（波动大就放宽） |
| 止盈 | 固定 +10% | 买入价 + ATR x tp_multiplier（盈亏比固定2:1） |
| 仓位 | 固定 20% 本金 | (balance x risk_pct) / (ATR x multiplier) |

**仓位计算公式**: `amount = (balance * risk_pct) / (ATR * atr_multiplier)`  
波动大 -> ATR大 -> 仓位小；波动小 -> ATR小 -> 仓位大。

### 4.2 移动止损（Trailing Stop）

1. 盈利达到 `trailing_pct`（默认2%）后激活
2. 激活后，止损线 = 最高价 - ATR x sl_multiplier
3. 止损线只上移不下移，锁住利润

### 4.3 市场状态检测

综合 ADX 和波动率分位数判断：
- ADX > 25 -> trending（趋势市）
- ADX < 20 -> ranging（震荡市）
- 20-25 中性区间，看波动率分位辅助判断

### 4.4 多时间框架确认（MTF）

日K线买入信号需要 4H K线趋势确认（4H EMA20 > EMA50）。防止日K假信号。

### 4.5 手续费感知

| 检查项 | 参数 | 说明 |
|---|---|---|
| 最小波动倍数 | MIN_MOVE_MULTIPLIER = 2.0 | 预期波动(ATR/price)至少为往返手续费2倍 |
| 止盈费率比 | MIN_TP_FEE_RATIO = 5.0 | 止盈目标至少为往返手续费5倍（>=1%） |
| 最低信号强度 | MIN_SIGNAL_STRENGTH = 0.1 | 信号强度低于0.1不交易 |

---

## 5. 组合层面风控

仅在 `scripts/rolling_window_10rounds.py` 研究层实现（生产层待移植）。

### 5.1 槽位系统（Slot System）

- **MAX_POSITIONS = 3**: 同时最多持有3个币种
- 买入信号按信号强度排序，从最强开始逐个买入
- 槽位满后，剩余信号记为 `slot_blocked`

### 5.2 相关性过滤（Correlation Filter）

- **CORR_THRESHOLD = 0.7**（当前值，从0.9下调）
- 回测开始前用训练期数据计算币种间日收益率相关系数矩阵
- 新币种买入时，检查与已持仓币种的相关性
- |corr| > 阈值 -> 拦截买入，记为 `corr_blocked`
- 防止同时持有高度相关的币种（变相加仓）

### 5.3 信号强度计算

| 组成 | 权重 | 说明 |
|---|---|---|
| RSI超卖程度 | 0~0.4 | RSI越低越强 |
| 布林带位置 | 0~0.3 | 价格越接近下轨越强 |
| 偏离EMA20 | 0~0.3 | 价格低于均线越多越强 |
| 多策略共识 | +15%/策略 | 多个策略同时喊buy，信号强度加权提升 |

### 5.4 三道门过滤流程

```
买入信号 -> [1.手续费可行性] -> [2.槽位检查] -> [3.相关性检查] -> 执行买入
              |                    |               |
           fee_blocked         slot_blocked     corr_blocked
```

---

## 6. 可用技术指标

统一指标库位于 `src/utils/indicators.py`，所有策略共用：

| 指标 | 函数 | 说明 |
|---|---|---|
| 简单移动平均 | `SMA(prices, period)` | |
| 指数移动平均 | `EMA(prices, period)` | |
| 相对强弱指标 | `RSI(prices, period=14)` | |
| MACD | `MACD(prices, fast=12, slow=26, signal=9)` | 返回 {macd, signal, histogram} |
| 布林带 | `BollingerBands(prices, period=20, std_dev=2.0)` | 返回 {upper, middle, lower} |
| 真实波幅均值 | `ATR(highs, lows, closes, period=14)` | |
| KDJ指标 | `KDJ(highs, lows, closes, period=9)` | 返回 {k, d, j} |
| 平均趋向指标 | `ADX(highs, lows, closes, period=14)` | 用于市场状态检测 |
| 成交量分布 | `VolumeProfile(prices, volumes, bins=20)` | |

---

## 7. 回测结果汇总

### 回测1: 生产框架单币种单策略

- **脚本**: `scripts/run_production_backtest.py`
- **配置**: 4策略 x 3币种(BTC/ETH/SOL) x 2风控模式(ATR/固定) = 24组
- **数据**: 300天日K（2025-10 ~ 2026-08），OKX真实数据
- **初始资金**: 10,000 USDT

| 策略 | ATR平均 | 固定平均 | 最佳单次 |
|---|---|---|---|
| MACD | +0.28% | +0.79% | ETH固定 +3.33% |
| DoubleMA | -1.64% | -1.63% | ETH ATR -0.65% |
| Breakout | -3.43% | -2.37% | SOL固定 -0.30% |
| RSI+Boll | -4.63% | -1.34% | BTC固定 -1.03% |

**结论**: 仅4组盈利（全是MACD），ATR风控在震荡市不如固定风控。

### 回测2: 组合回测（相关性阈值0.7）

- **脚本**: `scripts/rolling_window_10rounds.py`
- **配置**: 6币种 x 19策略 x 10轮滚动窗口
- **数据**: 300天日K + 4H MTF确认
- **初始资金**: 1,000 USDT

| 策略类型 | 平均收益 | 正收益率 | 评价 |
|---|---|---|---|
| DCA | +0.4% | 55.6% | 稳定盈利 |
| RSI | +0.2% | 37.5% | 稳定盈利 |
| DoubleMA/MACD/Breakout/VolAdapt | 0.0% | 0% | 无交易（信号被拦截） |
| EMAPullback | -0.2% | 0% | 小亏 |

**各币种PnL**: XRP +14.65 > SOL +13.62 > BNB +7.16 > BTC +1.57 > ETH -4.31

**相关性拦截**: 2234/2516 信号被拦截（88.8%），0.7阈值对加密货币太激进（BTC-ETH=0.91）。

---

## 8. 如何运行回测

### 生产框架回测（单币种单策略）

```bash
python scripts/run_production_backtest.py
```
输出: `backtest_report_production.json` + `backtest_production_chart.png`

### 组合回测（多币种多策略滚动窗口）

```bash
python scripts/rolling_window_10rounds.py
```
输出: `backtest_reports/rolling_window_portfolio.json`

### 关键参数修改

| 参数 | 位置 | 说明 |
|---|---|---|
| CORR_THRESHOLD | rolling_window_10rounds.py:32 | 相关性过滤阈值 |
| MAX_POSITIONS | rolling_window_10rounds.py:31 | 最大同时持仓数 |
| SYMBOLS | rolling_window_10rounds.py:24 | 回测币种列表 |
| NUM_ROUNDS | rolling_window_10rounds.py:26 | 滚动窗口轮数 |
| TRAIN_RATIO | rolling_window_10rounds.py:27 | 训练集占比 |
| STRATEGY_POOL | rolling_window_10rounds.py:590 | 策略池配置 |

---

## 更新日志

| 日期 | 变更 | Commit |
|---|---|---|
| 2026-08-04 | 创建策略文档，覆盖4个生产策略+19个研究策略配置+风控体系+2轮回测结果 | - |
| 2026-08-04 | 第3代策略能力回流到生产框架，新增MACD/RSI+Boll/Breakout策略 | be2739a |
| 2026-08-04 | 相关性阈值从0.9调整到0.7，组合回测验证 | dd9fc40 |
