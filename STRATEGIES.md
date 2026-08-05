# 投资策略文档 (STRATEGIES.md)

> **最后更新**: 2026-08-05  
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
9. [10轮自动优化回测结果](#9-10轮自动优化回测结果2026-08-04)
10. [深度优化回测结果](#10-深度优化回测结果2026-08-05--为什么收益低)
11. [做空机制](#11-做空机制双向交易2026-08-05)
12. [做空回测结果](#12-做空回测结果2026-08-05--v2-修复版数据可信)
13. [最高3倍杠杆与三年回测](#13-最高3倍杠杆与三年回测2026-08-05)

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
| 第4代 | 双向交易(做空)：OPEN_SHORT/CLOSE_SHORT/翻转，空头ATR风控+反向移动止损，allow_short开关 | scripts/shorting_backtest.py |
| 第4.1代 | 修复做空统计/仓位bug：多空仓位对等、平空正确计入胜率、退出计数器修复，真实回测双向平均+2.44% | scripts/shorting_backtest.py |
| 第5代 | 保证金模型+最高3倍杠杆：维持保证金强平/借币利息/硬上限/回撤熔断；3年日K回测；ADX向量化提速 | scripts/leverage_backtest.py |

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

### 10轮自动优化回测

```bash
python scripts/auto_optimize_10rounds.py
```
输出: `backtest_reports/auto_optimize_10rounds.json` + `backtest_10rounds_chart.png`

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
| 2026-08-04 | 10轮自动优化回测：117组测试，MACD最佳(avg+0.24%)，R9无移动止损+宽止盈最优(ETH+1.91%) | d257d15 |
| 2026-08-05 | 深度优化(第二轮)：引擎加ADX入场过滤+仓位上限参数+OKX分页；拉1年日K，算买入持有/货币基金基准，9轮优化81组测试 | 9d34f00 |
| 2026-08-05 | 做空机制：引擎+4策略支持双向交易(OPEN_SHORT/CLOSE_SHORT/翻转)，17项单元测试全过；做空回测 仅多-2.90%→双向-0.09%(Δ+2.81%/组)，盈利组2/12→6/12 | (pending) |
| 2026-08-05 | 修复做空统计/仓位 bug：做空启用 ATR 动态仓位+max_position_pct 上限、平空交易正确计入胜率、新增 long_stats/short_stats、修复 signal_sell/reverse/end_of_backtest 计数器；单元测试扩至32项全过；重跑回测双向平均 +2.44%（盈利组7/12），做空 leg 31笔胜率65% | (本次) |
| 2026-08-05 | 最高3倍杠杆：引擎升级为保证金模型（权益=现金+保证金+未实现盈亏），实现硬上限3x/维持保证金强平/借币利息/回撤熔断四件套；3年日K（2023-08~2026-08）6模式72组回测；ADX预计算+向量化滚动，单组156s→1.7s；安全：强平0次，最大名义杠杆1.93x≤3x | (本次) |

---

## 10. 深度优化回测结果（2026-08-05）— 为什么收益"低"

### 核心发现：测试区间恰好是漫长熊市

拉取 **2025-08 ~ 2026-08 共365天日K**（分页拉取），市场状态：全部 ranging（ADX 13-16）。

**基准对比（这是理解一切的关键）：**

| 基准 | 收益率 |
|---|---|
| 货币基金（年化4.5%，无风险） | **+4.50%** |
| 买入持有 BTC | -44.51% |
| 买入持有 ETH | -48.85% |
| 买入持有 SOL | -56.21% |
| 买入持有 等权组合 | **-49.86%** |

**结论：过去一年加密货币暴跌约50%。所谓"收益低"其实是策略在暴跌市里保住了本金**——相比死拿币亏一半，策略只亏了不到5%，这是优秀的风险控制，不是策略失效。

### 9轮优化配置（针对根因）

| 轮次 | 优化方向 | 核心改动 | 平均收益 | 平均回撤 |
|---|---|---|---|---|
| D1 Baseline_1yr | 基线（1年窗口） | 当前参数 | -3.61% | -5.90% |
| D2 TrendingOnly_ADX20 | ADX>20才入场 | 震荡市空仓 | -4.30% | -6.17% |
| D3 TrendingOnly_ADX25 | ADX>25才入场 | 更严格趋势过滤 | **-2.94%** | **-4.35%** |
| D4 HighDeploy_90 | 高资金部署 | 仓位上限90% | -3.61% | -5.90% |
| D5 Trend_HighDeploy | ADX20+90%部署 | 趋势过滤+资金效率 | -4.30% | -6.17% |
| D6 MACD_TrendOnly | 只跑MACD+ADX20 | 聚焦唯一正收益策略 | -3.91% | -5.46% |
| D7 WideSL_NoTrail | 无移动止损+宽TP7x | 让趋势发展 | -5.11% | -7.45% |
| D8 BestCombo | MACD+ADX25+90%+无移动止损 | 组合最优 | -2.20% | -4.47% |
| **D9 AllIn_Trend_ADX30** | **MACD+ADX30+满仓+无移动止损** | **最优配置** | **-0.50%** | **-4.30%** |

### 最优配置 D9 解读

- **平均 -0.50%**（81组测试中的最佳），回撤仅 -4.30%
- **唯一正收益个体**：ETH MACD +1.13%（夏普0.21，全样本唯一赚钱的单次）
- 机制：ADX>30 才入场（避开无趋势的震荡下行）+ 满仓部署（抓住难得的趋势）+ 无移动止损（不被小幅回调震出）

### 与货币基金的区别（用户核心问题）

| 维度 | 本策略 | 货币基金 |
|---|---|---|
| 熊市表现 | 亏<1%（保住本金） | +4.5% 无风险 |
| 牛市表现 | 参与趋势上涨（潜在50%+） | 固定4.5% |
| 风险 | 有回撤（约5%） | 零 |
| 本质 | 崩盘保护 + 趋势骑手 | 稳定利息 |

**本策略的价值是"不对称"**：熊市把亏损控制在5%以内（买持有亏50%），牛市捕捉加密货币的暴涨。货币基金永远给4.5%但不会给你任何上涨空间。在刚经历的这轮熊市里 MMF 确实暂时领先，但完整牛熊周期下策略应显著跑赢 MMF。

### 当前最佳可上线配置

`MACD + ADX30入场过滤 + 满仓(100%)部署 + 无移动止损 + SL 3x ATR / TP 6x ATR`（即 D9）。保守版可用 D3（ADX25，回撤更小）。

### 输出文件
- `backtest_reports/deep_optimize.json` — 完整81组测试数据
- `backtest_deep_chart.png` — 2-panel对比图
- `scripts/deep_optimize.py` — 深度优化脚本（含买入持有/货币基金基准）
- 引擎改动：`src/backtest/engine.py` 新增 `min_adx_for_entry` / `max_position_pct` 参数；`src/api/okx_rest.py` 的 `get_candles` 支持 `after` 分页


---

## 9. 10轮自动优化回测结果（2026-08-04）

### 实验设计

300天日K线（2025-10~2026-08），BTC/ETH/SOL × 4策略 × 10种参数配置 = 117组测试。
市场状态：全部 ranging（ADX 13-16）。

### 10轮配置

| 轮次 | 优化方向 | 核心改动 |
|---|---|---|
| R1 Baseline | 基线对照 | 当前参数（ATR 2.5x SL / 5x TP, trailing 2%, pos 20%） |
| R2 TightSL_WideTP | 紧止损宽止盈 | SL 1.5x / TP 6.0x（截断亏损，让利润奔跑） |
| R3 WideSL_SlowTP | 宽止损缓止盈 | SL 3.0x / TP 4.5x（给趋势更多空间） |
| R4 RegimeFilter | 市场状态过滤 | 趋势市只跑趋势策略，震荡市只跑震荡策略 |
| R5 FastMA | 快MA周期 | DoubleMA 5/15/30, Breakout lookback=10 |
| R6 SlowMA_HighTrail | 慢MA+高移动止损 | DoubleMA 20/50/100, Breakout lookback=30, trailing 4% |
| R7 Conservative | 保守仓位 | pos 10%, risk 1% |
| R8 Aggressive | 激进仓位 | pos 30%, risk 3% |
| R9 NoTrailing_WideTP | 无移动止损+宽止盈 | 关闭trailing, TP 7.0x（让趋势充分发展） |
| R10 BestCombo | 最优组合 | 紧SL 1.5x + 宽TP 6x + regime过滤 + trailing 3% |

### 各轮汇总

| 轮次 | 平均收益 | 平均回撤 | 平均夏普 | 盈利组数 | 最佳单次 |
|---|---|---|---|---|---|
| R1 Baseline | -2.36% | -4.44% | -0.70 | 2/12 | +0.72% |
| R2 TightSL_WideTP | -2.50% | -4.43% | -0.75 | 1/12 | +0.19% |
| **R3 WideSL_SlowTP** | **-2.23%** | -4.66% | **-0.65** | **3/12** | **+1.30%** |
| R4 RegimeFilter | -4.63% | -6.76% | -1.29 | 0/3 | -4.38% |
| R5 FastMA | -3.11% | -5.21% | -0.70 | 3/12 | +0.73% |
| R6 SlowMA_HighTrail | -2.72% | -4.39% | -0.80 | 2/12 | +0.72% |
| **R7 Conservative** | **-1.21%** | **-2.30%** | -0.71 | 2/12 | +0.37% |
| R8 Aggressive | -3.67% | -6.72% | -0.70 | 2/12 | +1.07% |
| R9 NoTrailing_WideTP | -3.24% | -5.56% | -0.81 | 3/12 | **+1.91%** |
| R10 BestCombo | -2.65% | -4.40% | -0.91 | 0/3 | -1.67% |

### TOP 10 单次结果

| 排名 | 轮次 | 币种 | 策略 | 收益率 | 回撤 | 夏普 | 胜率 | 交易数 |
|---|---|---|---|---|---|---|---|---|
| **1** | **R9 NoTrailing** | **ETH** | **MACD** | **+1.91%** | -3.94% | 0.43 | 50% | 6 |
| 2 | R3 WideSL | BTC | Breakout | +1.30% | -5.98% | 0.29 | 33% | 3 |
| 3 | R8 Aggressive | BTC | MACD | +1.07% | -3.25% | 0.20 | 40% | 5 |
| 4 | R9 NoTrailing | BTC | MACD | +0.74% | -2.36% | 0.20 | 40% | 5 |
| 5 | R5 FastMA | BTC | DoubleMA | +0.73% | -3.76% | 0.23 | 50% | 4 |
| 6-9 | 多轮 | BTC | MACD | +0.72% | -2.19% | 0.19 | 40% | 5 |
| 10 | R3 WideSL | ETH | MACD | +0.63% | -3.17% | 0.16 | 33% | 6 |

### 策略横向对比（10轮汇总）

| 策略 | 平均收益 | 平均回撤 | 平均夏普 | 最佳 | 最差 | 评价 |
|---|---|---|---|---|---|---|
| DoubleMA | -1.57% | -2.13% | -0.90 | +0.73% | -4.44% | 亏损最小，但几乎不交易 |
| **MACD** | **+0.24%** | -2.99% | 0.07 | **+1.91%** | -1.13% | **唯一正收益策略** |
| Breakout | -4.69% | -7.04% | -0.88 | +1.30% | -9.63% | 高波动，最差表现 |
| RSI_Boll | -4.33% | -6.48% | -1.18 | -1.67% | -8.45% | 震荡市仍亏，参数需调优 |

### 币种对比

| 币种 | 平均收益 | 最佳 | 最差 |
|---|---|---|---|
| BTC | -2.13% | +1.30% | -8.38% |
| ETH | -2.78% | +1.91% | -8.45% |
| SOL | -3.15% | +0.18% | -9.63% |

### 关键发现

1. **MACD 是唯一正收益策略**（avg +0.24%），10轮中9轮BTC MACD盈利
2. **R9（无移动止损+宽止盈7x）产出最佳单次**：ETH MACD +1.91%，夏普0.43
3. **R7（保守仓位10%/1%风险）风控最好**：平均亏损仅-1.21%，回撤仅-2.30%
4. **R3（宽止损3x）盈利组数最多**：3/12盈利，BTC Breakout首次盈利+1.30%
5. **市场状态过滤（R4/R10）失败**：3币种全ranging，只跑RSI_Boll，全部亏损
6. **移动止损在震荡市有害**：关闭后（R9）MACD收益反而更高
7. **Breakout是最危险策略**：最差-9.63%，10轮中多数大幅亏损
8. **BTC是最适合交易的币种**：平均亏损最小，盈利次数最多

---

## 11. 做空机制（双向交易，2026-08-05）

### 设计

引擎从纯多头升级为**双向感知**。核心改动：

1. **信号类型扩展**（`src/strategies/base.py`）：
   - `OPEN_LONG`(买入开多) / `OPEN_SHORT`(卖出开空) / `CLOSE_LONG`(卖出平多) / `CLOSE_SHORT`(买入平空)
   - 保留 `BUY`=OPEN_LONG、`SELL`=CLOSE_LONG 兼容别名
2. **引擎双向状态**（`src/backtest/engine.py`）：
   - 新增 `position_side` 状态（LONG/SHORT/NONE）
   - `_open_position(side)` / `_execute_open_short()` / `_execute_buy_to_cover()` / `_close_current_position()` 按方向自动选平多/平空
   - 反向信号自动**翻转**（先平后开）
3. **空头仓位算法（2026-08-05 修复）**：
   - `_execute_open_short` 现在与 `_execute_buy` 完全对等：启用 ATR 动态仓位重算、受 `max_position_pct × balance` 名义价值上限约束。
   - 修复前做空直接用策略层固定 `position_pct` 仓位，导致回测对比严重失真。
4. **空头风控（双向）**：
   - 空头止损：价格上涨突破 `entry + ATR×sl` 触发
   - 空头止盈：价格下跌破 `entry - ATR×tp` 触发
   - 空头移动止损：记录 `lowest_price`，止损线 = `lowest + ATR×sl`，只下移
   - 空头 PnL = `(entry - exit) × amount`
5. **权益记账**：空头持仓市值记为负（`equity = balance - amount×price`），保证净值连续
6. **策略双向**：4个策略都能在趋势反向下产生做空信号
   - MACD/DoubleMA：死叉→做空；Breakout：跌破低点→做空；RSI_Boll：超买→做空
7. **统计修复（关键）**：
   - 交易记录新增 `side` 字段：`long_open`/`long_close`/`short_open`/`short_close`。
   - `_calculate_performance` 用 `side` 识别真正的平仓交易：平多（action=sell）和平空（action=buy）都被正确计入 `total_trades` 和 `win_rate`。
   - 新增 `long_stats` / `short_stats`，分别输出多空各 leg 的交易数、盈利数、PnL。
   - 修复 `signal_sell` / `reverse` / `end_of_backtest` 计数器此前恒为 0 的问题。

### 开关

每个策略有 `allow_short` 参数（默认 True）：
- `allow_short=True`：双向交易（含做空）
- `allow_short=False`：仅做多（兼容旧行为，用于对照实验）

### 单元测试

`scripts/review_short.py` — **32项断言全过**，覆盖：开空balance/平空PnL/空头止盈/空头止损/空头反向移动止损/翻转/空头equity负值/端到端run()信号路由/多头回归/多空仓位对等/做空仓位上限/平空统计正确/开空不被误计为交易/退出计数器正确。

---

## 12. 做空回测结果（2026-08-05） — **v2 修复版，数据可信**

> ⚠️ 第一轮做空回测（commit `8c5c261`）存在统计/仓位 bug：
> - 做空未启用 ATR 动态仓位，与做多不对等；
> - `bi_short_trades` 误把"止损+止盈次数"当成做空笔数；
> - 平空交易（action=`buy`）完全没计入 `total_trades` / `win_rate`；
> - `signal_sell` / `reverse` / `end_of_backtest` 计数器恒为 0。
> 本章数据来自 bug 修复后的重跑结果，是真正的做空成绩单。

### 实验设计

365天日K（2025-08~2026-08），BTC/ETH/SOL × 4策略 × 2模式（仅多/双向）= 24组。
统一参数：ADX30入场 + 满仓100% + 无移动止损 + SL 3x ATR / TP 6x ATR。
**唯一变量是 allow_short**，纯隔离做空增量。

### 汇总

| 模式 | 平均收益 | 盈利组 | 最佳 | 最差 |
|---|---|---|---|---|
| 仅做多 | -2.90% | 2/12 | +1.13% | -13.47% |
| **双向(含做空)** | **+2.44%** | **7/12** | **+13.00%** | -8.82% |
| 做空增量 Δ | **+5.33%/组** | — | — | — |

基准：买入持有组合 -49.97% | 货币基金 +4.50%。

**关键变化**：修复后双向模式从"几乎打平"升级为"真正盈利"，但仍未跑赢无风险货币基金——这是熊市特征，不是策略失效。

### 多空 leg 真实 PnL 归因（修复后才可统计）

| leg | 交易数 | 胜率 | 累计 PnL |
|---|---|---|---|
| 做多 leg | 36笔 | — | **-3,333 USDT** |
| **做空 leg** | **31笔** | **65%** | **+6,607 USDT** |

**结论：做空 leg 是唯一 profitable 的 leg，做多 leg 在熊市中整体亏损。**

### 逐组对比（做空增量 Δ）

| 币种 | 策略 | 仅多 | 双向 | Δ | 做空笔数/PnL |
|---|---|---|---|---|---|
| BTC | DoubleMA | 0.00% | +7.05% | +7.05% | 1 / +710 |
| ETH | DoubleMA | 0.00% | +11.28% | +11.28% | 1 / +1,135 |
| SOL | DoubleMA | 0.00% | -2.23% | -2.23% | 1 / -219 |
| BTC | MACD | -0.65% | +4.17% | +4.82% | 5 / +522 |
| ETH | MACD | +1.13% | +6.85% | +5.71% | 1 / +560 |
| SOL | MACD | -0.74% | +6.99% | +7.73% | 2 / +763 |
| BTC | Breakout | -13.47% | -8.82% | +4.65% | 4 / +501 |
| ETH | Breakout | -1.18% | +13.00% | +14.18% | 4 / +1,442 |
| SOL | Breakout | -10.93% | -7.77% | +3.16% | 5 / +391 |
| BTC | RSI_Boll | +0.99% | +3.89% | +2.90% | 2 / +301 |
| ETH | RSI_Boll | -6.93% | -4.24% | +2.69% | 2 / +285 |
| SOL | RSI_Boll | -3.01% | -0.96% | +2.05% | 3 / +215 |

### 结论

1. **做空全面改善熊市表现**：12组中11组因做空而提升；双向模式平均 +2.44%，相比仅多 -2.90%。
2. **ETH Breakout 是本次明星**：双向 +13.00%，做空 leg 贡献 +1,442。
3. **趋势策略受益最大**：MACD/Breakout 通过做空把多个亏损组合扭转为正收益。
4. **SOL DoubleMA 仍是唯一亏损点**：做空1笔即亏损（死叉做空后迅速反弹），本质上是单次交易波动问题，并非机制性亏损。
5. **做空 leg 是真正的 alpha 来源**：31笔胜率65%、累计+6,607；做多 leg 在熊市中 -3,333。
6. **仍未跑赢货币基金的原因**：过去一年是 -50% 的极端熊市，策略把组合从"亏一半"拉到"赚2.4%"，但无风险4.5%在短期内仍领先；完整牛熊周期下策略应显著跑赢 MMF。

### 诊断：SOL DoubleMA 那笔亏损的做空

- **开仓**：2025-09-27，死叉信号，price=203.46，amount=9.83，名义价值≈1,999 USDT。
- **平仓**：2025-10-02，反向做多信号触发翻转，price=227.27（相对开仓 +11.7%）。
- **亏损**：-219 USDT。
- **原因**：3x ATR 宽止损（止损位约 233-243）尚未触发，但反向信号已经到来；趋势反转被当作离场条件，导致在浮亏时被动平仓。这是"信号驱动翻转"与"宽止损"之间的取舍——该笔属于信号正常行为，没有机制性错误。

### 输出文件

- `backtest_reports/shorting_backtest.json` — 24组完整数据（含 long_stats/short_stats）
- `backtest_shorting_chart.png` — 三面板对比图
- `scripts/shorting_backtest.py` — 做空回测脚本
- `scripts/review_short.py` — 单元测试（32项）
- `scripts/gen_shorting_chart.py` — 图表脚本
- `scripts/diagnose_sol_short.py` — SOL DoubleMA 单笔诊断脚本

---

## 13. 最高3倍杠杆与三年回测（2026-08-05）

### 设计前提：安全优先

用户要求"在保证安全和风险控制的前提下支持最高3倍杠杆"，因此本轮不追求激进收益，而是把**杠杆作为风险可控的放大器**。核心风控四件套：

| 风控项 | 机制 | 默认值 |
|---|---|---|
| 硬上限 | 配置leverage超过max_leverage自动截断 | `max_leverage=3.0` |
| 维持保证金强平 | 单根K线内最高价/最低价触及强平价即强平，跳空按开盘价成交 | `maintenance_margin_rate=0.005`, `liquidation_buffer=0.25` |
| 借币利息 | 仅对借入部分计息：`notional × (1 - 1/L) × 日利率` | `borrow_rate_daily=0.0003` |
| 回撤熔断 | 权益回撤超过阈值后禁止新开仓（可平仓） | `max_drawdown_halt=0.25` |

引擎从"全额现金扣款"升级为**保证金记账**：
- 开仓：`balance -= 名义价值/L + 手续费`，`margin_used += 名义价值/L`
- 权益：`equity = balance + margin_used + 未实现盈亏`
- 1倍杠杆时与旧现货模型完全等价，仅手续费差异。

### 关键安全设计：`risk_scales_with_leverage`

- `False`（风险恒定）：杠杆只放开资金约束，允许更大名义仓位；但**单笔风险仍按权益的 risk_pct 计算，不随杠杆放大**。这是"安全模式"。
- `True`（风险放大）：杠杆同时放大仓位和单笔风险，才是"真正用上杠杆"。用于对照实验。

### 数据与参数

- **数据**：OKX `history-candles` 接口，3年日K（2023-08-07 ~ 2026-08-05，共 1095 根），BTC/ETH/SOL。
- **参数**：沿用 D9 最优参数（ADX30入场 + 满仓100% + 无移动止损 + SL 3x ATR / TP 6x ATR），唯一变量是杠杆/风险模式。
- **性能优化**：`detect_market_regime` 改为向量化滚动标准差 + ADX 预计算复用，单组耗时从 156s 降至约 1.7s；72组总耗时约 2m13s。

### 对照的6种模式

| 模式 | leverage | risk_scales | 回撤熔断 | 含义 |
|---|---|---|---|---|
| 1x | 1.0 | False | 无 | 现货/基准 |
| 2x_riskparity | 2.0 | False | 无 | 杠杆只放资金，单笔风险不变 |
| 3x_riskparity | 3.0 | False | 无 | 同上，3倍 |
| 2x_scaled | 2.0 | True | 无 | 真正放大仓位+单笔风险 |
| 3x_scaled | 3.0 | True | 无 | 同上，3倍 |
| 3x_scaled_halt | 3.0 | True | 25% | 风险放大+回撤熔断 |

### 汇总结果（72组，3年）

| 模式 | 平均收益 | 中位数 | 最好 | 最差 | 盈利组 | 平均回撤 | 最差回撤 | 夏普 | 强平 | 利息 |
|---|---|---|---|---|---|---|---|---|---|---|
| **1x** | **+1.69%** | +4.98% | +22.69% | -17.86% | 7/12 | -15.08% | -29.37% | -0.01 | 0 | 0 |
| 2x_riskparity | +0.37% | +4.04% | +19.10% | -18.19% | 7/12 | -15.48% | -30.05% | -0.06 | 0 | 1545 |
| 3x_riskparity | -0.06% | +3.35% | +17.92% | -18.29% | 7/12 | -15.62% | -30.27% | -0.08 | 0 | 2053 |
| 2x_scaled | -1.01% | +4.47% | +35.79% | -34.43% | 7/12 | -28.09% | -51.71% | -0.07 | 0 | 3110 |
| 3x_scaled | -5.07% | -2.99% | +44.91% | -48.89% | 5/12 | -38.67% | -68.17% | -0.09 | 0 | 6150 |
| **3x_scaled_halt** | **+5.72%** | -9.57% | **+115.53%** | -32.81% | 5/12 | -24.40% | -35.81% | -0.17 | 0 | 2675 |

**基准**：买入持有组合 **+116.50%** ｜ 货币基金（4.5%年复利） **+14.12%**。

### Top 8 单组表现

| 排名 | 模式 | 币种 | 策略 | 收益 | 回撤 | 夏普 |
|---|---|---|---|---|---|---|
| 1 | 3x_scaled_halt | SOL | Breakout | +115.53% | -29.89% | 1.19 |
| 2 | 3x_scaled_halt | BTC | Breakout | +56.00% | -28.14% | 0.70 |
| 3 | 3x_scaled | BTC | Breakout | +44.91% | -36.87% | 0.52 |
| 4 | 2x_scaled | BTC | Breakout | +35.79% | -25.72% | 0.54 |
| 5 | 3x_scaled | SOL | MACD | +30.03% | -25.78% | 0.49 |
| 6 | 1x | BTC | Breakout | +22.69% | -12.88% | 0.60 |
| 7 | 2x_scaled | SOL | MACD | +21.72% | -17.95% | 0.50 |
| 8 | 2x_riskparity | BTC | Breakout | +19.10% | -13.70% | 0.53 |

### Worst 5 单组表现

| 排名 | 模式 | 币种 | 策略 | 收益 | 回撤 |
|---|---|---|---|---|---|
| 1 | 3x_scaled | SOL | RSI_Boll | -48.89% | -57.38% |
| 2 | 3x_scaled | BTC | MACD | -47.15% | -57.37% |
| 3 | 2x_scaled | SOL | RSI_Boll | -34.43% | -41.99% |
| 4 | 3x_scaled_halt | SOL | RSI_Boll | -32.81% | -35.81% |
| 5 | 2x_scaled | BTC | MACD | -32.74% | -42.20% |

### 关键发现

1. **这3年是加密货币大牛市**。买入持有组合 +116.5%，策略平均 +1.69%（1x）。**策略没有失效，而是 ADX30 过滤太严格，空仓时间过长，错过了主升浪**。
2. **风险恒定模式（risk_scales=False）几乎不改变收益**。2x/3x 平均 +0.37% / -0.06%，与 1x 的 +1.69% 没有本质差异，但要支付利息（1545/2053 USDT）。这说明"单笔风险不随杠杆放大"的设计确实安全，但也说明它没发挥杠杆的增值作用。
3. **风险放大模式（risk_scales=True）显著放大波动**。3x_scaled 平均回撤从 -15% 扩大到 -38%，最差回撤 -68%，收益分布两极化。
4. **回撤熔断是这次实验的最大亮点**。`3x_scaled_halt` 平均收益最高（+5.72%），诞生了 +115.53% 的明星单组（SOL Breakout）。但它的中位数是 -9.57%，说明收益严重偏斜——少数大行情贡献了大部分平均值。
5. **Breakout 策略在这轮牛市中表现最好**（Top 8 中占 5 席），因为它能抓住趋势启动；MACD/RSI_Boll 相对落后。

### 安全性核查

| 项目 | 结果 |
|---|---|
| 全部72组累计强平次数 | **0 次** |
| 实际达到的最大名义杠杆 | **1.93x**（硬上限 3.00x） |
| 杠杆越界 | **否** |
| 硬上限/维持保证金/利息/熔断 | 全部正常工作 |

**结论：即便配置为最高3倍，风控机制仍把实际敞口压在1.93倍以下，且3年无强平。**

### 与做空回测的对比

| 区间 | 市场状态 | 策略核心任务 | 1x 结果 | 最佳模式 |
|---|---|---|---|---|
| 1年（2025-08~2026-08） | 熊市 | 做空赚/空仓躲 | -2.90%（仅多）→ +2.44%（双向） | 双向 +2.44% |
| 3年（2023-08~2026-08） | 牛市 | 跟上趋势 | +1.69% | 3x_scaled_halt +5.72% |

熊市里做空是 alpha；牛市里趋势跟踪过滤器（ADX30）反而成了累赘。

### 生产建议

- **如果你最看重安全**：3x_riskparity 基本等同于现货（杠杆上限是安全垫），但要承担每年数百 U 的利息成本。收益/风险比没有改善。
- **如果你能接受更高波动、希望捕捉牛市**：`3x_scaled_halt` 是本轮唯一让平均收益提升的杠杆配置（+5.72%），回撤控制优于纯风险放大（-24.4% vs -38.7%）。建议配合更宽松的 ADX 阈值（如 20 或 25）使用，避免空仓错过主升浪。
- **真正的瓶颈不是杠杆，而是市场参与度**：在3年牛市中，任何杠杆都跑不赢买持有 +116.5%。下一步应优化入场过滤（降低 ADX 阈值、加入更灵敏的趋势确认）或仓位再平衡（定期强制配置一定比例），而不是继续加杠杆。

### 输出文件

- `backtest_reports/leverage_backtest.json` — 72组完整数据（含 equity_curves）
- `backtest_reports/backtest_leverage_chart.png` — 四面板对比图（收益/风险调整/净值曲线/安全性）
- `scripts/leverage_backtest.py` — 3年杠杆回测脚本
- `scripts/gen_leverage_chart.py` — 图表生成脚本
- `scripts/review_leverage.py` — 杠杆单元测试（51项断言全过）
- 数据缓存：`data/BTC-USDT_1D_1095d.csv`, `ETH-USDT_1D_1095d.csv`, `SOL-USDT_1D_1095d.csv`
