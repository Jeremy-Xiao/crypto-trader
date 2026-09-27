"""
永续资金费率（funding）数据加载与对齐。

用途：给回测引擎提供逐 bar 的 funding 费率序列。

背景（2026-09-28）：回测引擎原本没有任何 funding 模型，而实盘跑的是 USDT 永续 1x 全仓，
每 8 小时结算一次资金费率。数据由 scripts/fetch_funding_history.py 从币安公开数据仓库
（data.binance.vision）拉取，落盘为 data/<BASE>_funding.csv。

设计取舍：
- 日线 bar：一根 bar 覆盖 24h，含 3 次结算 → 该 bar 的费率 = 当日 3 次之和。
- 历史缺口：用 0.0 兜底，避免 NaN 污染权益（缺口期等价于「不计 funding」，与旧行为一致）。
- 时区：统一按 UTC 日期聚合，与回测日线数据口径一致。
"""
import csv
import os
from collections import defaultdict
from datetime import date, datetime, timezone

DEFAULT_DIR = "data"


def _to_date(value) -> date | None:
    """把 bar 的时间戳（字符串 / datetime / date / pandas Timestamp）转成 date。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        s = value.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(s[:19], fmt).date()
            except ValueError:
                continue
    return None


def load_daily_funding(symbol: str, data_dir: str = DEFAULT_DIR) -> dict:
    """读 funding CSV，按 UTC 日期聚合。返回 {date: 当日费率合计}，无文件返回 {}。"""
    path = os.path.join(data_dir, f"{symbol}_funding.csv")
    if not os.path.exists(path):
        return {}
    daily = defaultdict(float)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ms = int(row["calc_time_ms"])
                daily[datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()] += \
                    float(row["rate"])
            except (KeyError, TypeError, ValueError):
                continue
    return dict(daily)


def load_funding_series(symbol: str, timestamps, data_dir: str = DEFAULT_DIR):
    """生成与 bar 对齐的 funding 费率序列，可直接传给 BacktestConfig.funding_series。

    Args:
        symbol: 如 "BTC-USDT"（对应 data/BTC-USDT_funding.csv）
        timestamps: 逐 bar 的时间戳序列（与回测数据同序）

    Returns:
        与 timestamps 等长的 list[float]；若数据文件不存在则返回 None（回测将不计 funding）。
    """
    daily = load_daily_funding(symbol, data_dir)
    if not daily:
        return None
    series = []
    missing = 0
    for ts in timestamps:
        d = _to_date(ts)
        if d is None:
            series.append(0.0)
            continue
        if d in daily:
            series.append(daily[d])
        else:
            series.append(0.0)
            missing += 1
    if missing == len(series):
        return None          # 完全没对上，视为无数据
    return series


def summarize(series, label: str = "funding") -> str:
    """把序列折算成年化费率，便于人工核对。"""
    if not series:
        return f"{label}: 无数据"
    valid = [v for v in series if v == v]
    if not valid:
        return f"{label}: 无有效数据"
    avg_per_bar = sum(valid) / len(valid)
    annual = avg_per_bar * 365 * 100          # 每 bar 已含当日全部结算
    nonzero = sum(1 for v in valid if v != 0)
    return (f"{label}: bar数={len(series)} 非零={nonzero} "
            f"均值={avg_per_bar*100:+.6f}%/bar 年化={annual:+.2f}%")
