#!/usr/bin/env python
"""
抓取历史资金费率（funding rate），供回测计提永续持仓成本。

为什么需要：回测引擎此前完全没有 funding 模型——引擎里那个 borrow_rate_daily
只在 leverage>1 时生效（现货杠杆口径），而实盘跑的是 USDT 永续 1x 全仓，
每 8 小时结算一次资金费率，这是实盘真实存在而回测完全缺失的成本项。

数据源选择（2026-09-28 实测结论）：
  - OKX /api/v5/public/funding-rate-history  → ✅ 可用，但**只保留最近 3 个月**，拉不到长历史
  - Binance /fapi/v1/fundingRate             → ❌ 受限地区 451
  - Bybit /v5/market/funding/history         → ❌ CloudFront 地区封锁
  - Gate / Bitget                            → ✅ 可用（备选）
  - **data.binance.vision 公开数据仓库**      → ✅ 采用：月度 zip，覆盖 2021 至今，无频率限制

输出：data/<BASE>_funding.csv   列 = calc_time_ms, funding_time, rate, interval_hours
每次运行全量重建（月度文件很小，重跑成本低）。

用法：
    export HTTPS_PROXY=http://127.0.0.1:7897
    venv/bin/python scripts/fetch_funding_history.py --start 2021-01 --end 2026-09
"""
import argparse
import csv
import io
import os
import sys
import time
import zipfile
from datetime import datetime, date, timezone

import requests

# OKX 侧 instId -> 币安公开数据仓库的 symbol
SYMBOLS = {
    "BTC-USDT-SWAP": "BTCUSDT",
    "ETH-USDT-SWAP": "ETHUSDT",
    "SOL-USDT-SWAP": "SOLUSDT",
}
VISION = "https://data.binance.vision/data/futures/um"
OUT_DIR = "data"


def _get_zip_csv(url: str, session) -> str:
    """下载 zip 并返回其中 CSV 的文本；404 返回 None。"""
    for attempt in range(3):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                name = z.namelist()[0]
                return z.read(name).decode("utf-8")
        except Exception as e:
            if attempt == 2:
                print(f"    [skip] {url.split('/')[-1]}: {e}")
                return None
            time.sleep(1.0 * (attempt + 1))
    return None


def parse_rows(text: str) -> list:
    """解析币安 funding CSV -> [(ts_ms, rate, interval_hours)]"""
    out = []
    if not text:
        return out
    for line in text.strip().splitlines():
        parts = line.split(",")
        if len(parts) < 3 or not parts[0].isdigit():
            continue          # 跳过表头与异常行
        try:
            out.append((int(parts[0]), float(parts[2]), float(parts[1])))
        except ValueError:
            continue
    return out


def month_range(start: str, end: str):
    """生成 (year, month) 序列，闭区间。"""
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def fetch_symbol(bsym: str, start: str, end: str, session) -> dict:
    data = {}
    for y, m in month_range(start, end):
        url = f"{VISION}/monthly/fundingRate/{bsym}/{bsym}-fundingRate-{y:04d}-{m:02d}.zip"
        rows = parse_rows(_get_zip_csv(url, session))
        if rows:
            print(f"    {y}-{m:02d}: {len(rows)} 条")
        for ts, rate, iv in rows:
            data[ts] = (rate, iv)
        time.sleep(0.08)
    return data


def fetch_recent_daily(bsym: str, days: int, session) -> dict:
    """补最近若干天的日度文件（当月月度文件通常尚未发布）。"""
    from datetime import timedelta
    data = {}
    today = datetime.now(timezone.utc).date()
    for i in range(days, -1, -1):
        d = today - timedelta(days=i)
        url = f"{VISION}/daily/fundingRate/{bsym}/{bsym}-fundingRate-{d.isoformat()}.zip"
        rows = parse_rows(_get_zip_csv(url, session))
        for ts, rate, iv in rows:
            data[ts] = (rate, iv)
        time.sleep(0.08)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01", help="起始月份 YYYY-MM")
    ap.add_argument("--end", default=None, help="结束月份 YYYY-MM，默认上月")
    ap.add_argument("--daily-days", type=int, default=40,
                    help="额外补最近 N 天的日度文件（默认 40）")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    prev = (now.replace(day=1) - __import__("datetime").timedelta(days=1))
    end = args.end or f"{prev.year:04d}-{prev.month:02d}"

    os.makedirs(OUT_DIR, exist_ok=True)
    session = requests.Session()

    print(f"币安公开数据仓库 | 目标区间 {args.start} → {end}（+ 最近 {args.daily_days} 天）")
    for inst, bsym in SYMBOLS.items():
        print(f"\n[{inst}]")
        data = fetch_symbol(bsym, args.start, end, session)
        data.update(fetch_recent_daily(bsym, args.daily_days, session))
        if not data:
            print("  ⚠️ 无数据")
            continue
        base = inst.replace("-SWAP", "")
        path = os.path.join(OUT_DIR, f"{base}_funding.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["calc_time_ms", "funding_time", "rate", "interval_hours"])
            for ts in sorted(data):
                rate, iv = data[ts]
                iso = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
                w.writerow([ts, iso, f"{rate:.12f}", iv])
        vals = [v[0] for v in data.values()]
        avg = sum(vals) / len(vals)
        ts_sorted = sorted(data)
        first = datetime.fromtimestamp(ts_sorted[0] / 1000, tz=timezone.utc).date()
        last = datetime.fromtimestamp(ts_sorted[-1] / 1000, tz=timezone.utc).date()
        print(f"  ✅ {len(data)} 条  {first} → {last}")
        print(f"     均值 {avg*100:+.6f}%/8h  年化 {avg*3*365*100:+.2f}%")
        print(f"     落盘 {path}")


if __name__ == "__main__":
    sys.exit(main())
