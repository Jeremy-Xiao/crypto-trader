"""
历史 trading_bias 序列构建（供回测回放真实监测信号）

监测器(MarketMonitor)本身只产出「当前快照」。要回测过滤器效果，需要把
trading_bias 还原成一段历史序列，对齐到日K。本模块负责：

  1. 恐惧贪婪指数(F&G) 全历史（Alternative.me, 免费免key, limit=3000 覆盖多年）
     → 归一 norm_fg = (val-50)/50   （+1=极度贪婪, -1=极度恐惧）
  2. OKX 历史资金费率（公共接口免key, 每币种约保留最近~90天）→ 日聚合均值
     → 归一（复用 FundingRateSource._normalize 的分段逻辑）
        （+1=多头拥挤/过热, -1=空头拥挤）
  3. 按日合并：crowd_score = 各可用 norm 等权均值；
     trading_bias 取反向校准（与 monitor.MarketSnapshot 完全一致）：
        crowd_score >= 0.5 → risk_off（别追高）
        crowd_score <= -0.5 → risk_on（别做空接刀）
        其余 → neutral

结果缓存到 data/market_cache/historical_bias.json，重复运行不重复抓网。
对齐到回测K线用 asof（取该日期及之前最近一条），保证无缝接入。
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "market_cache"
)
HISTORY_CACHE = os.path.join(CACHE_DIR, "historical_bias.json")

FG_URL = "https://api.alternative.me/fng/?limit=3000"
OKX_BASE = "https://www.okx.com"
OKX_FUNDING_SYMS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]


def _http_get(url: str, params: Optional[dict] = None) -> dict:
    try:
        r = requests.get(url, params=params, timeout=30,
                         proxies={"http": None, "https": None})
        try:
            j = r.json() if r.content else {}
        except Exception:
            j = {}
        return {"ok": True, "status": r.status_code, "json": j}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _fg_norm(val: float) -> float:
    return (val - 50.0) / 50.0


def _fr_norm(fr: float) -> float:
    # 与 sources/funding_rate.py 中 FundingRateSource._normalize 一致
    if fr >= 0:
        if fr <= 0.0001:
            return fr / 0.0001 * 0.3
        return min(1.0, 0.3 + (fr - 0.0001) / 0.0007 * 0.7)
    a = -fr
    if a <= 0.0001:
        return -a / 0.0001 * 0.3
    return -min(1.0, 0.3 + (a - 0.0001) / 0.0007 * 0.7)


def _fetch_fg_history() -> Dict[str, float]:
    """返回 {date_str: fg_norm}。F&G 每日一条。"""
    out: Dict[str, float] = {}
    raw = _http_get(FG_URL)
    if not raw.get("ok"):
        return out
    for row in raw.get("json", {}).get("data", []):
        try:
            val = float(row["value"])
            ts = int(row["timestamp"])
            d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            out[d] = _fg_norm(val)
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _fetch_okx_funding_history(sym: str, max_pages: int = 20) -> Dict[str, float]:
    """抓取某币种 OKX 历史资金费率，返回 {date_str: 当日均值费率}。"""
    out: Dict[str, List[float]] = {}
    before = None
    for _ in range(max_pages):
        params = {"instId": sym, "limit": 100}
        if before is not None:
            params["before"] = before
        raw = _http_get(f"{OKX_BASE}/api/v5/public/funding-rate-history", params)
        if not raw.get("ok"):
            break
        data = raw.get("json", {}).get("data", [])
        if not data:
            break
        for row in data:
            try:
                fr = float(row["fundingRate"])
                ft = int(row["fundingTime"]) / 1000.0
                d = datetime.fromtimestamp(ft, tz=timezone.utc).strftime("%Y-%m-%d")
                out.setdefault(d, []).append(fr)
            except (KeyError, ValueError, TypeError):
                continue
        # 翻页：取本页最早 fundingTime 作为下页 before
        try:
            earliest = min(int(r["fundingTime"]) for r in data)
        except (ValueError, KeyError):
            break
        if before is not None and earliest >= before:
            break
        before = earliest
        time.sleep(0.08)
    # 日聚合：每个日期取均值
    return {d: sum(v) / len(v) for d, v in out.items()}


def build_historical_bias(use_cache: bool = True) -> List[dict]:
    """构建历史 bias 序列（按日期升序的 list of dict）。

    每条: {date, fg_value, fg_norm, fr_norm, crowd_score, trading_bias}
    trading_bias ∈ {'risk_off','risk_on','neutral'}
    """
    if use_cache and os.path.exists(HISTORY_CACHE):
        try:
            blob = json.loads(open(HISTORY_CACHE, encoding="utf-8").read())
            if blob.get("ok"):
                return blob["data"]
        except Exception:
            pass

    fg = _fetch_fg_history()
    fund = {}
    for sym in OKX_FUNDING_SYMS:
        part = _fetch_okx_funding_history(sym)
        for d, fr in part.items():
            fund.setdefault(d, []).append(fr)

    # 合并到 F&G 的日期轴上（F&G 覆盖最全）
    rows: List[dict] = []
    for d in sorted(fg.keys()):
        fg_n = fg[d]
        fg_val = round((fg_n * 50.0) + 50.0, 1)
        fr_vals = fund.get(d)
        fr_n = None
        if fr_vals:
            fr_n = _fr_norm(sum(fr_vals) / len(fr_vals))
        norms = [fg_n]
        if fr_n is not None:
            norms.append(fr_n)
        crowd = sum(norms) / len(norms)
        if crowd >= 0.5:
            bias = "risk_off"
        elif crowd <= -0.5:
            bias = "risk_on"
        else:
            bias = "neutral"
        rows.append({
            "date": d,
            "fg_value": fg_val,
            "fg_norm": round(fg_n, 4),
            "fr_norm": (round(fr_n, 4) if fr_n is not None else None),
            "crowd_score": round(crowd, 4),
            "trading_bias": bias,
        })

    os.makedirs(CACHE_DIR, exist_ok=True)
    open(HISTORY_CACHE, "w", encoding="utf-8").write(
        json.dumps({"ok": True, "generated_at": datetime.now(timezone.utc).isoformat(),
                    "data": rows}, ensure_ascii=False)
    )
    return rows


def bias_for_dates(dates: List[str]) -> List[str]:
    """把回测K线的日期列表对齐到历史 trading_bias（asof 前向填充）。

    dates: ['2023-08-07', ...]（升序或不要求）
    返回等长的 trading_bias 字符串列表（'risk_off'/'risk_on'/'neutral'）。
    某日期之前若无任何 bias（理论上不会，F&G 覆盖多年），填 'neutral'。
    """
    rows = build_historical_bias()
    # 构建 (date, bias) 已按日期升序；用 asof 查找
    bias_by_date = {r["date"]: r["trading_bias"] for r in rows}
    sorted_dates = sorted(bias_by_date.keys())
    out = []
    for d in dates:
        # 找 <= d 的最大日期
        chosen = None
        for sd in sorted_dates:
            if sd <= d:
                chosen = sd
            else:
                break
        out.append(bias_by_date[chosen] if chosen else "neutral")
    return out


if __name__ == "__main__":
    rows = build_historical_bias()
    from collections import Counter
    c = Counter(r["trading_bias"] for r in rows)
    print(f"历史 bias 总条数: {len(rows)}  ({rows[0]['date']} ~ {rows[-1]['date']})")
    print("分布:", dict(c))
    # 抽样近期
    for r in rows[-5:]:
        print(r)
