"""
行情监测框架 —— 基类与注册表

核心抽象：
- Signal        : 单个数据源归一化后的统一信号结构
- BaseSource    : 数据源基类（fetch_raw / parse / HTTP / 缓存 / 注册）
- RegimeBias    : 对「我们自己的策略」的交易含义（反向校准）
- register      : 装饰器，自动把数据源加入全局注册表
"""

import json
import logging
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import requests

try:
    from ..utils.logger import get_logger
except Exception:  # 兜底：独立运行时也能用
    get_logger = logging.getLogger


# ============================ 枚举 ============================

class Category(str, Enum):
    """数据源大类。"""
    SENTIMENT = "sentiment"      # 社交/情绪面（散户在说什么）
    DERIVATIVES = "derivatives"  # 衍生品/杠杆面（资金费率、未平仓）
    ONCHAIN = "onchain"          # 链上数据（真金白银在动）
    TECHNICAL = "technical"      # 技术面（价格/成交量）


class SignalDirection(str, Enum):
    """该信号原始指向（站在「人群」视角）。"""
    BULLISH = "bullish"    # 人群看涨/贪婪
    BEARISH = "bearish"    # 人群看跌/恐惧
    NEUTRAL = "neutral"


class RegimeBias(str, Enum):
    """合成后，对「我们自己策略」的交易含义（反向校准视角）。

    加密市场里情绪/杠杆极端值通常预示反转，所以这里取反向：
    - RISK_ON  : 市场极度恐惧/空头拥挤 → 对趋势策略反而是机会区，可偏多
    - RISK_OFF : 市场极度贪婪/多头拥挤 → 回调风险高，应减仓/谨慎
    - NEUTRAL  : 中性，按策略自身信号走
    """
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"


# ============================ 数据结构 ============================

@dataclass
class Signal:
    """统一信号结构。

    norm 约定：映射到 [-1, 1]，+1 表示「人群极度看涨/贪婪」，-1 表示「人群极度看跌/恐惧」。
    聚合器据此反向得出对策略的风险含义。
    """
    source: str
    category: str
    timestamp: datetime          # 该读数的时间（UTC）
    value: float                 # 原始数值
    norm: float                  # 归一化到 [-1, 1]
    direction: SignalDirection
    label: str                   # 人读解读
    meta: Dict[str, Any] = field(default_factory=dict)

    def is_error(self) -> bool:
        return bool(self.meta.get("error"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "norm": self.norm,
            "direction": self.direction.value,
            "label": self.label,
            "meta": self.meta,
        }


# ============================ 基类 ============================

class BaseSource(ABC):
    """数据源基类。

    子类只需实现：
        - fetch_raw() : 拉取原始数据（框架已负责 HTTP / 缓存 / 异常）
        - parse(raw)  : 把原始数据转成统一的 Signal
    其余（HTTP 请求、磁盘缓存、注册）由框架处理。
    """

    name: str = ""                 # 数据源唯一 id，如 "fear_greed"
    category: str = ""             # Category 的取值
    description: str = ""          # 一句话说明
    cache_ttl_hours: float = 6.0   # 缓存有效期（小时）

    def __init__(self, cache_dir: Optional[str] = None, timeout: int = 30):
        self.timeout = timeout
        root = Path(__file__).parent.parent.parent
        self.cache_dir = Path(cache_dir) if cache_dir else root / "data" / "market_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(f"monitor.{self.name or self.__class__.__name__}")

    # ---------- 必须由子类实现 ----------
    @abstractmethod
    def fetch_raw(self) -> Dict[str, Any]:
        """拉取原始数据，返回 dict（框架不假定结构）。"""
        raise NotImplementedError

    @abstractmethod
    def parse(self, raw: Dict[str, Any]) -> Signal:
        """把原始数据解析为统一 Signal。需自行处理 raw 中的错误标记。"""
        raise NotImplementedError

    # ---------- 框架提供的通用能力 ----------
    def _cache_path(self) -> Path:
        return self.cache_dir / f"{self.name}.json"

    def fetch(self) -> Signal:
        """拉取（优先用新鲜缓存）→ 解析 → 返回 Signal。"""
        raw = self._load_or_fetch()
        try:
            sig = self.parse(raw)
        except Exception as e:  # 解析失败也不让整个监视器崩
            self.logger.error(f"[{self.name}] parse 失败: {e}")
            sig = Signal(
                source=self.name, category=self.category,
                timestamp=datetime.now(timezone.utc), value=float("nan"),
                norm=0.0, direction=SignalDirection.NEUTRAL,
                label=f"解析异常: {e}", meta={"error": str(e)},
            )
        return sig

    def _load_or_fetch(self) -> Dict[str, Any]:
        cp = self._cache_path()
        if cp.exists():
            try:
                blob = json.loads(cp.read_text(encoding="utf-8"))
                fetched_at = datetime.fromisoformat(blob["fetched_at"])
                age_h = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600.0
                if age_h < self.cache_ttl_hours:
                    self.logger.debug(f"[{self.name}] 命中缓存（{age_h:.1f}h 前）")
                    return blob["raw"]
            except Exception:
                pass
        raw = self.fetch_raw()
        # 只缓存「成功」结果，避免把失败响应（ok=False / 网络错误）写进缓存
        # 导致后续一直命中坏数据。
        if isinstance(raw, dict) and raw.get("ok") is True:
            try:
                cp.write_text(json.dumps(
                    {"fetched_at": datetime.now(timezone.utc).isoformat(), "raw": raw},
                    ensure_ascii=False,
                ), encoding="utf-8")
            except Exception as e:
                self.logger.warning(f"[{self.name}] 写缓存失败: {e}")
        else:
            self.logger.warning(f"[{self.name}] 拉取失败，不写缓存（将下次重试）")
        return raw

    def _http_get(self, url: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """统一 HTTP GET，失败返回带 error 标记的 dict，不抛异常。

        代理策略：默认直连；若设置 CRYPTO_PROXY 环境变量（如 http://127.0.0.1:7897），
        则情绪数据源也走该代理（裸连 OKX 不稳定时使用）。
        """
        try:
            proxy = os.getenv("CRYPTO_PROXY", "")
            proxies = ({"http": proxy, "https": proxy} if proxy
                       else {"http": None, "https": None})
            resp = requests.get(
                url, params=params, timeout=self.timeout,
                proxies=proxies,
            )
            try:
                j = resp.json() if resp.content else {}
            except Exception:
                j = {}
            return {"ok": True, "status": resp.status_code, "json": j,
                    "text": resp.text[:500]}
        except Exception as e:
            self.logger.warning(f"[{self.name}] 请求失败 {url}: {e}")
            return {"ok": False, "error": str(e)}


# ============================ 注册表 ============================

REGISTERED_SOURCES: List[Type[BaseSource]] = []


def register(cls: Type[BaseSource]) -> Type[BaseSource]:
    """装饰器：把数据源类加入全局注册表（去重）。"""
    if cls not in REGISTERED_SOURCES:
        REGISTERED_SOURCES.append(cls)
    return cls


def all_sources() -> List[Type[BaseSource]]:
    """返回所有已注册的数据源类。"""
    return list(REGISTERED_SOURCES)


def get_source(name: str) -> Optional[Type[BaseSource]]:
    """按 name 取数据源类。"""
    for c in REGISTERED_SOURCES:
        if c.name == name:
            return c
    return None
