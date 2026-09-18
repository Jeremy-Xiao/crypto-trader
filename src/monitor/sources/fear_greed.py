"""
数据源：恐惧贪婪指数（Fear & Greed Index）

来源：Alternative.me 免费公开接口，无需认证、无需 API key。
- 接口：https://api.alternative.me/fng/?limit=2
- 返回 0~100：0=极度恐惧，100=极度贪婪；附带文字分类与前值（用于看趋势）

解读（对「我们自己的策略」）：
- 指数越高（人群越贪婪）→ 回调风险越高 → 反向视为 RISK_OFF（谨慎/减仓）
- 指数越低（人群越恐惧）→ 历史上是机会区 → 反向视为 RISK_ON
"""

from datetime import datetime, timezone

from ..base import (
    BaseSource,
    Category,
    Signal,
    SignalDirection,
    register,
)


@register
class FearGreedSource(BaseSource):
    name = "fear_greed"
    category = Category.SENTIMENT.value
    description = "Alternative.me 加密货币恐惧贪婪指数 (0=极度恐惧,100=极度贪婪)，免费免认证"
    cache_ttl_hours = 12.0

    URL = "https://api.alternative.me/fng/"

    def fetch_raw(self) -> dict:
        return self._http_get(self.URL, params={"limit": 2})

    def parse(self, raw: dict) -> Signal:
        if not raw.get("ok"):
            return Signal(
                source=self.name, category=self.category,
                timestamp=datetime.now(timezone.utc), value=float("nan"),
                norm=0.0, direction=SignalDirection.NEUTRAL,
                label=f"获取失败: {raw.get('error', '?')}",
                meta={"error": raw.get("error")},
            )

        data = raw.get("json", {}).get("data", [])
        if not data:
            return Signal(
                source=self.name, category=self.category,
                timestamp=datetime.now(timezone.utc), value=float("nan"),
                norm=0.0, direction=SignalDirection.NEUTRAL,
                label="接口返回空数据",
                meta={"error": "empty_data"},
            )

        cur = data[0]
        val = float(cur["value"])
        prev = float(data[1]["value"]) if len(data) > 1 else val
        ts = datetime.fromtimestamp(int(cur["timestamp"]), tz=timezone.utc)

        # 归一到 [-1, 1]：50 为中性
        norm = (val - 50.0) / 50.0
        direction = SignalDirection.BULLISH if val >= 50 else SignalDirection.BEARISH
        classification = cur.get("value_classification", "")
        trend = "↑" if val > prev else ("↓" if val < prev else "→")

        label = (f"恐惧贪婪指数 {val:.0f}（{classification}），"
                 f"前值 {prev:.0f} {trend}")

        return Signal(
            source=self.name, category=self.category,
            timestamp=ts, value=val, norm=norm,
            direction=direction, label=label,
            meta={"prev": prev, "classification": classification,
                  "url": self.URL},
        )
