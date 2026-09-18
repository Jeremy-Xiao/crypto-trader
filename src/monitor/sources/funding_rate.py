"""
数据源：永续合约资金费率（Funding Rate）

信号含义：资金费率是多头向空头（或反向）支付的费用，是衍生品市场杠杆情绪的
最直接温度计。
- 费率持续为正且偏高 → 多头拥挤、杠杆过热 → 历史上常领先回调 → 反向 RISK_OFF
- 费率持续为负 → 空头拥挤 → 反向 RISK_ON

接口选择：
- 默认用 **OKX** 公共接口（/api/v5/public/funding-rate），**免 API key、无地区限制**。
  聚合 BTC/ETH/SOL 三个永续合约费率，取均值代表"全市场杠杆情绪"，比单看一个币稳。
- Binance 公共接口（fapi/v1/premiumIndex）在受限地区返回 451，已降为备选。
- Coinglass 免费层现已需 key；如需切换，可传入 provider="coinglass" 与 coinglass_api_key。

归一化：以 0.01%（0.0001）为温和阈值、0.08%（0.0008）为极端阈值，做分段映射，
最终截断到 [-1, 1]。
"""

from datetime import datetime, timezone
from typing import Any

from ..base import (
    BaseSource,
    Category,
    Signal,
    SignalDirection,
    register,
)


@register
class FundingRateSource(BaseSource):
    name = "funding_rate"
    category = Category.DERIVATIVES.value
    description = "永续合约资金费率（OKX 公共接口免key），正=多头拥挤/回调风险"
    cache_ttl_hours = 3.0

    OKX_BASE = "https://www.okx.com"
    BINANCE_BASE = "https://fapi.binance.com"
    # Coinglass 备选（需 key）：https://open-api.coinglass.com/public/v2/funding_rate_chart
    SYMBOLS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]

    def __init__(self, symbols: Any = None, provider: str = "okx",
                 coinglass_api_key: str = "", **kw):
        super().__init__(**kw)
        # 接受单字符串或列表；统一转成 OKX 永续 instId 列表
        if symbols is None:
            self.symbols = list(self.SYMBOLS)
        elif isinstance(symbols, str):
            self.symbols = [symbols]
        else:
            self.symbols = list(symbols)
        self.provider = provider
        self.coinglass_api_key = coinglass_api_key

    def fetch_raw(self) -> dict:
        if self.provider == "coinglass":
            return self._fetch_coinglass()
        if self.provider == "binance":
            return self._fetch_binance()
        return self._fetch_okx()

    # ---------- OKX（默认，最稳） ----------
    def _fetch_okx(self) -> dict:
        rates = {}
        for sym in self.symbols:
            url = f"{self.OKX_BASE}/api/v5/public/funding-rate"
            r = self._http_get(url, params={"instId": sym})
            if not r.get("ok"):
                continue
            data = r.get("json", {}).get("data")
            if not data:
                continue
            try:
                rates[sym] = float(data[0].get("fundingRate", "nan"))
            except (TypeError, ValueError):
                continue
        if not rates:
            return {"ok": False, "error": "OKX 未返回任何费率"}
        return {"ok": True, "rates": rates}

    # ---------- Binance（备选，受限地区返回451） ----------
    def _fetch_binance(self) -> dict:
        rates = {}
        for sym in self.symbols:
            # Binance 现货/永续代号不带 -SWAP
            bsym = sym.replace("-SWAP", "")
            url = f"{self.BINANCE_BASE}/fapi/v1/premiumIndex"
            r = self._http_get(url, params={"symbol": bsym})
            if not r.get("ok"):
                continue
            d = r.get("json", {})
            if isinstance(d, list):
                d = d[-1] if d else {}
            if not isinstance(d, dict) or "lastFundingRate" not in d:
                continue
            try:
                rates[sym] = float(d["lastFundingRate"])
            except (TypeError, ValueError):
                continue
        if not rates:
            return {"ok": False, "error": "Binance 未返回任何费率"}
        return {"ok": True, "rates": rates}

    # ---------- Coinglass（需 key） ----------
    def _fetch_coinglass(self) -> dict:
        url = "https://open-api.coinglass.com/public/v2/funding_rate_chart"
        try:
            import requests
            resp = requests.get(
                url, params={"symbol": self.symbols[0], "interval": "4h"},
                headers={"coinglass-api-ey": self.coinglass_api_key},
                timeout=self.timeout,
            )
            return {"ok": True, "status": resp.status_code,
                    "json": resp.json() if resp.content else {},
                    "text": resp.text[:500]}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @staticmethod
    def _normalize(fr: float) -> float:
        # 分段映射：±0.0001 温和、±0.0008 极端，截断到 [-1, 1]
        if fr >= 0:
            if fr <= 0.0001:
                return fr / 0.0001 * 0.3
            return min(1.0, 0.3 + (fr - 0.0001) / 0.0007 * 0.7)
        else:
            a = -fr
            if a <= 0.0001:
                return -a / 0.0001 * 0.3
            return -min(1.0, 0.3 + (a - 0.0001) / 0.0007 * 0.7)

    def parse(self, raw: dict) -> Signal:
        if not raw.get("ok"):
            return self._error_signal(
                f"获取失败: {raw.get('error', '?')}", raw.get("error"))

        rates: dict = raw.get("rates", {})
        if not rates:
            return self._error_signal("返回结构异常（无费率）", "bad_structure")

        # 取均值代表全市场杠杆情绪
        vals = [v for v in rates.values() if v == v]
        avg_fr = float(sum(vals) / len(vals))
        norm = self._normalize(avg_fr)

        if avg_fr > 0:
            direction = SignalDirection.BULLISH
        elif avg_fr < 0:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        pct = avg_fr * 100.0
        if avg_fr >= 0.0005:
            note = "多头拥挤，警惕回调"
        elif avg_fr <= -0.0005:
            note = "空头拥挤，警惕反弹"
        else:
            note = "费率中性"

        per = ", ".join(f"{k.split('-')[0]}={v*100:.4f}%" for k, v in rates.items())
        label = f"资金费率均值 {pct:.4f}%（{note}）"
        detail = f"[{per}]"

        return Signal(
            source=self.name, category=self.category,
            timestamp=datetime.now(timezone.utc), value=avg_fr, norm=norm,
            direction=direction, label=label,
            meta={"rates": rates, "detail": detail,
                  "symbols": self.symbols, "provider": self.provider},
        )

    def _error_signal(self, label: str, err: Any) -> Signal:
        return Signal(
            source=self.name, category=self.category,
            timestamp=datetime.now(timezone.utc), value=float("nan"),
            norm=0.0, direction=SignalDirection.NEUTRAL,
            label=label, meta={"error": err},
        )
