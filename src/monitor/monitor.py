"""
行情聚合器（MarketMonitor）

把多个已接入的数据源信号，合成一个「市场状态」结论：
- crowd_score : 人群情绪综合分（各源 norm 的等权平均，[-1,1]）
- trading_bias: 反向校准后的交易含义（RISK_ON / RISK_OFF / NEUTRAL）
- risk_note   : 人读的风险解读

设计原则（呼应「情绪用来校准风险，不是精准择时」）：
人群越贪婪/多头越拥挤 → 对我们越应谨慎（RISK_OFF）；
人群越恐惧/空头越拥挤 → 对我们越是机会（RISK_ON）。
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from .base import BaseSource, RegimeBias, Signal, all_sources
from . import sources  # noqa: F401  触发数据源 @register 注册


@dataclass
class MarketSnapshot:
    """一次多源监测的聚合结果。"""
    signals: List[Signal]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def crowd_score(self) -> float:
        """人群情绪综合分：各信号 norm 的等权平均。"""
        vals = [s.norm for s in self.signals if s.norm == s.norm]  # 排除 nan
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def trading_bias(self) -> RegimeBias:
        c = self.crowd_score()
        if c >= 0.5:
            return RegimeBias.RISK_OFF
        if c <= -0.5:
            return RegimeBias.RISK_ON
        return RegimeBias.NEUTRAL

    def risk_note(self) -> str:
        c = self.crowd_score()
        bias = self.trading_bias()
        if bias == RegimeBias.RISK_OFF:
            return (f"人群综合情绪 {c:+.2f}（偏热/贪婪），回调风险偏高，"
                    f"建议减仓或给趋势策略加防护，勿追高。")
        if bias == RegimeBias.RISK_ON:
            return (f"人群综合情绪 {c:+.2f}（偏冷/恐惧），历史上是机会区，"
                    f"可偏多但需要确认信号，别一把梭。")
        return (f"人群综合情绪 {c:+.2f}（中性），无明显极端，"
                f"按策略自身信号正常交易即可。")

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at.isoformat(),
            "crowd_score": self.crowd_score(),
            "trading_bias": self.trading_bias().value,
            "risk_note": self.risk_note(),
            "signals": [s.to_dict() for s in self.signals],
        }

    def to_report(self) -> str:
        lines = []
        lines.append("=" * 64)
        lines.append("  行情监测快照（多源聚合）")
        lines.append("=" * 64)
        for s in self.signals:
            mark = "⚠" if s.is_error() else "✓"
            lines.append(f"  [{mark}] {s.source:<14} {s.label}")
            lines.append(f"      类别={s.category:<11} 原始={s.value} "
                         f"归一={s.norm:+.2f} 指向={s.direction.value}")
            detail = s.meta.get("detail")
            if detail:
                lines.append(f"      明细 {detail}")
        lines.append("-" * 64)
        lines.append(f"  人群综合情绪 crowd_score = {self.crowd_score():+.2f}")
        lines.append(f"  对策略的交易含义 trading_bias = {self.trading_bias().value}")
        lines.append(f"  风险解读：{self.risk_note()}")
        lines.append("=" * 64)
        return "\n".join(lines)


class MarketMonitor:
    """行情监视器：持有若干数据源，fetch_all 拉取并聚合。"""

    def __init__(self, sources: Optional[List[BaseSource]] = None,
                 use_registry: bool = True):
        if sources is not None:
            self.sources = sources
        elif use_registry:
            # 自动实例化所有已注册的数据源
            self.sources = [cls() for cls in all_sources()]
        else:
            self.sources = []

    def fetch_all(self) -> MarketSnapshot:
        signals: List[Signal] = []
        for src in self.sources:
            try:
                sig = src.fetch()
                signals.append(sig)
            except Exception as e:  # 单源失败不影响其他源
                try:
                    from ..utils.logger import get_logger
                except Exception:
                    import logging
                    get_logger = logging.getLogger
                get_logger("monitor").warning(
                    f"[{getattr(src, 'name', '?')}] fetch 失败: {e}")
        return MarketSnapshot(signals)
