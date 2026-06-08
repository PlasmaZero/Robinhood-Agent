"""Human-readable rendering of decisions and execution records for the CLI."""

from __future__ import annotations

from .engine import ExecutionRecord
from .strategy import BUY, HOLD, SELL, Decision

_ARROW = {BUY: "▲ BUY ", SELL: "▼ SELL", HOLD: "· HOLD"}


def _fmt_price(x: float) -> str:
    return f"{x:,.2f}" if x == x else "  n/a"  # x==x is False for NaN


def render_decisions(decisions: list[Decision], *, show_factors: bool = True) -> str:
    if not decisions:
        return "(no symbols)"
    rows = sorted(decisions, key=lambda d: d.confidence, reverse=True)
    out: list[str] = []
    out.append(f"{'SYMBOL':<8}{'SIGNAL':<8}{'CONF':>6}{'PRICE':>11}{'NET':>7}{'AGREE':>7}{'LIQ':>6}")
    out.append("-" * 53)
    for d in rows:
        out.append(
            f"{d.symbol:<8}{_ARROW.get(d.action, d.action):<8}"
            f"{d.confidence_pct:5.0f}%{_fmt_price(d.price):>11}"
            f"{d.weighted_score:>+7.2f}{d.agreement * 100:>6.0f}%{d.liquidity_gate * 100:>5.0f}%"
        )
    if show_factors:
        actionable = [d for d in rows if d.actionable]
        for d in actionable:
            out.append("")
            out.append(f"  {d.symbol} — {d.rationale}")
            for s in d.signals:
                bar = _evidence_bar(s.evidence)
                out.append(
                    f"    {s.name:<19}{s.score:>+5.2f} × {s.confidence:>4.2f}  {bar}  {s.rationale}"
                )
            if d.stop_price is not None and d.target_price is not None:
                out.append(
                    f"    entry≈{d.price:,.2f}  stop={d.stop_price:,.2f}  "
                    f"target={d.target_price:,.2f}  R:R={d.risk_reward:.2f}"
                )
    return "\n".join(out)


def _evidence_bar(evidence: float, width: int = 10) -> str:
    """A little -[====|    ]+ gauge of signed, confidence-weighted evidence."""
    half = width // 2
    filled = int(round(abs(evidence) * half))
    if evidence >= 0:
        return "[" + " " * half + "|" + "#" * filled + " " * (half - filled) + "]"
    return "[" + " " * (half - filled) + "#" * filled + "|" + " " * half + "]"


def render_records(records: list[ExecutionRecord]) -> str:
    if not records:
        return "(nothing to do)"
    out: list[str] = []
    icon = {"placed": "✓", "reviewed": "◷", "skipped": "·", "error": "✗"}
    for r in records:
        d = r.decision
        line = (
            f"{icon.get(r.status, '?')} {d.symbol:<6} {d.action:<5} "
            f"conf={d.confidence_pct:3.0f}%  {r.status:<8} {r.message}"
        )
        out.append(line)
        if r.order is not None and r.status in ("placed", "reviewed"):
            o = r.order
            qty = f"{o.quantity:g} sh" if o.quantity is not None else f"${o.dollar_amount:.2f}"
            px = f" @ {o.limit_price:.2f}" if o.limit_price is not None else " @ mkt"
            out.append(f"      {o.side} {qty}{px}  (~${o.notional:,.2f}, {o.order_type}, {o.time_in_force})")
    placed = sum(1 for r in records if r.status == "placed")
    reviewed = sum(1 for r in records if r.status == "reviewed")
    out.append("")
    out.append(f"Summary: {placed} placed, {reviewed} reviewed (dry-run), {len(records)} symbols.")
    return "\n".join(out)
