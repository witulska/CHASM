"""Shared helpers for reproducible experiment drivers."""

from __future__ import annotations

from collections.abc import Iterable


def aggregate_decisions(results: Iterable[dict]) -> dict[str, float]:
    """Convert nested CHASM results into per-metric detection rates."""
    rows = list(results)
    if not rows:
        return {}
    metric_names = sorted(rows[0])
    return {
        name: sum(float(row[name]["decision"]) for row in rows) / len(rows)
        for name in metric_names
    }
