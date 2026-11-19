"""Seed statistics for the paper tables."""
from __future__ import annotations

import statistics
from typing import Iterable, Tuple


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    """Mean and sample standard deviation (ddof=1) of ``values``."""
    values = [float(v) for v in values]
    if len(values) < 2:
        raise ValueError(f"mean_std needs at least two values, got {values}")
    return statistics.fmean(values), statistics.stdev(values)
