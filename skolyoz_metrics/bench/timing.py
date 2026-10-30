"""Wall-clock timing of a callable."""

from __future__ import annotations

import time
from collections.abc import Callable


def time_calls(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
    sync: Callable[[], None] | None = None,
) -> list[float]:
    """Run ``fn`` ``warmup`` times unmeasured, then ``repeat`` times measured.

    Returns per-call latencies in milliseconds. ``sync`` is called after each
    run so asynchronous devices (CUDA) are timed to completion.
    """
    if warmup < 0 or repeat < 1:
        raise ValueError("warmup must be >= 0 and repeat >= 1")
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        fn()
        if sync:
            sync()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples
