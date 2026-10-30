import pytest

from skolyoz_metrics.bench.timing import time_calls


def test_runs_warmup_then_repeat_and_syncs_every_timed_call():
    calls, syncs = [], []
    samples = time_calls(lambda: calls.append(1), warmup=3, repeat=5, sync=lambda: syncs.append(1))
    assert len(calls) == 8  # 3 warm-up + 5 timed
    assert len(samples) == 5
    assert all(s >= 0 for s in samples)
    assert len(syncs) == 1 + 5  # once after warm-up, once per timed call


def test_rejects_nonsense_counts():
    with pytest.raises(ValueError):
        time_calls(lambda: None, warmup=0, repeat=0)
    with pytest.raises(ValueError):
        time_calls(lambda: None, warmup=-1, repeat=1)
