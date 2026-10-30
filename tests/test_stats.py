import pytest

from skolyoz_metrics.stats import percentile, summarize, wilson_interval


def test_percentile_interpolates_linearly():
    data = [1, 2, 3, 4]
    assert percentile(data, 0) == 1
    assert percentile(data, 50) == pytest.approx(2.5)
    assert percentile(data, 100) == 4
    assert percentile(data, 90) == pytest.approx(3.7)


def test_percentile_is_order_independent_and_validates():
    assert percentile([4, 1, 3, 2], 50) == percentile([1, 2, 3, 4], 50)
    assert percentile([7], 99) == 7
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1, 2], 101)


def test_wilson_matches_hand_computation():
    # 15 of 16: phat .9375, z 1.96 -> centre .8528, half-width .1361
    low, high = wilson_interval(15, 16)
    assert low == pytest.approx(0.717, abs=1e-3)
    assert high == pytest.approx(0.989, abs=1e-3)


def test_wilson_is_not_degenerate_at_the_extremes():
    low, high = wilson_interval(4, 4)
    assert high == 1.0
    assert low == pytest.approx(4 / (4 + 1.959963984540054**2), abs=1e-9)  # 0.510
    low0, high0 = wilson_interval(0, 10)
    assert low0 == 0.0 and high0 > 0.2


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(5, 4)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_summarize_reports_tail_percentiles():
    s = summarize([float(i) for i in range(1, 101)])
    assert s["n"] == 100 and s["min"] == 1 and s["max"] == 100
    assert s["p50"] == pytest.approx(50.5)
    assert s["p99"] == pytest.approx(99.01)
    assert s["std"] == pytest.approx(29.011, abs=1e-3)
    assert summarize([3.0])["std"] == 0.0
