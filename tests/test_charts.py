import xml.etree.ElementTree as ET

import pytest

from skolyoz_metrics.charts import (
    CHARTS,
    W,
    chart_ablation,
    chart_latency,
    nice_axis,
    render_charts,
    tw,
)
from skolyoz_metrics.results import load_benchmarks

SVG = "{http://www.w3.org/2000/svg}"


@pytest.fixture()
def charts(results, root):
    return render_charts(results, load_benchmarks(root))


def test_every_chart_exists_in_both_themes(charts):
    expected = {f"{n}.svg" for n in CHARTS} | {f"{n}-dark.svg" for n in CHARTS}
    assert set(charts) == expected


def test_svgs_are_well_formed_and_accessible(charts):
    for name, svg in charts.items():
        root = ET.fromstring(svg)
        assert root.tag == f"{SVG}svg", name
        assert root.get("width") == str(W) and root.get("role") == "img", name
        assert root.find(f"{SVG}title").text and root.find(f"{SVG}desc").text, name


def test_output_is_deterministic(results, root):
    benchmarks = load_benchmarks(root)
    assert render_charts(results, benchmarks) == render_charts(results, benchmarks)


def test_charts_follow_the_data(results, root):
    base = render_charts(results, load_benchmarks(root))
    assert "24.35 px" in base["ablation.svg"] and "3.55 px" in base["ablation.svg"]
    stage1_only = next(r for r in results["two_stage_ablation"]["rows"]
                       if r["id"] == "keypoint_error_median")
    stage1_only["stage1"] = 30.0
    changed = render_charts(results, load_benchmarks(root))
    assert "30 px" in changed["ablation.svg"] and "24.35 px" not in changed["ablation.svg"]


def test_themes_use_their_own_series_colours(charts):
    assert "#2a78d6" in charts["ablation.svg"] and "#2a78d6" not in charts["ablation-dark.svg"]
    assert "#3987e5" in charts["ablation-dark.svg"]
    assert "#fcfcfb" in charts["ablation.svg"] and "#1a1a19" in charts["ablation-dark.svg"]


def test_decision_chart_shows_the_wide_interval_for_4_of_4(charts):
    # Wilson lower bound for 4/4 is 51.0 %: the chart must say so instead of "100 %".
    assert "100.0 %  (51.0–100.0)" in charts["decision.svg"]


def test_latency_chart_needs_a_benchmark(results):
    assert chart_latency(results, [], "light") is None


def test_no_text_runs_off_the_canvas(charts):
    """Estimated text extents must stay inside the card (catches clipped labels)."""
    for name, svg in charts.items():
        for node in ET.fromstring(svg).iter(f"{SVG}text"):
            x, size = float(node.get("x")), float(node.get("font-size"))
            width = tw(node.text or "", size, int(node.get("font-weight", 400)))
            anchor = node.get("text-anchor", "start")
            left = x - (width if anchor == "end" else width / 2 if anchor == "middle" else 0)
            assert left >= 0, (name, node.text)
            assert left + width <= W - 8, (name, node.text, round(left + width))


def test_nice_axis_picks_round_steps():
    assert nice_axis(24.35 * 1.25, 4) == (40.0, [0, 10.0, 20.0, 30.0, 40.0])
    top, ticks = nice_axis(2.2, 4)
    assert top >= 2.2 and ticks[0] == 0 and all(t > 0 for t in ticks[1:])


def test_ablation_requires_stable_row_ids(results):
    for row in results["two_stage_ablation"]["rows"]:
        del row["id"]
    with pytest.raises(KeyError):
        chart_ablation(results, "light")
