import pytest

from skolyoz_metrics.render import (
    proportion,
    render_headline,
    render_metrics_md,
    replace_block,
    table,
)
from skolyoz_metrics.results import load_benchmarks


def test_table_escapes_pipes_so_columns_do_not_shift():
    out = table(["a", "b"], [["x|y", "z"]])
    assert "x\\|y" in out
    assert out.splitlines()[2].count("|") - out.splitlines()[2].count("\\|") == 3


def test_proportion_includes_a_wilson_interval():
    assert proportion(15, 16) == "15/16 = 0.938 (0.717–0.989)"


def test_replace_block_swaps_only_the_marked_region():
    text = "keep\n<!-- headline:begin -->\nold\n<!-- headline:end -->\nkeep too\n"
    out = replace_block(text, "headline", "NEW")
    assert out == "keep\n<!-- headline:begin -->\nNEW\n<!-- headline:end -->\nkeep too\n"


def test_replace_block_fills_an_empty_block():
    out = replace_block("<!-- headline:begin -->\n<!-- headline:end -->\n", "headline", "X")
    assert out == "<!-- headline:begin -->\nX\n<!-- headline:end -->\n"


def test_replace_block_demands_markers():
    with pytest.raises(ValueError):
        replace_block("no markers here", "headline", "x")


def test_headline_numbers_come_from_the_json(results, root):
    benchmarks = load_benchmarks(root)
    en = render_headline(results, benchmarks, "en")
    assert "0.271° / 1.237°" in en
    assert "93.8 % (15/16; 95 % CI 71.7–98.9 %)" in en
    assert "6.44 MB" in en
    assert "2.69 M / 0.41 M" in en
    tr = render_headline(results, benchmarks, "tr")
    assert "ölçülen" in tr and "%95 GA" in tr and "Aksiller eğim hatası" in tr


def test_headline_follows_the_data(results, root):
    results["angle_error"]["rows"][0]["median_deg"] = 0.2
    assert "0.200° / 1.237°" in render_headline(results, load_benchmarks(root), "en")


def test_metrics_page_has_every_section_and_the_gaps(results, root):
    page = render_metrics_md(results, load_benchmarks(root))
    for heading in ("## 1. Doğruluk", "## 2. Dağıtım", "## 3. Gecikme", "## 4. Port",
                    "## 5. Dayanıklılık", "## 6. Veri kümesi", "## 7. Olumsuz sonuçlar",
                    "## 8. Bilinen boşluklar"):
        assert heading in page
    assert "cobb_estimate" in page and "unvalidated" in page
    assert "tanı aracı değildir" in page
    # Both protocols are described once, and every accuracy table points back to one.
    assert page.count("### Değerlendirme protokolleri") == 1
    assert "`cv5_stage2` — Aşama 2 için 5 katlı çapraz doğrulama" in page
    assert page.count("Protokol `holdout50`") == 3 and page.count("Protokol `cv5_stage2`") == 2


def test_metrics_page_embeds_every_chart_with_a_dark_variant(results, root):
    page = render_metrics_md(results, load_benchmarks(root))
    assert page.count("<picture>") == 5
    assert page.count("prefers-color-scheme: dark") == 5
    assert "img/latency-dark.svg" in page and 'alt="' in page


def test_ratio_columns_use_the_stored_numbers(results, root):
    page = render_metrics_md(results, load_benchmarks(root))
    assert "0.12" in page  # 0.271 / 2.21 error-to-signal for the axillary tilt
    assert "6.9×" in page  # 24.35 / 3.55
