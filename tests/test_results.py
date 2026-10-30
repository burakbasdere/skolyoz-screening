import json

import pytest

from skolyoz_metrics.cli import main
from skolyoz_metrics.results import (
    consistency_errors,
    load_benchmarks,
    schema_errors,
)

pytest.importorskip("jsonschema")


def test_results_file_matches_its_schema(root, results):
    assert schema_errors(results, root / "metrics" / "results.schema.json") == []


def test_every_committed_benchmark_matches_its_schema(root):
    runs = load_benchmarks(root)
    assert runs, "expected at least one recorded benchmark under metrics/benchmarks/"
    for run in runs:
        body = {k: v for k, v in run.items() if k != "_file"}
        assert schema_errors(body, root / "metrics" / "benchmark.schema.json") == [], run["_file"]


def test_committed_results_are_internally_consistent(results):
    assert consistency_errors(results) == []


def test_schema_rejects_a_missing_source(root, results):
    del results["landmark_error"]["source"]
    errors = schema_errors(results, root / "metrics" / "results.schema.json")
    assert any("source" in e for e in errors)


def test_schema_rejects_an_unknown_status(root, results):
    results["parity"]["status"] = "probably-fine"
    assert schema_errors(results, root / "metrics" / "results.schema.json")


def test_consistency_catches_a_mistyped_recall(results):
    results["screening_decision"]["rows"][0]["recall"] = 0.985  # 15/16 is 0.938
    assert any("recall" in e for e in consistency_errors(results))


def test_consistency_catches_row_counts_that_do_not_add_up(results):
    results["landmark_error"]["rows"][0]["n"] += 1
    assert any("sum to" in e for e in consistency_errors(results))


def test_consistency_catches_median_above_p90(results):
    results["angle_error"]["rows"][0]["median_deg"] = 9.9
    assert any("median exceeds p90" in e for e in consistency_errors(results))


def test_consistency_requires_exactly_one_selected_export(results):
    for row in results["export_variants"]["rows"]:
        row["selected"] = False
    assert any("selected" in e for e in consistency_errors(results))


def test_check_command_passes_on_the_repository(root, capsys):
    assert main(["--root", str(root), "check"]) == 0
    assert "OK" in capsys.readouterr().out


def test_check_command_detects_stale_generated_docs(root, tmp_path, capsys):
    # Work on a copy so the real docs are never touched.
    import shutil

    copy = tmp_path / "repo"
    (copy / "docs").mkdir(parents=True)
    shutil.copytree(root / "metrics", copy / "metrics")
    for name in ("README.md", "README.tr.md"):
        shutil.copy(root / name, copy / name)
    shutil.copy(root / "docs" / "METRICS.md", copy / "docs" / "METRICS.md")
    assert main(["--root", str(copy), "check"]) == 0

    data = json.loads((copy / "metrics" / "results.json").read_text(encoding="utf-8"))
    data["landmark_error"]["rows"][0]["median_px"] = 3.046
    (copy / "metrics" / "results.json").write_text(json.dumps(data), encoding="utf-8")
    capsys.readouterr()
    assert main(["--root", str(copy), "check"]) == 1
    assert "stale" in capsys.readouterr().out
