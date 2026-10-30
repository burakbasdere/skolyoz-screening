"""Load and validate the curated metrics and the benchmark run files."""

from __future__ import annotations

import json
from pathlib import Path

RECALL_TOLERANCE = 0.0005 + 1e-9  # values are published rounded to 3 decimals


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(root: Path) -> dict:
    return read_json(root / "metrics" / "results.json")


def load_benchmarks(root: Path) -> list[dict]:
    """All benchmark runs, oldest first. Each gets a ``_file`` key with its name."""
    runs = []
    for path in sorted((root / "metrics" / "benchmarks").glob("*.json")):
        run = read_json(path)
        run["_file"] = path.name
        runs.append(run)
    return sorted(runs, key=lambda r: r.get("created_utc", ""))


def latest(benchmarks: list[dict], kind: str) -> dict | None:
    matching = [b for b in benchmarks if b.get("kind") == kind]
    return matching[-1] if matching else None


def schema_errors(instance: dict, schema_path: Path) -> list[str]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError(
            "jsonschema is required for validation: pip install -r requirements/dev.txt"
        ) from exc
    validator = Draft202012Validator(read_json(schema_path))
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    return [
        f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors
    ]


def consistency_errors(results: dict) -> list[str]:
    """Cross-field checks that a JSON schema cannot express."""
    problems: list[str] = []
    evaluations = results["evaluations"]

    for name, section in results.items():
        if isinstance(section, dict) and "evaluation" in section:
            if section["evaluation"] not in evaluations:
                problems.append(f"{name}: unknown evaluation '{section['evaluation']}'")

    cv = evaluations.get(results["landmark_error"]["evaluation"], {})
    total = sum(r["n"] for r in results["landmark_error"]["rows"])
    if "n_landmarks" in cv and total != cv["n_landmarks"]:
        problems.append(
            f"landmark_error: row counts sum to {total}, evaluation says {cv['n_landmarks']}"
        )
    for r in results["landmark_error"]["rows"]:
        if r["median_px"] > r["p90_px"]:
            problems.append(f"landmark_error[{r['type']}]: median exceeds p90")
    for r in results["angle_error"]["rows"]:
        if r["median_deg"] > r["p90_deg"]:
            problems.append(f"angle_error[{r['measurement']}]: median exceeds p90")

    n = results["screening_decision"]["n"]
    for r in results["screening_decision"]["rows"]:
        tag = f"screening_decision[{r['threshold_deg']}]"
        if r["true_positives"] > r["positives"]:
            problems.append(f"{tag}: more true positives than positives")
        if r["correct"] > n:
            problems.append(f"{tag}: more correct than n")
        if abs(r["true_positives"] / r["positives"] - r["recall"]) > RECALL_TOLERANCE:
            problems.append(
                f"{tag}: recall {r['recall']} != {r['true_positives']}/{r['positives']}"
            )
        if abs(r["correct"] / n - r["accuracy"]) > RECALL_TOLERANCE:
            problems.append(f"{tag}: accuracy {r['accuracy']} != {r['correct']}/{n}")

    selected = [r["variant"] for r in results["export_variants"]["rows"] if r["selected"]]
    if len(selected) != 1:
        problems.append(f"export_variants: expected exactly one selected variant, got {selected}")

    ids = [g["id"] for g in results["gaps"]]
    if len(ids) != len(set(ids)):
        problems.append("gaps: duplicate ids")
    return problems
