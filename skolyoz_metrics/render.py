"""Render docs/METRICS.md and the README headline tables from the JSON sources.

Nothing in the rendered output is typed by hand: every number comes from
``metrics/results.json`` or ``metrics/benchmarks/*.json``, and the confidence
intervals are computed from the stored counts. ``check`` mode (used by CI)
fails when a rendered file on disk differs from what the JSON would produce,
so a README can never silently drift from the data.
"""

from __future__ import annotations

import re
from pathlib import Path

from .results import latest
from .stats import wilson_interval

DASH = "—"
METRICS_MD = Path("docs") / "METRICS.md"
README_TARGETS = {Path("README.md"): "en", Path("README.tr.md"): "tr"}
BLOCK = "headline"

LABELS = {
    "en": {
        "metric": "Metric", "value": "Value", "evidence": "Evidence",
        "axillary_tilt": "Axillary-tilt error, median / p90",
        "coronal": "Coronal-balance error, median / p90",
        "axillary_landmark": "Axillary landmark error, median (label noise floor)",
        "recall": "Screening recall at ≥ 3° tilt (held-out split, n = 50)",
        "size": "Model size, both stages, float16",
        "params": "Parameters, Stage 1 / Stage 2",
        "latency": "Model forward, 1 × Stage 1 + 7 × Stage 2 (TFLite, CPU, median / p90)",
        "parity": "Kotlin ↔ Python parity (golden data)",
        "gaps": "Open gaps",
        "reported": "reported", "measured": "measured",
        "ci": "95 % CI", "none": "not measured yet",
        "parity_value": "{jvm} JVM + {dev} device tests; reported diff {px:.3f} px / {deg:.4f}°",
        "gaps_value": "{n} listed — see [docs/METRICS.md](docs/METRICS.md#8-known-gaps)",
    },
    "tr": {
        "metric": "Metrik", "value": "Değer", "evidence": "Kanıt",
        "axillary_tilt": "Aksiller eğim hatası, medyan / p90",
        "coronal": "Koronal denge hatası, medyan / p90",
        "axillary_landmark": "Aksiller landmark hatası, medyan (etiket gürültü tabanı)",
        "recall": "Tarama duyarlılığı, eğim ≥ 3° (ayrılmış küme, n = 50)",
        "size": "Model boyutu, iki aşama, float16",
        "params": "Parametre sayısı, Aşama 1 / Aşama 2",
        "latency": "Model ileri geçişi, 1 × Aşama 1 + 7 × Aşama 2 (TFLite, CPU, medyan / p90)",
        "parity": "Kotlin ↔ Python eşdeğerliği (altın veri)",
        "gaps": "Açık boşluklar",
        "reported": "raporlanan", "measured": "ölçülen",
        "ci": "%95 GA", "none": "henüz ölçülmedi",
        "parity_value": "{jvm} JVM + {dev} cihaz testi; raporlanan fark {px:.3f} px / {deg:.4f}°",
        "gaps_value": "{n} madde — bkz. [docs/METRICS.md](docs/METRICS.md#8-known-gaps)",
    },
}


# ------------------------------------------------------------------ helpers

def num(value, digits: int = 3, unit: str = "") -> str:
    return DASH if value is None else f"{value:.{digits}f}{unit}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    def esc(cell: str) -> str:
        return str(cell).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(esc(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def proportion(successes: int, n: int) -> str:
    lo, hi = wilson_interval(successes, n)
    return f"{successes}/{n} = {successes / n:.3f} ({lo:.3f}–{hi:.3f})"


def title_case(key: str) -> str:
    return key.replace("_", " ").capitalize()


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


UNITS = {"deg": "°", "px": " px", "%": " %"}


def with_unit(value: float, unit: str) -> str:
    return f"{value:g}{UNITS.get(unit, ' ' + unit)}"


def evaluation_blurb(results: dict, section: dict) -> str:
    """One line tying a table to its protocol (the full text is in the protocols block)."""
    name = section["evaluation"]
    ev = results["evaluations"][name]
    repro = "; ".join(f"`{r}`" for r in ev["reproduce"])
    private = " Needs private data." if ev["requires_private_data"] else ""
    return (
        f"Protocol `{name}` ({ev['n_images']} images) · Status: `{section['status']}` · "
        f"Source: {section['source']} · Reproduce: {repro}.{private}"
    )


def _protocols(results: dict) -> str:
    lines = ["### Evaluation protocols\n\nThe accuracy tables come from two protocols. They are "
             "not interchangeable: read the protocol line under each table."]
    for name, ev in results["evaluations"].items():
        size = f"{ev['n_images']:,} images"
        if "n_landmarks" in ev:
            size += f", {ev['n_landmarks']:,} landmarks"
        lines.append(
            f"**`{name}` — {ev['title']}** ({size}; model: {ev['model']}). {ev['protocol']} "
            f"Source: {ev['source']}."
        )
    return "\n\n".join(lines)


def section_notes(section: dict) -> str:
    return ("\n\n" + bullets(section["notes"])) if section.get("notes") else ""


# ------------------------------------------------------------- METRICS.md

def _landmarks(results: dict) -> str:
    sec = results["landmark_error"]
    rows = [[title_case(r["type"]), str(r["n"]), num(r["median_px"]), num(r["p90_px"]),
             num(r["label_floor_px"], 2)] for r in sec["rows"]]
    return (f"### 1.1 Landmark localisation\n\n{evaluation_blurb(results, sec)}\n\n"
            + table(["Landmark", "n", "Median (px)", "p90 (px)", "Label noise floor (px)"], rows)
            + section_notes(sec))


def _angles(results: dict) -> str:
    sec = results["angle_error"]
    rows = [[title_case(r["measurement"]), num(r["median_deg"]), num(r["p90_deg"]),
             num(r["signal_median_deg"], 2), f"{r['median_deg'] / r['signal_median_deg']:.2f}",
             num(r["label_floor_deg"])] for r in sec["rows"]]
    return (f"### 1.2 Angle measurement\n\n{evaluation_blurb(results, sec)}\n\n"
            + table(["Measurement", "Median (°)", "p90 (°)", "Signal, median (°)",
                     "Error / signal", "Single-label floor (°)"], rows)
            + f"\n\nAveraging independent labels lowers the floor to "
              f"{sec['label_floor_averaged_deg']} °." + section_notes(sec))


def _decision(results: dict) -> str:
    sec = results["screening_decision"]
    n = sec["n"]
    rows = [[f"{r['threshold_deg']}°", str(r["positives"]),
             proportion(r["true_positives"], r["positives"]),
             proportion(r["correct"], n), num(r["macro_f1"])] for r in sec["rows"]]
    return (f"### 1.3 Screening decision\n\n{evaluation_blurb(results, sec)}\n\n"
            + table(["Tilt threshold", "Positives", "Recall (95 % Wilson CI)",
                     "Accuracy (95 % Wilson CI)", "Macro-F1"], rows)
            + section_notes(sec))


def _ablation(results: dict) -> str:
    sec = results["two_stage_ablation"]
    rows = []
    for r in sec["rows"]:
        if r["unit"] == "%":
            gain = f"+{r['stage1_plus_2'] - r['stage1']:.0f} pp"
        else:
            gain = f"{r['stage1'] / r['stage1_plus_2']:.1f}×"
        rows.append([r["metric"], with_unit(r["stage1"], r["unit"]),
                     with_unit(r["stage1_plus_2"], r["unit"]),
                     DASH if r["floor"] is None else with_unit(r["floor"], r["unit"]), gain])
    return (f"### 1.4 What the second stage buys\n\n{evaluation_blurb(results, sec)}\n\n"
            + table(["Metric", "Stage 1 only", "Stage 1 + 2", "Label floor", "Gain"], rows)
            + section_notes(sec))


def _deployment(results: dict) -> str:
    var = results["export_variants"]
    var_rows = [[(f"**{r['variant']}** (shipped)" if r["selected"] else r["variant"]),
                 num(r["size_mb"], 2), num(r["median_px"], 2), num(r["median_deg"]),
                 num(r["shoulder_accuracy"]), r["note"] or ""] for r in var["rows"]]
    art = results["artifacts"]
    art_rows = [[f"`{r['file']}`", f"{r['bytes']:,}", f"{r['bytes'] / 1e6:.2f}",
                 f"`{r['sha256'][:16]}…`", r["note"]] for r in art["rows"]]
    cx = results["model_complexity"]
    cx_rows = [[f"`{r['model']}`", f"{r['params']:,}", num(r["gflops"], 3), r["input"],
                r["note"] or ""] for r in cx["rows"]]
    return (
        "## 2. Deployment\n\n### 2.1 Export variants and quantisation cost\n\n"
        + evaluation_blurb(results, var) + "\n\n"
        + table(["Variant", "Size (MB)", "Median (px)", "Median (°)", "Shoulder acc.", "Note"],
                var_rows) + section_notes(var)
        + f"\n\n### 2.2 Shipped artifacts\n\nStatus: `{art['status']}` · {art['source']}\n\n"
        + table(["File", "Bytes", "MB", "SHA-256 (prefix)", "Note"], art_rows)
        + f"\n\n### 2.3 Model complexity\n\nStatus: `{cx['status']}` · {cx['source']}\n\n"
        + table(["Model", "Parameters", "GFLOPs", "Input", "Note"], cx_rows)
    )


def _benchmark_table(run: dict) -> str:
    env, cfg = run["environment"], run["config"]
    head = (f"`{run['_file']}` — {env['cpu']}, {env['os']}"
            + (f", GPU {env['cuda_device']}" if env.get("cuda_device") else "")
            + f". {cfg['threads']} threads, {cfg['warmup']} warm-up + {cfg['repeat']} timed runs, "
              "synthetic input.")
    rows = []
    for r in run["results"]:
        lat = r["latency_ms"]
        rows.append([r["name"], r["device"],
                     f"{r['params']:,}" if "params" in r else DASH,
                     num(r.get("gflops"), 3), num(lat["p50"], 2), num(lat["p90"], 2),
                     num(lat["p99"], 2)])
    return (head + "\n\n"
            + table(["Stage", "Device", "Parameters", "GFLOPs", "p50 (ms)", "p90 (ms)",
                     "p99 (ms)"], rows)
            + "\n\n" + bullets(run["notes"]))


def _latency(results: dict, benchmarks: list[dict]) -> str:
    out = ["## 3. Latency\n\n### 3.1 Benchmarks run from this repository\n\n"
           "Status: `measured`. Reproduce with `python -m skolyoz_metrics bench arch|tflite` "
           "(see [BENCHMARKING.md](BENCHMARKING.md)). Only the newest run of each kind is "
           "shown; older runs stay in `metrics/benchmarks/`."]
    for kind, heading in (("tflite", "Deployed TFLite models"),
                          ("arch", "PyTorch architectures (random weights)")):
        run = latest(benchmarks, kind)
        out.append(f"**{heading}**\n\n" + (_benchmark_table(run) if run else "_No run recorded._"))
    dev = results["device_reported"]
    ladder = [[c["change"], f"{c['total_ms']:g}"] for c in dev["shutter_latency_ms"]]
    imu = dev["imu"]
    out.append(
        f"### 3.2 Reported from a physical device\n\n{dev['device']} · Status: `{dev['status']}` "
        f"· Source: {dev['source']}\n\n"
        + table(["Shutter-latency change", "Total (ms)"], ladder)
        + f"\n\nIMU sampling period {imu['sensor_period_ms']} ms; rotation-induced error at "
          f"30 °/s fell from {imu['rotation_error_at_30_deg_per_s_nearest_deg']} ° (nearest "
          f"sample) to < {imu['rotation_error_at_30_deg_per_s_interpolated_deg']} ° "
          "(interpolation)." + section_notes(dev))
    jvm = results["jvm_preprocessing"]
    out.append(
        f"### 3.3 Kotlin pre-processing on a desktop JVM\n\nStatus: `{jvm['status']}` · "
        f"Source: {jvm['source']}\n\n"
        + table(["Step", "Mean (ms)"], [[r["step"], f"{r['ms']:.1f}"] for r in jvm["rows"]])
        + section_notes(jvm))
    return "\n\n".join(out)


def _parity(results: dict) -> str:
    p = results["parity"]
    rows = [["JVM unit tests", str(p["jvm_unit_tests"])],
            ["Instrumented (device) tests", str(p["instrumented_tests"])],
            ["Golden rows (olcum / cikarim / kucultme)",
             " / ".join(str(v) for v in p["golden_rows"].values())],
            ["Acceptance tolerance", f"{p['acceptance_tolerance_deg']} °"],
            ["Reported max difference", f"{p['reported_max_diff_px']:.3f} px / "
                                        f"{p['reported_max_diff_deg']:.4f} °"]]
    return (f"## 4. Port correctness (Kotlin ↔ Python)\n\nStatus: `{p['status']}` · "
            f"Source: {p['source']}\n\n" + table(["Item", "Value"], rows) + section_notes(p))


def _simple(results: dict, key: str, heading: str, columns: list[tuple[str, str]]) -> str:
    sec = results[key]
    rows = [[str(r[field]) if r[field] is not None else DASH for _, field in columns]
            for r in sec["rows"]]
    return (f"{heading}\n\nStatus: `{sec['status']}` · Source: {sec['source']}\n\n"
            + table([c for c, _ in columns], rows) + section_notes(sec))


def _gaps(results: dict) -> str:
    rows = [[f"`{g['id']}`", g["status"], g["what"], g["why"], g["next"]]
            for g in results["gaps"]]
    return ("## 8. Known gaps\n\nThings this project has **not** measured or validated. They are "
            "listed on purpose; a metrics page without them would overstate what is known.\n\n"
            + table(["Id", "Status", "What", "Why it is open", "Next step"], rows))


def render_metrics_md(results: dict, benchmarks: list[dict]) -> str:
    parts = [
        "# Metrics\n\n"
        "<!-- GENERATED by `python -m skolyoz_metrics render` from metrics/results.json and "
        "metrics/benchmarks/*.json. Do not edit by hand. -->\n\n"
        f"_Last updated {results['updated']}._ {results['policy']}\n\n"
        "**Read before quoting any number:** the accuracy figures describe offline experiments "
        "on web-sourced images. They are not clinical validation and not on-device results. "
        "This is a screening aid, not a diagnostic tool.",
        "## 1. Accuracy",
        _protocols(results),
        _landmarks(results), _angles(results), _decision(results), _ablation(results),
        _deployment(results),
        _latency(results, benchmarks),
        _parity(results),
        _simple(results, "robustness", "## 5. Robustness",
                [("Item", "item"), ("Value", "value"), ("Context", "context")]),
        _simple(results, "dataset", "## 6. Dataset", [("Item", "item"), ("Value", "value")]),
        _simple(results, "negative_results", "## 7. Negative results",
                [("Approach", "approach"), ("Result", "result"), ("Reason", "reason")]),
        _gaps(results),
    ]
    return "\n\n".join(parts) + "\n"


# --------------------------------------------------------------- README block

def render_headline(results: dict, benchmarks: list[dict], lang: str) -> str:
    t = LABELS[lang]
    ang = {r["measurement"]: r for r in results["angle_error"]["rows"]}
    axi = next(r for r in results["landmark_error"]["rows"] if r["type"] == "axillary")
    dec = next(r for r in results["screening_decision"]["rows"] if r["threshold_deg"] == 3)
    lo, hi = wilson_interval(dec["true_positives"], dec["positives"])
    stage = {m["model"]: m for m in results["model_complexity"]["rows"]}
    s1, s2 = stage["stage1_yolo11n_pose_7kpt"], stage["stage2_rafinaj7"]
    size_mb = sum(a["bytes"] for a in results["artifacts"]["rows"]
                  if a["file"].endswith(".tflite")) / 1e6
    par = results["parity"]

    run = latest(benchmarks, "tflite")
    if run:
        pipe = next(r for r in run["results"] if r["name"] == "pipeline")["latency_ms"]
        latency = f"{pipe['p50']:.1f} ms / {pipe['p90']:.1f} ms — {run['environment']['cpu']}"
    else:
        latency = t["none"]

    def a(m: dict) -> str:
        return f"{m['median_deg']:.3f}° / {m['p90_deg']:.3f}°"

    rows = [
        [t["axillary_tilt"], a(ang["axillary_tilt"]), t[results["angle_error"]["status"]]],
        [t["coronal"], a(ang["coronal_balance"]), t[results["angle_error"]["status"]]],
        [t["axillary_landmark"], f"{axi['median_px']:.3f} px ({axi['label_floor_px']} px)",
         t[results["landmark_error"]["status"]]],
        [t["recall"], f"{dec['recall'] * 100:.1f} % ({dec['true_positives']}/{dec['positives']}; "
                      f"{t['ci']} {lo * 100:.1f}–{hi * 100:.1f} %)",
         t[results["screening_decision"]["status"]]],
        [t["size"], f"{size_mb:.2f} MB", t[results["artifacts"]["status"]]],
        [t["params"], f"{s1['params'] / 1e6:.2f} M / {s2['params'] / 1e6:.2f} M",
         t[results["model_complexity"]["status"]]],
        [t["latency"], latency, t["measured"] if run else DASH],
        [t["parity"], t["parity_value"].format(jvm=par["jvm_unit_tests"],
                                              dev=par["instrumented_tests"],
                                              px=par["reported_max_diff_px"],
                                              deg=par["reported_max_diff_deg"]),
         t[par["status"]]],
        [t["gaps"], t["gaps_value"].format(n=len(results["gaps"])), DASH],
    ]
    return table([t["metric"], t["value"], t["evidence"]], rows)


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"<!-- {name}:begin -->.*?<!-- {name}:end -->", flags=re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"missing '<!-- {name}:begin -->' / '<!-- {name}:end -->' markers")
    block = f"<!-- {name}:begin -->\n{body}\n<!-- {name}:end -->"
    return pattern.sub(lambda _: block, text, count=1)


def normalise(text: str) -> str:
    """LF line endings, so the drift check is stable across Windows checkouts."""
    return text.replace("\r\n", "\n")


def render_all(root: Path, results: dict, benchmarks: list[dict]) -> dict[Path, str]:
    """Map each generated file to the full text it should contain."""
    out = {root / METRICS_MD: render_metrics_md(results, benchmarks)}
    for rel, lang in README_TARGETS.items():
        path = root / rel
        if path.exists():
            current = normalise(path.read_text(encoding="utf-8"))
            out[path] = replace_block(current, BLOCK, render_headline(results, benchmarks, lang))
    return out
