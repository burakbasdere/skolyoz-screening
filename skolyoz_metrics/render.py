"""Render docs/METRICS.md and the README headline tables from the JSON sources.

Nothing in the rendered output is typed by hand: every number comes from
``metrics/results.json`` or ``metrics/benchmarks/*.json``, and the confidence
intervals are computed from the stored counts. ``check`` mode (used by CI)
fails when a rendered file on disk differs from what the JSON would produce,
so a README can never silently drift from the data.

The generated page is in Turkish; the README headline table exists in Turkish
(README.md) and English (README.en.md).
"""

from __future__ import annotations

import re
from pathlib import Path

from .charts import picture, render_charts
from .results import latest
from .stats import wilson_interval

DASH = "—"
METRICS_MD = Path("docs") / "METRICS.md"
IMG_DIR = Path("docs") / "img"
README_TARGETS = {Path("README.md"): "tr", Path("README.en.md"): "en"}
BLOCK = "headline"
CHARTS_BLOCK = "charts"
README_CHARTS = ("ablation", "angle", "decision", "latency")
ALT = {
    "ablation": "Yalnızca Aşama 1 ile Aşama 1 + 2 için keypoint ve açı hatası çubuk grafiği",
    "angle": "Aksiller eğim ve koronal denge için medyan hata, p90 hata ve sinyal büyüklüğü",
    "decision": "3, 5 ve 7 derece eşiklerinde duyarlılık ve doğruluk, %95 Wilson aralıklarıyla",
    "latency": "Aşama 1, Aşama 2 ve tam zincir için TFLite ve PyTorch gecikme çubuk grafiği",
    "variants": "Dışa aktarma varyantlarının dosya boyutu ve açı hatası dağılım grafiği",
}
GAPS_LINK = "docs/METRICS.md#8-bilinen-boşluklar"

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
        "gaps_value": "{n} listed — see [METRICS.md (Turkish)](" + GAPS_LINK + ")",
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
        "gaps_value": "{n} madde — bkz. [docs/METRICS.md](" + GAPS_LINK + ")",
    },
}

# Display names for the machine ids used in results.json.
NAMES = {
    "axillary": "Aksiller", "c7": "C7", "sacrum": "Sakrum", "spine": "Omurga",
    "axillary_tilt": "Aksiller eğim", "coronal_balance": "Koronal denge",
}
STATUS_TR = {"reported": "raporlanan", "measured": "ölçülen"}


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


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


UNITS = {"deg": "°", "px": " px", "%": " %"}


def with_unit(value: float, unit: str) -> str:
    return f"{value:g}{UNITS.get(unit, ' ' + unit)}"


def status_line(section: dict) -> str:
    status = section["status"]
    return f"Durum: `{status}` ({STATUS_TR[status]}) · Kaynak: {section['source']}"


def evaluation_blurb(results: dict, section: dict) -> str:
    """One line tying a table to its protocol (the full text is in the protocols block)."""
    name = section["evaluation"]
    ev = results["evaluations"][name]
    repro = "; ".join(f"`{r}`" for r in ev["reproduce"])
    private = " Özel veri gerektirir." if ev["requires_private_data"] else ""
    return (
        f"Protokol `{name}` ({ev['n_images']} görüntü) · {status_line(section)} · "
        f"Yeniden üretmek için: {repro}.{private}"
    )


def _fig(fig: str) -> str:
    return fig + "\n\n" if fig else ""


def section_notes(section: dict) -> str:
    return ("\n\n" + bullets(section["notes"])) if section.get("notes") else ""


def _protocols(results: dict) -> str:
    lines = ["### Değerlendirme protokolleri\n\nDoğruluk tabloları iki protokolden gelir. Bunlar "
             "birbirinin yerine kullanılamaz: her tablonun altındaki protokol satırına bakın."]
    for name, ev in results["evaluations"].items():
        size = f"{ev['n_images']:,} görüntü"
        if "n_landmarks" in ev:
            size += f", {ev['n_landmarks']:,} landmark"
        lines.append(
            f"**`{name}` — {ev['title']}** ({size}; model: {ev['model']}). {ev['protocol']} "
            f"Kaynak: {ev['source']}."
        )
    return "\n\n".join(lines)


# ------------------------------------------------------------- METRICS.md

def _landmarks(results: dict) -> str:
    sec = results["landmark_error"]
    rows = [[NAMES.get(r["type"], r["type"]), str(r["n"]), num(r["median_px"]), num(r["p90_px"]),
             num(r["label_floor_px"], 2)] for r in sec["rows"]]
    return (f"### 1.1 Landmark lokalizasyonu\n\n{evaluation_blurb(results, sec)}\n\n"
            + table(["Landmark", "n", "Medyan (px)", "p90 (px)", "Etiket gürültü tabanı (px)"],
                    rows)
            + section_notes(sec))


def _angles(results: dict, fig: str = "") -> str:
    sec = results["angle_error"]
    rows = [[NAMES.get(r["measurement"], r["measurement"]), num(r["median_deg"]),
             num(r["p90_deg"]), num(r["signal_median_deg"], 2),
             f"{r['median_deg'] / r['signal_median_deg']:.2f}", num(r["label_floor_deg"])]
            for r in sec["rows"]]
    return (f"### 1.2 Açı ölçümü\n\n{evaluation_blurb(results, sec)}\n\n{_fig(fig)}"
            + table(["Ölçüm", "Medyan (°)", "p90 (°)", "Sinyal, medyan (°)",
                     "Hata / sinyal", "Tek etiketli taban (°)"], rows)
            + f"\n\nBağımsız etiketlerin ortalaması alınınca taban "
              f"{sec['label_floor_averaged_deg']} °'ye iner." + section_notes(sec))


def _decision(results: dict, fig: str = "") -> str:
    sec = results["screening_decision"]
    n = sec["n"]
    rows = [[f"{r['threshold_deg']}°", str(r["positives"]),
             proportion(r["true_positives"], r["positives"]),
             proportion(r["correct"], n), num(r["macro_f1"])] for r in sec["rows"]]
    return (f"### 1.3 Tarama kararı\n\n{evaluation_blurb(results, sec)}\n\n{_fig(fig)}"
            + table(["Eğim eşiği", "Pozitif", "Duyarlılık (%95 Wilson GA)",
                     "Doğruluk (%95 Wilson GA)", "Makro-F1"], rows)
            + section_notes(sec))


def _ablation(results: dict, fig: str = "") -> str:
    sec = results["two_stage_ablation"]
    rows = []
    for r in sec["rows"]:
        if r["unit"] == "%":
            gain = f"+{r['stage1_plus_2'] - r['stage1']:.0f} puan"
        else:
            gain = f"{r['stage1'] / r['stage1_plus_2']:.1f}×"
        rows.append([r["metric"], with_unit(r["stage1"], r["unit"]),
                     with_unit(r["stage1_plus_2"], r["unit"]),
                     DASH if r["floor"] is None else with_unit(r["floor"], r["unit"]), gain])
    return (f"### 1.4 Aşama 2'nin katkısı\n\n{evaluation_blurb(results, sec)}\n\n{_fig(fig)}"
            + table(["Metrik", "Yalnızca Aşama 1", "Aşama 1 + 2", "Etiket tabanı", "Kazanç"], rows)
            + section_notes(sec))


def _deployment(results: dict, fig: str = "") -> str:
    var = results["export_variants"]
    var_rows = [[(f"**{r['variant']}** (kullanılan)" if r["selected"] else r["variant"]),
                 num(r["size_mb"], 2), num(r["median_px"], 2), num(r["median_deg"]),
                 num(r["shoulder_accuracy"]), r["note"] or ""] for r in var["rows"]]
    art = results["artifacts"]
    art_rows = [[f"`{r['file']}`", f"{r['bytes']:,}", f"{r['bytes'] / 1e6:.2f}",
                 f"`{r['sha256'][:16]}…`", r["note"]] for r in art["rows"]]
    cx = results["model_complexity"]
    cx_rows = [[f"`{r['model']}`", f"{r['params']:,}", num(r["gflops"], 3), r["input"],
                r["note"] or ""] for r in cx["rows"]]
    return (
        "## 2. Dağıtım\n\n### 2.1 Dışa aktarma varyantları ve kuantizasyon maliyeti\n\n"
        + evaluation_blurb(results, var) + "\n\n" + _fig(fig)
        + table(["Varyant", "Boyut (MB)", "Medyan (px)", "Medyan (°)", "Omuz doğruluğu", "Not"],
                var_rows) + section_notes(var)
        + f"\n\n### 2.2 Gönderilen çıktılar\n\n{status_line(art)}\n\n"
        + table(["Dosya", "Bayt", "MB", "SHA-256 (önek)", "Not"], art_rows)
        + f"\n\n### 2.3 Model karmaşıklığı\n\n{status_line(cx)}\n\n"
        + table(["Model", "Parametre", "GFLOPs", "Girdi", "Not"], cx_rows)
    )


def _benchmark_table(run: dict) -> str:
    env, cfg = run["environment"], run["config"]
    head = (f"`{run['_file']}` — {env['cpu']}, {env['os']}"
            + (f", GPU {env['cuda_device']}" if env.get("cuda_device") else "")
            + f". {cfg['threads']} iş parçacığı, {cfg['warmup']} ısınma + {cfg['repeat']} "
              "ölçülen koşu, sentetik girdi.")
    rows = []
    for r in run["results"]:
        lat = r["latency_ms"]
        rows.append([r["name"], r["device"],
                     f"{r['params']:,}" if "params" in r else DASH,
                     num(r.get("gflops"), 3), num(lat["p50"], 2), num(lat["p90"], 2),
                     num(lat["p99"], 2)])
    return (head + "\n\n"
            + table(["Aşama", "Cihaz", "Parametre", "GFLOPs", "p50 (ms)", "p90 (ms)",
                     "p99 (ms)"], rows)
            + "\n\n" + bullets(run["notes"]))


def _latency(results: dict, benchmarks: list[dict], fig: str = "") -> str:
    out = ["## 3. Gecikme\n\n### 3.1 Bu depodan çalıştırılan benchmark'lar\n\n"
           "Durum: `measured` (ölçülen). Yeniden üretmek için: "
           "`python -m skolyoz_metrics bench arch|tflite` "
           "(bkz. [BENCHMARKING.md](BENCHMARKING.md)). Her türün yalnızca en yeni koşusu "
           "gösterilir; eskiler `metrics/benchmarks/` içinde geçmiş olarak kalır."
           + ("\n\n" + fig if fig else "")]
    for kind, heading in (("tflite", "Dağıtılan TFLite modelleri"),
                          ("arch", "PyTorch mimarileri (rastgele ağırlık)")):
        run = latest(benchmarks, kind)
        out.append(f"**{heading}**\n\n" + (_benchmark_table(run) if run else "_Kayıtlı koşu yok._"))
    dev = results["device_reported"]
    ladder = [[c["change"], f"{c['total_ms']:g}"] for c in dev["shutter_latency_ms"]]
    imu = dev["imu"]
    out.append(
        f"### 3.2 Fiziksel cihazdan raporlananlar\n\n{dev['device']} · {status_line(dev)}\n\n"
        + table(["Deklanşör gecikmesi değişikliği", "Toplam (ms)"], ladder)
        + f"\n\nIMU örnekleme periyodu {imu['sensor_period_ms']} ms; 30 °/s dönüşte dönmeden "
          "doğan hata, en yakın örnek seçilince "
          f"{imu['rotation_error_at_30_deg_per_s_nearest_deg']} ° iken enterpolasyonla < "
          f"{imu['rotation_error_at_30_deg_per_s_interpolated_deg']} °'ye indi."
          + section_notes(dev))
    jvm = results["jvm_preprocessing"]
    out.append(
        f"### 3.3 Masaüstü JVM'de Kotlin ön işleme\n\n{status_line(jvm)}\n\n"
        + table(["Adım", "Ortalama (ms)"], [[r["step"], f"{r['ms']:.1f}"] for r in jvm["rows"]])
        + section_notes(jvm))
    return "\n\n".join(out)


def _parity(results: dict) -> str:
    p = results["parity"]
    rows = [["JVM birim testleri", str(p["jvm_unit_tests"])],
            ["Cihaz (instrumented) testleri", str(p["instrumented_tests"])],
            ["Altın veri satırları (olcum / cikarim / kucultme)",
             " / ".join(str(v) for v in p["golden_rows"].values())],
            ["Kabul toleransı", f"{p['acceptance_tolerance_deg']} °"],
            ["Raporlanan en büyük fark", f"{p['reported_max_diff_px']:.3f} px / "
                                         f"{p['reported_max_diff_deg']:.4f} °"]]
    return (f"## 4. Port doğruluğu (Kotlin ↔ Python)\n\n{status_line(p)}\n\n"
            + table(["Madde", "Değer"], rows) + section_notes(p))


def _simple(results: dict, key: str, heading: str, columns: list[tuple[str, str]]) -> str:
    sec = results[key]
    rows = [[str(r[field]) if r[field] is not None else DASH for _, field in columns]
            for r in sec["rows"]]
    return (f"{heading}\n\n{status_line(sec)}\n\n"
            + table([c for c, _ in columns], rows) + section_notes(sec))


def _gaps(results: dict) -> str:
    rows = [[f"`{g['id']}`", g["status"], g["what"], g["why"], g["next"]]
            for g in results["gaps"]]
    return ("## 8. Bilinen boşluklar\n\nBu projenin **ölçmediği** ya da doğrulamadığı şeyler. "
            "Bilerek listelenir; bunlar olmadan bir metrik sayfası bilinenin kapsamını "
            "abartırdı.\n\n"
            + table(["Kimlik", "Durum", "Ne", "Neden açık", "Sonraki adım"], rows))


def render_metrics_md(results: dict, benchmarks: list[dict]) -> str:
    have = {n[:-4] for n in render_charts(results, benchmarks) if not n.endswith("-dark.svg")}

    def fig(name: str) -> str:
        return picture(name, ALT[name]) if name in have else ""

    parts = [
        "# Metrikler\n\n"
        "<!-- `python -m skolyoz_metrics render` ile metrics/results.json ve "
        "metrics/benchmarks/*.json dosyalarından ÜRETİLİR. Elle düzenlemeyin. -->\n\n"
        f"_Son güncelleme {results['updated']}._ {results['policy']}\n\n"
        "**Herhangi bir sayıyı alıntılamadan önce okuyun:** doğruluk rakamları web kaynaklı "
        "görüntüler üzerindeki çevrimdışı deneyleri anlatır. Klinik doğrulama değildir ve "
        "cihaz üstü sonuç değildir. Bu bir tarama yardımcısıdır, tanı aracı değildir.",
        "## 1. Doğruluk",
        _protocols(results),
        _landmarks(results), _angles(results, fig("angle")), _decision(results, fig("decision")),
        _ablation(results, fig("ablation")),
        _deployment(results, fig("variants")),
        _latency(results, benchmarks, fig("latency")),
        _parity(results),
        _simple(results, "robustness", "## 5. Dayanıklılık",
                [("Madde", "item"), ("Değer", "value"), ("Bağlam", "context")]),
        _simple(results, "dataset", "## 6. Veri kümesi", [("Madde", "item"), ("Değer", "value")]),
        _simple(results, "negative_results", "## 7. Olumsuz sonuçlar",
                [("Yaklaşım", "approach"), ("Sonuç", "result"), ("Gerekçe", "reason")]),
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


def render_gallery(charts: dict[str, str]) -> str:
    return "\n\n".join(picture(n, ALT[n], prefix="docs/img/") for n in README_CHARTS
                       if f"{n}.svg" in charts)


def render_all(root: Path, results: dict, benchmarks: list[dict]) -> dict[Path, str]:
    """Map each generated file to the full text it should contain."""
    charts = render_charts(results, benchmarks)
    out = {root / METRICS_MD: render_metrics_md(results, benchmarks)}
    out.update({root / IMG_DIR / name: svg for name, svg in charts.items()})
    for rel, lang in README_TARGETS.items():
        path = root / rel
        if path.exists():
            text = normalise(path.read_text(encoding="utf-8"))
            text = replace_block(text, BLOCK, render_headline(results, benchmarks, lang))
            out[path] = replace_block(text, CHARTS_BLOCK, render_gallery(charts))
    return out
