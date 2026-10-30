"""Deterministic SVG charts drawn from metrics/results.json and metrics/benchmarks/*.json.

Pure standard library: no matplotlib, no fonts to install. The same JSON always
produces byte-identical SVG, so ``skolyoz_metrics check`` can verify the images
the same way it verifies the generated tables.

Design follows the dataviz rules of the project: one axis per chart, thin bars
(<= 24 px, 4 px rounded data end, square baseline), hairline solid grid, a
legend whenever there are two or more series, direct labels only where they
help, text in ink tokens (never the series colour), and a separately selected
dark variant. Every chart's numbers are also in a table in docs/METRICS.md.
"""

from __future__ import annotations

import math
from xml.sax.saxutils import escape

from .results import latest
from .stats import wilson_interval

W = 760
FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'

# Reference palette (validated with the dataviz validator: slots 1-3, both modes).
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "base": "#c3c2b7",
              "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "gray": "#898781"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "base": "#383835",
             "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "gray": "#898781"},
}


def tw(text: str, size: float, weight: int = 400) -> float:
    """Conservative text width estimate (SVG cannot measure; sans-serif averages ~0.56 em)."""
    return len(text) * size * (0.58 if weight >= 600 else 0.55)


def nice_axis(max_value: float, target: int = 5) -> tuple[float, list[float]]:
    """(axis_max, ticks) with round steps like 1, 2, 2.5, 5 x 10^k."""
    raw = max_value / target
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag)
    top = math.ceil(max_value / step - 1e-9) * step
    return top, [round(i * step, 10) for i in range(int(round(top / step)) + 1)]


def tick_text(v: float) -> str:
    return f"{v:g}"


class Canvas:
    def __init__(self, height: int, theme: str):
        self.h, self.t, self.parts = height, THEMES[theme], []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, size=12, fill=None, anchor="start", weight=400) -> None:
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill or self.t["ink2"]}"'
                 f' text-anchor="{anchor}"' + (f' font-weight="{weight}"' if weight != 400 else "")
                 + f'>{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke=None, width=1) -> None:
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{stroke or self.t["grid"]}" stroke-width="{width}" '
                 f'stroke-linecap="round"/>')

    def bar(self, x, y, w, h, fill, r=4) -> None:
        """Horizontal bar: square at the baseline (left), rounded at the data end."""
        w = max(w, 1.5)
        if w <= r * 2:
            self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                     f'height="{h:.1f}" fill="{fill}"/>')
            return
        self.add(f'<path d="M{x:.1f},{y:.1f}H{x + w - r:.1f}A{r},{r} 0 0 1 {x + w:.1f},{y + r:.1f}'
                 f'V{y + h - r:.1f}A{r},{r} 0 0 1 {x + w - r:.1f},{y + h:.1f}'
                 f'H{x:.1f}Z" fill="{fill}"/>')

    def dot(self, cx, cy, fill, r=5) -> None:
        """Marker with a 2 px ring in the surface colour."""
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" '
                 f'stroke="{self.t["surface"]}" stroke-width="2"/>')

    def legend(self, x, y, items: list[tuple[str, str]]) -> None:
        for label, color in items:
            self.add(f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2" '
                     f'fill="{color}"/>')
            self.text(x + 15, y, label, 12, self.t["ink2"])
            x += 15 + tw(label, 12) + 20

    def header(self, title: str, subtitle: str) -> None:
        self.text(24, 32, title, 16, self.t["ink"], weight=600)
        self.text(24, 52, subtitle, 12, self.t["ink2"])

    def footer(self, *lines: str) -> None:
        for i, line in enumerate(lines):
            self.text(24, self.h - 14 - 15 * (len(lines) - 1 - i), line, 11, self.t["muted"])

    def axis_x(self, x0, x1, y_top, y_bottom, ticks, scale, fmt=tick_text, unit="") -> None:
        """Vertical hairline grid + tick labels along the bottom."""
        for v in ticks:
            x = x0 + v * scale
            self.line(x, y_top, x, y_bottom, self.t["base"] if v == 0 else self.t["grid"])
            self.text(x, y_bottom + 16, fmt(v) + (unit if v == ticks[-1] else ""), 11,
                      self.t["muted"], "middle")

    def svg(self, title: str, desc: str) -> str:
        t = self.t
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{self.h}" '
            f'viewBox="0 0 {W} {self.h}" role="img" aria-labelledby="t d" font-family=\'{FONT}\'>\n'
            f'<title id="t">{escape(title)}</title>\n<desc id="d">{escape(desc)}</desc>\n'
            f'<rect x="0.5" y="0.5" width="{W - 1}" height="{self.h - 1}" rx="8" '
            f'fill="{t["surface"]}" stroke="{t["grid"]}"/>\n'
            + "\n".join(self.parts) + "\n</svg>\n"
        )


def _source(section: dict, results: dict) -> str:
    st = {"reported": "raporlanan", "measured": "ölçülen"}[section["status"]]
    ev = section.get("evaluation")
    n = f" · n = {results['evaluations'][ev]['n_images']} görüntü" if ev else ""
    return f"Durum: {st}{n} · tablo: docs/METRICS.md"


# ------------------------------------------------------------------ charts

def chart_ablation(results: dict, theme: str) -> tuple[str, str] | None:
    sec = results["two_stage_ablation"]
    rows = {r["id"]: r for r in sec["rows"]}
    panels = [("Keypoint hatası, medyan", rows["keypoint_error_median"]),
              ("Açı hatası, medyan", rows["angle_error_median"])]
    c = Canvas(280, theme)
    t = c.t
    c.header("İkinci aşama hatayı büyük ölçüde düşürüyor",
             "Yalnızca Aşama 1 ile Aşama 1 + 2; kısa çubuk daha iyi. "
             "Turuncu çizgi: etiket gürültü tabanı.")
    c.legend(24, 84, [("Yalnızca Aşama 1", t["gray"]), ("Aşama 1 + 2", t["s1"])])
    x_off, pw = 24, 332
    for i, (name, r) in enumerate(panels):
        x0 = x_off + i * (pw + 40)
        unit = " px" if r["unit"] == "px" else "°"
        top, ticks = nice_axis(max(r["stage1"], r["stage1_plus_2"]) * 1.25, 4)
        scale = (pw - 20) / top
        y0 = 126
        c.text(x0, 112, name + f" ({unit.strip()})", 12, t["ink"], weight=600)
        c.axis_x(x0, x0 + pw, y0 - 6, y0 + 74, ticks, scale)
        for j, (key, color) in enumerate((("stage1", t["gray"]), ("stage1_plus_2", t["s1"]))):
            y = y0 + 4 + j * 34
            c.bar(x0, y, r[key] * scale, 22, color)
            c.text(x0 + r[key] * scale + 8, y + 15, f"{r[key]:g}{unit}", 12, t["ink"], weight=600)
        if r["floor"] is not None:
            xf = x0 + r["floor"] * scale
            c.line(xf, y0 - 4, xf, y0 + 70, t["s2"], 2)
            c.text(xf + 4, y0 + 112, f"etiket tabanı {r['floor']:g}{unit}", 11, t["ink2"])
    gain = rows["keypoint_error_median"]
    ratio = gain["stage1"] / gain["stage1_plus_2"]
    c.footer(f"Keypoint hatası {ratio:.1f}× düştü. " + _source(sec, results))
    return (c.svg("Aşama 1 ve Aşama 1 + 2 hata karşılaştırması",
                  f"Keypoint medyan hatası {gain['stage1']:g} pikselden {gain['stage1_plus_2']:g} "
                  f"piksele, açı medyan hatası {rows['angle_error_median']['stage1']:g} dereceden "
                  f"{rows['angle_error_median']['stage1_plus_2']:g} dereceye iniyor."),
            "ablation")


def chart_angle(results: dict, theme: str) -> tuple[str, str] | None:
    sec = results["angle_error"]
    names = {"axillary_tilt": "Aksiller eğim", "coronal_balance": "Koronal denge"}
    top, ticks = nice_axis(max(max(r["p90_deg"], r["signal_median_deg"]) for r in sec["rows"]), 4)
    c = Canvas(96 + len(sec["rows"]) * 110 + 70, theme)
    t = c.t
    c.header("Aksiller eğimde hata, sinyalin çok altında; koronal dengede sınırda",
             "Hata küçük ve sinyal büyükse ölçüm anlamlıdır. "
             "Çubuk ne kadar kısaysa o kadar iyi (sinyal hariç).")
    c.legend(24, 84, [("Hata, medyan", t["s1"]), ("Hata, p90", t["s2"]),
                      ("Sinyal (ölçülen şeyin büyüklüğü)", t["gray"])])
    gutter, x0, pw = 130, 24, 480
    x_plot = x0 + gutter
    scale = pw / top
    y_start = 108
    c.axis_x(x_plot, x_plot + pw, y_start - 4, y_start + len(sec["rows"]) * 110 - 24, ticks, scale)
    for i, r in enumerate(sec["rows"]):
        y = y_start + i * 110
        c.text(x0, y + 44, names[r["measurement"]], 13, t["ink"], weight=600)
        for j, (key, color) in enumerate((("median_deg", t["s1"]), ("p90_deg", t["s2"]),
                                          ("signal_median_deg", t["gray"]))):
            yy = y + j * 28
            c.bar(x_plot, yy, r[key] * scale, 22, color)
            c.text(x_plot + r[key] * scale + 8, yy + 15, f"{r[key]:g}°", 12, t["ink"])
        ratio = r["median_deg"] / r["signal_median_deg"]
        c.text(x_plot + pw + 28, y + 36, "hata / sinyal", 11, t["muted"])
        c.text(x_plot + pw + 28, y + 58, f"{ratio:.2f}", 20, t["ink"], weight=600)
    c.footer("Aşama 2, benzetilmiş ±35 px kaba noktalarla; uçtan uca ya da cihaz üstü ölçüm değil.",
             _source(sec, results))
    return (c.svg("Açı hatası ve sinyal büyüklüğü",
                  "; ".join(f"{names[r['measurement']]}: medyan hata {r['median_deg']:g}°, p90 "
                            f"{r['p90_deg']:g}°, sinyal {r['signal_median_deg']:g}°"
                            for r in sec["rows"])), "angle")


def chart_decision(results: dict, theme: str) -> tuple[str, str] | None:
    sec = results["screening_decision"]
    n = sec["n"]
    c = Canvas(96 + len(sec["rows"]) * 62 + 50, theme)
    t = c.t
    c.header("Tarama duyarlılığı: az sayıda pozitif vaka yüzünden belirsizlik geniş",
             "Nokta ve çizgi: değer ve %95 Wilson güven aralığı. "
             "Ayrılmış test kümesi, az sayıda pozitif vaka.")
    c.legend(24, 84, [("Duyarlılık (pozitifleri yakalama)", t["s1"]), ("Doğruluk", t["s2"])])
    x0, gutter, pw = 24, 150, 380
    xp = x0 + gutter
    scale = pw / 100
    y_start = 112
    c.axis_x(xp, xp + pw, y_start - 8, y_start + len(sec["rows"]) * 62 - 14,
             [0, 20, 40, 60, 80, 100], scale, unit=" %")
    for i, r in enumerate(sec["rows"]):
        y = y_start + i * 62
        c.text(x0, y + 20, f"Eşik ≥ {r['threshold_deg']}°", 13, t["ink"], weight=600)
        c.text(x0, y + 38, f"{r['positives']} pozitif vaka", 11, t["muted"])
        measures = ((r["true_positives"], r["positives"], t["s1"], 12),
                    (r["correct"], n, t["s2"], 32))
        for k, (succ, tot, color, dy) in enumerate(measures):
            lo, hi = wilson_interval(succ, tot)
            yy = y + dy
            c.line(xp + lo * 100 * scale, yy, xp + hi * 100 * scale, yy, color, 2)
            c.dot(xp + succ / tot * 100 * scale, yy, color)
            if k == 0:
                c.text(xp + pw + 20, yy + 4,
                       f"{succ / tot * 100:.1f} %  ({lo * 100:.1f}–{hi * 100:.1f})", 12, t["ink"])
    c.footer(_source(sec, results))
    return (c.svg("Tarama kararı duyarlılığı ve doğruluğu, güven aralıklarıyla",
                  "; ".join(
                      f"eşik {r['threshold_deg']}°: duyarlılık "
                      f"{r['true_positives']}/{r['positives']}" for r in sec["rows"])),
            "decision")


def chart_latency(results: dict, benchmarks: list[dict], theme: str) -> tuple[str, str] | None:
    tfl, arch = latest(benchmarks, "tflite"), latest(benchmarks, "arch")
    series: list[tuple[str, str, dict]] = []   # (label, colour key, {stage: latency})
    if tfl:
        series.append((f"TFLite · CPU, {tfl['config']['threads']} iş parçacığı", "s1",
                       {r["name"]: r["latency_ms"] for r in tfl["results"]}))
    if arch:
        for dev, key, label in (("cpu", "s2", "PyTorch · CPU"), ("cuda", "s3", "PyTorch · CUDA")):
            lat = {r["name"]: r["latency_ms"] for r in arch["results"] if r["device"] == dev}
            if lat:
                series.append((label, key, lat))
    if not series:
        return None
    stages = [("stage1", "Aşama 1 (tek çağrı)"), ("stage2", "Aşama 2 (tek çağrı)"),
              ("pipeline", "Zincir: 1 × Aşama 1 + 7 × Aşama 2")]
    top, ticks = nice_axis(max(s[2][k]["p99"] for s in series for k, _ in stages) * 1.15, 5)
    bar_h, gap, grp = 16, 4, 24
    group_h = len(series) * (bar_h + gap) + grp
    c = Canvas(112 + len(stages) * group_h + 66, theme)
    t = c.t
    env = (tfl or arch)["environment"]
    cpu = env["cpu"].replace("(R)", "").replace("(TM)", "")
    c.header("Gecikme: dağıtılan TFLite modelleri, aynı CPU'da PyTorch'tan hızlı",
             f"Yalnızca model ileri geçişi, sentetik girdi · {cpu}")
    c.legend(24, 84, [(lbl, t[key]) for lbl, key, _ in series])
    x0, gutter, pw = 24, 218, 430
    xp = x0 + gutter
    scale = pw / top
    y_start = 108
    c.axis_x(xp, xp + pw, y_start - 6, y_start + len(stages) * group_h - grp + 6, ticks, scale,
             unit=" ms")
    for i, (stage, title) in enumerate(stages):
        y = y_start + i * group_h
        c.text(x0, y + (len(series) * (bar_h + gap)) / 2 + 2, title, 12, t["ink"], weight=600)
        for j, (_, key, lat) in enumerate(series):
            yy = y + j * (bar_h + gap)
            m = lat[stage]
            c.bar(xp, yy, m["p50"] * scale, bar_h, t[key])
            # thin whisker from the median to the 99th percentile: the tail
            xe, xw = xp + m["p50"] * scale, xp + m["p99"] * scale
            c.line(xe, yy + bar_h / 2, xw, yy + bar_h / 2, t["ink2"], 1.5)
            c.line(xw, yy + 3, xw, yy + bar_h - 3, t["ink2"], 1.5)
            c.text(xw + 8, yy + 12, f"{m['p50']:.1f} ms", 11, t["ink"])
    c.footer("Çubuk: medyan (p50); ince çizgi: p99'a kadar kuyruk. Android gecikmesi DEĞİL.",
             "Durum: ölçülen · tablo: docs/METRICS.md")
    return (c.svg("Model ileri geçişi gecikmesi",
                  "; ".join(f"{lbl}: zincir medyan {lat['pipeline']['p50']:.1f} ms"
                            for lbl, _, lat in series)), "latency")


def chart_variants(results: dict, theme: str) -> tuple[str, str] | None:
    sec = results["export_variants"]
    pts = [r for r in sec["rows"] if r["size_mb"] is not None and r["median_deg"] is not None]
    ref = next((r for r in sec["rows"] if r["variant"] == "pytorch_reference"), None)
    labels = {"float16": "float16 (kullanılan)", "float32": "float32",
              "dynamic_range_quant": "dinamik aralık",
              "int8_int16_activations": "INT8 + int16 aktivasyon"}
    c = Canvas(414, theme)
    t = c.t
    c.header("float16: boyut yarıya iniyor, doğruluk neredeyse aynı kalıyor",
             "Her nokta bir TFLite varyantı; sol-alt daha iyi (küçük ve doğru). "
             "2 landmark'lı modeller, n = 50.")
    px0, py0, pw, ph = 80, 84, 600, 230          # plot box
    xmax = 14.0
    ytop, yticks = nice_axis(max(r["median_deg"] for r in pts) * 1.1, 4)
    sx, sy = pw / xmax, ph / ytop
    for v in yticks:
        y = py0 + ph - v * sy
        c.line(px0, y, px0 + pw, y, t["base"] if v == 0 else t["grid"])
        c.text(px0 - 8, y + 4, f"{v:g}°", 11, t["muted"], "end")
    for v in range(0, 15, 2):
        x = px0 + v * sx
        c.text(x, py0 + ph + 17, f"{v}" + (" MB" if v == 14 else ""), 11, t["muted"], "middle")
    if ref:
        yr = py0 + ph - ref["median_deg"] * sy
        c.line(px0, yr, px0 + pw, yr, t["ink2"], 1)
        c.text(px0 + pw - 4, yr + 16, f"PyTorch referansı {ref['median_deg']:g}°", 11,
               t["ink2"], "end")
    for r in sorted(pts, key=lambda r: r["size_mb"]):
        x, y = px0 + r["size_mb"] * sx, py0 + ph - r["median_deg"] * sy
        chosen = r["selected"]
        c.dot(x, y, t["s1"] if chosen else t["gray"], 6 if chosen else 5)
        name = labels.get(r["variant"], r["variant"])
        c.text(x, y - 28, name, 11, t["ink"], "middle", 600 if chosen else 400)
        c.text(x, y - 14, f"{r['size_mb']:g} MB · {r['median_deg']:g}°", 11, t["ink2"], "middle")
    c.text(px0 + pw / 2, py0 + ph + 34, "Model boyutu (MB)", 11, t["ink2"], "middle")
    c.text(24, py0 - 14, "Medyan açı hatası (°)", 11, t["ink2"])
    failed = [r for r in sec["rows"] if r["median_deg"] is None and r["size_mb"] is not None]
    if failed:
        r = failed[0]
        c.text(px0, py0 + ph + 48,
               f"Tam tamsayı INT8 ({r['size_mb']:g} MB): 50 görüntünün hiçbirinde "
               "tespit yok, bu yüzden çizilemedi.", 11, t["ink2"])
    c.footer(_source(sec, results))
    return (c.svg("Dışa aktarma varyantları: boyut ve açı hatası",
                  "; ".join(f"{r['variant']}: {r['size_mb']:g} MB, {r['median_deg']:g}°"
                            for r in pts)),
            "variants")


CHARTS = ("ablation", "angle", "decision", "latency", "variants")


def render_charts(results: dict, benchmarks: list[dict]) -> dict[str, str]:
    """Map 'name[-dark].svg' to SVG text for every chart that has data."""
    out: dict[str, str] = {}
    for theme in ("light", "dark"):
        built = [chart_ablation(results, theme), chart_angle(results, theme),
                 chart_decision(results, theme), chart_latency(results, benchmarks, theme),
                 chart_variants(results, theme)]
        for item in built:
            if item:
                svg, name = item
                out[f"{name}{'-dark' if theme == 'dark' else ''}.svg"] = svg
    return out


def picture(name: str, alt: str, prefix: str = "img/") -> str:
    """HTML <picture> so GitHub shows the dark variant in dark theme."""
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" '
            f'srcset="{prefix}{name}-dark.svg">\n  <img alt="{alt}" src="{prefix}{name}.svg" '
            f'width="{W}">\n</picture>')
