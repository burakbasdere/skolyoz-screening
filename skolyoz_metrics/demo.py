"""Sample-output demo: the on-device chain, mirrored in Python, drawn on example images.

Five examples use a *drawn* back figure whose armpit tilt is known by construction; no real
person appears in them. The sixth (``kind="photo"``) is the single real photograph the README
opens with, run through the same chain. The chain is a line-for-line mirror of
android/.../cikarim/OlcumZinciri.kt (stretch to 640, Stage 1 decode, 128 px crops with
black padding, soft-argmax with temperature 10, entropy / peak-distance gate, map back to the
original resolution, angle at the original resolution), run on the real ``.tflite`` files.

Needs numpy, Pillow and a TFLite runtime (``requirements/bench.txt``) plus the two model files,
which are not distributed. The pure helpers (``soft_argmax``, ``decode_stage1``, ``figure``,
``overlay``) are unit-tested without models.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

MODEL_SIZE, CROP = 640, 128
CONFIDENCE_MIN = 0.25          # Asama1Coz.GUVEN_ESIGI
ENTROPY_MAX = 2.5              # OnIsleme.ENTROPI_ESIGI
PEAK_DISTANCE_MAX = 15.0       # OnIsleme.TEPE_UZAKLIK_ESIGI
TEMPERATURE = 10.0             # OnIsleme.SICAKLIK
PITCH_MAX = 8.0                # Olcum / guven_kontrolu
TYPE = [0, 0, 1, 2, 2, 2, 3]   # landmark index -> Stage-2 channel
NAMES = ["sağ koltuk altı", "sol koltuk altı", "C7", "omurga 1", "omurga 2", "omurga 3",
         "sakrum"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DISPLAY_WIDTH = 900            # width the overlay is drawn at: the drawn figure's own width
SYNTHETIC_FOOTER = "Tanı aracı değildir · tarama amaçlıdır · SENTETİK ÖRNEK"
PHOTO_FOOTER = "Tanı aracı değildir · tarama amaçlıdır · GERÇEK FOTOĞRAF, TEK ÖRNEK"


# --------------------------------------------------------------- pure helpers

def soft_argmax(heatmap, temperature: float = TEMPERATURE) -> tuple[float, float, float, float]:
    """(x, y, entropy, peak_distance) in 128-px crop coordinates, like OnIsleme.yumusakArgmax."""
    import numpy as np

    flat = np.asarray(heatmap, dtype=np.float64).ravel()
    n = int(round(math.sqrt(flat.size)))
    if n * n != flat.size:
        raise ValueError(f"heatmap must be square, got {flat.size} values")
    p = np.exp((flat - flat.max()) * temperature)
    p /= p.sum()
    rows, cols = np.divmod(np.arange(flat.size), n)
    scale = (CROP - 1) / (n - 1)
    x = float((p * cols).sum()) * scale
    y = float((p * rows).sum()) * scale
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    peak = int(p.argmax())
    peak_x, peak_y = (peak % n) * scale, (peak // n) * scale
    return x, y, entropy, math.hypot(x - peak_x, y - peak_y)


def decode_stage1(out, confidence_min: float = CONFIDENCE_MIN):
    """Best candidate of a (26, 8400) Stage-1 output: ([(x, y)] * 7, confidence) or None.

    Like Asama1Coz.coz: no NMS, the highest-confidence candidate wins.
    """
    import numpy as np

    out = np.asarray(out)
    best = int(out[4].argmax())
    confidence = float(out[4, best])
    if confidence < confidence_min:
        return None
    points = [(float(out[5 + 3 * k, best]), float(out[6 + 3 * k, best])) for k in range(7)]
    return points, confidence


def reference(root: Path):
    """Import the measurement maths from scripts/19 (the file name starts with a digit)."""
    path = root / "scripts" / "19_kilavuz_cizim.py"
    spec = importlib.util.spec_from_file_location("kilavuz_cizim_ref", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def figure(tilt_deg: float = 0.0, width: int = 900, height: int = 1200):
    """A drawn back figure (head, neck, trapezius, arms, torso, shorts) on a studio gradient.

    ``tilt_deg`` tilts the shoulder line and both armpit junctions together; positive = the
    right side of the frame is higher, i.e. the armpit line has that slope by construction.
    """
    from PIL import Image, ImageDraw, ImageFilter

    cx, tan_t = width // 2, math.tan(math.radians(tilt_deg))

    def up(dx: float) -> float:          # vertical shift of a point dx right of centre
        return -dx * tan_t

    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        v = int(236 - 26 * y / height)
        draw.line([(0, y), (width, y)], fill=(v, v + 1, v + 3))

    mask = Image.new("L", (width, height), 0)
    m = ImageDraw.Draw(mask)
    shoulder = 420
    m.polygon([(cx - 250, shoulder + up(-250)), (cx - 120, shoulder + up(-120) - 40),
               (cx + 120, shoulder + up(120) - 40), (cx + 250, shoulder + up(250)),
               (cx + 215, 760), (cx + 170, 980), (cx - 170, 980), (cx - 215, 760)], fill=255)
    for sign in (-1, 1):                 # arms; the inner end is the armpit junction
        m.polygon([(cx + sign * 250, shoulder + up(sign * 250)), (cx + sign * 330, 780),
                   (cx + sign * 270, 800), (cx + sign * 205, 470 + up(sign * 205))], fill=255)
    m.rectangle((cx - 55, shoulder - 130, cx + 55, shoulder - 20), fill=255)       # neck
    m.ellipse((cx - 105, shoulder - 330, cx + 105, shoulder - 100), fill=255)      # head

    skin = Image.new("RGB", (width, height), (214, 168, 134))
    shade = Image.new("L", (width, height), 0)
    s = ImageDraw.Draw(shade)
    s.line([(cx, shoulder - 20), (cx, 960)], fill=90, width=14)                    # spine groove
    for sign in (-1, 1):                                                            # scapulae
        x0, x1 = sorted((cx + sign * 60, cx + sign * 190))
        s.ellipse((x0, 520, x1, 700), outline=60, width=10)
    shade = shade.filter(ImageFilter.GaussianBlur(9))
    layer = Image.composite(Image.new("RGB", (width, height), (120, 80, 60)), skin, shade)
    hair = Image.new("L", (width, height), 0)
    ImageDraw.Draw(hair).ellipse((cx - 112, shoulder - 340, cx + 112, shoulder - 150), fill=255)
    layer = Image.composite(Image.new("RGB", (width, height), (40, 30, 28)), layer, hair)
    shorts = Image.new("L", (width, height), 0)
    ImageDraw.Draw(shorts).polygon([(cx - 175, 900), (cx + 175, 900), (cx + 170, 980),
                                    (cx - 170, 980)], fill=255)
    layer = Image.composite(Image.new("RGB", (width, height), (60, 70, 95)), layer, shorts)
    img.paste(layer, (0, 0), mask.filter(ImageFilter.GaussianBlur(1.2)))
    return img


def blank(width: int = 900, height: int = 1200):
    """An empty studio background: nobody in the frame."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        v = int(236 - 26 * y / height)
        draw.line([(0, y), (width, y)], fill=(v, v + 1, v + 3))
    return img


# ------------------------------------------------------------------ the chain

@dataclass
class Measurement:
    size: tuple[int, int]
    confidence: float
    found: bool = False
    coarse: list[tuple[float, float]] = field(default_factory=list)       # 640 space
    points: list[tuple[float, float]] = field(default_factory=list)       # original resolution
    gates: list[tuple[float, float]] = field(default_factory=list)        # (entropy, peak dist)

    @property
    def reliable(self) -> bool:
        """Like OlcumZinciri: only the two axillary points decide."""
        return self.found and all(e <= ENTROPY_MAX and d <= PEAK_DISTANCE_MAX
                                  for e, d in self.gates[:2])


class Chain:
    """Stage 1 + 7 x Stage 2 on the shipped TFLite files."""

    def __init__(self, stage1: Path, stage2: Path, threads: int = 4):
        from .bench.tflite import load_interpreter_class

        cls = load_interpreter_class()
        self.stage1, self.stage2 = (cls(model_path=str(p), num_threads=threads)
                                    for p in (stage1, stage2))
        for interp in (self.stage1, self.stage2):
            interp.allocate_tensors()
        self.hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (stage1, stage2)}

    @staticmethod
    def _run(interp, x):
        interp.set_tensor(interp.get_input_details()[0]["index"], x.astype("float32"))
        interp.invoke()
        return interp.get_tensor(interp.get_output_details()[0]["index"]).copy()

    def measure(self, img) -> Measurement:
        import numpy as np
        from PIL import Image

        width, height = img.size
        small = img.convert("RGB").resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.BILINEAR)
        out = self._run(self.stage1, (np.asarray(small, np.float32) / 255.0)[None])[0]
        decoded = decode_stage1(out)
        if decoded is None:
            return Measurement((width, height), float(out[4].max()))
        coarse, confidence = decoded
        mean, std = np.array(IMAGENET_MEAN, np.float32), np.array(IMAGENET_STD, np.float32)
        points, gates = [], []
        for k, (cx, cy) in enumerate(coarse):
            left, top = round(cx - CROP / 2), round(cy - CROP / 2)
            crop = small.crop((left, top, left + CROP, top + CROP))      # black outside the frame
            tensor = ((np.asarray(crop, np.float32) / 255.0) - mean) / std
            heat = self._run(self.stage2, tensor[None])[0]               # (64, 64, 4)
            x, y, entropy, peak = soft_argmax(heat[:, :, TYPE[k]])
            points.append(((left + x) * width / MODEL_SIZE, (top + y) * height / MODEL_SIZE))
            gates.append((entropy, peak))
        return Measurement((width, height), confidence, True, coarse, points, gates)


def scaled(m: Measurement, factor: float) -> Measurement:
    """``m`` with the refined points mapped onto an image ``factor`` times larger.

    The coarse points and the gate numbers are unchanged: they are not in image pixels.
    """
    return replace(m, size=(round(m.size[0] * factor), round(m.size[1] * factor)),
                   points=[(x * factor, y * factor) for x, y in m.points])


# ------------------------------------------------------------------- drawing

def _font(size: int):
    from PIL import ImageFont

    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _dashed(draw, p1, p2, color, width=3, dash=22, gap=14):
    length = math.dist(p1, p2)
    if length < 1:
        return
    ux, uy = (p2[0] - p1[0]) / length, (p2[1] - p1[1]) / length
    t = 0.0
    while t < length:
        e = min(t + dash, length)
        draw.line([(p1[0] + ux * t, p1[1] + uy * t), (p1[0] + ux * e, p1[1] + uy * e)],
                  fill=color, width=width)
        t = e + gap


def overlay(img, m: Measurement, lines: list[str], roll_deg: float = 0.0,
            verdict: str | None = None, guides: bool = True, *,
            footer: str = SYNTHETIC_FOOTER, band_below: bool = False):
    """Draw the guide the app shows: armpit line, gravity references, landmarks, reading.

    With ``guides=False`` (a rejected measurement) only hollow points are drawn: the app would
    show nothing, the demo shows where the unreliable points landed. With ``band_below`` the
    text goes in a band under the frame (the verdict as its first row) instead of over it, so
    no landmark is hidden on a short image.
    """
    from PIL import Image, ImageDraw

    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    if m.found and not guides:
        for x, y in m.points:
            d.ellipse((x - 11, y - 11, x + 11, y + 11), outline=(255, 255, 255, 230), width=4)
    elif m.found:
        r = math.radians(roll_deg)
        horiz, vert = (math.cos(r), math.sin(r)), (-math.sin(r), math.cos(r))
        sag, sol, c7, o1, o2, o3, sak = m.points
        mid = ((sag[0] + sol[0]) / 2, (sag[1] + sol[1]) / 2)
        reach = w * 0.42
        _dashed(d, (mid[0] - horiz[0] * reach, mid[1] - horiz[1] * reach),
                (mid[0] + horiz[0] * reach, mid[1] + horiz[1] * reach), (255, 255, 255, 230))
        _dashed(d, (sak[0] + vert[0] * h * 0.05, sak[1] + vert[1] * h * 0.05),
                (sak[0] - vert[0] * h * 0.70, sak[1] - vert[1] * h * 0.70), (190, 120, 255, 230))
        d.line([c7, o1, o2, o3, sak], fill=(40, 200, 200, 200), width=4, joint="curve")
        d.line([sag, sol], fill=(255, 205, 0, 255), width=6)
        palette = [(255, 70, 70), (70, 160, 255), (190, 120, 255), (40, 200, 200), (40, 200, 200),
                   (40, 200, 200), (190, 120, 255)]
        labels = ["R", "L", "C7", "", "", "", "S"]
        font_small = _font(28)
        for (x, y), color, label in zip(m.points, palette, labels, strict=True):
            d.ellipse((x - 11, y - 11, x + 11, y + 11), fill=color + (255,),
                      outline=(255, 255, 255),
                      width=3)
            if label:
                d.text((x + 16, y - 16), label, fill=(255, 255, 255), font=font_small,
                       stroke_width=3, stroke_fill=(0, 0, 0))
    font = _font(34)
    rows = ([verdict] if verdict and band_below else []) + lines
    band = 46 * (len(rows) + 1) + 24
    if band_below:
        canvas = Image.new("RGB", (w, h + band), (0, 0, 0))
        canvas.paste(out, (0, 0))
        out, top = canvas, h
        d = ImageDraw.Draw(out, "RGBA")
    else:
        top = h - band
        d.rectangle((0, top, w, h), fill=(0, 0, 0, 190))
    y = top + 14
    for i, line in enumerate(rows):
        warn = bool(verdict) and band_below and i == 0
        d.text((24, y), line, fill=(255, 130, 130) if warn else (255, 255, 255), font=font)
        y += 46
    d.text((24, y), footer, fill=(255, 205, 0), font=_font(28))
    if verdict and not band_below:
        d.rectangle((0, 0, w, 74), fill=(200, 40, 40, 235))
        d.text((24, 16), verdict, fill=(255, 255, 255), font=_font(36))
    return out


# --------------------------------------------------------------------- the run

@dataclass
class Example:
    name: str
    title: str
    note: str
    kind: str                      # figure | blur | pitch | blank | photo
    tilt: float = 0.0              # drawn armpit tilt (figure / blur / pitch)
    pitch: float = 0.0             # simulated phone pitch (pitch)
    source: str | None = None      # photo: image path relative to the repository root
    measurement: Measurement | None = None
    angle: float | None = None
    coronal: tuple[float, float] | None = None
    rejected: str | None = None    # why the app would show no measurement


EXAMPLES = [
    Example("duz", "Simetrik figür",
            "Omuz ve koltuk altı çizgisi yatay çizildi; beklenen eğim ≈ 0°.", "figure"),
    Example("asimetri", "Sağ omuz yüksek",
            "Omuz ve koltuk altı çizgisi 5° eğik çizildi (çizimde bilinen eğim, sağ taraf yüksek).",
            "figure", tilt=5.0),
    Example("bulanik", "Bulanık çekim: güven kapısı reddeder",
            "Aynı simetrik figür, 12 piksellik Gauss bulanıklığıyla. Model yine de kişiyi bulur "
            "(Aşama 1 güveni yüksek) ama Aşama 2 ısı haritaları dağılır; entropi eşiği aşar ve "
            "uygulama yanlış bir açı göstermek yerine yeniden çekim ister. Beyaz içi boş daireler, "
            "uygulamanın göstermeyeceği güvenilmez noktalardır.", "blur"),
    Example("pitch", "Telefon öne yatık: çekim reddedilir",
            "Telefon 11° öne yatıkken perspektif açıyı bozar; |pitch| > 8° olduğunda deklanşör "
            "kilitlenir ve ölçüm yapılmaz. Bu karar yalnızca IMU'dan gelir, model çalıştırılmaz.",
            "pitch", pitch=11.0),
    Example("kisi-yok", "Karede kimse yok",
            "Boş zemin: Aşama 1 güveni eşiğin (0.25) altında kalır, ölçüm yapılmaz.", "blank"),
    Example("referans", "Gerçek fotoğraf: README'deki referans görsel",
            "README'nin başındaki referans fotoğraf, çizim yerine **gerçek bir sırt fotoğrafı**; "
            "gönderilen modellerden olduğu gibi geçirildi. Sayılar bir başarım ölçüsü değildir; "
            "nedenleri sayıların altında.", "photo", source="docs/img/referans/skolyoz.webp"),
]


def _render(ex: Example, chain: Chain, ref, root: Path, out_dir: Path) -> None:
    from PIL import Image, ImageFilter

    if ex.kind == "blank":
        base = blank()
    elif ex.kind == "photo":
        base = Image.open(root / ex.source).convert("RGB")
    else:
        base = figure(ex.tilt)
        if ex.kind == "blur":
            base = base.filter(ImageFilter.GaussianBlur(12))

    lines: list[str] = []
    verdict = None
    m = Measurement(base.size, 0.0)
    if ex.kind == "pitch":
        assert not ref.guven_kontrolu(ex.pitch, PITCH_MAX)
        ex.rejected = f"|pitch| {ex.pitch:.1f}° > {PITCH_MAX:g}°"
        verdict = "ÖLÇÜM GEÇERSİZ"
        lines = [f"Telefon {ex.pitch:+.1f}° öne/arkaya yatık — dikleştirin",
                 "Deklanşör kilitli, model çalıştırılmadı"]
    else:
        m = chain.measure(base)
        if not m.found:
            ex.rejected = f"kişi bulunamadı (güven {m.confidence:.2f} < {CONFIDENCE_MIN})"
            verdict = "ÖLÇÜM YOK"
            lines = [f"Kişi bulunamadı (güven {m.confidence:.2f} < {CONFIDENCE_MIN})",
                     "Ölçüm yapılmadı"]
        else:
            sag, sol, c7, *_, sak = m.points
            ex.angle = ref.aksiller_aci(sag, sol, 0.0)
            dev_px, dev_deg, _ = ref.koronal_denge(c7, sak, 0.0)
            ex.coronal = (dev_px, dev_deg)
            ent = max(e for e, _ in m.gates[:2])
            peak = max(p for _, p in m.gates[:2])
            if m.reliable:
                lines = [f"Aksiller eğim: {ex.angle:+.2f}°",
                         f"Koronal sapma: {dev_px:+.1f} px  ({dev_deg:+.2f}°)",
                         f"Güven kapısı: geçti  (entropi {ent:.2f} ≤ {ENTROPY_MAX},"
                         f" tepe {peak:.1f} ≤ {PEAK_DISTANCE_MAX:g} px)"]
            else:
                failed = []
                if ent > ENTROPY_MAX:
                    failed.append(f"entropi {ent:.2f} > {ENTROPY_MAX}")
                if peak > PEAK_DISTANCE_MAX:
                    failed.append(f"tepe uzaklığı {peak:.1f} > {PEAK_DISTANCE_MAX:g} px")
                ex.rejected = " ve ".join(failed)
                verdict = "ÖLÇÜM GÜVENİLMEZ — yeniden çekin"
                lines = [f"Güven kapısı: REDDEDİLDİ  ({ex.rejected})",
                         "Açı gösterilmez; yeniden çekim istenir"]
    ex.measurement = m
    if ex.kind == "photo":      # measured at its own size, drawn at the figure's width
        factor = DISPLAY_WIDTH / base.width
        big = base.resize((DISPLAY_WIDTH, round(base.height * factor)), Image.Resampling.LANCZOS)
        shown = overlay(big, scaled(m, factor), lines, 0.0, verdict, guides=m.reliable,
                        footer=PHOTO_FOOTER, band_below=True)
    else:
        shown = overlay(base, m, lines, 0.0, verdict, guides=m.reliable)
    width = 600
    shown.resize((width, round(shown.height * width / shown.width)),
                 Image.Resampling.LANCZOS).save(out_dir / f"{ex.name}.png", optimize=True)


def run(root: Path, stage1: Path, stage2: Path, out_dir: Path, threads: int = 4) -> list[Example]:
    """Run every example, write PNGs to ``out_dir`` and docs/ORNEKLER.md."""
    ref = reference(root)
    chain = Chain(stage1, stage2, threads)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ex in EXAMPLES:
        _render(ex, chain, ref, root, out_dir)
    write_summary(root, out_dir, chain.hashes, threads)
    return EXAMPLES


def _photo_caveats(m: Measurement) -> str:
    """Why the real-photo output is not evidence, as a Markdown list."""
    width, height = m.size
    items = [
        "**Doğru değer yok.** Bu fotoğraf için aksiller eğimin gerçek değeri bilinmiyor; "
        "yukarıdaki sayı zincirin çıktısıdır, bir karşılaştırma değil.",
        f"**IMU yok.** Telefon sensörü olmadığından roll ve pitch 0 alındı. Piksel değerleri "
        f"{width}×{height} kaynak çözünürlüğündedir.",
        f"**Düşük çözünürlük.** {width}×{height} piksel, modele 640×640'a büyütülerek verildi; "
        "uygulamada telefon fotoğrafı küçültülür.",
        "**Eğitim verisinde.** Aynı kare, projenin yerel veri klasöründe (`raw/aag/train`, depoya "
        "girmez) en az üç artırılmış kopyayla bulundu (renk ve kırpma farkıyla; gözle "
        "doğrulandı). Bu kopyalarda yalnızca C7 ve sakrum etiketli. Aşama 1 eğitimi kısmi "
        "etiketli görüntüleri de kullanır ([TEKNIK_DOKUMAN §5.3](tr/TEKNIK_DOKUMAN.md)); bu "
        "yüzden model bu kareyi eğitimde görmüş olabilir ve sonuç genelleme göstermez. "
        "Koronal sapmayı belirleyen C7 ve sakrum o kopyalarda etiketliydi; eğimi belirleyen "
        "koltuk altı noktaları etiketli değildi.",
    ]
    outside = [(NAMES[i], e, dist) for i, (e, dist) in enumerate(m.gates)
               if i >= 2 and (e > ENTROPY_MAX or dist > PEAK_DISTANCE_MAX)]
    if outside:
        listed = ", ".join(f"{n} (entropi {e:.2f}, tepe {dist:.1f} px)" for n, e, dist in outside)
        item = ("**Kapı yalnızca iki koltuk altı noktasına bakar** (`OlcumZinciri.kt`). Kapının "
                f"dışında kalan şu noktalar aynı eşikleri aşıyor: {listed}.")
        if any(n in ("C7", "sakrum") for n, _, _ in outside):
            item += " Koronal sapma C7 ve sakruma dayandığı için o sayıya kapı güvencesi verilmez."
        items.append(item)
    return "\n".join(f"- {i}" for i in items) + "\n"


def write_summary(root: Path, out_dir: Path, hashes: dict[str, str], threads: int) -> None:
    """docs/ORNEKLER.md: the same examples with their numbers."""
    rel = out_dir.relative_to(root / "docs").as_posix()
    parts = [
        "# Örnek çıktılar\n",
        "<!-- `python -m skolyoz_metrics demo` ile ÜRETİLİR (model dosyaları gerekir). "
        "Elle düzenlemeyin. -->\n",
        "İlk beş görüntü gerçek bir kişiye ait değildir: **çizilmiş bir sırt figürü** kullanıldı "
        "(omuz eğimi çizimde bilinir). Son bölüm, README'nin başındaki tek **gerçek "
        "fotoğrafı** gösterir. İşaretler, uygulamayla aynı zincirin (`OlcumZinciri.kt`'nin "
        "Python aynası) gönderilen gerçek `.tflite` modelleri üzerindeki çıktısıdır.\n",
        "> [!IMPORTANT]\n> Bunlar **başarım kanıtı değil, boru hattının ne yaptığının "
        "gösterimidir.** Modeller gerçek fotoğraflarla eğitildi; çizilmiş bir figürde çalışması "
        "gerçek fotoğraflarda doğru çalışacağı anlamına gelmez (örneğin 5° çizilen eğim 3.3° "
        "bulundu: bu figürde nokta hatası etiket gürültü tabanının çok üzerinde). Gerçek telefon "
        "fotoğraflarında kararsızlık ölçüldü, bkz. [METRICS.md](METRICS.md). Gerçek fotoğraf "
        "bölümü de kanıt değildir; nedenleri o bölümde.\n",
        "> [!NOTE]\n> IMU roll düzeltmesi bu örneklerde **gösterilmiyor** (roll = 0): düzeltmenin "
        "işareti cihazda doğrulanmadı, bkz. METRICS.md → Bilinen boşluklar → `imu_roll_sign`.\n",
        f"Modeller: `asama1.tflite` `{hashes['asama1.tflite'][:16]}…`, `asama2.tflite` "
        f"`{hashes['asama2.tflite'][:16]}…` · {threads} iş parçacığı · {date.today():%Y-%m-%d}.\n",
    ]
    for ex in EXAMPLES:
        m = ex.measurement
        parts.append(f"## {ex.title}\n\n![{ex.title}]({rel}/{ex.name}.png)\n\n{ex.note}\n")
        if ex.kind == "pitch":
            parts.append(f"Karar: ölçüm reddedildi ({ex.rejected}).\n")
            continue
        if not m.found:
            parts.append(f"Aşama 1 en yüksek güven: {m.confidence:.3f} (eşik {CONFIDENCE_MIN}). "
                         "Ölçüm yapılmadı.\n")
            continue
        truth = f"Çizimde bilinen eğim: **{ex.tilt:+.1f}°** · " if ex.kind == "figure" else ""
        verdict = "geçti" if m.reliable else f"**reddedildi** ({ex.rejected})"
        parts.append(f"{truth}Aksiller eğim: **{ex.angle:+.2f}°** · koronal sapma: "
                     f"{ex.coronal[0]:+.1f} px ({ex.coronal[1]:+.2f}°) · güven kapısı: {verdict}"
                     f" · Aşama 1 güveni {m.confidence:.3f}\n")
        rows = ["| Nokta | Kaba (640 uzayı) | Rafine (orijinal çözünürlük) | Entropi "
                "| Tepe uzaklığı (px) |",
                "|---|---|---|---|---|"]
        for name, c, pt, (e, dist) in zip(NAMES, m.coarse, m.points, m.gates, strict=True):
            rows.append(f"| {name} | ({c[0]:.1f}, {c[1]:.1f}) | ({pt[0]:.1f}, {pt[1]:.1f}) "
                        f"| {e:.2f} | {dist:.1f} |")
        parts.append("\n".join(rows) + "\n")
        if ex.kind == "photo":
            parts.append("Bu sayıları okurken:\n\n" + _photo_caveats(m))
    parts.append(
        "## Nasıl yeniden üretilir\n\n```bash\npip install -r requirements/bench.txt pillow\n"
        "python -m skolyoz_metrics demo --stage1 asama1.tflite --stage2 asama2.tflite\n```\n")
    (root / "docs" / "ORNEKLER.md").write_text("\n".join(parts), encoding="utf-8", newline="\n")
    (out_dir / "ornekler.json").write_text(json.dumps(
        [{"name": ex.name, "found": ex.measurement.found,
          "confidence": round(ex.measurement.confidence, 4), "angle": ex.angle,
          "rejected": ex.rejected} for ex in EXAMPLES], indent=2) + "\n",
        encoding="utf-8", newline="\n")
