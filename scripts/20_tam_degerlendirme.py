"""Asama 5 - Tam boru hatti degerlendirmesi + urun metrikleri.

Boru hatti:
    Asama 1 (YOLO-pose)  -> kaba nokta (~24 px)
    Asama 2 (rafinaj agi) -> kirpim uzerinde heatmap + soft-argmax -> hassas nokta

METRIKLER
---------
F1 / recall / accuracy siniflandirma metrikleridir; koordinat regresyonuna
dogrudan uygulanmaz. Bu yuzden uc katmanda raporluyoruz:

  1) Lokalizasyon : piksel hatasi + PCK (keypoint dunyasinin "accuracy"si)
  2) Olcum        : aci hatasi (derece)
  3) Karar        : urunun gercekten SINIFLANDIRMA yaptigi iki nokta
       a) hangi omuz yuksek  (sag / sol / simetrik)  -> 3 sinif
       b) tarama karari |aci| >= esik                -> 2 sinif
     Bunlarda accuracy / precision / recall / F1 gercekten anlamli.
     Tarama aracinda RECALL kritiktir: vakayi kacirmamak gerekir.
"""

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
RafinajAgi = import_module("17_rafinaj_model").RafinajAgi
kilavuz = import_module("19_kilavuz_cizim")

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "final"
YOLO_AGIRLIK = KOK / "runs" / "baseline" / "weights" / "best.pt"
RAF_AGIRLIK = KOK / "runs" / "rafinaj" / "best.pt"

KIRPIM = 128
ORT = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIMETRI_BANDI = 1.0     # |aci| < 1 derece -> simetrik say
TARAMA_ESIKLERI = (3.0, 5.0, 7.0)


def yuzdelik(v, p):
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    a, u = math.floor(k), math.ceil(k)
    return s[int(k)] if a == u else s[a] * (u - k) + s[u] * (k - a)


def rafine_et(model, im, kaba, cihaz):
    """Kaba nokta etrafindan kirpip alt-piksel nokta dondurur."""
    sol = kaba[0] - KIRPIM / 2
    ust = kaba[1] - KIRPIM / 2
    kirpim = im.crop((int(round(sol)), int(round(ust)),
                      int(round(sol)) + KIRPIM, int(round(ust)) + KIRPIM))
    a = np.asarray(kirpim.convert("RGB"), dtype=np.float32) / 255.0
    a = (a - ORT) / STD
    x = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(cihaz)
    with torch.no_grad():
        koord, _ = model(x)
    kx, ky = koord[0].cpu().tolist()
    return (round(sol) + kx, round(ust) + ky)


def sinif_metrikleri(gercek, tahmin, etiketler):
    """Cok sinifli precision / recall / F1 + accuracy."""
    n = len(gercek)
    dogru = sum(1 for g, t in zip(gercek, tahmin) if g == t)
    acc = dogru / n if n else 0.0
    satir = {}
    for e in etiketler:
        tp = sum(1 for g, t in zip(gercek, tahmin) if g == e and t == e)
        fp = sum(1 for g, t in zip(gercek, tahmin) if g != e and t == e)
        fn = sum(1 for g, t in zip(gercek, tahmin) if g == e and t != e)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        satir[e] = {"destek": tp + fn, "precision": p, "recall": r, "f1": f}
    makro = sum(v["f1"] for v in satir.values()) / len(etiketler)
    return acc, satir, makro


def sinif_yaz(baslik, gercek, tahmin, etiketler):
    acc, satir, makro = sinif_metrikleri(gercek, tahmin, etiketler)
    print(f"\n  {baslik}")
    print(f"    accuracy : {acc:.3f}   makro-F1 : {makro:.3f}")
    print(f"    {'sinif':<12}{'destek':>7}{'precision':>11}{'recall':>9}{'F1':>8}")
    for e in etiketler:
        v = satir[e]
        print(f"    {str(e):<12}{v['destek']:>7}{v['precision']:>11.3f}"
              f"{v['recall']:>9.3f}{v['f1']:>8.3f}")
    print(f"    karisiklik: ", end="")
    kar = Counter(zip(gercek, tahmin))
    print("  ".join(f"{g}->{t}:{c}" for (g, t), c in sorted(kar.items()) if g != t)
          or "hata yok")
    return acc, makro


def main():
    from ultralytics import YOLO
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"

    y = YOLO(str(YOLO_AGIRLIK))
    d = torch.load(RAF_AGIRLIK, weights_only=False)
    raf = RafinajAgi(pretrained=False).to(cihaz)
    raf.load_state_dict(d["model"])
    raf.eval()
    print(f"Rafinaj agirligi: epoch {d['epoch']}, val medyan {d['val_medyan']:.3f} px")

    kayitlar = [r for r in json.loads((VERI / "kayitlar.json").read_text(encoding="utf-8"))
                if r["split"] == "test"]
    print(f"Test goruntusu: {len(kayitlar)}")

    px1, px2, aci1, aci2 = [], [], [], []
    g_aci_ler, t_aci_ler = [], []
    atlanan = 0

    for r in kayitlar:
        yol = VERI / r["split"] / r["dosya"]
        im = Image.open(yol).convert("RGB")
        g = {"sag": tuple(r["sag"]), "sol": tuple(r["sol"])}

        s = y.predict(str(yol), verbose=False, conf=0.25)[0]
        if s.keypoints is None or len(s.boxes) == 0:
            atlanan += 1
            continue
        i = int(np.argmax(s.boxes.conf.cpu().numpy()))
        kp = s.keypoints.data.cpu().numpy()[i]
        kaba = {"sag": (float(kp[0][0]), float(kp[0][1])),
                "sol": (float(kp[1][0]), float(kp[1][1]))}
        ince = {k: rafine_et(raf, im, v, cihaz) for k, v in kaba.items()}

        for k in ("sag", "sol"):
            px1.append(math.dist(kaba[k], g[k]))
            px2.append(math.dist(ince[k], g[k]))

        ga = kilavuz.aksiller_aci(g["sag"], g["sol"])
        aci1.append(abs(kilavuz.aksiller_aci(kaba["sag"], kaba["sol"]) - ga))
        ta = kilavuz.aksiller_aci(ince["sag"], ince["sol"])
        aci2.append(abs(ta - ga))
        g_aci_ler.append(ga)
        t_aci_ler.append(ta)

    n = len(g_aci_ler)
    print(f"Degerlendirilen: {n}  (tespit edilemeyen {atlanan})")

    print()
    print("=" * 70)
    print("1. LOKALIZASYON")
    print("=" * 70)
    print(f"  {'':22}{'medyan':>10}{'ortalama':>11}{'p90':>9}")
    for ad, v in (("Asama 1 (YOLO)", px1), ("Asama 1+2 (rafine)", px2)):
        print(f"  {ad:<22}{statistics.median(v):>9.2f}p{np.mean(v):>10.2f}"
              f"{yuzdelik(v,90):>9.2f}")
    print(f"  {'etiket gurultusu':<22}{2.45:>10.2f}")
    print(f"\n  iyilesme: {statistics.median(px1)/statistics.median(px2):.1f}x")

    print("\n  PCK (esik altinda kalan nokta orani) = keypoint 'accuracy'")
    print(f"    {'esik':<10}{'Asama 1':>10}{'Asama 1+2':>12}")
    for e in (2.45, 5, 10, 20):
        a = 100.0 * sum(1 for v in px1 if v <= e) / len(px1)
        b = 100.0 * sum(1 for v in px2 if v <= e) / len(px2)
        print(f"    {e:<10.2f}{a:>9.1f}%{b:>11.1f}%")

    print()
    print("=" * 70)
    print("2. OLCUM (aci hatasi, derece)")
    print("=" * 70)
    print(f"  {'':22}{'medyan':>10}{'ortalama':>11}{'p90':>9}")
    for ad, v in (("Asama 1 (YOLO)", aci1), ("Asama 1+2 (rafine)", aci2)):
        print(f"  {ad:<22}{statistics.median(v):>10.3f}{np.mean(v):>11.3f}"
              f"{yuzdelik(v,90):>9.3f}")
    print(f"  {'etiket gurultusu':<22}{0.435:>10.3f}")
    print(f"\n  iyilesme: {statistics.median(aci1)/statistics.median(aci2):.1f}x")

    print()
    print("=" * 70)
    print("3. KARAR METRIKLERI  (F1 / recall / accuracy burada anlamli)")
    print("=" * 70)

    def sinifla(a):
        if abs(a) < SIMETRI_BANDI:
            return "simetrik"
        return "sag" if a > 0 else "sol"

    sinif_yaz("a) Hangi omuz yuksek (3 sinif)",
              [sinifla(a) for a in g_aci_ler], [sinifla(a) for a in t_aci_ler],
              ["sag", "sol", "simetrik"])

    for esik in TARAMA_ESIKLERI:
        sinif_yaz(f"b) Tarama karari |aci| >= {esik:.0f}° (2 sinif)",
                  ["pozitif" if abs(a) >= esik else "negatif" for a in g_aci_ler],
                  ["pozitif" if abs(a) >= esik else "negatif" for a in t_aci_ler],
                  ["pozitif", "negatif"])


if __name__ == "__main__":
    main()
