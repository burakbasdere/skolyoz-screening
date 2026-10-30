"""TFLite boru hattinin hiz olcumu (mobil hedef).

Calistirma (.venv-tflite icinde):
    .venv-tflite/Scripts/python.exe scripts/31_tflite_performans.py
"""

import json
import math
import statistics
import sys
import time
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
degerlendirme = import_module("26_tflite_degerlendirme")

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "final"
TFLITE1 = KOK / "android" / "app" / "src" / "main" / "assets" / "asama1.tflite"
TFLITE2 = KOK / "android" / "app" / "main" / "assets" / "asama2.tflite" if False else KOK / "android" / "app" / "src" / "main" / "assets" / "asama2.tflite"

ISITMA = 5
TEKRAR = 20


def yuzdelik(v, p):
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    a, u = math.floor(k), math.ceil(k)
    return s[int(k)] if a == u else s[a] * (u - k) + s[u] * (k - a)


def olc(m1, m2, yol):
    from PIL import Image
    import numpy as np

    t0 = time.perf_counter()
    im = Image.open(yol).convert("RGB")
    x = np.asarray(im.resize((640, 640), Image.BILINEAR), dtype=np.float32) / 255.0
    kaba = degerlendirme.asama1_coz(m1(x[None, ...]))
    t1 = time.perf_counter()
    if kaba is None:
        return None, (t1 - t0) * 1000, 0.0, 0.0
    for nokta in (kaba["sag"], kaba["sol"]):
        degerlendirme.rafine_et(m2, im, nokta, heatmap_mi=True)
    t2 = time.perf_counter()
    return True, (t2 - t0) * 1000, (t1 - t0) * 1000, (t2 - t1) * 1000


def main():
    kayitlar = [r for r in json.loads((VERI / "kayitlar.json").read_text(encoding="utf-8"))
                if r["split"] == "test"][:TEKRAR]
    print(f"TFLite performans olcumu — n={len(kayitlar)}")
    print(f"Modeller: {TFLITE1.name} ({TFLITE1.stat().st_size/1e6:.2f} MB), "
          f"{TFLITE2.name} ({TFLITE2.stat().st_size/1e6:.2f} MB)")

    m1 = degerlendirme.TFLite(TFLITE1)
    m2 = degerlendirme.TFLite(TFLITE2)
    print(f"XNNPACK: asama1={m1.xnnpack}, asama2={m2.xnnpack}")

    for r in kayitlar[:ISITMA]:
        olc(m1, m2, VERI / r["split"] / r["dosya"])

    toplamlar, a1ler, a2ler = [], [], []
    for r in kayitlar:
        ok, toplam, a1, a2 = olc(m1, m2, VERI / r["split"] / r["dosya"])
        if ok:
            toplamlar.append(toplam)
            a1ler.append(a1)
            a2ler.append(a2)

    print(f"\nBasarili olcum: {len(toplamlar)}/{len(kayitlar)}")
    print(f"  toplam ms   medyan={statistics.median(toplamlar):6.1f}  "
          f"p90={yuzdelik(toplamlar, 90):6.1f}  "
          f"min={min(toplamlar):6.1f}  max={max(toplamlar):6.1f}")
    print(f"  asama1 ms   medyan={statistics.median(a1ler):6.1f}  p90={yuzdelik(a1ler, 90):6.1f}")
    print(f"  asama2 ms   medyan={statistics.median(a2ler):6.1f}  p90={yuzdelik(a2ler, 90):6.1f}")
    print(f"\nMobil hedef (MOBIL_YAPILACAKLAR): ~100-200 ms/cihaz")
    print(f"Bu olcum: masaustu CPU, PIL resize dahil (Android'de farkli olabilir)")


if __name__ == "__main__":
    main()
