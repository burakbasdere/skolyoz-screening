"""Asama 2d - 40 px hata nereden geliyor?

Iki ihtimal:
  A) SISTEMATIK kayma - model noktalari tutarli bir yone koyuyor
     (ornek: gercek koltuk alti yerine omuz ucu). Duzeltilebilir.
  B) RASTGELE sacilim - model kararsiz. Mimari/veri sorunu.

Isaretli hata bilesenlerinin ORTALAMASI sistematigi, STANDART SAPMASI
sacilimi verir. Ayrica tahmin+gercek ayni karede ust uste cizilir.
"""

import math
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "yolo"
AGIRLIK = KOK / "runs" / "baseline" / "weights" / "best.pt"
CIKTI = KOK / "runs" / "baseline"
KARE, SUTUN, SATIR = 300, 5, 3


def main():
    from ultralytics import YOLO
    model = YOLO(str(AGIRLIK))

    yollar = sorted((VERI / "test" / "images").glob("*.jpg"))
    kayit = []

    for p in yollar:
        im = Image.open(p).convert("RGB")
        W, H = im.size
        t = (VERI / "test" / "labels" / (p.stem + ".txt")).read_text().split()
        g = [(float(t[5]) * W, float(t[6]) * H),
             (float(t[8]) * W, float(t[9]) * H)]

        # ultralytics numpy diziyi BGR varsayar; dosya yolu vermek dogrusu
        r = model.predict(str(p), verbose=False, conf=0.25)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            continue
        i = int(np.argmax(r.boxes.conf.cpu().numpy()))
        kp = r.keypoints.data.cpu().numpy()[i]
        t_pts = [(float(kp[0][0]), float(kp[0][1])), (float(kp[1][0]), float(kp[1][1]))]
        kayit.append((p, g, t_pts))

    print("=" * 64)
    print("HATA BILESENLERI  (tahmin - gercek, piksel)")
    print("=" * 64)
    for idx, ad in ((0, "SAG"), (1, "SOL")):
        dx = [t[idx][0] - g[idx][0] for _, g, t in kayit]
        dy = [t[idx][1] - g[idx][1] for _, g, t in kayit]
        print(f"  {ad}:")
        print(f"    dx  ort {statistics.mean(dx):+7.2f}  std {statistics.stdev(dx):6.2f}")
        print(f"    dy  ort {statistics.mean(dy):+7.2f}  std {statistics.stdev(dy):6.2f}")

    print()
    print("  YORUM: |ort| >> std ise SISTEMATIK kayma (duzeltilebilir).")
    print("         std  >> |ort| ise RASTGELE sacilim (mimari/veri sorunu).")

    # sistematigi cikarinca kalan aci hatasi ne olurdu?
    duz = []
    for idx in (0, 1):
        duz.append((statistics.mean([t[idx][0] - g[idx][0] for _, g, t in kayit]),
                    statistics.mean([t[idx][1] - g[idx][1] for _, g, t in kayit])))

    ham_aci, duz_aci = [], []
    for _, g, t in kayit:
        def aci(a, b):
            return math.degrees(math.atan2(b[1] - a[1], abs(b[0] - a[0])))
        ham_aci.append(abs(aci(t[0], t[1]) - aci(g[0], g[1])))
        td = [(t[0][0] - duz[0][0], t[0][1] - duz[0][1]),
              (t[1][0] - duz[1][0], t[1][1] - duz[1][1])]
        duz_aci.append(abs(aci(td[0], td[1]) - aci(g[0], g[1])))

    print()
    print("=" * 64)
    print("SISTEMATIK KAYMA CIKARILIRSA")
    print("=" * 64)
    print(f"  ham aci hatasi (medyan)      : {statistics.median(ham_aci):.3f}°")
    print(f"  kayma cikarilmis (medyan)    : {statistics.median(duz_aci):.3f}°")
    print(f"  kazanc                       : "
          f"{statistics.median(ham_aci)/max(statistics.median(duz_aci),1e-9):.2f}x")

    # ---- gorsel: tahmin (kirmizi) vs gercek (yesil) ----
    tuval = Image.new("RGB", (SUTUN * KARE, SATIR * KARE + 20), (18, 18, 18))
    d0 = ImageDraw.Draw(tuval)
    d0.text((6, 5), "YESIL = gercek etiket    KIRMIZI = model tahmini", fill=(255, 255, 255))
    for i, (p, g, t) in enumerate(kayit[:SUTUN * SATIR]):
        im = Image.open(p).convert("RGB")
        o = KARE / im.size[0]
        im = im.resize((KARE, KARE), Image.BILINEAR)
        d = ImageDraw.Draw(im)
        d.line([(g[0][0] * o, g[0][1] * o), (g[1][0] * o, g[1][1] * o)],
               fill=(0, 230, 0), width=3)
        d.line([(t[0][0] * o, t[0][1] * o), (t[1][0] * o, t[1][1] * o)],
               fill=(255, 40, 40), width=3)
        for pt, renk in ((g, (0, 230, 0)), (t, (255, 40, 40))):
            for q in pt:
                x, y = q[0] * o, q[1] * o
                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=renk,
                          outline=(255, 255, 255))
        tuval.paste(im, ((i % SUTUN) * KARE, (i // SUTUN) * KARE + 20))
    yol = CIKTI / "teshis_tahmin_vs_gercek.jpg"
    tuval.save(yol, quality=92)
    print(f"\n  yazildi: {yol}")


if __name__ == "__main__":
    main()
