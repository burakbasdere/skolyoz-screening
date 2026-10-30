"""Asama 1a - Oryantasyon tespiti icin varsayim dogrulama.

Kurtarma boru hattini yazmadan once iki seyi veriyle sinamaliyiz:

  V1) Arkadan cekimde kisinin sag koltuk alti karede SAGDA mi?
      (etiketlerin ne kadari bu kurala uyuyor?)

  V2) Dik konumu icerikten bulabilir miyiz?
      Fikir: ten rengi maskesinin agirlik merkezi, koltuk alti
      cizgisinin ALT tarafinda kalir (govde + kalca asagida).
      Bu 4 rotasyonun hangisinin dik oldugunu tek basina belirler.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

KLASOR = Path(__file__).resolve().parent.parent / "data" / "coco" / "train"
COCO_JSON = KLASOR / "_annotations.coco.json"
SAG, SOL = 1, 2


def ten_maskesi(rgb):
    """Klasik RGB ten rengi kurali. Duz beyaz/bej duvar elenir (max-min kucuk)."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    return ((r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) &
            ((mx - mn) > 15) & (np.abs(r - g) > 15))


def main():
    d = json.loads(COCO_JSON.read_text(encoding="utf-8"))
    isim = {g["id"]: g["file_name"] for g in d["images"]}
    per = defaultdict(list)
    for a in d["annotations"]:
        per[a["image_id"]].append(a)

    kayit = {}
    for img_id, anots in per.items():
        if sorted(a["category_id"] for a in anots) != [SAG, SOL]:
            continue
        kayit[isim[img_id]] = {
            a["category_id"]: (a["bbox"][0] + a["bbox"][2] / 2,
                               a["bbox"][1] + a["bbox"][3] / 2)
            for a in anots
        }

    print(f"Kullanilabilir goruntu: {len(kayit)}")

    # ---------- V1 ----------
    print()
    print("=" * 64)
    print("V1: 'sag koltuk' karede sagda mi?")
    print("=" * 64)
    dik_sag, dik_sol, donuk = 0, 0, 0
    for ad, m in kayit.items():
        dx = m[SOL][0] - m[SAG][0]
        dy = m[SOL][1] - m[SAG][1]
        if abs(math.degrees(math.atan2(dy, abs(dx)))) > 45:
            donuk += 1
            continue
        if m[SAG][0] > m[SOL][0]:
            dik_sag += 1
        else:
            dik_sol += 1
    top = dik_sag + dik_sol
    print(f"  Yatay (dik/ters) goruntu : {top}")
    print(f"    sag koltuk SAGDA  : {dik_sag:4d}  ({100.0*dik_sag/top:.1f}%)")
    print(f"    sag koltuk SOLDA  : {dik_sol:4d}  ({100.0*dik_sol/top:.1f}%)")
    print(f"  Dikey (90 donuk) atlandi : {donuk}")
    print()
    print("  Yorum: ~%50/50 ise kural yok, etiket konuma gore rastgele.")
    print("         Baskin bir taraf varsa kural var, azinlik = hatali etiket.")

    # ---------- V2 ----------
    print()
    print("=" * 64)
    print("V2: ten maskesi agirlik merkezi koltuk alti cizgisinin altinda mi?")
    print("=" * 64)

    ornek = sorted(kayit)[:250]
    sonuc = Counter()
    mesafeler = []

    for ad in ornek:
        m = kayit[ad]
        im = Image.open(KLASOR / ad).convert("RGB")
        rgb = np.asarray(im)
        mask = ten_maskesi(rgb)
        n = int(mask.sum())
        if n < 500:
            sonuc["ten_bulunamadi"] += 1
            continue
        ys, xs = np.nonzero(mask)
        cx, cy = xs.mean(), ys.mean()

        (xs_, ys_), (xl, yl) = m[SAG], m[SOL]
        mx, my = (xs_ + xl) / 2.0, (ys_ + yl) / 2.0
        dx, dy = xl - xs_, yl - ys_
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        # cizgiye dik birim vektor (goruntu koordinatinda +y asagi)
        px, py = -dy / L, dx / L
        # agirlik merkezinin cizgiye gore isaretli uzakligi
        s = (cx - mx) * px + (cy - my) * py
        # "asagi" yonu +y ise, dik goruntude s ile py ayni isarette olmali
        yon = s * (1 if py > 0 else -1)
        mesafeler.append(abs(s) / L)
        sonuc["ten_altta" if yon > 0 else "ten_ustte"] += 1

    print(f"  Incelenen: {len(ornek)}")
    for k, v in sonuc.most_common():
        print(f"    {k:18s}: {v}")
    if mesafeler:
        mesafeler.sort()
        print(f"  |uzaklik| / D  ->  medyan {mesafeler[len(mesafeler)//2]:.3f}   "
              f"p10 {mesafeler[len(mesafeler)//10]:.3f}")
        print()
        print("  Yorum: 'ten_altta' baskinsa ve uzaklik/D belirgin (>0.15) ise")
        print("         bu olcut dik konumu guvenilir sekilde belirler.")


if __name__ == "__main__":
    main()
