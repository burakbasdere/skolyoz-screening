"""Asama 1c - Gri tonlamada da calisan oryantasyon olcutu.

Sorun: ten rengi maskesi siyah-beyaz goruntulerde sifir sinyal veriyor,
oryantasyon karari korlemesine veriliyor.

Yeni olcut - SILUET GENISLIK PROFILI:
  Koltuk alti cizgisinin USTUNDE govde boyuna dogru daralir (dar).
  ALTINDA govde genis kalir (torso).
  Cizgiye dik yonde +0.3D ve -0.3D mesafede siluet genisligini olcup
  genis olan tarafi "asagi" kabul ediyoruz. Renkten bagimsiz.

Siluet: kenar piksellerinden arka plan rengi tahmin edilir, ondan yeterince
farkli pikseller on plan sayilir.

DOGRULAMA: renk ipucunun emin oldugu (|guven| yuksek) vakalarda iki olcut
ayni cevabi veriyor mu? Veriyorsa yeni olcut gri goruntulerde de guvenilir.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

KAYNAK = Path(__file__).resolve().parent.parent / "data" / "coco" / "train"
COCO_JSON = KAYNAK / "_annotations.coco.json"
SAG, SOL = 1, 2


def ten_maskesi(rgb):
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    return ((r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) &
            ((mx - mn) > 15) & (np.abs(r - g) > 15))


def siluet(rgb):
    """Kenar piksellerinden arka plan rengi -> ondan uzak pikseller on plan."""
    h, w = rgb.shape[:2]
    kenar = np.concatenate([
        rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3),
        rgb[:, :8].reshape(-1, 3), rgb[:, -8:].reshape(-1, 3)])
    arka = np.median(kenar, axis=0)
    uzaklik = np.linalg.norm(rgb.astype(np.float32) - arka, axis=2)
    esik = max(28.0, float(np.percentile(uzaklik, 45)))
    return uzaklik > esik


def genislik_skoru(mask, p_sag, p_sol):
    """Cizgiye dik +/- yonde siluet genisligi. Pozitif => govde +dik yonde."""
    mx = (p_sag[0] + p_sol[0]) / 2.0
    my = (p_sag[1] + p_sol[1]) / 2.0
    dx, dy = p_sol[0] - p_sag[0], p_sol[1] - p_sag[1]
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return 0.0
    ux, uy = dx / L, dy / L           # cizgi yonu
    px, py = -uy, ux                  # dik yon
    h, w = mask.shape

    def genislik(isaret):
        toplam, n = 0, 0
        for t in (0.25, 0.35, 0.45):
            cx = mx + px * isaret * t * L
            cy = my + py * isaret * t * L
            sayac = 0
            for s in np.arange(-0.75, 0.75, 0.02):
                x = int(round(cx + ux * s * L))
                y = int(round(cy + uy * s * L))
                if 0 <= x < w and 0 <= y < h and mask[y, x]:
                    sayac += 1
            toplam += sayac
            n += 1
        return toplam / max(n, 1)

    arti = genislik(+1)
    eksi = genislik(-1)
    if arti + eksi < 1e-6:
        return 0.0
    return (arti - eksi) / (arti + eksi)


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

    # sadece yatay (dik ya da ters) goruntuler uzerinde sina
    yatay = []
    for ad, m in kayit.items():
        dx = m[SOL][0] - m[SAG][0]
        dy = m[SOL][1] - m[SAG][1]
        if abs(math.degrees(math.atan2(dy, abs(dx)))) <= 45:
            yatay.append(ad)
    yatay.sort()
    print(f"Yatay goruntu: {len(yatay)}  (ilk 400 uzerinde sinaniyor)")

    sonuc = Counter()
    renk_emin_uyum = [0, 0]

    for ad in yatay[:400]:
        rgb = np.asarray(Image.open(KAYNAK / ad).convert("RGB"))
        m = kayit[ad]
        ps, pl = m[SAG], m[SOL]

        # --- renk ipucu ---
        tm = ten_maskesi(rgb)
        renk_yon, renk_guc = 0, 0.0
        if tm.sum() >= 500:
            ys, _ = np.nonzero(tm)
            my = (ps[1] + pl[1]) / 2.0
            alt = float((ys > my).sum()) / len(ys)
            renk_yon = 1 if alt > 0.5 else -1
            renk_guc = abs(alt - 0.5) * 2

        # --- genislik ipucu ---
        sm = siluet(rgb)
        skor = genislik_skoru(sm, ps, pl)
        # +dik yon (px,py) goruntude asagiyi mi gosteriyor?
        dx, dy = pl[0] - ps[0], pl[1] - ps[1]
        L = math.hypot(dx, dy)
        py = dx / L
        gen_yon = (1 if skor > 0 else -1) * (1 if py > 0 else -1)

        sonuc["gen_altta" if gen_yon > 0 else "gen_ustte"] += 1

        if renk_guc > 0.5:
            renk_emin_uyum[0 if renk_yon == gen_yon else 1] += 1

    print()
    print("=" * 62)
    print("GENISLIK OLCUTU SONUCU")
    print("=" * 62)
    for k, v in sonuc.most_common():
        print(f"  {k:12s}: {v}")

    u, f = renk_emin_uyum
    print()
    print("=" * 62)
    print("DOGRULAMA: renk ipucunun EMIN oldugu vakalarda uyum")
    print("=" * 62)
    print(f"  uyusuyor : {u}")
    print(f"  celisiyor: {f}")
    if u + f:
        print(f"  -> uyum orani: {100.0*u/(u+f):.1f}%")
        print()
        print("  >%90 ise genislik olcutu tek basina guvenilir, gri")
        print("  goruntulerde de kullanabiliriz.")


if __name__ == "__main__":
    main()
