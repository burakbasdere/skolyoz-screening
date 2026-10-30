"""Deney: C7-sakrum arasında omurga orta hattı OpenCV ile izlenebilir mi?

FİKİR
-----
Sırt fotoğrafında omurga bir gölge oluğu (sulcus) oluşturur. Serbest arama
kırılgan; ama artık C7 ve sakrum etiketli, yani aramayı bu iki çapa arasındaki
dar bir şeride hapsedebiliriz. Arama uzayı küçülünce yöntem sağlamlaşır.

YÖNTEM
------
1. C7 -> sakrum eksenini kur, N seviyeye böl.
2. Her seviyede eksene DİK bir profil örnekle (±w piksel).
3. Profildeki aydınlanma eğimini çıkar (düşük dereceli polinom), kalan
   sinyalin minimumu = oluk. Bu, ışık gradyanına karşı bağışıklık sağlar.
4. Bulunan noktalara 3. derece polinom uydur; kalıntı RMS = GÜVEN ölçütü.
5. C7-sakrum düz çizgisinden sapma profili = eğrilik. Tepe noktası ve yönü
   klinik tarifin karşılığıdır ("sağa konveks torakal eğrilik").

BU BİR ÖLÇÜM DEĞİL. Yüzey orta hattı sapmasıdır, Cobb açısı değildir.
Amaç: güvenilir mi, ne kadar sık çalışıyor — bunu SAYIYLA görmek.
"""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

KOK = Path(__file__).resolve().parent.parent
AAG = KOK / "data" / "aag" / "train"
FINAL = KOK / "data" / "final"
CIKTI = KOK / "runs" / "omurga"

C7, SAKRUM = 1, 2
SEVIYE = 40          # C7-sakrum arası kaç seviyede örnekle
PROFIL_ORAN = 0.22   # arama yarı genişliği, gövde boyunun oranı olarak
POLI_DERECE = 3


def taban(ad):
    for k in ("_jpg.rf.", "_JPG_jpg.rf.", ".rf."):
        if k in ad:
            ad = ad.split(k)[0]
            break
    else:
        ad = Path(ad).stem
    for o in ("train_", "val_", "test_"):
        if ad.startswith(o + o):
            return ad[len(o):]
    return ad


def orta_hat_izle(gri, c7, sakrum, yari_genislik):
    """C7->sakrum ekseni boyunca oluk noktalarını bul.

    Döner: (t dizisi, sapma dizisi, gecerli maske)
    sapma: eksene dik yönde, sağa pozitif.
    """
    H, W = gri.shape
    ax = np.array(sakrum, dtype=np.float64) - np.array(c7, dtype=np.float64)
    L = np.linalg.norm(ax)
    if L < 20:
        return None
    u = ax / L                      # eksen yönü (aşağı)
    p = np.array([-u[1], u[0]])     # dik yön (sağa)

    ts = np.linspace(0.08, 0.92, SEVIYE)   # uçlardan kaçın
    ofs = np.arange(-yari_genislik, yari_genislik + 1, 1.0)
    sapmalar = np.full(SEVIYE, np.nan)

    for i, t in enumerate(ts):
        merkez = np.array(c7, dtype=np.float64) + u * (t * L)
        xs = merkez[0] + p[0] * ofs
        ys = merkez[1] + p[1] * ofs
        ic = (xs >= 0) & (xs < W - 1) & (ys >= 0) & (ys < H - 1)
        if ic.sum() < len(ofs) * 0.6:
            continue
        # bilinear örnekleme
        x0 = np.clip(xs[ic].astype(int), 0, W - 2)
        y0 = np.clip(ys[ic].astype(int), 0, H - 2)
        fx = xs[ic] - x0
        fy = ys[ic] - y0
        prof = (gri[y0, x0] * (1 - fx) * (1 - fy) + gri[y0, x0 + 1] * fx * (1 - fy) +
                gri[y0 + 1, x0] * (1 - fx) * fy + gri[y0 + 1, x0 + 1] * fx * fy)
        o = ofs[ic]
        if len(prof) < 15:
            continue
        # aydınlanma eğimini çıkar -> kalan sinyalin minimumu oluk
        kat = np.polyfit(o, prof, 2)
        kalan = prof - np.polyval(kat, o)
        # yumuşat
        cekirdek = np.ones(5) / 5.0
        kalan = np.convolve(kalan, cekirdek, mode="same")
        sapmalar[i] = o[int(np.argmin(kalan))]

    gecerli = ~np.isnan(sapmalar)
    if gecerli.sum() < SEVIYE * 0.6:
        return None
    return ts, sapmalar, gecerli


def main():
    d = json.loads((AAG / "_annotations.coco.json").read_text(encoding="utf-8"))
    isim = {g["id"]: g["file_name"] for g in d["images"]}
    per = defaultdict(list)
    for a in d["annotations"]:
        per[a["image_id"]].append(a)

    fin = {Path(r["dosya"]).stem: r for r in
           json.loads((FINAL / "kayitlar.json").read_text(encoding="utf-8"))}

    CIKTI.mkdir(parents=True, exist_ok=True)
    kalintilar, tepe_sapma, basarisiz = [], [], 0
    kareler = []

    for img_id, anots in per.items():
        if sorted(a["category_id"] for a in anots) != [C7, SAKRUM]:
            continue
        m = {a["category_id"]: (a["bbox"][0] + a["bbox"][2] / 2,
                                a["bbox"][1] + a["bbox"][3] / 2) for a in anots}
        c7p, skp = m[C7], m[SAKRUM]

        t = taban(isim[img_id])
        r = fin.get(t)
        if r is None:
            continue
        yol = FINAL / r["split"] / r["dosya"]
        if not yol.exists():
            continue

        im = Image.open(yol).convert("RGB")
        gri = np.asarray(im.convert("L"), dtype=np.float64)
        govde = math.dist(c7p, skp)

        sonuc = orta_hat_izle(gri, c7p, skp, govde * PROFIL_ORAN)
        if sonuc is None:
            basarisiz += 1
            continue
        ts, sap, gec = sonuc

        # düzgün eğri uydur; kalıntı = güven ölçütü
        kat = np.polyfit(ts[gec], sap[gec], POLI_DERECE)
        duzgun = np.polyval(kat, ts)
        kalinti = float(np.sqrt(np.mean((sap[gec] - duzgun[gec]) ** 2)))
        kalintilar.append(kalinti / govde * 100.0)   # gövde boyunun yüzdesi

        tepe_i = int(np.argmax(np.abs(duzgun)))
        tepe_sapma.append(float(duzgun[tepe_i]) / govde * 100.0)

        if len(kareler) < 12:
            kareler.append((yol, c7p, skp, ts, sap, gec, duzgun, kalinti, govde))

    print("=" * 66)
    print("OMURGA IZI FIZIBILITE")
    print("=" * 66)
    print(f"  denenen goruntu     : {len(kalintilar) + basarisiz}")
    print(f"  iz cikarilabildi    : {len(kalintilar)}")
    print(f"  basarisiz           : {basarisiz}")
    print()
    print("  KALINTI RMS (govde boyunun %'si) -- DUSUK = duzgun/guvenilir iz")
    k = sorted(kalintilar)
    print(f"    medyan {statistics.median(k):.2f}%   p75 {k[int(len(k)*.75)]:.2f}%"
          f"   p90 {k[int(len(k)*.9)]:.2f}%   max {max(k):.2f}%")
    print()
    print("  Yorum: kalinti %1'in altindaysa iz duzgun, omurga olugu izlenmis.")
    print("         %3 ustu ise gurultu izliyoruz, guvenilmez.")
    for esik in (0.5, 1.0, 2.0, 3.0):
        n = sum(1 for x in kalintilar if x <= esik)
        print(f"    kalinti <= {esik:.1f}% : {n:4d} / {len(kalintilar)}"
              f"  ({100.0*n/len(kalintilar):.1f}%)")

    print()
    print("  TEPE SAPMA (govde boyunun %'si, + = saga)")
    ts_ = sorted(abs(x) for x in tepe_sapma)
    print(f"    |tepe| medyan {statistics.median(ts_):.2f}%   "
          f"p90 {ts_[int(len(ts_)*.9)]:.2f}%   max {max(ts_):.2f}%")

    # ---- gorsel ----
    K = 320
    tuval = Image.new("RGB", (4 * K, 3 * K + 20), (16, 16, 16))
    dd = ImageDraw.Draw(tuval)
    dd.text((6, 5), "YESIL = C7-sakrum duz cizgi   KIRMIZI = izlenen omurga hatti"
                    "   (basliktaki sayi: kalinti %)", fill=(255, 255, 255))
    for i, (yol, c7p, skp, ts, sap, gec, duzgun, kalinti, govde) in enumerate(kareler):
        im = Image.open(yol).convert("RGB")
        o = K / im.size[0]
        im = im.resize((K, K), Image.BILINEAR)
        g = ImageDraw.Draw(im)
        g.line([(c7p[0] * o, c7p[1] * o), (skp[0] * o, skp[1] * o)],
               fill=(0, 230, 0), width=2)
        ax = np.array(skp) - np.array(c7p)
        L = np.linalg.norm(ax)
        u = ax / L
        p = np.array([-u[1], u[0]])
        nokta = []
        for t, s in zip(ts, duzgun):
            q = np.array(c7p) + u * (t * L) + p * s
            nokta.append((q[0] * o, q[1] * o))
        g.line(nokta, fill=(255, 60, 60), width=3)
        g.rectangle([0, 0, K, 16], fill=(0, 0, 0))
        g.text((4, 3), f"kalinti {kalinti/govde*100:.2f}%", fill=(255, 255, 255))
        tuval.paste(im, ((i % 4) * K, (i // 4) * K + 20))
    p_ = CIKTI / "omurga_izi.jpg"
    tuval.save(p_, quality=92)
    print(f"\n  gorsel: {p_}")


if __name__ == "__main__":
    main()
