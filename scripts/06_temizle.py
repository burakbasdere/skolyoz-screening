"""Asama 1 - Kurtarma boru hatti.

Girdi : data/coco/train  (1119 dosya, 2.54x rotasyon kopyasi, etiketleri karisik)
Cikti : data/clean       (kopyasiz, dik, etiketi tutarli, sizintisiz bolunmus)

Adimlar
-------
1. Kopya kumelemesi        : 64x64 imza, 4 rotasyonda korelasyon, union-find
2. Temsilci secimi         : her kumeden en dik olan
3. Kanonik dondurme        : koltuk alti cizgisi yatay + govde asagida
4. Sag/sol yeniden atama   : arkadan cekimde sag koltuk karede sagda (%89 kural)
5. Kullanilamazlari ayikla : tek taraf / cift kutu
6. Sizintisiz bolme        : kume bazinda train/val/test

90 derecenin katlariyla dondurme kayipsizdir (interpolasyon yok).
"""

import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

KOK = Path(__file__).resolve().parent.parent
KAYNAK = KOK / "data" / "coco" / "train"
HEDEF = KOK / "data" / "clean"
COCO_JSON = KAYNAK / "_annotations.coco.json"

SAG, SOL = 1, 2
IMZA = 64
ESIK = 0.98
TOHUM = 1337
ORAN = (0.70, 0.15, 0.15)  # train / val / test


# ----------------------------------------------------------------- yardimci

def ileri_map(x, y, k, s):
    """A karesindeki (x,y) noktasini rot90(A,k) karesine tasir."""
    for _ in range(k % 4):
        x, y = y, s - 1 - x
    return x, y


def ten_maskesi(rgb):
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    return ((r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) &
            ((mx - mn) > 15) & (np.abs(r - g) > 15))


def aci_of(p_sag, p_sol):
    dx = p_sol[0] - p_sag[0]
    dy = p_sol[1] - p_sag[1]
    return math.degrees(math.atan2(dy, abs(dx))), math.hypot(dx, dy)


# ----------------------------------------------------------------- 1) yukle

def kayitlari_yukle():
    d = json.loads(COCO_JSON.read_text(encoding="utf-8"))
    isim = {g["id"]: g["file_name"] for g in d["images"]}
    per = defaultdict(list)
    for a in d["annotations"]:
        per[a["image_id"]].append(a)

    iyi, kotu = {}, []
    for img_id, anots in per.items():
        kats = sorted(a["category_id"] for a in anots)
        ad = isim[img_id]
        if kats != [SAG, SOL]:
            kotu.append((ad, "+".join(map(str, kats))))
            continue
        iyi[ad] = {a["category_id"]: [a["bbox"][0] + a["bbox"][2] / 2,
                                      a["bbox"][1] + a["bbox"][3] / 2]
                   for a in anots}
    return iyi, kotu


# ----------------------------------------------------------------- 2) kumele

def kumele(adlar):
    ham = np.zeros((len(adlar), IMZA, IMZA), dtype=np.float32)
    for i, ad in enumerate(adlar):
        ham[i] = np.asarray(
            Image.open(KAYNAK / ad).convert("L").resize((IMZA, IMZA), Image.BILINEAR),
            dtype=np.float32)

    def nrm(a):
        v = a.reshape(a.shape[0], -1)
        v = v - v.mean(axis=1, keepdims=True)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n < 1e-6] = 1.0
        return v / n

    taban = nrm(ham)
    ebeveyn = list(range(len(adlar)))

    def bul(x):
        while ebeveyn[x] != x:
            ebeveyn[x] = ebeveyn[ebeveyn[x]]
            x = ebeveyn[x]
        return x

    for k in range(4):
        kor = taban @ nrm(np.rot90(ham, k, axes=(1, 2)).copy()).T
        for i, j in zip(*np.where(kor > ESIK)):
            i, j = int(i), int(j)
            if i == j:
                continue
            ra, rb = bul(i), bul(j)
            if ra != rb:
                ebeveyn[max(ra, rb)] = min(ra, rb)

    kume = defaultdict(list)
    for i in range(len(adlar)):
        kume[bul(i)].append(adlar[i])
    return list(kume.values())


# --------------------------------------------------- 3) kanonik rotasyon bul

def kanonik_rotasyon(ad, merkez):
    """Goruntuyu dik hale getiren k (0..3) ve karar guveni."""
    im = Image.open(KAYNAK / ad).convert("RGB")
    rgb = np.asarray(im)
    s = rgb.shape[0]
    mask = ten_maskesi(rgb)

    adaylar = []
    for k in range(4):
        ps = ileri_map(*merkez[SAG], k, s)
        pl = ileri_map(*merkez[SOL], k, s)
        aci, D = aci_of(ps, pl)
        if abs(aci) > 45 or D < 1e-6:
            continue  # yatay degil, eleme
        m2 = np.rot90(mask, k)
        ys, xs = np.nonzero(m2)
        if len(ys) < 500:
            adaylar.append((k, aci, 0.0))
            continue
        my = (ps[1] + pl[1]) / 2.0
        # cizginin altinda kalan ten pikseli orani
        alt = float((ys > my).sum()) / len(ys)
        adaylar.append((k, aci, alt))

    if not adaylar:
        return None, 0.0
    # govdenin en cok asagida kaldigi aday
    k, aci, alt = max(adaylar, key=lambda t: t[2])
    rakip = min(adaylar, key=lambda t: t[2])
    guven = alt - rakip[2] if len(adaylar) > 1 else alt
    return k, guven


# ----------------------------------------------------------------- ana akis

def main():
    random.seed(TOHUM)
    iyi, kotu = kayitlari_yukle()
    print(f"[1] Kullanilabilir goruntu : {len(iyi)}")
    print(f"    Ayiklanan (tek taraf/cift kutu) : {len(kotu)}")

    adlar = sorted(iyi)
    kumeler = kumele(adlar)
    print(f"[2] Benzersiz kume : {len(kumeler)}  (sisme {len(adlar)/len(kumeler):.2f}x)")

    if HEDEF.exists():
        shutil.rmtree(HEDEF)
    (HEDEF / "images").mkdir(parents=True)

    kayitlar = []
    basarisiz = []

    for kume in kumeler:
        # temsilci: zaten en dik olan (en kucuk |aci|) -> en az islem
        en_iyi, en_iyi_aci = None, 1e9
        for ad in kume:
            aci, _ = aci_of(iyi[ad][SAG], iyi[ad][SOL])
            if abs(aci) < en_iyi_aci:
                en_iyi, en_iyi_aci = ad, abs(aci)
        ad = en_iyi

        k, guven = kanonik_rotasyon(ad, iyi[ad])
        if k is None:
            basarisiz.append(ad)
            continue

        im = Image.open(KAYNAK / ad).convert("RGB")
        arr = np.asarray(im)
        s = arr.shape[0]
        yeni = Image.fromarray(np.rot90(arr, k).copy())

        p1 = ileri_map(*iyi[ad][SAG], k, s)
        p2 = ileri_map(*iyi[ad][SOL], k, s)

        # 4) sag/sol yeniden atama: arkadan cekimde sag koltuk karede SAGDA
        p_sag, p_sol = (p1, p2) if p1[0] > p2[0] else (p2, p1)
        duzeltildi = not (p1[0] > p2[0])

        aci, D = aci_of(p_sag, p_sol)
        cikti_ad = f"{len(kayitlar):04d}_{ad.split('_jpg')[0].split('_JPG')[0]}.jpg"
        yeni.save(HEDEF / "images" / cikti_ad, quality=95)

        kayitlar.append({
            "dosya": cikti_ad,
            "kaynak": ad,
            "kume_boyu": len(kume),
            "rotasyon": k * 90,
            "guven": round(guven, 4),
            "sag": [round(p_sag[0], 2), round(p_sag[1], 2)],
            "sol": [round(p_sol[0], 2), round(p_sol[1], 2)],
            "aci": round(aci, 3),
            "D": round(D, 2),
            "etiket_duzeltildi": duzeltildi,
        })

    print(f"[3] Kanoniklestirildi : {len(kayitlar)}   basarisiz: {len(basarisiz)}")
    duz = sum(1 for r in kayitlar if r["etiket_duzeltildi"])
    print(f"[4] Sag/sol duzeltilen : {duz}  ({100.0*duz/len(kayitlar):.1f}%)")

    # 6) sizintisiz bolme  (kume zaten tekillestirildi, yine de kaynak bazinda)
    random.shuffle(kayitlar)
    n = len(kayitlar)
    n_tr = int(n * ORAN[0])
    n_va = int(n * ORAN[1])
    for i, r in enumerate(kayitlar):
        r["split"] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    sayim = defaultdict(int)
    for r in kayitlar:
        sayim[r["split"]] += 1
    print(f"[5] Bolme : " + "  ".join(f"{k}={v}" for k, v in sorted(sayim.items())))

    kayitlar.sort(key=lambda r: r["dosya"])
    (HEDEF / "kayitlar.json").write_text(
        json.dumps(kayitlar, ensure_ascii=False, indent=2), encoding="utf-8")

    guvenler = sorted(r["guven"] for r in kayitlar)
    print()
    print(f"    Oryantasyon karar guveni: medyan {guvenler[len(guvenler)//2]:.3f}   "
          f"p10 {guvenler[len(guvenler)//10]:.3f}   min {guvenler[0]:.3f}")
    dusuk = [r for r in kayitlar if r["guven"] < 0.10]
    print(f"    Guveni dusuk (<0.10) : {len(dusuk)}  -> gozle kontrol gerekir")
    print()
    print(f"Cikti: {HEDEF}")


if __name__ == "__main__":
    main()
