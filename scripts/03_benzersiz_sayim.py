"""Asama 0d - Tum veri setinde benzersiz fotograf sayisi.

1119 goruntunun kaci gercekten farkli fotograf? Rotasyon kopyalarini ve
birebir kopyalari kumeleyip efektif veri buyuklugunu cikarir.

Yontem: 64x64 gri imza -> 4 rotasyonda korelasyon matrisi -> esik ustu
baglantilarla birlesim-bul (union-find) kumelemesi.
"""

import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

KLASOR = Path(__file__).resolve().parent.parent / "data" / "coco" / "train"
N = 64
ESIK = 0.98


def main():
    yollar = sorted(KLASOR.glob("*.jpg"))
    print(f"Goruntu okunuyor: {len(yollar)}")

    ham = np.zeros((len(yollar), N, N), dtype=np.float32)
    for i, p in enumerate(yollar):
        ham[i] = np.asarray(Image.open(p).convert("L").resize((N, N), Image.BILINEAR),
                            dtype=np.float32)

    def norm_duz(a):
        v = a.reshape(a.shape[0], -1)
        v = v - v.mean(axis=1, keepdims=True)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n < 1e-6] = 1.0
        return v / n

    taban = norm_duz(ham)

    # union-find
    ebeveyn = list(range(len(yollar)))

    def bul(x):
        while ebeveyn[x] != x:
            ebeveyn[x] = ebeveyn[ebeveyn[x]]
            x = ebeveyn[x]
        return x

    def birlestir(a, b):
        ra, rb = bul(a), bul(b)
        if ra != rb:
            ebeveyn[max(ra, rb)] = min(ra, rb)

    rot_sayaci = Counter()
    for k in range(4):
        dondurulmus = norm_duz(np.rot90(ham, k, axes=(1, 2)).copy())
        korelasyon = taban @ dondurulmus.T
        i_idx, j_idx = np.where(korelasyon > ESIK)
        for i, j in zip(i_idx, j_idx):
            if i < j or (k != 0 and i != j):
                birlestir(int(i), int(j))
                if i != j:
                    rot_sayaci[k * 90] += 1

    kumeler = defaultdict(list)
    for i, p in enumerate(yollar):
        kumeler[bul(i)].append(p.name)

    boyutlar = Counter(len(v) for v in kumeler.values())

    print()
    print("=" * 66)
    print("SONUC")
    print("=" * 66)
    print(f"  Toplam goruntu dosyasi   : {len(yollar)}")
    print(f"  BENZERSIZ fotograf       : {len(kumeler)}")
    print(f"  Kopya nedeniyle sisme    : {len(yollar)/len(kumeler):.2f}x")
    print()
    print("  Kume buyuklugu dagilimi:")
    for b, adet in sorted(boyutlar.items()):
        print(f"    {b} kopyali kume -> {adet:4d} kume  ({b*adet:5d} dosya)")

    # numarali vs IMG batch ayrimi
    print()
    print("=" * 66)
    print("BATCH BAZINDA")
    print("=" * 66)
    for etiket, desen in [("numarali (NN-N)", r"^\d+-\d+_"), ("IMG_*", r"^IMG_")]:
        ait = [k for k, v in kumeler.items() if any(re.match(desen, ad) for ad in v)]
        dosya = sum(1 for p in yollar if re.match(desen, p.name))
        print(f"  {etiket:18s}: {dosya:4d} dosya -> {len(ait):4d} benzersiz kume")

    # ornek kumeler
    print()
    print("  Ornek cok-kopyali kumeler:")
    ornek = sorted(kumeler.values(), key=len, reverse=True)[:3]
    for v in ornek:
        print(f"    [{len(v)} kopya]")
        for ad in sorted(v)[:5]:
            print(f"       {ad}")
        if len(v) > 5:
            print(f"       ... +{len(v)-5}")


if __name__ == "__main__":
    main()
