"""Asama 0c - Kopya (duplicate) tespiti.

Supheli durum: ayni hastanin 1/2/3/4 numarali cekimlerinden bazilari
AYNI fotografin farkli oryantasyonda kaydedilmis kopyasi olabilir.
Eger oyleyse veri seti sanildigi kadar buyuk degil ve train/val ayrimi
kopya sizdirir.

Yontem: her goruntuyu 64x64 griye indir, ortalamasi cikarilmis vektor
olarak normalize et. Her hasta icin tum cekim ciftlerini 4 rotasyonda
karsilastir, en iyi eslesmeyi raporla.
"""

import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

KLASOR = Path(__file__).resolve().parent.parent / "data" / "coco" / "train"
N = 64


def imza(yol):
    im = Image.open(yol).convert("L").resize((N, N), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    return a


def normalize(a):
    v = a.ravel() - a.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def en_iyi_eslesme(a, b):
    """4 rotasyonda en yuksek korelasyon ve hangi rotasyon."""
    nb = {k: normalize(np.rot90(b, k)) for k in range(4)}
    na = normalize(a)
    skorlar = {k: float(na @ nb[k]) for k in range(4)}
    k = max(skorlar, key=skorlar.get)
    return skorlar[k], k * 90


def main():
    hasta = defaultdict(dict)
    for p in KLASOR.glob("*.jpg"):
        m = re.match(r"^(\d+)-(\d+)_", p.name)
        if m:
            hasta[m.group(1)][m.group(2)] = p

    tam = {h: v for h, v in hasta.items() if len(v) == 4}
    print(f"4 cekimi de olan hasta sayisi: {len(tam)}")
    print()

    onbellek = {}

    def al(p):
        if p not in onbellek:
            onbellek[p] = imza(p)
        return onbellek[p]

    ciftler = [("1", "2"), ("3", "4"), ("1", "3"), ("2", "4"), ("1", "4"), ("2", "3")]
    sonuc = {c: [] for c in ciftler}
    rot = {c: [] for c in ciftler}

    for h, v in tam.items():
        for c in ciftler:
            a, b = c
            if a in v and b in v:
                s, r = en_iyi_eslesme(al(v[a]), al(v[b]))
                sonuc[c].append(s)
                rot[c].append(r)

    print(f"{'cift':8s} {'n':>4s} {'medyan korelasyon':>19s} {'min':>7s}  {'baskin rotasyon':>16s}")
    print("-" * 62)
    for c in ciftler:
        s = sonuc[c]
        if not s:
            continue
        r = rot[c]
        baskin = max(set(r), key=r.count)
        oran = 100.0 * r.count(baskin) / len(r)
        print(f"{c[0]}-{c[1]:6s} {len(s):4d} {statistics.median(s):19.4f} {min(s):7.4f}  "
              f"{baskin:>10d}° ({oran:.0f}%)")

    print()
    print("YORUM: korelasyon ~1.0 (>0.98) => ayni fotograf, kopya.")
    print("       korelasyon ~0.5-0.9   => ayni kisi, farkli cekim.")

    # kac cift kopya sayilir
    print()
    esik = 0.98
    for c in ciftler:
        s = sonuc[c]
        if not s:
            continue
        kopya = sum(1 for x in s if x > esik)
        print(f"  {c[0]}-{c[1]}: {kopya}/{len(s)} cift korelasyon > {esik}  "
              f"({100.0*kopya/len(s):.0f}% KOPYA)")


if __name__ == "__main__":
    main()
