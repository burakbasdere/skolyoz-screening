"""Asama 0e - Rotasyon kopyalarinda sag/sol etiketi tutarli mi?

Kritik soru: ayni fotografin 0/90/180/270 kopyalarinda "sag koltuk"
etiketi ayni ANATOMIK noktayi mi gosteriyor, yoksa etiketci goruntudeki
konuma gore mi isaretlemis?

Ikincisi ise model celiskili denetim aliyor demektir: ayni anatomik nokta
bazen "sag" bazen "sol" olarak ogretiliyor. Bu, hesaplanan acinin
ISARETINI bozar.

Yontem: her kopya kumesinde bir temsilci sec, digerlerinin rotasyonunu
korelasyonla bul, kutu merkezlerini temsilcinin karesine geri map'le,
"sag koltuk" noktalarinin ne kadar dagildigina bak.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

KLASOR = Path(__file__).resolve().parent.parent / "data" / "coco" / "train"
COCO_JSON = KLASOR / "_annotations.coco.json"
N, S, ESIK = 64, 640, 0.98
SAG, SOL = 1, 2


def geri_map(x, y, k, s=S):
    """rot90(A,k) karesindeki (x,y) noktasini A karesine dondurur."""
    for _ in range(k % 4):
        x, y = s - 1 - y, x
    return x, y


def main():
    d = json.loads(COCO_JSON.read_text(encoding="utf-8"))
    isim = {g["id"]: g["file_name"] for g in d["images"]}
    per = defaultdict(list)
    for a in d["annotations"]:
        per[a["image_id"]].append(a)

    merkezler = {}
    for img_id, anots in per.items():
        if sorted(a["category_id"] for a in anots) != [SAG, SOL]:
            continue
        merkezler[isim[img_id]] = {
            a["category_id"]: (a["bbox"][0] + a["bbox"][2] / 2,
                               a["bbox"][1] + a["bbox"][3] / 2)
            for a in anots
        }

    yollar = sorted(p for p in KLASOR.glob("*.jpg") if p.name in merkezler)
    ham = np.zeros((len(yollar), N, N), dtype=np.float32)
    for i, p in enumerate(yollar):
        ham[i] = np.asarray(Image.open(p).convert("L").resize((N, N), Image.BILINEAR),
                            dtype=np.float32)

    def nrm(a):
        v = a.reshape(a.shape[0], -1)
        v = v - v.mean(axis=1, keepdims=True)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n < 1e-6] = 1.0
        return v / n

    taban = nrm(ham)
    # i -> (j, k): i, rot90(j,k) ile eslesiyor
    eslesme = defaultdict(list)
    for k in range(4):
        kor = taban @ nrm(np.rot90(ham, k, axes=(1, 2)).copy()).T
        for i, j in zip(*np.where(kor > ESIK)):
            if i != j:
                eslesme[int(i)].append((int(j), k))

    # kumeler
    ebeveyn = list(range(len(yollar)))

    def bul(x):
        while ebeveyn[x] != x:
            ebeveyn[x] = ebeveyn[ebeveyn[x]]
            x = ebeveyn[x]
        return x

    for i, lst in eslesme.items():
        for j, _ in lst:
            ra, rb = bul(i), bul(j)
            if ra != rb:
                ebeveyn[max(ra, rb)] = min(ra, rb)

    kume = defaultdict(list)
    for i in range(len(yollar)):
        kume[bul(i)].append(i)

    coklu = {k: v for k, v in kume.items() if len(v) >= 2}
    print(f"Incelenen cok-kopyali kume: {len(coklu)}")

    tutarli, celiskili, atlanan = 0, 0, 0
    sapmalar = []

    for temsilci, uyeler in coklu.items():
        # temsilciye gore her uyenin rotasyonu
        rot = {temsilci: 0}
        for j, k in eslesme.get(temsilci, []):
            if j in uyeler and j not in rot:
                rot[j] = k
        if len(rot) < len(uyeler):
            atlanan += 1
        noktalar = []
        for i, k in rot.items():
            m = merkezler[yollar[i].name]
            noktalar.append(geri_map(*m[SAG], k))
        if len(noktalar) < 2:
            continue
        xs = [p[0] for p in noktalar]
        ys = [p[1] for p in noktalar]
        yayilim = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        sapmalar.append(yayilim)
        if yayilim < 40:
            tutarli += 1
        else:
            celiskili += 1

    print()
    print("=" * 64)
    print("SONUC: 'sag koltuk' noktasinin kopyalar arasi yayilimi")
    print("=" * 64)
    if sapmalar:
        sapmalar.sort()
        print(f"  medyan yayilim : {sapmalar[len(sapmalar)//2]:7.1f} px")
        print(f"  p90 yayilim    : {sapmalar[int(len(sapmalar)*0.9)]:7.1f} px")
        print(f"  max yayilim    : {max(sapmalar):7.1f} px")
    print()
    print(f"  TUTARLI  (<40 px, ayni anatomik nokta) : {tutarli}")
    print(f"  CELISKILI (>40 px, etiket karismis)    : {celiskili}")
    if tutarli + celiskili:
        print(f"  -> celiskili oran: {100.0*celiskili/(tutarli+celiskili):.1f}%")


if __name__ == "__main__":
    main()
