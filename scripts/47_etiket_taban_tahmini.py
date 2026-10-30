"""Omurga etiketlerinin gürültü tabanını YENİDEN ETİKETLEMEDEN tahmin eder.

FİKİR
-----
Gerçek bir omurga düzgün bir eğridir; zikzak yapmaz. Etiketlenen 5 nokta
(c7, o1, o2, o3, sakrum) düzgün bir eğri üzerinde oturmuyorsa, aradaki
sapma etiketleme titremesidir.

YÖNTEM — birini dışarıda bırak (leave-one-out)
Her iç nokta (o1/o2/o3) için: diğer 4 noktadan parabol uydur, dışarıdaki
noktanın yanal sapmasını tahmin et, gerçekle karşılaştır. Kalan hata,
o noktanın komşularının tarif ettiği düzgün eğriden ne kadar saptığıdır.

YORUM
  LOO kalıntısı ~1-2 px  -> etiketler düzgün; modelin 5.88 px'i MODELIN sınırı,
                            mimari değişikliği (Model B) anlamlı
  LOO kalıntısı ~5-6 px  -> etiketlerin kendisi bu kadar oynuyor; model zaten
                            tabanda, hiçbir mimari kurtarmaz

UYARI
Bu bir ÜST SINIR. Gerçek omurga paraboldan sapabilir (S eğrisi), o sapma da
kalıntıya karışır. Yani çıkan sayı etiket gürültüsünü OLDUĞUNDAN BÜYÜK
gösterebilir; küçük gösteremez. Küçük çıkarsa güvenebiliriz.
"""

import math
import statistics as st
from pathlib import Path

import numpy as np
from PIL import Image

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "yolo7"


def yuzdelik(v, p):
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    a, u = math.floor(k), math.ceil(k)
    return s[int(k)] if a == u else s[a] * (u - k) + s[u] * (k - a)


def main():
    loo = {3: [], 4: [], 5: []}
    egrilik = []
    n = 0

    for split in ("train", "val", "test"):
        for lb in sorted((VERI / split / "labels").glob("*.txt")):
            im = VERI / split / "images" / (lb.stem + ".jpg")
            if not im.exists():
                continue
            W, H = Image.open(im).size
            t = lb.read_text(encoding="utf-8").split()
            g = {}
            for i in range(7):
                x, y, v = float(t[5+i*3])*W, float(t[6+i*3])*H, float(t[7+i*3])
                if v:
                    g[i] = (x, y)
            if not all(i in g for i in (2, 3, 4, 5, 6)):
                continue

            c7, sk = g[2], g[6]
            ax = (sk[0]-c7[0], sk[1]-c7[1])
            L = math.hypot(*ax)
            if L < 50:
                continue
            u = (ax[0]/L, ax[1]/L)
            pn = (-u[1], u[0])
            ts, ds = {}, {}
            for i in (2, 3, 4, 5, 6):
                v = (g[i][0]-c7[0], g[i][1]-c7[1])
                ts[i] = (v[0]*u[0] + v[1]*u[1]) / L
                ds[i] = v[0]*pn[0] + v[1]*pn[1]
            n += 1
            egrilik.append(max(abs(ds[i]) for i in (3, 4, 5)))

            for hedef in (3, 4, 5):
                kalan = [i for i in (2, 3, 4, 5, 6) if i != hedef]
                x = np.array([ts[i] for i in kalan])
                y = np.array([ds[i] for i in kalan])
                kat = np.polyfit(x, y, 2)          # 4 nokta, 3 parametre
                tahmin = float(np.polyval(kat, ts[hedef]))
                loo[hedef].append(abs(tahmin - ds[hedef]))

    print("=" * 62)
    print(f"OMURGA ETIKET TABANI — leave-one-out kalintisi   (n={n} goruntu)")
    print("=" * 62)
    ad = {3: "omurga_1", 4: "omurga_2", 5: "omurga_3"}
    print(f"  {'nokta':<12}{'medyan':>9}{'p90':>9}")
    for i in (3, 4, 5):
        v = loo[i]
        print(f"  {ad[i]:<12}{st.median(v):>9.2f}{yuzdelik(v,90):>9.2f}")
    tum = [x for i in (3, 4, 5) for x in loo[i]]
    print(f"  {'TOPLAM':<12}{st.median(tum):>9.2f}{yuzdelik(tum,90):>9.2f}  px")

    print()
    print(f"  karsilastirma:")
    print(f"    modelin YANAL hatasi (CV)     : 5.88 px")
    print(f"    etiketlerdeki sinyal          : 6.5 px")
    print(f"    bu LOO kalintisi (ust sinir)  : {st.median(tum):.2f} px")
    print()
    taban = st.median(tum)
    if taban >= 4.5:
        print("  YORUM: etiketlerin kendisi modelin hatasi kadar oynuyor.")
        print("         Model zaten tabanda; mimari degisikligi kurtarmaz.")
    elif taban <= 2.5:
        print("  YORUM: etiketler duzgun. Modelin 5.88 px'i MODELIN sinirlarindan")
        print("         geliyor -> mimari degisikligi (Model B) anlamli.")
    else:
        print("  YORUM: ara bolge. Mimari degisikligi kismi kazanc verebilir")
        print("         ama tavan yakin.")

    print()
    print(f"  referans: omurga noktalarinin gercek sapma buyuklugu")
    print(f"    medyan {st.median(egrilik):.2f} px   p90 {yuzdelik(egrilik,90):.2f} px")


if __name__ == "__main__":
    main()
