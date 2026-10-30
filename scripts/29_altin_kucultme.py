"""Kucultme icin PIL'e karsi ALTIN VERI uretir.

    python scripts/29_altin_kucultme.py

NEDEN AYRI BIR TEST
-------------------
Kucultme, Kotlin ile Python arasindaki tek olcumsel farkti: gercek telefon
fotografinda 0.29 dereceye kadar aci farki uretiyordu (bkz.
28_kotlin_karsilastirma.py). Android'in createScaledBitmap'i aliasing
birakiyor, PIL ucgen filtrenin destegini olcek kadar genisletiyor.

Kotlin tarafi artik Pillow'un algoritmasini birebir uyguluyor. Bu dosya o
uygulamanin gercekten esleştigini dogrulayacak veriyi uretiyor.

TOLERANS
--------
Pillow konvolusyonu sabit noktali tam sayiyla yapiyor (22 bit), Kotlin float
ile. Bu yuzden tek tuk piksel +-1 sapabilir; test o toleransla karsilastiriyor.
Onemli olan piksellerin birebir esitligi degil, acinin esitligi.
"""

from pathlib import Path

import numpy as np
from PIL import Image

KOK = Path(__file__).resolve().parent.parent
CIKTI = (KOK / "android" / "app" / "src" / "test" / "resources"
         / "kucultme_altin_veri.csv")

# Gercek cihazda gorulen sekiller + kenar durumlar
VAKALAR = [
    (750, 1000, 640, 640),    # telefonun urettigi sekil (3:4 -> kare stretch)
    (1000, 750, 640, 640),    # yatay
    (640, 640, 640, 640),     # olcek yok
    (300, 400, 640, 640),     # BUYUTME (filtre genisletilmemeli)
    (1234, 567, 640, 640),    # asal-ish, tam bolunmeyen
    (61, 43, 32, 32),         # kucuk, hizli kontrol
]

# Her vakadan kac ornek piksel yazilacak
ORNEK = 24


def sentetik(g, y):
    """Yuksek frekansli desen — aliasing farkini ortaya cikarir.

    Duz gradyan kullanmak yanlis olurdu: her iki algoritma da duz gradyanda
    neredeyse ayni sonucu verir, test yesil kalir ve gercek farki kacirirdi.

    RASTGELELIK YOK. Ilk surumde gurultu ekleniyordu, ama Kotlin tarafi
    numpy'nin uretecini birebir kopyalayamaz; o zaman karsilastirmayi gevsek
    bir toleransa dusurmek gerekiyordu ve test neredeyse hicbir sey
    dogrulamiyordu. Desen tamamen deterministik olunca iki taraf AYNI kaynagi
    uretiyor ve karsilastirma +-1 piksele kadar sikilastirilabiliyor.
    """
    xx, yy = np.meshgrid(np.arange(g), np.arange(y))
    r = 127 + 127 * np.sin(xx * 0.7) * np.cos(yy * 0.4)
    gc = 127 + 127 * np.sin((xx + yy) * 0.9)
    b = (xx * 3 + yy * 5) % 256
    a = np.stack([r, gc, b], axis=-1)
    # int'e cevirmede Python/Kotlin ayni davransin diye once kirp sonra kes
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def main():
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    satirlar = ["# tip,kaynakG,kaynakY,hedefG,hedefY,tohum,ornekler"
                "  — 29_altin_kucultme.py uretir"]

    for i, (kg, ky, hg, hy) in enumerate(VAKALAR):
        im = sentetik(kg, ky)
        kucuk = im.resize((hg, hy), Image.BILINEAR)
        a = np.asarray(kucuk, dtype=np.int64)

        rng = np.random.default_rng(1000 + i)
        noktalar = set()
        # koseler her zaman dahil: filtrenin sinir davranisi orada belirginlesir
        for p in [(0, 0), (hg - 1, 0), (0, hy - 1), (hg - 1, hy - 1)]:
            noktalar.add(p)
        while len(noktalar) < ORNEK:
            noktalar.add((int(rng.integers(0, hg)), int(rng.integers(0, hy))))

        ornekler = ";".join(
            f"{x}:{y}:{a[y, x, 0]}:{a[y, x, 1]}:{a[y, x, 2]}"
            for x, y in sorted(noktalar))
        satirlar.append(f"kucult,{kg},{ky},{hg},{hy},{i},{ornekler}")

    CIKTI.write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    print(f"yazildi: {CIKTI}  ({CIKTI.stat().st_size / 1e3:.1f} KB)")
    for kg, ky, hg, hy in VAKALAR:
        yon = "buyutme" if hg > kg else "kucultme"
        print(f"  {kg}x{ky} -> {hg}x{hy}   ({yon}, olcek {kg/hg:.2f}x{ky/hy:.2f})")


if __name__ == "__main__":
    main()
