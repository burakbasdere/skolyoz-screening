"""Etiketleme kolaylaştırıcı: omurga oluğunu görünür kılan kontrast işlemi.

NEDEN
-----
Omurga oluğu bir gölge vadisi; ham fotoğrafta çoğu zaman zar zor seçiliyor.
Otomatik izleme bu yüzden başarısız oldu (scripts/34, kalıntı RMS %9.20).
Ama insan gözü, kontrast artırılmış görüntüde oluğu rahat seçiyor.

KRİTİK: kare 640x640 olarak AYNI kalıyor. Yani işlenmiş görüntü üzerinde
konulan etiketin koordinatı, orijinal görüntüde birebir aynı yeri gösterir.
Etiketleme işlenmiş kopyada yapılır, EĞİTİM temiz orijinalle.

    python scripts/36_etiketleme_gorunurluk.py            # karsilastirma sayfasi
    python scripts/36_etiketleme_gorunurluk.py --tumu     # tum seti uret
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "final"
CIKTI = KOK / "data" / "etiketleme_gorunur"
KARSILASTIRMA = KOK / "runs" / "omurga" / "gorunurluk_karsilastirma.jpg"


def belirginlestir(bgr):
    """Oluğu öne çıkaran yerel kontrast. Geometriyi DEĞİŞTİRMEZ."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # yerel histogram esitleme: bolgesel golgeleri ortaya cikarir
    l2 = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(l)
    # unsharp: genis olcekli aydinlanmayi cikar, ince detayi birak
    bulanik = cv2.GaussianBlur(l2, (0, 0), 11)
    detay = cv2.addWeighted(l2, 1.7, bulanik, -0.7, 0)
    detay = np.clip(detay, 0, 255).astype(np.uint8)
    # renk bilgisini koru ama zayiflat: dikkat oluga gitsin
    lab2 = cv2.merge([detay,
                      cv2.addWeighted(a, 0.35, np.full_like(a, 128), 0.65, 0),
                      cv2.addWeighted(b, 0.35, np.full_like(b, 128), 0.65, 0)])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def _yazi(boyut, kalin=False):
    adlar = (("segoeuib.ttf", "arialbd.ttf") if kalin else ("segoeui.ttf", "arial.ttf"))
    for ad in adlar:
        try:
            return ImageFont.truetype(ad, boyut)
        except OSError:
            continue
    return ImageFont.load_default()


def karsilastirma(ornekler):
    K = 330
    n = len(ornekler)
    tuval = Image.new("RGB", (n * K, 2 * K + 56), (20, 20, 24))
    d = ImageDraw.Draw(tuval)
    f = _yazi(22, True)
    fk = _yazi(15)
    d.text((14, 12), "ÜST: orijinal        ALT: kontrast artırılmış "
                     "(oluk görünür hale gelir)", font=f, fill=(255, 255, 255))
    for i, yol in enumerate(ornekler):
        bgr = cv2.imread(str(yol))
        for j, g in enumerate((bgr, belirginlestir(bgr))):
            im = Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB)).resize(
                (K, K), Image.LANCZOS)
            tuval.paste(im, (i * K, 44 + j * K))
    d.text((14, 44 + 2 * K + 4),
           "Kare 640x640 aynı kaldığı için etiket koordinatları orijinale birebir taşınır.",
           font=fk, fill=(170, 170, 170))
    KARSILASTIRMA.parent.mkdir(parents=True, exist_ok=True)
    tuval.save(KARSILASTIRMA, quality=94)
    print(f"karsilastirma: {KARSILASTIRMA}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tumu", action="store_true", help="tum seti uret")
    a = ap.parse_args()

    kayitlar = json.loads((VERI / "kayitlar.json").read_text(encoding="utf-8"))

    if not a.tumu:
        secim = ["train_0114.jpg", "train_0175.jpg", "train_0172.jpg",
                 "train_0023.jpg", "train_0232.jpg"]
        yollar = []
        for r in kayitlar:
            if r["dosya"] in secim:
                yollar.append(VERI / r["split"] / r["dosya"])
        karsilastirma(yollar[:5])
        print("\nTum seti uretmek icin: --tumu")
        return

    CIKTI.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in kayitlar:
        kaynak = VERI / r["split"] / r["dosya"]
        if not kaynak.exists():
            continue
        bgr = cv2.imread(str(kaynak))
        if bgr is None:
            continue
        cv2.imwrite(str(CIKTI / r["dosya"]), belirginlestir(bgr),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    print(f"uretilen: {n} -> {CIKTI}")
    print("Bunlari Roboflow'a yukleyip etiketle; koordinatlar orijinale birebir uyar.")


if __name__ == "__main__":
    main()
