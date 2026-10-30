"""INT8 quantization icin kalibrasyon (temsilci) veri seti uretir.

NEDEN GEREKLI
-------------
INT8 quantization, her tensorun gercekte hangi deger araliginda gezdigini
bilmek zorunda. Bu araliklar agirliklardan cikarilamaz; ag'i GERCEK girdilerle
calistirip aktivasyon dagilimlarina bakmak gerekiyor. Temsilci veri bunun icin.

Rastgele gurultu vermek en sik yapilan hata: aktivasyon araliklari gercekte
olduklarindan cok daha genis cikar, olcek katsayilari bozulur ve model INT8'de
cok daha fazla dogruluk kaybeder.

IKI AYRI SET, CUNKU IKI ASAMANIN GIRDISI FARKLI
-----------------------------------------------
  Asama 1 : 640x640 tam goruntu, sadece /255      (YOLO konvansiyonu)
  Asama 2 : 128x128 kirpim, ImageNet normalizasyonu

Bunlari karistirmak MOBIL_YAPILACAKLAR.md §5.6'da ayrica uyarilan bir tuzak.

CIKTI NHWC DUZENINDE — bu onemli
--------------------------------
Ilk denemede NCHW kaydedildi (ONNX'in duzeni) ve INT8 donusumu sessizce
duserek yalnizca float/dynamic-range dosyalari uretildi. Ayrintili log'da
sebep goruldu:

    input_channel % filter_input_channel != 0 (2 != 0)  Node 1 (CONV_2D)

Cunku onnx2tf kalibrasyon verisini ONNX grafina degil, DONUSTURULMUS TFLite
modeline besliyor; o model NHWC. NCHW verince (N,3,128,128) tensorunun son
ekseni 128 kanal sayiliyor ve 128 % 3 = 2 kaliyor.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "final"
CIKTI = KOK / "runs" / "export"

MODEL_BOYUT = 640
KIRPIM = 128
ORT = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Kalibrasyon icin kac ornek. 100-200 arasi standart; daha fazlasi olcuyu
# belirgin iyilestirmiyor.
#
# Asama 1'de 100'de kaliyoruz cunku 640x640 float32'de her ornek ~4.9 MB:
# 200 ornek 983 MB dosya demek ve donusum sirasinda tamami bellege yukleniyor.
# Tek sinifli, iki noktali bir model icin 100 ornek aktivasyon araliklarini
# fazlasiyla temsil ediyor.
ASAMA1_ADET = 100
ASAMA2_ADET = 400


def kayitlar_train():
    k = json.loads((VERI / "kayitlar.json").read_text(encoding="utf-8"))
    return [r for r in k if r["split"] == "train"]


def asama1_veri(kayitlar):
    """640x640 stretch + /255 -> (N, 640, 640, 3) NHWC."""
    yigin = []
    for r in kayitlar[:ASAMA1_ADET]:
        im = Image.open(VERI / r["split"] / r["dosya"]).convert("RGB")
        # STRETCH, letterbox degil — egitimde de stretch kullanilmisti.
        im = im.resize((MODEL_BOYUT, MODEL_BOYUT), Image.BILINEAR)
        yigin.append(np.asarray(im, dtype=np.float32) / 255.0)
    return np.stack(yigin)


def asama2_veri(kayitlar):
    """Gercek nokta etrafindan 128x128 kirpim + ImageNet norm.

    Kirpimlar GERCEK etiketten aliniyor ama Asama 1 uygulamada ~24 px hatayla
    kaba nokta veriyor. Kalibrasyonun gercek dagilimi gormesi icin merkeze
    ayni mertebede rastgele kayma ekliyoruz — aksi halde model her zaman
    noktanin tam ortada oldugu kirpimlarla kalibre edilir ve uretimde
    gorecegi dagilimi temsil etmez.
    """
    rng = np.random.default_rng(0)
    yigin = []
    for r in kayitlar:
        im = Image.open(VERI / r["split"] / r["dosya"]).convert("RGB")
        for taraf in ("sag", "sol"):
            if len(yigin) >= ASAMA2_ADET:
                break
            cx, cy = r[taraf]
            cx += float(rng.normal(0, 12))
            cy += float(rng.normal(0, 12))
            sol = int(round(cx - KIRPIM / 2))
            ust = int(round(cy - KIRPIM / 2))
            kirpim = im.crop((sol, ust, sol + KIRPIM, ust + KIRPIM))
            a = np.asarray(kirpim.convert("RGB"), dtype=np.float32) / 255.0
            yigin.append((a - ORT) / STD)
    return np.stack(yigin)


def main():
    CIKTI.mkdir(parents=True, exist_ok=True)
    kayitlar = kayitlar_train()
    print(f"train kaydi: {len(kayitlar)}")

    a1 = asama1_veri(kayitlar)
    yol1 = CIKTI / "kalibrasyon_asama1.npy"
    np.save(yol1, a1)
    print(f"asama1: {a1.shape} {a1.dtype}  "
          f"[{a1.min():.3f}, {a1.max():.3f}]  -> {yol1.name} "
          f"({yol1.stat().st_size / 1e6:.1f} MB)")

    a2 = asama2_veri(kayitlar)
    yol2 = CIKTI / "kalibrasyon_asama2.npy"
    np.save(yol2, a2)
    print(f"asama2: {a2.shape} {a2.dtype}  "
          f"[{a2.min():.3f}, {a2.max():.3f}]  -> {yol2.name} "
          f"({yol2.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
