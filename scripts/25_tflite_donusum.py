"""ONNX -> TFLite donusumu (float32 / float16 / int8).

DIKKAT: bu script ANA ortamda degil, .venv-tflite icinde calisir:

    .venv-tflite/Scripts/python.exe scripts/25_tflite_donusum.py

Ayri ortamda olmasinin sebebi tensorflow + onnx2tf ile torch'un ayni ortamda
bagimlilik catismasi cikarmasi. Ara format ONNX oldugu icin bu ortamin torch'a
ihtiyaci yok.

UC VARYANT NEDEN
----------------
  float32 : referans. TFLite'a gecerken kayip olup olmadigini gorurüz.
  float16 : yarim boyut, mobilde neredeyse kayipsiz. Guvenli secim.
  int8    : ceyrek boyut ve en hizli, ama kayip riski en yuksek.

MOBIL_YAPILACAKLAR.md Faz 1 kabul kriteri: aci hatasi medyani <= 0.50 derece
(float sürümde 0.380 derece). INT8 bunu asarsa float16'ya duseriz. Bu karar
tahminle degil, 22'de yazilan degerlendirmeyle verilecek.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import onnx2tf
import onnx2tf.onnx2tf as _o2t_modul
import onnx2tf.utils.common_functions as _o2t_ortak

KOK = Path(__file__).resolve().parent.parent
EXPORT = KOK / "runs" / "export"


def _test_verisini_degistir(kalibrasyon_yolu):
    """onnx2tf'in indirdigi varsayilan test goruntulerini bizimkiyle degistirir.

    IKI SEBEP
    ---------
    1) Calismiyor. onnx2tf o dosyayi pickle'li .npy olarak indiriyor; numpy
       2.x guvenlik geregi pickle'li dosyayi acmayi reddediyor ve donusum
       ValueError ile duruyor.

    2) Calissa bile yanlis veri. O dosya jenerik ImageNet karesi; onnx2tf onu
       ONNX ve TF ciktilarini karsilastirmak icin kullaniyor. Bizim alanimiz
       (arkadan cekilmis sirt fotografi, kendi normalizasyonumuzla) oradan
       tamamen farkli. Kendi kalibrasyon verimizi vermek karsilastirmayi
       anlamli kiliyor.

    mmap ile aciyoruz: asama1 kalibrasyonu 491 MB, tamamini bellege almanin
    gereği yok, ilk birkac ornek yetiyor.
    """
    def sahte_indirme():
        # Kalibrasyon dosyasi zaten NHWC (24_temsilci_veri oyle uretiyor).
        v = np.load(kalibrasyon_yolu, mmap_mode="r")
        return np.ascontiguousarray(v[:20])

    _o2t_ortak.download_test_image_data = sahte_indirme
    _o2t_modul.download_test_image_data = sahte_indirme


def donustur(ad, onnx_ad, girdi_adi, kalibrasyon):
    """Tek modeli TFLite'in uc varyantina cevirir."""
    onnx_yolu = EXPORT / onnx_ad
    cikti = EXPORT / f"tflite_{ad}"
    if cikti.exists():
        shutil.rmtree(cikti)

    print()
    print("=" * 70)
    print(f"{ad}  ({onnx_ad})")
    print("=" * 70)

    _test_verisini_degistir(EXPORT / kalibrasyon)

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_yolu),
        output_folder_path=str(cikti),
        output_integer_quantized_tflite=True,
        # Kalibrasyon verisi ZATEN normalize edilmis durumda (24_temsilci_veri
        # her asamanin kendi normalizasyonunu uyguluyor), o yuzden burada
        # mean=0 std=1 veriyoruz — onnx2tf tekrar normalize etmesin.
        custom_input_op_name_np_data_path=[
            [girdi_adi, str(EXPORT / kalibrasyon), 0.0, 1.0],
        ],
        # NOT: quant_norm_mean / quant_norm_std'ye dokunmuyoruz. Onlar
        # onnx2tf'in varsayilan (indirilen) ornek verisi icin; bizim
        # kalibrasyonumuz custom_input_op_name_np_data_path uzerinden geliyor
        # ve kendi mean/std'sini tasiyor. Ayrica bu iki parametre Python API'de
        # de STRING bekliyor (iceride ast.literal_eval ediliyor), liste
        # verilince "malformed node or string" ile donusum duruyor.
        # TFLite girdi/cikti adlari ONNX'teki adlarla ayni kalsin; Kotlin
        # tarafinda indeks yerine ada gore baglamak istiyoruz.
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )

    print(f"\n  uretilen dosyalar ({cikti.name}):")
    for f in sorted(cikti.glob("*.tflite")):
        print(f"    {f.name:<45} {f.stat().st_size / 1e6:>7.2f} MB")
    return cikti


def main():
    if not EXPORT.exists():
        sys.exit("runs/export yok — once 22_onnx_export.py calistir")

    donustur("asama1", "asama1.onnx", "images", "kalibrasyon_asama1.npy")
    donustur("asama2_koord", "asama2_koord.onnx", "girdi",
             "kalibrasyon_asama2.npy")
    donustur("asama2_heatmap", "asama2_heatmap.onnx", "girdi",
             "kalibrasyon_asama2.npy")

    print()
    print("=" * 70)
    print("BITTI — siradaki adim: 26_tflite_degerlendirme.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
