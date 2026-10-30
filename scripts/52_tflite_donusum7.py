"""7 Landmark ONNX -> TFLite Dönüşümü (float16 / float32).

.venv-tflite/Scripts/python.exe scripts/52_tflite_donusum7.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import onnx2tf
import onnx2tf.onnx2tf as _o2t_modul
import onnx2tf.utils.common_functions as _o2t_ortak

KOK = Path(__file__).resolve().parent.parent
EXPORT = KOK / "runs" / "export7"


def _test_verisini_degistir():
    def sahte_indirme():
        return np.random.randn(1, 3, 640, 640).astype(np.float32)
    _o2t_ortak.download_test_image_data = sahte_indirme
    _o2t_modul.download_test_image_data = sahte_indirme


def donustur7(ad, onnx_ad):
    _test_verisini_degistir()
    onnx_yolu = EXPORT / onnx_ad
    cikti = EXPORT / f"tflite_{ad}"
    if cikti.exists():
        shutil.rmtree(cikti)

    print()
    print("=" * 70)
    print(f"7 LANDMARK TFLITE DÖNÜŞÜMÜ: {ad} ({onnx_ad})")
    print("=" * 70)

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_yolu),
        output_folder_path=str(cikti),
        output_integer_quantized_tflite=False,
        output_signaturedefs=True,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )

    print(f"\n  üretilen dosyalar ({cikti.name}):")
    for f in sorted(cikti.glob("*.tflite")):
        print(f"    {f.name:<45} {f.stat().st_size / 1e6:>7.2f} MB")
    return cikti


def main():
    if not EXPORT.exists():
        sys.exit("runs/export7 yok — önce 51_onnx_export7.py çalıştırın")

    donustur7("asama1_7", "asama1_7.onnx")
    donustur7("asama2_7_koord", "asama2_7_koord.onnx")
    donustur7("asama2_7_heatmap", "asama2_7_heatmap.onnx")

    print()
    print("=" * 70)
    print("DÖNÜŞÜM BİTTİ — TFLite dosyaları runs/export7 altında hazır.")
    print("=" * 70)


if __name__ == "__main__":
    main()
