"""Aşama 1'i 7 keypoint ile eğitir.

    python scripts/42_egit7.py --epochs 200 --batch 16

14_egit.py'den tek kritik fark: kobj=0.0

Ultralytics maskeli koordinat kaybının yanında ayrı bir GÖRÜNÜRLÜK başlığı
eğitiyor (utils/loss.py:805). Etiketlenmemiş ama aslında görünen noktaları
v=0 yaptığımız için o başlık "bu nokta yok" diye yanlış öğrenir. Kapatıyoruz;
koordinat kaybı zaten v=0'ları maskeliyor (aynı dosya, satır 801).

Diğer ayarlar 2 noktalı eğitimle aynı gerekçelere dayanıyor:
  fliplr=0  sağ/sol koltuk altı anlam taşıyor, flip etiketi bozar
  degrees=10 telefonun hafif eğikliğini taklit eder; geniş rotasyon
            ölçmeye çalıştığımız sinyali yok eder
"""

import argparse
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "yolo7" / "skolyoz7.yaml"
PROJE = KOK / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ad", default="baseline7")
    a = ap.parse_args()

    from ultralytics import YOLO
    YOLO(a.model).train(
        data=str(VERI), epochs=a.epochs, batch=a.batch, imgsz=a.imgsz,
        device=0, project=str(PROJE), name=a.ad, exist_ok=True,
        seed=1337, deterministic=True, patience=50,

        kobj=0.0,          # KRITIK - dosya basligindaki gerekceye bak

        fliplr=0.0, flipud=0.0, degrees=10.0, translate=0.1, scale=0.4,
        shear=0.0, perspective=0.0,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        mosaic=0.4, close_mosaic=30,
        plots=False,       # matplotlib Agg DLL'i bu makinede engelli
        val=True,
    )
    print(f"\nBitti -> {PROJE / a.ad}")


if __name__ == "__main__":
    main()
