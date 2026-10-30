"""Asama 2b - Baseline egitimi (YOLO11-pose, transfer ogrenme).

Amac skor kovalamak degil: TEK bir sayi uretmek ->
    keypoint hatasi (piksel) -> dereceye cevrilince ne kadar?

Karsilastirma tabani: etiket gurultusu 2.45 px / 0.435°.
Model bunun altina INEMEZ; oraya yaklasirsa veri sinirina gelmisiz demektir.

Hiperparametre notlari
----------------------
epochs=200   : kullanicinin istedigi gibi
batch=16     : 128 istenmisti ama 236 goruntude epoch basina 1.8 batch eder
               (200 epoch = ~400 optimizer adimi, yakinsamaz). 16 ile
               epoch basina 15 adim, toplam ~3000 adim.
fliplr=0     : KRITIK. Yatay flip sag/sol koltuk altini takas eder,
               etiketi bozar. Ayni sebeple flipud=0.
degrees=10   : telefonun hafif egik tutulmasini taklit eder. Genis rotasyon
               olcmeye calistigimiz sinyali yok eder (bkz. veri analizi).
mosaic=0.4   : 236 goruntude ezberlemeye karsi yardimci, ama nokta
               hassasiyetini bozdugu icin dusuk + son 30 epoch kapali.
"""

import argparse
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "yolo" / "skolyoz.yaml"
PROJE = KOK / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ad", default="baseline")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(
        data=str(VERI),
        epochs=a.epochs,
        batch=a.batch,
        imgsz=a.imgsz,
        device=0,
        project=str(PROJE),
        name=a.ad,
        exist_ok=True,
        seed=1337,
        deterministic=True,
        patience=50,

        # --- augmentasyon: bu goreve ozel ---
        fliplr=0.0,        # sag/sol anlam tasiyor, flip YASAK
        flipud=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.4,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        mosaic=0.4,
        close_mosaic=30,

        plots=True,
        val=True,
    )
    print(f"\nBitti -> {PROJE / a.ad}")


if __name__ == "__main__":
    main()
