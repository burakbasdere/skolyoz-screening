"""Asama 2a - data/final -> YOLO-pose formati.

YOLO-pose etiket satiri (normalize, 0-1):
    sinif  xc yc w h   px1 py1 v1   px2 py2 v2

Nokta sirasi (sabit, degistirilemez):
    0 = sag koltuk alti
    1 = sol koltuk alti

Kutu: pose modelinin buldugu kisi kutusu. Boylece model once kisiyi
lokalize edip noktalari kirpim uzerinde tahmin eder - bu, tam kare uzerinde
regresyona gore piksel hassasiyetini artirir (bizim tek onemsedigimiz sey).
Kisi bulunamazsa iki noktayi kapsayan genis bir kutu uretilir.

flip_idx [1,0] yazilir cunku yatay flip sag/sol'u takas eder. Egitimde
fliplr=0 kullanacagiz ama format dogru olsun.
"""

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

KOK = Path(__file__).resolve().parent.parent
KAYNAK = KOK / "data" / "final"
HEDEF = KOK / "data" / "yolo"
KENAR_PAY = 0.06   # kisi kutusuna eklenen oransal pay


def main():
    from ultralytics import YOLO
    pose = YOLO("yolo11n-pose.pt")

    kayitlar = json.loads((KAYNAK / "kayitlar.json").read_text(encoding="utf-8"))
    print(f"Kayit: {len(kayitlar)}")

    if HEDEF.exists():
        shutil.rmtree(HEDEF)
    for s in ("train", "val", "test"):
        (HEDEF / s / "images").mkdir(parents=True)
        (HEDEF / s / "labels").mkdir(parents=True)

    kutu_pose, kutu_yedek = 0, 0

    for r in kayitlar:
        s = r["split"]
        kaynak_yol = KAYNAK / s / r["dosya"]
        im = Image.open(kaynak_yol).convert("RGB")
        W, H = im.size
        im.save(HEDEF / s / "images" / r["dosya"], quality=95)

        sag, sol = r["sag"], r["sol"]

        # --- kisi kutusu ---
        # ultralytics numpy diziyi BGR varsayar (OpenCV gelenegi); PIL RGB verir.
        # Kanallari cevirmeden vermek tespiti bozar.
        sonuc = pose.predict(np.asarray(im)[:, :, ::-1], verbose=False, conf=0.25)[0]
        if len(sonuc.boxes) > 0:
            i = int(np.argmax(sonuc.boxes.conf.cpu().numpy()))
            x1, y1, x2, y2 = sonuc.boxes.xyxy.cpu().numpy()[i]
            kutu_pose += 1
        else:
            # yedek: iki noktayi kapsayan genis kutu
            mx = (sag[0] + sol[0]) / 2
            my = (sag[1] + sol[1]) / 2
            yari = abs(sag[0] - sol[0]) * 0.9 + 40
            x1, y1, x2, y2 = mx - yari, my - yari, mx + yari, my + yari * 1.6
            kutu_yedek += 1

        # noktalari kapsadigindan emin ol + pay ekle
        x1 = min(x1, sag[0], sol[0]); x2 = max(x2, sag[0], sol[0])
        y1 = min(y1, sag[1], sol[1]); y2 = max(y2, sag[1], sol[1])
        pw, ph = (x2 - x1) * KENAR_PAY, (y2 - y1) * KENAR_PAY
        x1 = max(0.0, x1 - pw); y1 = max(0.0, y1 - ph)
        x2 = min(float(W), x2 + pw); y2 = min(float(H), y2 + ph)

        xc = (x1 + x2) / 2 / W
        yc = (y1 + y2) / 2 / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H

        parcalar = [f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"]
        for p in (sag, sol):   # sira: 0=sag, 1=sol
            parcalar.append(f"{p[0]/W:.6f} {p[1]/H:.6f} 2")
        satir = " ".join(parcalar)

        (HEDEF / s / "labels" / (Path(r["dosya"]).stem + ".txt")).write_text(
            satir + "\n", encoding="utf-8")

    yaml = f"""# Skolyoz - aksiller nokta tespiti
# 0 = sag koltuk alti, 1 = sol koltuk alti
path: {HEDEF.as_posix()}
train: train/images
val: val/images
test: test/images

kpt_shape: [2, 3]
flip_idx: [1, 0]      # yatay flip sag/sol'u takas eder (egitimde fliplr=0)

names:
  0: kisi
"""
    (HEDEF / "skolyoz.yaml").write_text(yaml, encoding="utf-8")

    print(f"  kisi kutusu pose'dan : {kutu_pose}")
    print(f"  yedek kutu uretilen  : {kutu_yedek}")
    for s in ("train", "val", "test"):
        n = len(list((HEDEF / s / "images").glob("*.jpg")))
        print(f"  {s:6s}: {n}")
    print(f"\nYazildi: {HEDEF / 'skolyoz.yaml'}")


if __name__ == "__main__":
    main()
