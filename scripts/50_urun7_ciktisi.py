"""7 Landmark Görselleştirme ve Ürün Çıktısı (C7, Omurga 1-3, Sakrum, Koltuk Altı).

7 landmark PyTorch modellerini (baseline7 + rafinaj7) test fotoğrafları üzerinde
çalıştırır ve dikey C7-Sakrum şakül hattı, yatay aksiller hat, omurga eğrisi
ve kalibre edilmiş Cobb açısını çizerek kaydeder.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
Rafinaj7 = import_module("39_rafinaj7_model").Rafinaj7
NOKTA_TIPI = import_module("39_rafinaj7_model").NOKTA_TIPI
kilavuz = import_module("19_kilavuz_cizim")
cobb_mod = import_module("48_cobb_kalibrasyon")

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "yolo7"
CIKTI = KOK / "runs" / "kilavuz"
YOLO_A = KOK / "runs" / "baseline7" / "weights" / "best.pt"
RAF_A = KOK / "runs" / "rafinaj7" / "best.pt"

KIRPIM = 128
ORT = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def rafine(model, im, kaba, tip, cihaz):
    sol = int(round(kaba[0] - KIRPIM / 2))
    ust = int(round(kaba[1] - KIRPIM / 2))
    k = im.crop((sol, ust, sol + KIRPIM, ust + KIRPIM))
    a = (np.asarray(k.convert("RGB"), dtype=np.float32) / 255.0 - ORT) / STD
    x = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(cihaz)
    with torch.no_grad():
        koord, _ = model(x)
    kx, ky = koord[0, tip].cpu().tolist()
    return (sol + kx, ust + ky)


def main():
    from ultralytics import YOLO
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    CIKTI.mkdir(parents=True, exist_ok=True)

    y = YOLO(str(YOLO_A))
    d = torch.load(RAF_A, weights_only=False)
    raf = Rafinaj7(pretrained=False).to(cihaz)
    raf.load_state_dict(d["model"])
    raf.eval()

    kareler = []
    yollar = sorted((VERI / "test" / "images").glob("*.jpg"))
    for p in yollar[:9]:
        im = Image.open(p).convert("RGB")
        s = y.predict(str(p), verbose=False, conf=0.25)[0]
        if s.keypoints is None or len(s.boxes) == 0:
            continue
        j = int(np.argmax(s.boxes.conf.cpu().numpy()))
        kp = s.keypoints.data.cpu().numpy()[j]

        tah = {}
        for i in range(7):
            tah[i] = rafine(raf, im, (float(kp[i][0]), float(kp[i][1])),
                             NOKTA_TIPI[i], cihaz)

        sag_ka, sol_ka = tah[0], tah[1]
        c7, sakrum = tah[2], tah[6]

        # 7 landmarklı kılavuz çizimi (dikey C7-Sakrum şakülü dahil)
        kare = kilavuz.kilavuz_ciz(im, sag_ka, sol_ka, roll_derece=0.0, pitch_derece=0.0,
                                   c7=c7, sakrum=sakrum)

        # Omurga noktalarını (3, 4, 5) da mor halkalar şeklinde çiz
        d_draw = ImageDraw.Draw(kare, "RGBA")
        olcek = max(im.size) / 640.0
        nr = max(5, int(6 * olcek))
        for idx in (3, 4, 5):
            x, y_pt = tah[idx]
            d_draw.ellipse([x - nr, y_pt - nr, x + nr, y_pt + nr],
                           fill=(190, 120, 255, 200), outline=(255, 255, 255),
                           width=max(1, int(1.5 * olcek)))

        kareler.append(kare)

    if kareler:
        K = 420
        satir_sayisi = math.ceil(len(kareler) / 3)
        tuval = Image.new("RGB", (3 * K, satir_sayisi * K), (16, 16, 16))
        for i, k in enumerate(kareler):
            tuval.paste(k.resize((K, K), Image.BILINEAR), ((i % 3) * K, (i // 3) * K))
        yol = CIKTI / "urun7_ciktisi.jpg"
        tuval.save(yol, quality=93)
        print(f"7-landmark kılavuz çizimi üretildi: {yol}")


if __name__ == "__main__":
    main()
