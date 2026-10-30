"""Asama 6 - Urunun gercek ciktisi.

Tam boru hattini test goruntuleri uzerinde calistirir ve kilavuz cizimini
TAHMIN EDILEN noktalarla uretir (gercek etiketle degil). Yani kullanicinin
telefonda gorecegi sey budur.

Karsilastirma icin her karede gercek etiket cizgisi de ince yesil olarak
gosterilir; sapma gozle gorulebilsin.
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
RafinajAgi = import_module("17_rafinaj_model").RafinajAgi
kilavuz = import_module("19_kilavuz_cizim")
tam = import_module("20_tam_degerlendirme")

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "final"
CIKTI = KOK / "runs" / "kilavuz"
YOLO_AGIRLIK = KOK / "runs" / "baseline" / "weights" / "best.pt"
RAF_AGIRLIK = KOK / "runs" / "rafinaj" / "best.pt"


def main():
    from ultralytics import YOLO
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    y = YOLO(str(YOLO_AGIRLIK))
    d = torch.load(RAF_AGIRLIK, weights_only=False)
    raf = RafinajAgi(pretrained=False).to(cihaz)
    raf.load_state_dict(d["model"])
    raf.eval()

    kayitlar = [r for r in json.loads((VERI / "kayitlar.json").read_text(encoding="utf-8"))
                if r["split"] == "test"]

    kareler = []
    for r in kayitlar[:9]:
        yol = VERI / r["split"] / r["dosya"]
        im = Image.open(yol).convert("RGB")
        s = y.predict(str(yol), verbose=False, conf=0.25)[0]
        if s.keypoints is None or len(s.boxes) == 0:
            continue
        i = int(np.argmax(s.boxes.conf.cpu().numpy()))
        kp = s.keypoints.data.cpu().numpy()[i]
        ince = {}
        for j, ad in ((0, "sag"), (1, "sol")):
            ince[ad] = tam.rafine_et(raf, im, (float(kp[j][0]), float(kp[j][1])), cihaz)

        kare = kilavuz.kilavuz_ciz(im, ince["sag"], ince["sol"], 0.0, 0.0)

        # gercek etiket: ince yesil referans
        dd = ImageDraw.Draw(kare)
        dd.line([tuple(r["sag"]), tuple(r["sol"])], fill=(0, 230, 0), width=2)

        g_aci = kilavuz.aksiller_aci(tuple(r["sag"]), tuple(r["sol"]))
        t_aci = kilavuz.aksiller_aci(ince["sag"], ince["sol"])
        dd.text((14, kare.size[1] - 52),
                f"gercek {g_aci:+.2f}°   tahmin {t_aci:+.2f}°   "
                f"fark {abs(t_aci-g_aci):.2f}°", fill=(0, 255, 0))
        kareler.append(kare)

    K = 420
    tuval = Image.new("RGB", (3 * K, 3 * K), (16, 16, 16))
    for i, k in enumerate(kareler[:9]):
        tuval.paste(k.resize((K, K), Image.BILINEAR), ((i % 3) * K, (i // 3) * K))
    yol = CIKTI / "urun_ciktisi.jpg"
    tuval.save(yol, quality=93)
    print(f"yazildi: {yol}")


if __name__ == "__main__":
    main()
