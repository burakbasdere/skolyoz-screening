"""PyTorch boru hattinin hiz olcumu (egitim referansi).

Calistirma:
    python scripts/32_pytorch_performans.py
"""

import json
import math
import statistics
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
RafinajAgi = import_module("17_rafinaj_model").RafinajAgi

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "final"
YOLO_AGIRLIK = KOK / "runs" / "baseline" / "weights" / "best.pt"
RAF_AGIRLIK = KOK / "runs" / "rafinaj" / "best.pt"
KIRPIM = 128
ORT = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ISITMA = 3
TEKRAR = 20


def yuzdelik(v, p):
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    a, u = math.floor(k), math.ceil(k)
    return s[int(k)] if a == u else s[a] * (u - k) + s[u] * (k - a)


def rafine_et(model, im, kaba, cihaz):
    sol = kaba[0] - KIRPIM / 2
    ust = kaba[1] - KIRPIM / 2
    kirpim = im.crop((int(round(sol)), int(round(ust)),
                      int(round(sol)) + KIRPIM, int(round(ust)) + KIRPIM))
    a = np.asarray(kirpim.convert("RGB"), dtype=np.float32) / 255.0
    a = (a - ORT) / STD
    x = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(cihaz)
    with torch.no_grad():
        koord, _ = model(x)
    kx, ky = koord[0].cpu().tolist()
    return (round(sol) + kx, round(ust) + ky)


def olc(y, raf, yol, cihaz):
    t0 = time.perf_counter()
    im = Image.open(yol).convert("RGB")
    s = y.predict(str(yol), verbose=False, conf=0.25)[0]
    t1 = time.perf_counter()
    if s.keypoints is None or len(s.boxes) == 0:
        return None, (t1 - t0) * 1000, 0.0, 0.0
    i = int(np.argmax(s.boxes.conf.cpu().numpy()))
    kp = s.keypoints.data.cpu().numpy()[i]
    kaba = {"sag": (float(kp[0][0]), float(kp[0][1])),
            "sol": (float(kp[1][0]), float(kp[1][1]))}
    t2 = time.perf_counter()
    for v in kaba.values():
        rafine_et(raf, im, v, cihaz)
    t3 = time.perf_counter()
    return True, (t3 - t0) * 1000, (t1 - t0) * 1000, (t3 - t2) * 1000


def main():
    from ultralytics import YOLO

    kayitlar = [r for r in json.loads((VERI / "kayitlar.json").read_text(encoding="utf-8"))
                if r["split"] == "test"][:TEKRAR]
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch performans olcumu — cihaz: {cihaz}, n={len(kayitlar)}")

    y = YOLO(str(YOLO_AGIRLIK))
    d = torch.load(RAF_AGIRLIK, weights_only=False)
    raf = RafinajAgi(pretrained=False).to(cihaz)
    raf.load_state_dict(d["model"])
    raf.eval()

    for r in kayitlar[:ISITMA]:
        olc(y, raf, VERI / r["split"] / r["dosya"], cihaz)

    toplamlar, a1ler, a2ler = [], [], []
    for r in kayitlar:
        ok, toplam, a1, a2 = olc(y, raf, VERI / r["split"] / r["dosya"], cihaz)
        if ok:
            toplamlar.append(toplam)
            a1ler.append(a1)
            a2ler.append(a2)

    print(f"Basarili olcum: {len(toplamlar)}/{len(kayitlar)}")
    print(f"  toplam ms   medyan={statistics.median(toplamlar):6.1f}  "
          f"p90={yuzdelik(toplamlar, 90):6.1f}  "
          f"min={min(toplamlar):6.1f}  max={max(toplamlar):6.1f}")
    print(f"  asama1 ms   medyan={statistics.median(a1ler):6.1f}  p90={yuzdelik(a1ler, 90):6.1f}")
    print(f"  asama2 ms   medyan={statistics.median(a2ler):6.1f}  p90={yuzdelik(a2ler, 90):6.1f}")


if __name__ == "__main__":
    main()
