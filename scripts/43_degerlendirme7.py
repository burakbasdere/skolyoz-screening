"""7 landmark boru hattı değerlendirmesi — hatayı BİLEŞENLERİNE ayırır.

Neden bileşen ayrıştırması
--------------------------
Omurga noktalarının öklid hatası 22.8 px çıktı ama bu yanıltıcı: hatanın
büyük kısmı C7-sakrum EKSENİ BOYUNCA. Noktayı omurga boyunca yukarı/aşağı
kaydırmak eğrinin şeklini değiştirmez — ölçtüğümüz şey YANAL sapma.

Etiket analizinde ölçüldü:
  yükseklik belirsizliği ~21 px (std t=0.043 x 476 px gövde)
  yanal sapma (asıl sinyal) 6.5 px medyan

Dolayısıyla anlamlı metrik: eksene DİK bileşendeki hata.
"""

import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
Rafinaj7 = import_module("39_rafinaj7_model").Rafinaj7
NOKTA_TIPI = import_module("39_rafinaj7_model").NOKTA_TIPI
kilavuz = import_module("19_kilavuz_cizim")

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "yolo7"
YOLO_A = KOK / "runs" / "baseline7" / "weights" / "best.pt"
RAF_A = KOK / "runs" / "rafinaj7" / "best.pt"

KIRPIM = 128
ORT = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
AD = ["sag koltuk", "sol koltuk", "c7", "omurga_1", "omurga_2", "omurga_3", "sakrum"]


def yuzdelik(v, p):
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    a, u = math.floor(k), math.ceil(k)
    return s[int(k)] if a == u else s[a] * (u - k) + s[u] * (k - a)


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
    y = YOLO(str(YOLO_A))
    d = torch.load(RAF_A, weights_only=False)
    raf = Rafinaj7(pretrained=False).to(cihaz)
    raf.load_state_dict(d["model"]); raf.eval()
    print(f"rafinaj: epoch {d['epoch']}  val medyan {d['val_medyan']:.2f} px")

    px = {i: [] for i in range(7)}
    eksen = {i: [] for i in range(3, 6)}      # C7-sakrum ekseni boyunca
    yanal = {i: [] for i in range(3, 6)}      # eksene dik (ASIL)
    aci_h, denge_h = [], []
    n = 0

    for p in sorted((VERI / "test" / "images").glob("*.jpg")):
        lb = VERI / "test" / "labels" / (p.stem + ".txt")
        if not lb.exists():
            continue
        t = lb.read_text().split()
        W, H = Image.open(p).size
        g = {}
        for i in range(7):
            x, yy, v = float(t[5+i*3])*W, float(t[6+i*3])*H, float(t[7+i*3])
            if v:
                g[i] = (x, yy)
        if 0 not in g or 1 not in g:
            continue

        s = y.predict(str(p), verbose=False, conf=0.25)[0]
        if s.keypoints is None or len(s.boxes) == 0:
            continue
        j = int(np.argmax(s.boxes.conf.cpu().numpy()))
        kp = s.keypoints.data.cpu().numpy()[j]
        im = Image.open(p).convert("RGB")

        tah = {}
        for i in range(7):
            tah[i] = rafine(raf, im, (float(kp[i][0]), float(kp[i][1])),
                            NOKTA_TIPI[i], cihaz)
        n += 1

        for i in g:
            px[i].append(math.dist(tah[i], g[i]))

        # --- omurga hatasini eksen/yanal olarak ayristir ---
        if 2 in g and 6 in g:
            ax = (g[6][0]-g[2][0], g[6][1]-g[2][1])
            L = math.hypot(*ax)
            if L > 50:
                u = (ax[0]/L, ax[1]/L); pn = (-u[1], u[0])
                for i in (3, 4, 5):
                    if i not in g:
                        continue
                    dx = tah[i][0]-g[i][0]; dy = tah[i][1]-g[i][1]
                    eksen[i].append(abs(dx*u[0] + dy*u[1]))
                    yanal[i].append(abs(dx*pn[0] + dy*pn[1]))

        aci_h.append(abs(kilavuz.aksiller_aci(tah[0], tah[1]) -
                         kilavuz.aksiller_aci(g[0], g[1])))
        if 2 in g and 6 in g:
            _, dt, _ = kilavuz.koronal_denge(tah[2], tah[6])
            _, dg, _ = kilavuz.koronal_denge(g[2], g[6])
            denge_h.append(abs(dt - dg))

    print(f"degerlendirilen: {n}\n")
    print("=" * 66)
    print("1. NOKTA HATASI (oklid, px)")
    print("=" * 66)
    print(f"  {'nokta':<12}{'n':>5}{'medyan':>9}{'p90':>9}")
    for i in range(7):
        if px[i]:
            print(f"  {AD[i]:<12}{len(px[i]):>5}{st.median(px[i]):>9.2f}"
                  f"{yuzdelik(px[i],90):>9.2f}")

    print()
    print("=" * 66)
    print("2. OMURGA HATASI BILESENLERI  <-- ASIL SORU")
    print("=" * 66)
    print(f"  {'nokta':<12}{'eksen boyu':>12}{'YANAL':>9}   (medyan px)")
    for i in (3, 4, 5):
        if yanal[i]:
            print(f"  {AD[i]:<12}{st.median(eksen[i]):>12.2f}"
                  f"{st.median(yanal[i]):>9.2f}")
    tum_y = [v for i in (3,4,5) for v in yanal[i]]
    tum_e = [v for i in (3,4,5) for v in eksen[i]]
    if tum_y:
        print(f"\n  TOPLAM   eksen {st.median(tum_e):.2f} px   "
              f"YANAL {st.median(tum_y):.2f} px   p90 {yuzdelik(tum_y,90):.2f} px")
        print(f"  karsilastirma: etiketlerdeki yanal sapma sinyali 6.5 px medyan")
        print(f"  -> yanal hata sinyalden kucukse egri kullanilabilir")

    print()
    print("=" * 66)
    print("3. OLCUMLER (derece)")
    print("=" * 66)
    print(f"  aksiller aci hatasi : medyan {st.median(aci_h):.3f}   "
          f"p90 {yuzdelik(aci_h,90):.3f}   (2 noktali modelde 0.380)")
    if denge_h:
        print(f"  koronal denge hatasi: medyan {st.median(denge_h):.3f}   "
              f"p90 {yuzdelik(denge_h,90):.3f}   (n={len(denge_h)})")


if __name__ == "__main__":
    main()
