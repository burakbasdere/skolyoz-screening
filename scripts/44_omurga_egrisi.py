"""Omurga eğrisi: nokta değil EĞRİ seviyesinde değerlendirme.

NEDEN
-----
Nokta bazında omurga yanal hatası 6.85 px, sinyal 6.5 px — oran ~1, yani
nokta seviyesinde kullanılamaz. Ama ürünün çıktısı nokta değil eğri.
5 noktadan (c7 + o1/o2/o3 + sakrum) eğri uydurunca nokta gürültüsü ortalanır;
apeks ve konveksite yönü tek tek noktalardan daha kararlı olabilir.

YÖNTEM
------
C7-sakrum eksenini referans al. Her nokta için:
    t = eksen boyunca normalize yükseklik (C7=0, sakrum=1)
    d = eksene DİK sapma (px, sağa pozitif)
(t, d) çiftlerine polinom uydur. Uçlar sıfıra sabit (c7 ve sakrum eksenin
üzerinde), yani eğri gövde eksenine göre sapmayı tarif eder.

ÜRÜN ÇIKTISI
------------
  apeks t   : eğriliğin tepe noktasının yüksekliği (torakal/lomber ayrımı)
  apeks d   : sapma büyüklüğü (px)
  konveksite: sapmanın işareti -> "sağa konveks" / "sola konveks"

Konveksite yönü klinik tarifin karşılığıdır ve SINIFLANDIRMA olduğu için
accuracy/F1 anlamlıdır.
"""

import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
_m = import_module("39_rafinaj7_model")
Rafinaj7, NOKTA_TIPI = _m.Rafinaj7, _m.NOKTA_TIPI

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "yolo7"
YOLO_A = KOK / "runs" / "baseline7" / "weights" / "best.pt"
RAF_A = KOK / "runs" / "rafinaj7" / "best.pt"

KIRPIM = 128
ORT = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DERECE = 2          # 5 nokta icin guvenli; 3 de raporlaniyor
SIMETRI_BANDI = 3.0  # |apeks| < 3 px -> "duz" say


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


def egri(noktalar, derece):
    """noktalar: {idx: (x,y)}, 2=c7 6=sakrum 3,4,5=omurga.

    Doner: (kat, apeks_t, apeks_d) veya None
    """
    if not all(i in noktalar for i in (2, 3, 4, 5, 6)):
        return None
    c7, sk = noktalar[2], noktalar[6]
    ax = (sk[0] - c7[0], sk[1] - c7[1])
    L = math.hypot(*ax)
    if L < 50:
        return None
    u = (ax[0] / L, ax[1] / L)
    pn = (-u[1], u[0])
    ts, ds = [], []
    for i in (2, 3, 4, 5, 6):
        v = (noktalar[i][0] - c7[0], noktalar[i][1] - c7[1])
        ts.append((v[0] * u[0] + v[1] * u[1]) / L)
        ds.append(v[0] * pn[0] + v[1] * pn[1])
    kat = np.polyfit(ts, ds, derece)
    tt = np.linspace(0.05, 0.95, 100)
    dd = np.polyval(kat, tt)
    j = int(np.argmax(np.abs(dd)))
    return kat, float(tt[j]), float(dd[j])


def main():
    from ultralytics import YOLO
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    y = YOLO(str(YOLO_A))
    d = torch.load(RAF_A, weights_only=False)
    raf = Rafinaj7(pretrained=False).to(cihaz)
    raf.load_state_dict(d["model"]); raf.eval()

    sonuc = {2: [], 3: []}
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
        if not all(i in g for i in (2, 3, 4, 5, 6)):
            continue

        s = y.predict(str(p), verbose=False, conf=0.25)[0]
        if s.keypoints is None or len(s.boxes) == 0:
            continue
        j = int(np.argmax(s.boxes.conf.cpu().numpy()))
        kp = s.keypoints.data.cpu().numpy()[j]
        im = Image.open(p).convert("RGB")
        tah = {i: rafine(raf, im, (float(kp[i][0]), float(kp[i][1])),
                         NOKTA_TIPI[i], cihaz) for i in (2, 3, 4, 5, 6)}

        for dr in (2, 3):
            eg = egri(g, dr)
            et = egri(tah, dr)
            if eg and et:
                sonuc[dr].append((eg, et))

    for dr in (2, 3):
        v = sonuc[dr]
        if not v:
            continue
        print("=" * 66)
        print(f"POLINOM DERECESI {dr}   (n={len(v)})")
        print("=" * 66)

        apeks_t = [abs(a[1] - b[1]) for a, b in v]
        apeks_d = [abs(a[2] - b[2]) for a, b in v]
        gercek_d = [abs(a[2]) for a, b in v]

        print(f"  apeks yuksekligi hatasi : medyan {st.median(apeks_t):.3f}"
              f"   p90 {yuzdelik(apeks_t,90):.3f}   (t birimi, 0-1)")
        print(f"  apeks sapmasi hatasi    : medyan {st.median(apeks_d):.2f} px"
              f"   p90 {yuzdelik(apeks_d,90):.2f} px")
        print(f"  gercek apeks sapmasi    : medyan {st.median(gercek_d):.2f} px"
              f"   <- SINYAL")
        oran = st.median(apeks_d) / max(st.median(gercek_d), 1e-9)
        print(f"  hata/sinyal             : {oran:.2f}"
              f"   {'<-- kullanilabilir' if oran < 0.6 else '<-- zayif'}")

        # --- konveksite yonu: siniflandirma ---
        def sinif(dv):
            if abs(dv) < SIMETRI_BANDI:
                return "duz"
            return "saga" if dv > 0 else "sola"

        gs = [sinif(a[2]) for a, b in v]
        ts_ = [sinif(b[2]) for a, b in v]
        dogru = sum(1 for x, z in zip(gs, ts_) if x == z)
        print(f"\n  KONVEKSITE YONU (|apeks| < {SIMETRI_BANDI:.0f} px -> duz)")
        print(f"    accuracy : {dogru/len(gs):.3f}  ({dogru}/{len(gs)})")
        print(f"    {'sinif':<8}{'destek':>7}{'precision':>11}{'recall':>9}{'F1':>8}")
        f1ler = []
        for e in ("saga", "sola", "duz"):
            tp = sum(1 for x, z in zip(gs, ts_) if x == e and z == e)
            fp = sum(1 for x, z in zip(gs, ts_) if x != e and z == e)
            fn = sum(1 for x, z in zip(gs, ts_) if x == e and z != e)
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2*pr*rc/(pr+rc) if pr+rc else 0.0
            f1ler.append(f1)
            print(f"    {e:<8}{tp+fn:>7}{pr:>11.3f}{rc:>9.3f}{f1:>8.3f}")
        print(f"    makro-F1 : {sum(f1ler)/3:.3f}")

        # sadece belirgin egriligi olanlarda (|apeks| >= 8 px)
        belirgin = [(a, b) for a, b in v if abs(a[2]) >= 8]
        if belirgin:
            dg = sum(1 for a, b in belirgin if (a[2] > 0) == (b[2] > 0))
            print(f"\n    belirgin egrilikte (|apeks| >= 8 px, n={len(belirgin)}):"
                  f" yon dogrulugu {dg/len(belirgin):.3f}")
        print()


if __name__ == "__main__":
    main()
