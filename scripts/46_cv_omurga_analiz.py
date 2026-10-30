"""CV modelleriyle omurga hatasını bileşenlerine ayırır ve eğri seviyesinde ölçer.

    python scripts/46_cv_omurga_analiz.py

Her kat kendi ölçüm parçasında değerlendirilir (o modelin hiç görmediği
görüntüler), sonuçlar birleştirilir. Böylece 182 omurgalı görüntünün TAMAMI
dürüstçe ölçülmüş olur — sabit split'te bu sayı 14'tü.

ÖLÇÜLEN
-------
1. Omurga hatasının eksen/yanal ayrışması
   Eksen boyunca kayma eğrinin şeklini değiştirmez; anlamlı olan yanal.
2. Eğri seviyesinde apeks ve konveksite yönü
   Ürünün gerçek çıktısı bu.

NOT: Bu ölçüm rafinaj ağını YALNIZ başına değerlendirir (kırpımlar gerçek
noktanın etrafından jitter'lı alınır, CV ile aynı kurulum). Tam boru hattı
hatası bundan bir miktar yüksek olur çünkü Aşama 1'in kaba hatası eklenir.
"""

import json
import math
import statistics as st
import sys
from collections import defaultdict
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
CV = KOK / "runs" / "rafinaj7_cv"

KIRPIM, JITTER = 128, 35
ORT = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIMETRI = 3.0


def yuzdelik(v, p):
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    a, u = math.floor(k), math.ceil(k)
    return s[int(k)] if a == u else s[a] * (u - k) + s[u] * (k - a)


def etiket_oku(yol, W, H):
    t = yol.read_text(encoding="utf-8").split()
    out = {}
    for i in range(7):
        x, y, v = float(t[5+i*3])*W, float(t[6+i*3])*H, float(t[7+i*3])
        if v:
            out[i] = (x, y)
    return out


def egri(nok, derece=3, indeksler=(2, 3, 4, 5, 6)):
    """indeksler: egriye dahil edilecek noktalar.

    (2,3,4,5,6) = c7 + o1 + o2 + o3 + sakrum  -> derece 3, S egrisi temsil eder
    (2,4,6)     = c7 + o2 + sakrum            -> derece 2, sadece C egrisi
    """
    if not all(i in nok for i in indeksler):
        return None
    c7, sk = nok[2], nok[6]
    ax = (sk[0]-c7[0], sk[1]-c7[1])
    L = math.hypot(*ax)
    if L < 50:
        return None
    u = (ax[0]/L, ax[1]/L)
    pn = (-u[1], u[0])
    ts, ds = [], []
    for i in indeksler:
        v = (nok[i][0]-c7[0], nok[i][1]-c7[1])
        ts.append((v[0]*u[0] + v[1]*u[1]) / L)
        ds.append(v[0]*pn[0] + v[1]*pn[1])
    kat = np.polyfit(ts, ds, derece)
    tt = np.linspace(0.05, 0.95, 100)
    dd = np.polyval(kat, tt)
    j = int(np.argmax(np.abs(dd)))
    return float(tt[j]), float(dd[j])


def main():
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    katlar = sorted(CV.glob("kat*.pt"))
    if not katlar:
        raise SystemExit(f"CV modeli yok: {CV}")
    print(f"kat modeli: {len(katlar)}")

    # CV'deki kat atamasini yeniden uret (ayni tohum, ayni gruplama)
    _cv = import_module("45_rafinaj7_cv")
    import random as _r
    tum = _cv.Havuz()
    goruntuler = sorted({p for p, _, _ in tum.ornekler})
    grup = _cv.gruplar(goruntuler)
    _r.seed(1337)
    torch.manual_seed(1337)
    np.random.seed(1337)
    _r.shuffle(grup)
    dolu = [0] * len(katlar)
    goruntu_kat = {}
    for g in sorted(grup, key=len, reverse=True):
        k = dolu.index(min(dolu))
        for i in g:
            goruntu_kat[goruntuler[i]] = k
        dolu[k] += len(g)
    print(f"kat buyuklukleri: {dolu}")

    modeller = {}
    for p in katlar:
        d = torch.load(p, weights_only=False)
        m = Rafinaj7(pretrained=False).to(cihaz)
        m.load_state_dict(d["model"])
        m.eval()
        modeller[d["kat"]] = m

    eksen, yanal = [], []
    varyant = {}
    rng = np.random.RandomState(7)

    for yol in goruntuler:
        k = goruntu_kat[yol]
        model = modeller[k]
        lb = yol.parent.parent / "labels" / (yol.stem + ".txt")
        if not lb.exists():
            continue
        W, H = Image.open(yol).size
        g = etiket_oku(lb, W, H)
        if not all(i in g for i in (2, 3, 4, 5, 6)):
            continue
        im = Image.open(yol).convert("RGB")

        tah = {}
        for i in (2, 3, 4, 5, 6):
            gx, gy = g[i]
            jx, jy = rng.uniform(-JITTER, JITTER, 2)
            sol = int(round(gx + jx - KIRPIM/2))
            ust = int(round(gy + jy - KIRPIM/2))
            kirp = im.crop((sol, ust, sol+KIRPIM, ust+KIRPIM))
            a = (np.asarray(kirp, dtype=np.float32)/255.0 - ORT) / STD
            x = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(cihaz)
            with torch.no_grad():
                koord, _ = model(x)
            kx, ky = koord[0, NOKTA_TIPI[i]].cpu().tolist()
            tah[i] = (sol + kx, ust + ky)

        c7, sk = g[2], g[6]
        ax = (sk[0]-c7[0], sk[1]-c7[1])
        L = math.hypot(*ax)
        if L < 50:
            continue
        u = (ax[0]/L, ax[1]/L)
        pn = (-u[1], u[0])
        for i in (3, 4, 5):
            dx = tah[i][0]-g[i][0]
            dy = tah[i][1]-g[i][1]
            eksen.append(abs(dx*u[0] + dy*u[1]))
            yanal.append(abs(dx*pn[0] + dy*pn[1]))

        for ad, ix, dr in (("5 nokta (c7+o1+o2+o3+sakrum), derece 3", (2, 3, 4, 5, 6), 3),
                           ("3 nokta (c7+o2+sakrum), derece 2", (2, 4, 6), 2)):
            eg, et = egri(g, dr, ix), egri(tah, dr, ix)
            if eg and et:
                varyant.setdefault(ad, []).append((eg, et))

    print()
    print("=" * 62)
    print(f"1. OMURGA HATASI BILESENLERI   (n={len(yanal)} nokta)")
    print("=" * 62)
    print(f"  eksen boyunca : medyan {st.median(eksen):6.2f} px   "
          f"p90 {yuzdelik(eksen,90):6.2f} px   (zararsiz)")
    print(f"  YANAL         : medyan {st.median(yanal):6.2f} px   "
          f"p90 {yuzdelik(yanal,90):6.2f} px   <-- ASIL")
    print(f"  sinyal (etiketlerdeki yanal sapma): 6.5 px")
    print(f"  hata/sinyal   : {st.median(yanal)/6.5:.2f}")

    def sinif(d):
        return "duz" if abs(d) < SIMETRI else ("saga" if d > 0 else "sola")

    for ad, egriler in varyant.items():
        print()
        print("=" * 62)
        print(f"2. EGRI SEVIYESI — {ad}   (n={len(egriler)})")
        print("=" * 62)
        ap_d = [abs(a[1]-b[1]) for a, b in egriler]
        ger = [abs(a[1]) for a, b in egriler]
        print(f"  apeks sapmasi hatasi : medyan {st.median(ap_d):.2f} px")
        print(f"  gercek apeks sapmasi : medyan {st.median(ger):.2f} px  <- SINYAL")
        print(f"  hata/sinyal          : "
              f"{st.median(ap_d)/max(st.median(ger),1e-9):.2f}")

        gs = [sinif(a[1]) for a, b in egriler]
        ts_ = [sinif(b[1]) for a, b in egriler]
        dg = sum(1 for x, z in zip(gs, ts_) if x == z)
        print(f"\n  KONVEKSITE YONU: accuracy {dg/len(gs):.3f}  ({dg}/{len(gs)})")
        f1s = []
        for e in ("saga", "sola", "duz"):
            tp = sum(1 for x, z in zip(gs, ts_) if x == e and z == e)
            fp = sum(1 for x, z in zip(gs, ts_) if x != e and z == e)
            fn = sum(1 for x, z in zip(gs, ts_) if x == e and z != e)
            pr = tp/(tp+fp) if tp+fp else 0.0
            rc = tp/(tp+fn) if tp+fn else 0.0
            f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0.0)
            print(f"    {e:<6} destek {tp+fn:>3}  P {pr:.3f}  R {rc:.3f}  F1 {f1s[-1]:.3f}")
        print(f"  makro-F1: {sum(f1s)/3:.3f}")
        # taban: hep en sik sinifi soylersen ne olur
        from collections import Counter as _C
        taban = _C(gs).most_common(1)[0][1] / len(gs)
        print(f"  (taban: hep '{_C(gs).most_common(1)[0][0]}' desen {taban:.3f})")

        for esik in (6, 8, 10):
            alt = [(a, b) for a, b in egriler if abs(a[1]) >= esik]
            if alt:
                d2 = sum(1 for a, b in alt if (a[1] > 0) == (b[1] > 0))
                print(f"    |apeks| >= {esik:2d} px (n={len(alt):3d}): "
                      f"yon dogrulugu {d2/len(alt):.3f}")


if __name__ == "__main__":
    main()
