"""Asama 2c - Baseline'in ASIL metrigi: aci hatasi.

mAP bu projede anlamsiz. Onemli olan tek sey:
    tahmin edilen iki nokta -> aci   vs   gercek iki nokta -> aci
    aradaki fark, DERECE cinsinden.

Referans noktalari:
    etiket gurultusu     0.435°  (model bunun altina inemez, fizik siniri)
    veri medyan |aci|    2.21°   (olcmeye calistigimiz sinyalin buyuklugu)

Cikti hem test hem val icin uretilir; test'e bakip karar veririz.
"""

import json
import math
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "yolo"
AGIRLIK = KOK / "runs" / "baseline" / "weights" / "best.pt"

ETIKET_GURULTUSU_PX = 2.45
ETIKET_GURULTUSU_DEG = 0.435
MEDYAN_SINYAL = 2.21


def yuzdelik(v, p):
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    a, u = math.floor(k), math.ceil(k)
    return s[int(k)] if a == u else s[a] * (u - k) + s[u] * (k - a)


def aci_of(sag, sol):
    dx, dy = sol[0] - sag[0], sol[1] - sag[1]
    return math.degrees(math.atan2(dy, abs(dx))), math.hypot(dx, dy)


def degerlendir(model, split):
    img_dir = VERI / split / "images"
    lbl_dir = VERI / split / "labels"
    yollar = sorted(img_dir.glob("*.jpg"))

    px_hata, aci_hata, D_ler = [], [], []
    bulunamadi = 0
    isaret_hatasi = 0
    detay = []

    for p in yollar:
        im = Image.open(p).convert("RGB")
        W, H = im.size
        satir = (lbl_dir / (p.stem + ".txt")).read_text(encoding="utf-8").split()
        g_sag = (float(satir[5]) * W, float(satir[6]) * H)
        g_sol = (float(satir[8]) * W, float(satir[9]) * H)

        # DIKKAT: ultralytics numpy dizisini BGR varsayar (OpenCV gelenegi).
        # PIL'den gelen RGB diziyi dogrudan vermek kanallari ters cevirir ve
        # keypoint hatasini 24 px'den 40 px'e cikarir. Dosya yolu vermek
        # ultralytics'in kendi (dogru) yukleyicisini kullanir.
        r = model.predict(str(p), verbose=False, conf=0.25)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            bulunamadi += 1
            continue
        i = int(np.argmax(r.boxes.conf.cpu().numpy()))
        kp = r.keypoints.data.cpu().numpy()[i]
        t_sag, t_sol = (kp[0][0], kp[0][1]), (kp[1][0], kp[1][1])

        px_hata.append(math.dist(t_sag, g_sag))
        px_hata.append(math.dist(t_sol, g_sol))

        g_aci, g_D = aci_of(g_sag, g_sol)
        t_aci, _ = aci_of(t_sag, t_sol)
        fark = abs(t_aci - g_aci)
        aci_hata.append(fark)
        D_ler.append(g_D)
        if (t_aci > 0) != (g_aci > 0) and min(abs(t_aci), abs(g_aci)) > 1.0:
            isaret_hatasi += 1
        detay.append({"dosya": p.name, "gercek_aci": round(g_aci, 2),
                      "tahmin_aci": round(t_aci, 2), "fark": round(fark, 2)})

    return px_hata, aci_hata, D_ler, bulunamadi, isaret_hatasi, detay


def rapor(ad, px, aci, D, bulunamadi, isaret, n):
    print()
    print("=" * 66)
    print(f"{ad.upper()}  (n={n}, tespit edilemeyen={bulunamadi})")
    print("=" * 66)
    if not aci:
        print("  sonuc yok")
        return
    print("  KEYPOINT HATASI (piksel)")
    print(f"    medyan {statistics.median(px):6.2f}   p90 {yuzdelik(px,90):6.2f}"
          f"   p95 {yuzdelik(px,95):6.2f}   max {max(px):7.2f}")
    print(f"    etiket gurultusu tabani: {ETIKET_GURULTUSU_PX:.2f} px")

    print()
    print("  ACI HATASI (derece)  <-- ASIL METRIK")
    print(f"    medyan {statistics.median(aci):6.3f}°  ort {statistics.mean(aci):6.3f}°")
    print(f"    p90    {yuzdelik(aci,90):6.3f}°  p95 {yuzdelik(aci,95):6.3f}°"
          f"  max {max(aci):7.3f}°")
    print(f"    etiket gurultusu tabani: {ETIKET_GURULTUSU_DEG:.3f}°")

    med = statistics.median(aci)
    print()
    print(f"    tabana oran      : {med/ETIKET_GURULTUSU_DEG:.2f}x")
    print(f"    sinyale oran     : {med/MEDYAN_SINYAL:.2f}  "
          f"(1.0 = hata sinyal kadar buyuk, ise yaramaz)")
    print(f"    isaret hatasi    : {isaret}/{len(aci)}  "
          "(hangi omuz yuksek sorusunu ters bilme)")

    print()
    print("  Aci hatasi dagilimi:")
    for alt, ust in [(0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 999)]:
        adet = sum(1 for x in aci if alt <= x < ust)
        oran = 100.0 * adet / len(aci)
        etiket = f"{alt:.2f}-{ust:.2f}°" if ust < 999 else f">{alt:.0f}°     "
        print(f"    {etiket:12s}: {adet:4d}  ({oran:5.1f}%) {'#'*int(oran/2)}")

    print()
    print("  KARAR:")
    if med <= ETIKET_GURULTUSU_DEG * 1.5:
        print("    Model veri sinirina gelmis. Kutu merkezleri yeterli,")
        print("    yeniden etiketleme gereksiz -> mobil entegrasyona gec.")
    elif med <= 1.0:
        print("    Kullanilabilir ama tabanin uzerinde. Iki asamali kirpma +")
        print("    heatmap head (Lite-HRNet tarzi) ile kazanc alinabilir.")
    else:
        print("    Hata sinyale gore buyuk. Once mimari degisikligi:")
        print("    top-down kirpma + yuksek cozunurluklu heatmap sart.")


def main():
    from ultralytics import YOLO
    if not AGIRLIK.exists():
        raise SystemExit(f"Agirlik yok: {AGIRLIK}\nEgitim bitmemis olabilir.")
    model = YOLO(str(AGIRLIK))

    tum = {}
    for split in ("val", "test"):
        px, aci, D, bul, isa, detay = degerlendir(model, split)
        rapor(split, px, aci, D, bul, isa, len(aci))
        tum[split] = detay

    (KOK / "runs" / "baseline" / "aci_hatasi.json").write_text(
        json.dumps(tum, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
