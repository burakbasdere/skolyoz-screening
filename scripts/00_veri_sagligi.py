"""Asama 0 - Veri sagligi kontrolu.

Roboflow export'u KEYPOINT degil, 2 sinifli BOUNDING BOX detection:
  category 1 = "sag koltuk", category 2 = "sol koltuk"
Bu yuzden "nokta" olarak kutu merkezini kullaniyoruz.

Not: Roboflow "Resize to 640x640 (Stretch)" uygulamis. Stretch en-boy
oranini bozdugu icin buradaki acilar GERCEK aci DEGILDIR; sadece etiket
kalitesini ve tutarliligi incelemek icin kullanilir.
"""

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

COCO_JSON = Path(__file__).resolve().parent.parent / "data" / "coco" / "train" / "_annotations.coco.json"

SAG = 1  # "sag koltuk"
SOL = 2  # "sol koltuk"


def yuzdelik(degerler, p):
    if not degerler:
        return float("nan")
    s = sorted(degerler)
    k = (len(s) - 1) * p / 100.0
    alt, ust = math.floor(k), math.ceil(k)
    if alt == ust:
        return s[int(k)]
    return s[alt] * (ust - k) + s[ust] * (k - alt)


def ozet(baslik, degerler, birim=""):
    if not degerler:
        print(f"  {baslik}: veri yok")
        return
    print(f"  {baslik}")
    print(f"    min {min(degerler):7.1f}{birim}   p5 {yuzdelik(degerler,5):7.1f}{birim}   "
          f"medyan {statistics.median(degerler):7.1f}{birim}   "
          f"p95 {yuzdelik(degerler,95):7.1f}{birim}   max {max(degerler):7.1f}{birim}")


def main():
    data = json.loads(COCO_JSON.read_text(encoding="utf-8"))
    goruntuler = {g["id"]: g for g in data["images"]}

    # --- goruntu basina anotasyonlari topla ---
    per_img = defaultdict(list)
    for a in data["annotations"]:
        per_img[a["image_id"]].append(a)

    print("=" * 66)
    print("1. ETIKET KAPSAMI")
    print("=" * 66)
    sinif_dagilimi = Counter()
    for anots in per_img.values():
        sinif_dagilimi[tuple(sorted(a["category_id"] for a in anots))] += 1
    print(f"  Toplam goruntu   : {len(goruntuler)}")
    print(f"  Toplam anotasyon : {len(data['annotations'])}")
    print()
    print("  Goruntu basina sinif kombinasyonu:")
    isim = {SAG: "sag", SOL: "sol"}
    for kombo, adet in sorted(sinif_dagilimi.items(), key=lambda x: -x[1]):
        etiketler = "+".join(isim.get(c, str(c)) for c in kombo)
        durum = "OK" if kombo == (SAG, SOL) else "<-- KULLANILAMAZ"
        print(f"    {etiketler:20s} -> {adet:4d} goruntu   {durum}")

    kullanilabilir = sinif_dagilimi.get((SAG, SOL), 0)
    print()
    print(f"  ACI HESAPLANABILIR goruntu : {kullanilabilir} / {len(goruntuler)}"
          f"  ({100.0*kullanilabilir/len(goruntuler):.1f}%)")

    # --- geometri ---
    mesafeler, acilar, kutu_kenarlari, en_boy = [], [], [], []
    dikeyler = []  # 45 dereceden dik olanlar = dondurulmus goruntu suphesi

    for img_id, anots in per_img.items():
        kats = sorted(a["category_id"] for a in anots)
        if kats != [SAG, SOL]:
            continue
        merkez = {}
        for a in anots:
            x, y, w, h = a["bbox"]
            merkez[a["category_id"]] = (x + w / 2.0, y + h / 2.0)
            kutu_kenarlari.extend([w, h])
            if h > 0:
                en_boy.append(w / h)
        (xs, ys), (xl, yl) = merkez[SAG], merkez[SOL]
        dx, dy = xl - xs, yl - ys
        d = math.hypot(dx, dy)
        if d < 1e-6:
            continue
        mesafeler.append(d)
        aci = math.degrees(math.atan2(dy, abs(dx)))
        acilar.append(aci)
        if abs(aci) > 45:
            dikeyler.append((goruntuler[img_id]["file_name"], aci))

    print()
    print("=" * 66)
    print("2. KUTU BOYUTLARI  (merkez hassasiyetini belirler)")
    print("=" * 66)
    ozet("Kutu kenar uzunlugu (px)", kutu_kenarlari, " px")
    ozet("Kutu en/boy orani", en_boy)
    print()
    print("  Yorum: kutu ne kadar buyuk ve degisken cizildiyse, merkezi")
    print("         o kadar 'gercek koltuk alti noktasi'ndan sapabilir.")

    print()
    print("=" * 66)
    print("3. D = IKI KOLTUK ALTI ARASI MESAFE  (640x640 uzayinda)")
    print("=" * 66)
    ozet("D (px)", mesafeler, " px")
    print()
    print("  1 piksel merkez hatasi kac derece eder:")
    for etiket, d in [("min", min(mesafeler)), ("p5", yuzdelik(mesafeler, 5)),
                      ("medyan", statistics.median(mesafeler)), ("p95", yuzdelik(mesafeler, 95))]:
        print(f"    D={d:6.1f} px ({etiket:6s}) -> 1 px = {math.degrees(math.atan(1.0/d)):.3f}°")

    print()
    print("=" * 66)
    print("4. ACI DAGILIMI  (stretch nedeniyle gercek aci DEGIL)")
    print("=" * 66)
    mutlak = [abs(x) for x in acilar]
    print(f"  min {min(acilar):.2f}°   medyan {statistics.median(acilar):.2f}°   "
          f"max {max(acilar):.2f}°   ort|aci| {statistics.mean(mutlak):.2f}°")
    print()
    kovalar = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 45), (45, 91)]
    for alt, ust in kovalar:
        adet = sum(1 for x in mutlak if alt <= x < ust)
        oran = 100.0 * adet / len(mutlak)
        uyari = "  <-- goruntu dondurulmus olmali" if alt >= 45 else ""
        print(f"    {alt:2d}-{ust:2d}° : {adet:4d}  ({oran:5.1f}%) {'#'*int(oran/2)}{uyari}")

    print()
    print("=" * 66)
    print("5. ORYANTASYON AYKIRILARI  (|aci| > 45°)")
    print("=" * 66)
    print(f"  Adet: {len(dikeyler)} / {len(acilar)}")
    for ad, aci in dikeyler[:15]:
        print(f"    {aci:7.2f}°  {ad}")
    if len(dikeyler) > 15:
        print(f"    ... ve {len(dikeyler)-15} tane daha")


if __name__ == "__main__":
    main()
