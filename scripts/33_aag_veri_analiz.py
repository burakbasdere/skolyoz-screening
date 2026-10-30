"""C7 + sakrum export'unun (aag.v1i.coco) sağlık kontrolü.

Cevaplanacak sorular
--------------------
1. Bu 122 görüntü mevcut 337'nin alt kümesi mi?  -> split KORUNMALI, yoksa
   Aşama 1'in train'indeki bir görüntü yeni testimize düşer ve skor sahte olur.
2. Kutular ne kadar sıkı?  -> merkez hassasiyetinin göstergesi.
3. C7→sakrum tabanı gerçekten koltuk altı tabanından uzun mu?  (scripts/30
   "~2 kat" iddia ediyordu, ölçelim.)
4. Yeni sinyal ne kadar büyük: C7'nin sakrum dikeyinden yatay sapması.
5. Eksik/bozuk kayıtlar.
"""

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
AAG = KOK / "data" / "aag" / "train" / "_annotations.coco.json"
FINAL = KOK / "data" / "final" / "kayitlar.json"

C7, SAKRUM = 1, 2


def yuzdelik(v, p):
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    a, u = math.floor(k), math.ceil(k)
    return s[int(k)] if a == u else s[a] * (u - k) + s[u] * (k - a)


def ozet(ad, v, birim=""):
    if not v:
        print(f"  {ad}: veri yok")
        return
    print(f"  {ad:<32} min {min(v):6.1f}{birim}  medyan {statistics.median(v):6.1f}{birim}"
          f"  p90 {yuzdelik(v,90):6.1f}{birim}  max {max(v):6.1f}{birim}")


def main():
    d = json.loads(AAG.read_text(encoding="utf-8"))
    isim = {g["id"]: g["file_name"] for g in d["images"]}
    per = defaultdict(list)
    for a in d["annotations"]:
        per[a["image_id"]].append(a)

    # --- Roboflow dosya adından orijinali çıkar: "ad_jpg.rf.<hash>.jpg" ---
    # Ayrıca split öneki İKİ KEZ eklenmiş ("train_train_0140"); hazırlık
    # scriptinde yeni_dosya = split + "_" + orijinal_dosya yapılmış. Tekilleştir.
    def taban(ad):
        for kes in ("_jpg.rf.", "_JPG_jpg.rf.", ".rf."):
            if kes in ad:
                ad = ad.split(kes)[0]
                break
        else:
            ad = Path(ad).stem
        for onek in ("train_", "val_", "test_"):
            if ad.startswith(onek + onek):
                return ad[len(onek):]
        return ad

    final = json.loads(FINAL.read_text(encoding="utf-8"))
    final_ix = {Path(r["dosya"]).stem: r for r in final}

    print("=" * 68)
    print("1. MEVCUT VERI SETIYLE ESLESME")
    print("=" * 68)
    eslesen, eslesmeyen = {}, []
    for img_id, ad in isim.items():
        t = taban(ad)
        if t in final_ix:
            eslesen[img_id] = final_ix[t]
        else:
            eslesmeyen.append(ad)
    print(f"  aag goruntusu           : {len(isim)}")
    print(f"  data/final ile eslesen  : {len(eslesen)}")
    print(f"  eslesmeyen (YENI)       : {len(eslesmeyen)}")
    for a in eslesmeyen[:5]:
        print(f"      {a}")
    if eslesen:
        sp = Counter(r["split"] for r in eslesen.values())
        print(f"  eslesenlerin ORIJINAL split'i: " +
              "  ".join(f"{k}={v}" for k, v in sorted(sp.items())))
        print("  -> yeni model bu atamayi AYNEN korumali")

    print()
    print("=" * 68)
    print("2. ETIKET KAPSAMI")
    print("=" * 68)
    kombo = Counter()
    for anots in per.values():
        kombo[tuple(sorted(a["category_id"] for a in anots))] += 1
    ad_ = {C7: "c7", SAKRUM: "sakrum"}
    for k, v in sorted(kombo.items(), key=lambda x: -x[1]):
        etiket = "+".join(ad_.get(c, str(c)) for c in k)
        durum = "OK" if k == (C7, SAKRUM) else "<-- KULLANILAMAZ"
        print(f"  {etiket:<16} {v:4d} goruntu   {durum}")
    for img_id, anots in per.items():
        if sorted(a["category_id"] for a in anots) != [C7, SAKRUM]:
            print(f"      eksik: {isim[img_id]}")

    print()
    print("=" * 68)
    print("3. KUTU SIKILIGI  (merkez hassasiyeti gostergesi)")
    print("=" * 68)
    kenar = {C7: [], SAKRUM: []}
    for anots in per.values():
        for a in anots:
            if a["category_id"] in kenar:
                kenar[a["category_id"]].extend([a["bbox"][2], a["bbox"][3]])
    for c, ad2 in ad_.items():
        ozet(f"{ad2} kutu kenari", kenar[c], " px")
    print("  karsilastirma: koltuk alti kutu kenari medyan 59.5 px idi")

    print()
    print("=" * 68)
    print("4. TABAN UZUNLUGU KARSILASTIRMASI")
    print("=" * 68)
    govde, aksiller, oran = [], [], []
    sapma, sapma_derece = [], []

    for img_id, anots in per.items():
        kats = sorted(a["category_id"] for a in anots)
        if kats != [C7, SAKRUM]:
            continue
        m = {a["category_id"]: (a["bbox"][0] + a["bbox"][2] / 2,
                                a["bbox"][1] + a["bbox"][3] / 2) for a in anots}
        c7p, skp = m[C7], m[SAKRUM]
        h = math.dist(c7p, skp)
        govde.append(h)
        # C7'nin sakrum dikeyinden YATAY sapmasi = koronal denge sinyali
        dx = c7p[0] - skp[0]
        sapma.append(abs(dx))
        dy = abs(skp[1] - c7p[1])
        if dy > 1:
            sapma_derece.append(math.degrees(math.atan2(dx, dy)))
        r = eslesen.get(img_id)
        if r:
            ax = math.dist(tuple(r["sag"]), tuple(r["sol"]))
            aksiller.append(ax)
            if ax > 1:
                oran.append(h / ax)

    ozet("C7->sakrum (govde) tabani", govde, " px")
    ozet("koltuk alti tabani (ayni goruntu)", aksiller, " px")
    if oran:
        print(f"\n  ORAN (govde / koltukalti): medyan {statistics.median(oran):.2f}x")
        print(f"    scripts/30 '~2 kat' diyordu.")
        k = statistics.median(oran)
        print(f"    gercek: {k:.2f}x  -> ayni piksel hatasinda aci hatasi "
              f"{1/k:.2f} katina duser")

    print()
    print("=" * 68)
    print("5. YENI SINYAL: C7 sakrum dikeyinden sapma (koronal denge)")
    print("=" * 68)
    ozet("|yatay sapma|", sapma, " px")
    if sapma_derece:
        mut = [abs(x) for x in sapma_derece]
        print(f"\n  aci olarak: medyan {statistics.median(mut):.2f}°   "
              f"p90 {yuzdelik(mut,90):.2f}°   max {max(mut):.2f}°")
        print(f"  1 px yatay hata = "
              f"{math.degrees(math.atan(1.0/statistics.median(govde))):.3f}° "
              f"(koltuk altinda 0.140° idi)")
        print("\n  dagilim:")
        for alt, ust in [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, 90)]:
            n = sum(1 for x in mut if alt <= x < ust)
            o = 100.0 * n / len(mut)
            print(f"    {alt:4.1f}-{ust:4.1f}° : {n:4d}  ({o:5.1f}%) {'#'*int(o/2)}")


if __name__ == "__main__":
    main()
