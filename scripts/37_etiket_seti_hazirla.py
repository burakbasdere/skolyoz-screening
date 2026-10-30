"""7 landmark etiketlemesi için görüntü setini hazırlar.

    python scripts/37_etiket_seti_hazirla.py

NE YAPAR
--------
1. data/final'deki 337 görüntüden bu şema için UYGUN olanları ayırır.
2. Her birine kontrast işlemi uygular (omurga oluğu görünür hale gelir).
   Kare 640x640 aynı kalır -> etiket koordinatları orijinale birebir oturur.
3. Önceliğe göre alt klasörlere ayırır ve bir manifest CSV yazar.

UYGUNLUK
--------
Sakrum için ALT SIRT, C7 için ÜST SIRT görünmeli. İki sinyal kullanıyoruz:
  - Zaten C7+sakrum etiketi VAR  -> kanıtlanmış uygun (en güvenilir sinyal)
  - Pose modelinin kalca/omuz güveni -> etiketlenmemişler için tahmin

Not: mevcut etiketlerin 21 tanesi heuristiğin "zayıf" dediği görüntülerde.
Yani heuristik fazla muhafazakâr; sınırdakileri atmak yerine ayrı klasöre
koyup kullanıcıya bıraktık.

ÇIKTI (data/ altında — git'e girmez)
  data/etiket_seti/1_oncelik_kanitli/   C7+sakrum etiketi olanlar
  data/etiket_seti/2_duzeltilecek/      etiketi eksik/mükerrer olanlar
  data/etiket_seti/3_yeni_iyi/          etiketlenmemiş, heuristik "iyi"
  data/etiket_seti/4_sinirda/           heuristik zayıf — bakıp karar ver
  data/etiket_seti/MANIFEST.csv
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "final"
AAG = KOK / "data" / "aag" / "train" / "_annotations.coco.json"
CIKTI = KOK / "data" / "etiket_seti"

KLASORLER = {
    "1_oncelik_kanitli": "C7+sakrum etiketi var — 7 noktayı yeniden çiz",
    "2_duzeltilecek": "etiket eksik/mükerrer — 7 noktayı temiz çiz",
    "3_yeni_iyi": "hiç etiketlenmemiş, alt+üst sırt görünüyor",
    "4_sinirda": "alt sırt şüpheli — bak, sakrum seçilmiyorsa atla",
}

OMUZ_ESIK = 0.5
KALCA_ESIK = 0.3


def belirginlestir(bgr):
    """Oluğu öne çıkaran yerel kontrast. Geometriyi DEĞİŞTİRMEZ."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(l)
    bulanik = cv2.GaussianBlur(l2, (0, 0), 11)
    detay = np.clip(cv2.addWeighted(l2, 1.7, bulanik, -0.7, 0), 0, 255).astype(np.uint8)
    lab2 = cv2.merge([detay,
                      cv2.addWeighted(a, 0.35, np.full_like(a, 128), 0.65, 0),
                      cv2.addWeighted(b, 0.35, np.full_like(b, 128), 0.65, 0)])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def aag_durumu():
    """{dosya_kok: 'tam' | 'bozuk'} — mevcut C7/sakrum etiketinin durumu."""
    if not AAG.exists():
        return {}
    d = json.loads(AAG.read_text(encoding="utf-8"))
    isim = {g["id"]: g["file_name"] for g in d["images"]}
    per = defaultdict(list)
    for a in d["annotations"]:
        per[a["image_id"]].append(a)

    def kok(ad):
        for k in ("_jpg.rf.", "_JPG_jpg.rf.", ".rf."):
            if k in ad:
                ad = ad.split(k)[0]
                break
        else:
            ad = Path(ad).stem
        for o in ("train_", "val_", "test_"):
            if ad.startswith(o + o):
                return ad[len(o):]
        return ad

    out = {}
    for img_id, anots in per.items():
        kats = sorted(a["category_id"] for a in anots)
        out[kok(isim[img_id])] = "tam" if kats == [1, 2] else "bozuk"
    return out


def main():
    kayitlar = json.loads((VERI / "kayitlar.json").read_text(encoding="utf-8"))
    durum = aag_durumu()

    if CIKTI.exists():
        import shutil
        shutil.rmtree(CIKTI)
    for k in KLASORLER:
        (CIKTI / k).mkdir(parents=True)

    satirlar = []
    sayim = Counter()

    for r in kayitlar:
        kok = Path(r["dosya"]).stem
        kaynak = VERI / r["split"] / r["dosya"]
        if not kaynak.exists():
            continue

        d = durum.get(kok)
        renkli = r.get("renkli", False)
        omuz = r.get("omuz_guven", 0) or 0
        kalca = r.get("kalca_guven", 0) or 0

        if d == "tam":
            grup = "1_oncelik_kanitli"
        elif d == "bozuk":
            grup = "2_duzeltilecek"
        elif renkli and kalca >= KALCA_ESIK and omuz >= OMUZ_ESIK:
            grup = "3_yeni_iyi"
        elif renkli:
            grup = "4_sinirda"
        else:
            continue          # gri/illüstrasyon — bu şema için kullanılamaz

        bgr = cv2.imread(str(kaynak))
        if bgr is None:
            continue
        # dosya adi data/final ile AYNI: sonradan eslestirmek icin sart
        cv2.imwrite(str(CIKTI / grup / r["dosya"]), belirginlestir(bgr),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        sayim[grup] += 1

        satirlar.append({
            "dosya": r["dosya"],
            "grup": grup,
            "split": r["split"],          # ORIJINAL split — korunmalı
            "mevcut_c7_sakrum": d or "yok",
            "kalca_guven": round(kalca, 3),
            "omuz_guven": round(omuz, 3),
            "aksiller_aci": r.get("aci"),
        })

    satirlar.sort(key=lambda s: (s["grup"], -s["kalca_guven"]))
    with (CIKTI / "MANIFEST.csv").open("w", newline="", encoding="utf-8-sig") as f:
        y = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()))
        y.writeheader()
        y.writerows(satirlar)

    print("=" * 62)
    print("ETIKET SETI HAZIR")
    print("=" * 62)
    toplam = 0
    for k, aciklama in KLASORLER.items():
        print(f"  {k:<22} {sayim[k]:4d}   {aciklama}")
        toplam += sayim[k]
    print(f"  {'TOPLAM':<22} {toplam:4d}")
    atlanan = len(kayitlar) - toplam
    print(f"\n  atlanan (gri/illüstrasyon): {atlanan}")

    sp = Counter(s["split"] for s in satirlar)
    print(f"\n  orijinal split dağılımı: " +
          "  ".join(f"{k}={v}" for k, v in sorted(sp.items())))
    print("  -> bu atama KORUNMALI, yeni modelde aynen kullanılacak")
    print(f"\n  manifest: {CIKTI / 'MANIFEST.csv'}")
    print(f"  klasör  : {CIKTI}")


if __name__ == "__main__":
    main()
