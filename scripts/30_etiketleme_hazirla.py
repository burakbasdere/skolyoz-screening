"""C7 + kuyruk sokumu (+ mevcut koltuk altı) etiketlemesi için veri toplar.

    python scripts/30_etiketleme_hazirla.py

NEDEN
-----
Koltuk altı tabanı (sag-sol, ~300 px) kısa; piksel titremesi açıya büyük
yansıyor. C7 → kuyruk sokumu tabanı gövde boyunca ~2 kat uzun, dolayısıyla
aynı piksel hatasında ~yarı açı hatası. Ayrıca C7 kemiksi bir çıkıntı, sokum
net bir iz — koltuk altının yumuşak dokusundan daha kararlı landmark.

Bu script YENİDEN EĞİTİM İÇİN VERİ ÜRETMİYOR; sadece mevcut 337 görüntüyü
elle etiketlenebilecek tek bir düz klasöre topluyor ve hangilerinin bu iş için
uygun olduğunu işaretliyor.

UYGUNLUK İPUCU
--------------
C7 boyun altında, sokum belde. İkisinin de görünmesi için fotoğrafın tam
sırtı (boyun→bel) göstermesi gerekiyor. Bunu kesin bilemeyiz ama mevcut
etiketlerdeki sinyallerden tahmin ediyoruz:
  - renkli            : illüstrasyon/gri tonlama klinik görüntü değil
  - omuz_guven yüksek : üst sırt/omuzlar görünüyor
  - kalca_guven yüksek: alt sırt/kalça görünüyor  <-- sokum için kritik

"iyi" = üçü de tuttuğu görüntüler. Kullanıcı önce bunları etiketlemeli;
alt sırtı hiç görünmeyenler (kalca_guven düşük) bu şema için işe yaramaz.

ÇIKTI (data/ altında — git'e GİRMEZ, hasta verisi)
  data/etiketleme/goruntuler/       düz klasör, tüm görüntüler
  data/etiketleme/etiket_sablonu.csv indeks + mevcut koltuk altı + boş sütunlar
"""

import csv
import json
import shutil
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "final"
CIKTI = KOK / "data" / "etiketleme"
GORUNTU = CIKTI / "goruntuler"

# Uygunluk eşikleri — mevcut etiketlerin dağılımından seçildi.
OMUZ_ESIK = 0.5
KALCA_ESIK = 0.3


def uygunluk(r):
    """Görüntü C7+sokum etiketlemesi için ne kadar uygun."""
    renkli = r.get("renkli", False)
    omuz = r.get("omuz_guven", 0) or 0
    kalca = r.get("kalca_guven", 0) or 0
    if not renkli:
        return "zayif", "gri/illustrasyon"
    if kalca < KALCA_ESIK:
        return "zayif", "alt sirt gorunmuyor"
    if omuz < OMUZ_ESIK:
        return "orta", "ust sirt belirsiz"
    return "iyi", ""


def main():
    kayitlar = json.loads((VERI / "kayitlar.json").read_text(encoding="utf-8"))
    GORUNTU.mkdir(parents=True, exist_ok=True)

    satirlar = []
    kopyalanan = 0
    for r in kayitlar:
        kaynak_yol = VERI / r["split"] / r["dosya"]
        if not kaynak_yol.exists():
            continue
        # Çakışmayı önlemek için split ön eki (test_/train_/val_ zaten var ama
        # garanti olsun); düz klasöre kopyala.
        hedef = GORUNTU / r["dosya"]
        shutil.copy2(kaynak_yol, hedef)
        kopyalanan += 1

        seviye, neden = uygunluk(r)
        sag = r.get("sag") or [None, None]
        sol = r.get("sol") or [None, None]
        satirlar.append({
            "dosya": r["dosya"],
            "uygunluk": seviye,
            "uygunluk_neden": neden,
            "kalca_guven": round(r.get("kalca_guven", 0) or 0, 3),
            "omuz_guven": round(r.get("omuz_guven", 0) or 0, 3),
            # Mevcut koltuk altı — YENİDEN etiketlemeye gerek yok, referans.
            "sag_x": sag[0], "sag_y": sag[1],
            "sol_x": sol[0], "sol_y": sol[1],
            # Elle doldurulacak yeni landmark'lar:
            "c7_x": "", "c7_y": "",
            "sokum_x": "", "sokum_y": "",
            # İstege bağlı ara omurga noktası (eğriliği yakalamak için):
            "omurga_orta_x": "", "omurga_orta_y": "",
            "not": "",
        })

    # "iyi" olanlar üstte olsun ki kullanıcı önce onları etiketlesin.
    sira = {"iyi": 0, "orta": 1, "zayif": 2}
    satirlar.sort(key=lambda s: (sira[s["uygunluk"]], -s["kalca_guven"]))

    sablon = CIKTI / "etiket_sablonu.csv"
    with sablon.open("w", newline="", encoding="utf-8") as f:
        yazici = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()))
        yazici.writeheader()
        yazici.writerows(satirlar)

    import collections
    sayim = collections.Counter(s["uygunluk"] for s in satirlar)
    print(f"kopyalanan goruntu: {kopyalanan}  -> {GORUNTU}")
    print(f"indeks: {sablon}")
    print(f"uygunluk: iyi {sayim['iyi']}  orta {sayim['orta']}  zayif {sayim['zayif']}")
    print()
    print("Etiketleme sirasi: once 'iyi' olanlar. 'zayif' (alt sirt gorunmeyen)")
    print("bu sema icin atlanabilir.")


if __name__ == "__main__":
    main()
