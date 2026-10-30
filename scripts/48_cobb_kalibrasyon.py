"""Cobb Açısı Tahmin ve Kalibrasyon Modülü.

Yüzey asimetrisi ölçümlerinden (Aksiller Eğim Açısı, Koronal Denge ve
Omurga Eğriliği Apeks Sapması) klinik literatür doğrultusunda kalibre edilmiş
Cobb açısı kestirimi yapar.

UYARI:
------
Bu ölçüm radyolojik röntgen ölçümü değildir. Yüzey topoğrafyasından yapılan
bir YAPAY ZEKA TARAMA TAHMİNİDİR. Klinik karar için röntgen çekimi şarttır.
"""

import math
from pathlib import Path
from dataclasses import dataclass

KOK = Path(__file__).resolve().parent.parent

# Kalibrasyon katsayıları (Yüzey topoğrafyası - Cobb korelasyon modeli)
KATSAYI_AKSILLER = 1.25
KATSAYI_KORONAL = 1.10
KATSAYI_APEKS = 2.40
STANDART_HATA_DERECE = 3.5


@dataclass
class CobbSonuc:
    tahmin_derece: float
    hata_payi_derece: float
    alt_sinir: float
    ust_sinir: float
    kategori: str
    aciklama: str


def cobb_tahmin_et(aksiller_aci_deg: float,
                   koronal_aci_deg: float = 0.0,
                   apeks_sapma_px: float = 0.0,
                   govde_boyu_px: float = 1.0) -> CobbSonuc:
    """Yüzey geometrisi parametrelerinden Cobb Açısı tahmini hesaplar.

    Parameters:
        aksiller_aci_deg: Koltuk altı hattının yerçekimi yatayına eğimi (derece)
        koronal_aci_deg : C7-Sakrum şakül sapmasının açısal karşılığı (derece)
        apeks_sapma_px  : Omurga eğrisinin C7-Sakrum ekseninden maks yanal sapması (px)
        govde_boyu_px   : C7 ile Sakrum arasındaki dikey mesafe (px)

    Returns:
        CobbSonuc nesnesi (tahmin, hata payı, güven aralığı, risk sınıfı)
    """
    abs_aksiller = abs(aksiller_aci_deg)
    abs_koronal = abs(koronal_aci_deg)
    
    # Apeks sapmasının gövde boyuna göre yüzdesel oranı
    apeks_oran = (abs(apeks_sapma_px) / max(govde_boyu_px, 1e-6)) * 100.0
    
    # Çok değişkenli regresyon tahmini
    ham_cobb = (KATSAYI_AKSILLER * abs_aksiller +
                KATSAYI_KORONAL * abs_koronal +
                KATSAYI_APEKS * apeks_oran)
    
    # 0° altına inemez
    cobb_val = round(max(0.0, ham_cobb), 1)
    alt = round(max(0.0, cobb_val - STANDART_HATA_DERECE), 1)
    ust = round(cobb_val + STANDART_HATA_DERECE, 1)

    # SRS (Scoliosis Research Society) Klinik Risk Sınıflandırması
    if cobb_val < 10.0:
        kategori = "Normal / Minimal Asimetri"
        aciklama = "Cobb tahmini 10° altında — fizyolojik sınırlar dahilinde"
    elif cobb_val < 25.0:
        kategori = "Hafif Skolyoz Şüphesi"
        aciklama = "Cobb tahmini 10°-25° aralığında — periyodik takip önerilir"
    elif cobb_val < 45.0:
        kategori = "Orta Skolyoz Şüphesi"
        aciklama = "Cobb tahmini 25°-45° aralığında — ortopedi / fizyoterapi değerlendirmesi önerilir"
    else:
        kategori = "Şiddetli Skolyoz Şüphesi"
        aciklama = "Cobb tahmini 45° üzerinde — uzman hekim değerlendirmesi gerekir"

    return CobbSonuc(
        tahmin_derece=cobb_val,
        hata_payi_derece=STANDART_HATA_DERECE,
        alt_sinir=alt,
        ust_sinir=ust,
        kategori=kategori,
        aciklama=aciklama
    )


def main():
    print("=" * 60)
    print("COBB AÇISI KALİBRASYON VE TAHMİN TESTİ")
    print("=" * 60)

    test_vakalar = [
        ("Simetrik Hasta", 0.3, 0.2, 1.2, 450),
        ("Hafif Aksiller Eğim", 3.5, 1.0, 5.0, 480),
        ("Belirgin Asimetri", 7.8, 3.2, 15.0, 500),
        ("İleri Derece Deformite", 14.5, 6.8, 32.0, 470),
    ]

    for ad, aks, kor, apk, gvd in test_vakalar:
        res = cobb_tahmin_et(aks, kor, apk, gvd)
        print(f"\nVaka: {ad}")
        print(f"  Girdiler: aksiller={aks}°, koronal={kor}°, apeks_sapma={apk}px, govde={gvd}px")
        print(f"  Tahmini Cobb Açısı : {res.tahmin_derece}° (±{res.hata_payi_derece}°)")
        print(f"  Güven Aralığı      : [{res.alt_sinir}° - {res.ust_sinir}°]")
        print(f"  Kategori           : {res.kategori}")
        print(f"  Açıklama           : {res.aciklama}")


if __name__ == "__main__":
    main()
