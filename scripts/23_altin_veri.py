"""Kotlin portu icin ALTIN VERI uretir.

NEDEN
-----
MOBIL_YAPILACAKLAR.md Faz 3: "Birim testler: Python ciktisiyla ayni goruntude
<=0.05 derece farkla eslesmeli. Bu testi yazmadan devam etme."

Elle yazilmis beklenen degerler bu isi GORMEZ: insan hesabi da Kotlin kodu da
ayni yanlis varsayimi paylasabilir (ozellikle isaret ve eksen yonu
konvansiyonlarinda). Bu yuzden beklenen degerleri referans uygulamanin
KENDISINDEN uretiyoruz.

Cikti: android/app/src/test/resources/olcum_altin_veri.csv
Kotlin tarafi bu dosyayi okuyup karsilastirir; JSON ayristiricisi gerekmesin
diye duz CSV.

Python tarafinda olcum matematigi degisirse bu script yeniden calistirilmali;
Kotlin testi o an kirilir ve iki taraf ayrisirsa haberimiz olur.
"""

import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
kilavuz = import_module("19_kilavuz_cizim")

KOK = Path(__file__).resolve().parent.parent
CIKTI = KOK / "android" / "app" / "src" / "test" / "resources" / "olcum_altin_veri.csv"

# (sag, sol, roll) — sag/sol ORIJINAL goruntu pikselinde, y asagi artar.
ACI_VAKALARI = [
    # duz duran kisi, telefon dik
    ((300.0, 500.0), (700.0, 500.0), 0.0),
    # sag omuz yuksek (sag noktanin y'si KUCUK) -> pozitif beklenir
    ((300.0, 460.0), (700.0, 500.0), 0.0),
    # sol omuz yuksek -> negatif beklenir
    ((300.0, 500.0), (700.0, 460.0), 0.0),
    # telefon 6 derece egik tutulmus, kisi duz -> roll duzeltmesi -6 vermeli
    ((300.0, 500.0), (700.0, 500.0), 6.0),
    # telefon -4 derece egik + kisi de egri
    ((300.0, 470.0), (700.0, 500.0), -4.0),
    # buyuk aci
    ((280.0, 620.0), (760.0, 480.0), 0.0),
    # dar aralik (noktalar birbirine yakin) - sayisal duyarlilik
    ((498.0, 500.0), (502.0, 499.0), 0.0),
    # tam ust uste (dejenere) -> 0.0 donmeli
    ((500.0, 500.0), (500.0, 500.0), 0.0),
    # gercek telefon cozunurlugu olcegi
    ((1180.0, 2050.0), (1900.0, 1985.0), 1.7),
    # negatif roll + sol yuksek
    ((400.0, 900.0), (1400.0, 940.0), -11.25),
]

# (gx, gy, gz) — Android sensor ekseni: x saga, y yukari, z ekrandan disari.
# Telefon dik tutuldugunda g = (0, +9.81, 0).
DURUS_VAKALARI = [
    (0.0, 9.81, 0.0),          # tam dik
    (1.0, 9.75, 0.0),          # hafif yana egik
    (-1.0, 9.75, 0.0),
    (0.0, 9.70, 1.5),          # one/arkaya yatik
    (0.0, 9.70, -1.5),
    (2.5, 9.40, 0.8),          # ikisi birden
    (0.0, 0.0, 9.81),          # masada duz yatiyor -> pitch 90
    (9.81, 0.0, 0.0),          # yan yatmis -> roll 90
]


def main():
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    satirlar = ["tip,a,b,c,d,e,beklenen1,beklenen2"]

    for sag, sol, roll in ACI_VAKALARI:
        aci = kilavuz.aksiller_aci(sag, sol, roll)
        satirlar.append(
            f"aci,{sag[0]:.6f},{sag[1]:.6f},{sol[0]:.6f},{sol[1]:.6f},"
            f"{roll:.6f},{aci:.10f},"
        )

    for gx, gy, gz in DURUS_VAKALARI:
        roll = kilavuz.roll_acisi(gx, gy)
        pitch = kilavuz.pitch_acisi(gx, gy, gz)
        satirlar.append(
            f"durus,{gx:.6f},{gy:.6f},{gz:.6f},,,{roll:.10f},{pitch:.10f}"
        )

    CIKTI.write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    print(f"yazildi: {CIKTI}")
    print(f"  aci vakasi   : {len(ACI_VAKALARI)}")
    print(f"  durus vakasi : {len(DURUS_VAKALARI)}")
    print()
    for s in satirlar[:4]:
        print("  " + s)
    print("  ...")


if __name__ == "__main__":
    main()
