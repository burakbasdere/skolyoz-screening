"""Kotlin cikarim zincirini Python referansiyla karsilastirir.

DIKKAT: .venv-tflite icinde calisir:

    .venv-tflite/Scripts/python.exe scripts/28_kotlin_karsilastirma.py <teshis_dizini>

NE ISE YARAR
------------
MOBIL_YAPILACAKLAR.md Faz 3 sarti: "Birim testler: Python ciktisiyla ayni
goruntude <=0.05 derece farkla eslesmeli."

Birim testler zincirin saf parcalarini (decode, kirpim, normalizasyon,
mapleme) zaten kapsiyor. Kapsamadigi sey UCTAN UCA davranis: gercek bir
fotografta iki taraf ayni aciyi veriyor mu?

IKI AYRI KARSILASTIRMA — ve neden ayri
--------------------------------------
  1) girdi640.png : Kotlin'in modele verdigi TAM TENSOR. Python ayni tensoru
     isliyor. Burada yeniden boyutlandirma devrede degil, yani olculen sey
     zincirin kendisi. DOKUMANDAKI <=0.05 DERECE SARTI BURADA ARANIR.

  2) orijinal.png : Python kendi yeniden boyutlandirmasini (PIL BILINEAR)
     uyguluyor. Android'in createScaledBitmap'i ile PIL'inki AYNI DEGIL;
     buradaki fark o farkin sonuca yansimasi.

Ikisi ayrilmazsa cikan fark nereden geliyor bilinemez — port hatasi mi,
yeniden boyutlandirma mi?
"""

import math
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
degerlendirme = import_module("26_tflite_degerlendirme")
kilavuz = import_module("19_kilavuz_cizim")

KOK = Path(__file__).resolve().parent.parent
MODELLER = KOK / "android" / "app" / "src" / "main" / "assets"
ESIK_DERECE = 0.05


def sayi(metin):
    """Ondalik ayirici hem nokta hem VIRGUL olabilir.

    Eski teshis dosyalari cihazin Turkce yerel ayariyla yazildi ve "0,341"
    uretti. Kotlin tarafi Locale.ROOT'a cevrildi ama elimizdeki dosyalar
    okunabilsin diye ikisini de kabul ediyoruz.
    """
    return float(metin.strip().replace(",", "."))


def ikili(metin):
    """'464,944,237,704' veya '464.944,237.704' -> (x, y)."""
    p = [s for s in metin.strip().split(",") if s != ""]
    if len(p) == 2:
        return float(p[0]), float(p[1])
    if len(p) == 4:            # virgul hem ondalik hem ayirici
        return float(f"{p[0]}.{p[1]}"), float(f"{p[2]}.{p[3]}")
    raise ValueError(f"ayristirilamayan ikili: {metin!r}")


def sonuc_oku(yol):
    d = {}
    for satir in yol.read_text(encoding="utf-8").splitlines():
        if "=" not in satir:
            continue
        k, v = satir.split("=", 1)
        d[k.strip()] = v.strip()
    return d


def zinciri_kosur(m1, m2, im640, orijinal_boyut):
    """Python tarafinda ayni zincir. 26_tflite_degerlendirme ile ayni geometri."""
    x = np.asarray(im640.convert("RGB"), dtype=np.float32) / 255.0
    kaba = degerlendirme.asama1_coz(m1(x[None, ...]))
    if kaba is None:
        return None
    ince = {k: degerlendirme.rafine_et(m2, im640, v, False)
            for k, v in kaba.items()}
    G, Y = orijinal_boyut
    orj = {k: (v[0] * G / 640.0, v[1] * Y / 640.0) for k, v in ince.items()}
    return {"kaba": kaba, "ince": ince, "orijinal": orj}


def karsilastir(baslik, py, kt_sag, kt_sol, kt_aci, roll):
    if py is None:
        print(f"  {baslik:<22} PYTHON TESPIT EDEMEDI")
        return None
    p_sag, p_sol = py["orijinal"]["sag"], py["orijinal"]["sol"]
    p_aci = kilavuz.aksiller_aci(p_sag, p_sol, roll)
    d_sag = math.dist(p_sag, kt_sag)
    d_sol = math.dist(p_sol, kt_sol)
    d_aci = abs(p_aci - kt_aci)
    print(f"  {baslik:<22} nokta farki {d_sag:6.3f} / {d_sol:6.3f} px   "
          f"aci {p_aci:+8.4f} vs {kt_aci:+8.4f}  ->  {d_aci:.4f} derece")
    return d_aci


def main():
    if len(sys.argv) < 2:
        sys.exit("kullanim: 28_kotlin_karsilastirma.py <teshis_dizini>")
    dizin = Path(sys.argv[1])
    setler = sorted(dizin.rglob("*_sonuc.txt"))
    if not setler:
        sys.exit(f"{dizin} icinde *_sonuc.txt yok")

    m1 = degerlendirme.TFLite(MODELLER / "asama1.tflite")
    m2 = degerlendirme.TFLite(MODELLER / "asama2.tflite")
    print(f"Modeller: asama1 {m1.boyut_mb:.2f} MB, asama2 {m2.boyut_mb:.2f} MB")
    print(f"Karsilastirilacak cekim: {len(setler)}")

    farklar_640, farklar_orj = [], []

    for s in setler:
        onek = s.name.replace("_sonuc.txt", "")
        d = sonuc_oku(s)
        G, Y = (int(v) for v in d["orijinal"].split("x"))
        roll = sayi(d["roll"])
        kt_sag = ikili(d["orijinal_sag"])
        kt_sol = ikili(d["orijinal_sol"])
        kt_aci = sayi(d["aci"])

        print()
        print("=" * 78)
        print(f"{onek}   orijinal {G}x{Y}   roll {roll:+.4f} derece   "
              f"Kotlin aci {kt_aci:+.4f}")
        print("=" * 78)

        # 1) AYNI TENSOR — zincirin kendisi
        yol640 = s.parent / f"{onek}_girdi640.png"
        if yol640.exists():
            py = zinciri_kosur(m1, m2, Image.open(yol640), (G, Y))
            f = karsilastir("ayni 640 girdi", py, kt_sag, kt_sol, kt_aci, roll)
            if f is not None:
                farklar_640.append(f)

        # 2) AYNI ORIJINAL — yeniden boyutlandirma farki dahil
        yolorj = s.parent / f"{onek}_orijinal.png"
        if yolorj.exists():
            im = Image.open(yolorj).convert("RGB")
            py = zinciri_kosur(m1, m2, im.resize((640, 640), Image.BILINEAR), (G, Y))
            f = karsilastir("orijinal + PIL resize", py, kt_sag, kt_sol, kt_aci, roll)
            if f is not None:
                farklar_orj.append(f)

    print()
    print("=" * 78)
    print("SONUC")
    print("=" * 78)
    if farklar_640:
        e = max(farklar_640)
        print(f"  Ayni 640 girdi        : en buyuk fark {e:.4f} derece   "
              f"{'GECTI' if e <= ESIK_DERECE else 'KALDI'} (esik {ESIK_DERECE})")
        print("    Bu, portun kendisini olcer: decode, kirpim, mapleme, TFLite.")
    if farklar_orj:
        e = max(farklar_orj)
        print(f"  Orijinal + PIL resize : en buyuk fark {e:.4f} derece")
        print("    Bu esige tabi DEGIL. Android'in createScaledBitmap'i ile")
        print("    PIL BILINEAR ayni algoritma degil; fark buradan geliyor.")


if __name__ == "__main__":
    main()
