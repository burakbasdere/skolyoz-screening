"""Etiketleme referans sayfası — 7 landmark'ın nereye konacağını gösterir.

Roboflow'da etiketlerken bakılacak tek sayfa. Örnek görüntü train_0175;
koltuk altı, C7 ve sakrum GERÇEK etiketlerden, 3 omurga noktası ise
görünür oluk izlenerek elle yerleştirildi (örnek amaçlı).

Kural özeti sağ panelde.
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

KOK = Path(__file__).resolve().parent.parent
# train_0114: belirgin saga konveks torakal egrilik, oluk net gorunuyor.
# (Ilk denemede train_0175 secilmisti; o kiside egrilik ~8 px, ogretici degil.)
GORUNTU = KOK / "data" / "final" / "train" / "train_0114.jpg"
CIKTI = KOK / "runs" / "omurga" / "ETIKET_ORNEGI.jpg"

# --- landmark'lar: (ad, x, y, renk, aciklama) ---
SAG = (255, 70, 70)
SOL = (70, 160, 255)
MOR = (190, 120, 255)
TURUNCU = (255, 150, 40)

# c7 / sakrum / koltuk alti: GERCEK etiketler.
# omurga_1..3: kontrast artirilmis buyutmede oluk izlenerek okundu.
#   düz çizgi        okunan       sapma
#   %25 -> 314.2     320          +5.8
#   %50 -> 297.5     318         +20.5
#   %75 -> 280.8     295         +14.2
NOKTALAR = [
    ("sag koltuk", 570.0, 214.6, SAG,     "mevcut etiket — yeniden çizme"),
    ("sol koltuk", 133.9, 238.3, SOL,     "mevcut etiket — yeniden çizme"),
    ("c7",         331.0,  41.0, MOR,     "boyun dibindeki kemik çıkıntı"),
    ("omurga_1",   320.0, 169.5, TURUNCU, "oluk üzerinde, üst bölge"),
    ("omurga_2",   318.0, 298.0, TURUNCU, "oluk üzerinde, orta bölge"),
    ("omurga_3",   295.0, 426.5, TURUNCU, "oluk üzerinde, alt bölge"),
    ("sakrum",     264.0, 555.0, MOR,     "kalça yarığının üst ucu"),
]

KURALLAR = [
    ("SIRA ÖNEMLİ", None),
    ("omurga_1 en ÜSTTE (C7'ye yakın), omurga_3 en ALTTA", (215, 215, 215)),
    ("(sakruma yakın). Karıştırırsan eğri ters çıkar.", (215, 215, 215)),
    ("", None),
    ("YÜKSEKLİK SERBEST", None),
    ("Tam %25/%50/%75 olmak zorunda değil. Sonradan eğri", (215, 215, 215)),
    ("uyduruyoruz. Gereken: C7 ile sakrum arasına makul", (215, 215, 215)),
    ("şekilde YAYILMIŞ olmaları.", (215, 215, 215)),
    ("", None),
    ("OLUK GÖRÜNMÜYORSA", None),
    ("Gövdenin iki kenarından eşit uzaklıktaki hattı işaretle.", (215, 215, 215)),
    ("Tahmin etmekten çekinme — tutarlı ol, kusursuz olma.", (215, 215, 215)),
    ("", None),
    ("YAPMA", None),
    ("• C7-sakrum düz çizgisi üzerine hizalama — eğriliği", (255, 170, 170)),
    ("  yok eder, ölçmeye çalıştığımız şey o.", (255, 170, 170)),
    ("• Omurları saymaya çalışma, fotoğraftan seçilmez.", (255, 170, 170)),
    ("• Kürek kemiği kenarını oluk sanma (soldaki koyu", (255, 170, 170)),
    ("  şerit oluk değil, kas gölgesi).", (255, 170, 170)),
    ("", None),
    ("ALT SIRT GÖRÜNMÜYORSA", None),
    ("Sakrum yoksa o görüntüyü atla. C7'siz de olmaz.", (215, 215, 215)),
]


def yazi(boyut, kalin=False):
    adlar = (("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf") if kalin
             else ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"))
    for ad in adlar:
        try:
            return ImageFont.truetype(ad, boyut)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    im = Image.open(GORUNTU).convert("RGB")
    G = 720
    o = G / im.size[0]
    im = im.resize((G, G), Image.LANCZOS)

    PANEL = 620
    UST = 50            # baslik seridi
    ALT = 34            # alt not seridi
    tuval = Image.new("RGB", (G + PANEL, UST + G + ALT), (22, 22, 26))
    d = ImageDraw.Draw(tuval)

    f_baslik = yazi(30, True)
    f_ad = yazi(19, True)
    f_metin = yazi(17)
    f_kucuk = yazi(15)

    d.text((18, 16), "ETİKETLEME REFERANSI — 7 landmark", font=f_baslik, fill=(255, 255, 255))

    kat = ImageDraw.Draw(im, "RGBA")

    # omurga egrisi (noktalari birlestir) — egriligi gorunur kilar
    omurga = [(x * o, y * o) for ad, x, y, *_ in NOKTALAR if ad.startswith("omurga")]
    c7 = next((x * o, y * o) for ad, x, y, *_ in NOKTALAR if ad == "c7")
    sk = next((x * o, y * o) for ad, x, y, *_ in NOKTALAR if ad == "sakrum")
    # duz cizgi ile egri ARASINI boya — olctugumuz sey tam olarak bu fark
    duz_karsilik = []
    for px, py in omurga:
        t = (py - c7[1]) / (sk[1] - c7[1]) if sk[1] != c7[1] else 0.0
        duz_karsilik.append((c7[0] + (sk[0] - c7[0]) * t, py))
    kat.polygon([c7] + omurga + [sk] + list(reversed(duz_karsilik)),
                fill=(255, 150, 40, 70))

    kat.line([c7] + omurga + [sk], fill=(*TURUNCU, 230), width=5)
    # referans: C7-sakrum duz cizgisi (kesikli izlenim)
    for t in range(0, 100, 6):
        a = t / 100.0
        b = min(1.0, (t + 3) / 100.0)
        kat.line([(c7[0] + (sk[0] - c7[0]) * a, c7[1] + (sk[1] - c7[1]) * a),
                  (c7[0] + (sk[0] - c7[0]) * b, c7[1] + (sk[1] - c7[1]) * b)],
                 fill=(255, 255, 255, 200), width=3)

    for ad, x, y, renk, _ in NOKTALAR:
        px, py = x * o, y * o
        kat.ellipse([px - 9, py - 9, px + 9, py + 9], fill=renk,
                    outline=(255, 255, 255), width=3)
        # etiket kutusu, anatomiyi kapatmasin diye disari kacir
        sag_tarafta = px > G / 2
        ex = px + (26 if sag_tarafta else -26 - d.textlength(ad, font=f_kucuk) - 12)
        ex = max(4, min(ex, G - d.textlength(ad, font=f_kucuk) - 16))
        kat.line([(px, py), (ex + (0 if sag_tarafta else
                                   d.textlength(ad, font=f_kucuk) + 12), py)],
                 fill=(*renk, 180), width=2)
        w = d.textlength(ad, font=f_kucuk) + 12
        kat.rounded_rectangle([ex, py - 12, ex + w, py + 12], radius=6,
                              fill=(0, 0, 0, 200), outline=renk, width=2)
        kat.text((ex + 6, py - 8), ad, font=f_kucuk, fill=renk)

    tuval.paste(im, (0, UST))

    # --- sag panel ---
    px0 = G + 24
    y = 62
    d.text((px0, y), "ROBOFLOW SINIF ADLARI", font=f_ad, fill=(255, 255, 255))
    y += 30
    for ad, _, _, renk, aciklama in NOKTALAR:
        d.ellipse([px0, y + 4, px0 + 14, y + 18], fill=renk, outline=(255, 255, 255), width=2)
        d.text((px0 + 24, y), ad, font=f_ad, fill=renk)
        d.text((px0 + 24 + 130, y + 2), aciklama, font=f_kucuk, fill=(180, 180, 180))
        y += 27

    y += 14
    d.line([(px0, y), (px0 + PANEL - 48, y)], fill=(70, 70, 80), width=2)
    y += 16

    for metin, renk in KURALLAR:
        if metin == "":
            y += 8
            continue
        if renk is None:
            d.text((px0, y), metin, font=f_ad, fill=(255, 210, 80))
            y += 26
        else:
            d.text((px0, y), metin, font=f_metin, fill=renk)
            y += 22

    # alt not — goruntunun ALTINDA, uzerinde degil
    d.text((18, UST + G + 9),
           "Turuncu eğri = omurga hattı   ·   Beyaz kesikli = C7-sakrum düz çizgisi   ·   "
           "aradaki fark ölçtüğümüz eğrilik",
           font=f_kucuk, fill=(160, 160, 160))

    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    tuval.save(CIKTI, quality=95)
    print(f"yazildi: {CIKTI}  ({tuval.size[0]}x{tuval.size[1]})")


if __name__ == "__main__":
    main()
