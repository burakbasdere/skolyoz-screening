"""Asama 1b - Gorsel kontrol sayfasi.

Otomatik dondurme ve sag/sol atamasi dogru mu? Kod dogru diyor diye
kabul etmiyoruz; izgara halinde cizip goze bakiyoruz.

Her karede:
  - R (sag koltuk) ve L (sol koltuk) noktalari
  - aralarindaki cizgi
  - yatay referans (kesikli)
  - aci ve oryantasyon karar guveni

Uretilen sayfalar:
  qa_rastgele.jpg      -> rastgele 24 ornek (genel dogruluk)
  qa_dusuk_guven.jpg   -> en dusuk guvenli 24 ornek (en riskli vakalar)
"""

import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

# varsayilan clean2 (pose tabanli); argumanla baska klasor verilebilir
KOK = Path(sys.argv[1]) if len(sys.argv) > 1 else (Path(__file__).resolve().parent.parent / "data" / "clean2")
KAYIT = KOK / "kayitlar.json"
KARE = 220
SUTUN, SATIR = 6, 4
TOHUM = 7


def goruntu_yolu(r):
    """clean2 duz 'images/' kullanir; final split alt klasorlerine ayirir."""
    aday = KOK / "images" / r["dosya"]
    if aday.exists():
        return aday
    return KOK / r.get("split", "train") / r["dosya"]


def kare_ciz(r):
    im = Image.open(goruntu_yolu(r)).convert("RGB")
    s = im.size[0]
    im = im.resize((KARE, KARE), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    o = KARE / s

    sx, sy = r["sag"][0] * o, r["sag"][1] * o
    lx, ly = r["sol"][0] * o, r["sol"][1] * o

    # yatay referans (kesikli), noktalarin orta yuksekliginde
    my = (sy + ly) / 2
    for x in range(0, KARE, 12):
        d.line([(x, my), (x + 6, my)], fill=(90, 90, 90), width=1)

    # koltuk alti cizgisi
    d.line([(sx, sy), (lx, ly)], fill=(255, 210, 0), width=3)

    for (x, y), etiket, renk in ((( sx, sy), "R", (255, 60, 60)),
                                 ((lx, ly), "L", (60, 160, 255))):
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=renk, outline=(255, 255, 255), width=2)
        d.text((x + 9, y - 7), etiket, fill=renk)

    # ust serit: aci / rotasyon / guven
    d.rectangle([0, 0, KARE, 17], fill=(0, 0, 0))
    g = r.get("pose_guven", r.get("guven", 0.0))
    bayrak = "" if r.get("renkli", True) else " GRI"
    d.text((4, 4), f"{r['aci']:+.1f}°  rot{r['rotasyon']}  p={g:.2f}{bayrak}",
           fill=(255, 255, 255))
    d.rectangle([0, 0, KARE - 1, KARE - 1], outline=(40, 40, 40))
    return im


def sayfa(kayitlar, cikti, baslik):
    tuval = Image.new("RGB", (SUTUN * KARE, SATIR * KARE + 22), (18, 18, 18))
    d = ImageDraw.Draw(tuval)
    d.text((6, 6), baslik, fill=(255, 255, 255))
    for i, r in enumerate(kayitlar[:SUTUN * SATIR]):
        x = (i % SUTUN) * KARE
        y = (i // SUTUN) * KARE + 22
        tuval.paste(kare_ciz(r), (x, y))
    tuval.save(cikti, quality=92)
    print(f"  yazildi: {cikti}")


def main():
    kayitlar = json.loads(KAYIT.read_text(encoding="utf-8"))
    print(f"Toplam kayit: {len(kayitlar)}")

    random.seed(TOHUM)
    rast = random.sample(kayitlar, min(SUTUN * SATIR, len(kayitlar)))
    sayfa(rast, KOK / "qa_rastgele.jpg",
          "RASTGELE 24 - dogru mu? R sagda, L solda, kisi dik olmali")

    anahtar = (lambda r: r.get("pose_guven", r.get("guven", 0.0)))
    sayfa(sorted(kayitlar, key=anahtar), KOK / "qa_dusuk_guven.jpg",
          "EN DUSUK POSE GUVENI 24 - en riskli vakalar")


if __name__ == "__main__":
    main()
