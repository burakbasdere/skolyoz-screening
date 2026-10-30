"""Asama 4 - Kilavuz cizim / olcum katmani.

Bu dosya iki is yapar:
  1) OLCUM MATEMATIGI - Android tarafina birebir portlanacak referans.
     Buradaki fonksiyonlar goruntu isleme icermez, saf geometridir.
  2) CIZIM - hastanin fotografi uzerine sari koltuk alti cizgisi,
     kesikli YERCEKIMI yatayi, aci yayi ve okuma.

KRITIK: kesikli referans cizgisi goruntunun yatayi DEGIL, IMU'dan gelen
gercek yercekimi yatayidir. Telefon 5 derece egik tutulursa duz bir hasta
5 derece egri olcusur; roll duzeltmesi bunu sifirlar.

Aci isareti: pozitif = kisinin SAG omzu yuksek (karede sagda, arkadan cekim).
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SARI = (255, 205, 0)
KIRMIZI = (255, 70, 70)
MAVI = (70, 160, 255)
MOR = (190, 120, 255)      # C7 / sakrum ve koronal denge
GRI = (170, 170, 170)
BEYAZ = (255, 255, 255)


# ----------------------------------------------------------- olcum (portlanacak)

def roll_acisi(g_x, g_y):
    """IMU yercekimi vektorunden telefonun roll acisi (derece).

    g_x, g_y : yercekimi vektorunun cihaz duzlemindeki bilesenleri.
    Telefon dik tutuldugunda g_x=0, g_y=+9.81 -> roll 0.
    """
    return math.degrees(math.atan2(g_x, g_y))


def pitch_acisi(g_x, g_y, g_z):
    """IMU yercekimi vektorunden telefonun pitch acisi (derece).

    Yercekimi vektorunun EKRAN DUZLEMIYLE yaptigi aci. Bu tanim roll'dan
    bagimsizdir: telefonu kendi ekseninde dondurmek pitch'i degistirmez,
    sadece one/arkaya yatirmak degistirir.

    Ekran duzlemi dikeyken g_z = 0 -> pitch 0.
    Pozitif = telefonun ustu geriye yatik (ekran yukari bakiyor).
    """
    return math.degrees(math.atan2(g_z, math.hypot(g_x, g_y)))


def aksiller_aci(sag, sol, roll_derece=0.0):
    """Iki koltuk alti noktasindan yercekimine gore egim acisi.

    sag, sol : (x, y) goruntu pikselinde. y asagi dogru artar.
    roll     : telefonun roll acisi (derece). Goruntu yatayini gercek
               yataya cevirmek icin cikarilir.

    Donen deger: pozitif = kisinin SAG omzu daha yuksek.
    """
    dx = sol[0] - sag[0]
    dy = sol[1] - sag[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    ham = math.degrees(math.atan2(dy, abs(dx)))
    return ham - roll_derece


def guven_kontrolu(pitch_derece, esik=8.0):
    """Telefon one/arkaya yatikken perspektif aciyi bozar; cekimi reddet.

    IMU pitch'i bu esigi asiyorsa olcum guvenilmez.
    """
    return abs(pitch_derece) <= esik


# --- olculen acinin ETIKETLI VERI SETIMIZDEKI konumu -------------------------
# Klinik esikler (10-25 hafif, 25-45 orta, >45 siddetli) COBB acisi icindir.
# Bizim olctugumuz aksiller egim farkli bir buyukluk (medyan 2.21, p90 7.68).
# Aksiller acisi 8 olan birinin Cobb'u 5 de olabilir 35 de; klinik etiket
# yapistiramayiz.
#
# KISIT: bu yuzdelikler 337 goruntuluk etiketli setimizden geliyor. O set okul
# taramasindan gelen rastgele bir orneklem DEGIL, klinik goruntulerle
# zenginlesmis. "Genel nufusun %90'indan yuksek" DENMEZ; "olctuklerimizin ust
# %10'unda" denir.
DAGILIM_TABLO = [(2.21, 50.0), (2.73, 60.0), (3.27, 70.0), (3.75, 75.0),
                 (4.45, 80.0), (5.46, 85.0), (7.68, 90.0), (10.27, 95.0),
                 (12.46, 97.5), (14.15, 99.0)]
YONLENDIRME_ESIGI = 7.68     # veri setinin p90'i


def dagilim_yuzdelik(aci_derece):
    """|aci| -> yuzdelik (0-100), tablo arasi dogrusal aradegerleme."""
    a = abs(aci_derece)
    x0, y0 = DAGILIM_TABLO[0]
    if a <= x0:
        return a / x0 * y0
    for (x0, y0), (x1, y1) in zip(DAGILIM_TABLO, DAGILIM_TABLO[1:]):
        if a <= x1:
            return y0 + (a - x0) / (x1 - x0) * (y1 - y0)
    return 99.5


def belirgin_asimetri(aci_derece):
    return abs(aci_derece) >= YONLENDIRME_ESIGI


def dagilim_aciklama(aci_derece):
    """Arayuz satiri. Klinik derece IDDIA ETMEZ.

    Turkce ek uyumu her sayida degistigi icin ("%10'unda" ama "%2'sinde")
    iyelik ekinden kacinip sabit "dilimde" kullaniyoruz.
    """
    ust = 100.0 - dagilim_yuzdelik(aci_derece)
    if ust >= 50:
        konum = "veri setimizin alt yarısında"
    elif ust < 1:
        konum = "veri setimizde üst %1 dilimde"
    else:
        konum = f"veri setimizde üst %{ust:.0f} dilimde"
    if belirgin_asimetri(aci_derece):
        konum += " — belirgin asimetri, hekime başvurun"
    return konum


def koronal_denge(c7, sakrum, roll_derece=0.0):
    """C7'nin sakrumdan cekilen YERCEKIMI SAKULUNDEN yatay sapmasi.

    Skolyoz muayenesinde C7'den sarkitilan iplik olcumunun karsiligidir.

    c7, sakrum : (x, y) goruntu pikselinde, y asagi artar.
    roll       : telefon roll acisi (derece); sakul goruntu dikeyi degil,
                 yercekimi dikeyidir.

    Doner: (sapma_px, sapma_derece, govde_px)
      sapma_px > 0  -> C7 sakrumun SAGINDA (govde saga kaymis)
      sapma_derece  -> sapmanin govde boyuna gore acisal karsiligi
    """
    r = math.radians(roll_derece)
    # yercekimi cerceves: dikey birim vektor (asagi), yatay birim vektor (saga)
    dik_x, dik_y = -math.sin(r), math.cos(r)
    yat_x, yat_y = math.cos(r), math.sin(r)

    vx = c7[0] - sakrum[0]
    vy = c7[1] - sakrum[1]
    sapma = vx * yat_x + vy * yat_y          # yercekimi yatayindaki bilesen
    govde = abs(vx * dik_x + vy * dik_y)     # yercekimi dikeyindeki bilesen
    if govde < 1e-6:
        return sapma, 0.0, govde
    return sapma, math.degrees(math.atan2(sapma, govde)), govde


# ----------------------------------------------------------------------- cizim

def _yazi(boyut):
    for ad in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(ad, boyut)
        except OSError:
            continue
    return ImageFont.load_default()


def _kesikli(d, p1, p2, renk, kalinlik=2, bosluk=10, dolu=14):
    uzunluk = math.dist(p1, p2)
    if uzunluk < 1:
        return
    ux = (p2[0] - p1[0]) / uzunluk
    uy = (p2[1] - p1[1]) / uzunluk
    t = 0.0
    while t < uzunluk:
        s = min(t + dolu, uzunluk)
        d.line([(p1[0] + ux * t, p1[1] + uy * t),
                (p1[0] + ux * s, p1[1] + uy * s)], fill=renk, width=kalinlik)
        t = s + bosluk


def kilavuz_ciz(im, sag, sol, roll_derece=0.0, pitch_derece=0.0,
                c7=None, sakrum=None):
    """Fotograf uzerine olcum kilavuzunu cizer, yeni goruntu dondurur.

    c7 / sakrum verilirse ikinci olcum de cizilir: sakrumdan cekilen
    YERCEKIMI SAKULU (dikey kilavuz) ve C7'nin bu sakulden yatay sapmasi.
    Ikisi de None ise davranis eskisiyle aynidir.
    """
    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im, "RGBA")
    W, H = im.size
    olcek = max(W, H) / 640.0

    aci = aksiller_aci(sag, sol, roll_derece)
    gecerli = guven_kontrolu(pitch_derece)

    mx = (sag[0] + sol[0]) / 2.0
    my = (sag[1] + sol[1]) / 2.0

    # --- yercekimi cerceves birim vektorleri (ikisi de roll kadar doner) ---
    r = math.radians(roll_derece)
    yat = (math.cos(r), math.sin(r))     # yercekimi yatayi, saga
    dik = (-math.sin(r), math.cos(r))    # yercekimi dikeyi, asagi

    # --- YATAY ve DIKEY kilavuz çizgileri (yerçekimi yatayı ve dikey şakülü) ---
    uzan_y = W * 0.75
    uzan_d = H * 0.75
    _kesikli(d, (mx - yat[0] * uzan_y, my - yat[1] * uzan_y),
             (mx + yat[0] * uzan_y, my + yat[1] * uzan_y), GRI,
             max(1, int(2 * olcek)), int(10 * olcek), int(14 * olcek))
    _kesikli(d, (mx - dik[0] * uzan_d, my - dik[1] * uzan_d),
             (mx + dik[0] * uzan_d, my + dik[1] * uzan_d), GRI,
             max(1, int(2 * olcek)), int(10 * olcek), int(14 * olcek))

    # --- DIKEY kilavuz: sakrumdan cekilen sakul (koronal denge referansi) ---
    if c7 is not None and sakrum is not None:
        uz = H * 0.95
        _kesikli(d, (sakrum[0] - dik[0] * uz, sakrum[1] - dik[1] * uz),
                 (sakrum[0] + dik[0] * uz * 0.15, sakrum[1] + dik[1] * uz * 0.15),
                 MOR, max(1, int(2 * olcek)), int(10 * olcek), int(14 * olcek))

    # --- aci yayi: yercekimi yatayindan koltuk alti cizgisine ---
    # Cizgi yonunu SOLDAN SAGA olcuyoruz (sol -> sag). sag karede sagda
    # oldugu icin bu yon ~0 derecedir; tersi alinirsa yay 180 derece taranir.
    yaricap = max(26.0, math.dist(sag, sol) * 0.20)
    a_cizgi = math.degrees(math.atan2(sag[1] - sol[1], sag[0] - sol[0]))
    a_yatay = roll_derece
    d.arc([mx - yaricap, my - yaricap, mx + yaricap, my + yaricap],
          start=min(a_yatay, a_cizgi), end=max(a_yatay, a_cizgi),
          fill=SARI, width=max(2, int(3 * olcek)))

    # --- koltuk alti cizgisi ---
    d.line([tuple(sag), tuple(sol)], fill=SARI, width=max(3, int(4 * olcek)))

    nr = max(6, int(8 * olcek))
    for nokta, renk, etiket in ((sag, KIRMIZI, "R"), (sol, MAVI, "L")):
        x, y = nokta
        d.ellipse([x - nr, y - nr, x + nr, y + nr], fill=renk, outline=BEYAZ,
                  width=max(2, int(2 * olcek)))

    # --- koronal denge: C7'nin sakulden sapmasi ---
    sapma_px = sapma_der = None
    if c7 is not None and sakrum is not None:
        sapma_px, sapma_der, govde = koronal_denge(c7, sakrum, roll_derece)
        # C7'den sakule dik inen kisa segment: sapmayi gorunur kilar
        ayak = (c7[0] - yat[0] * sapma_px, c7[1] - yat[1] * sapma_px)
        d.line([tuple(c7), ayak], fill=MOR, width=max(3, int(4 * olcek)))
        # C7 -> sakrum govde ekseni (ince, referans)
        d.line([tuple(c7), tuple(sakrum)], fill=(*MOR, 140),
               width=max(2, int(2 * olcek)))
        for nokta in (c7, sakrum):
            x, y = nokta
            d.ellipse([x - nr, y - nr, x + nr, y + nr], fill=MOR, outline=BEYAZ,
                      width=max(2, int(2 * olcek)))

    # --- okuma paneli ---
    f_buyuk = _yazi(int(34 * olcek))
    f_kucuk = _yazi(int(17 * olcek))
    pad = int(14 * olcek)
    alt_satirlar = []
    if not gecerli:
        baslik = "ÖLÇÜM GEÇERSİZ"
        alt_satirlar.append(
            (f"Telefon {pitch_derece:+.1f}° öne/arkaya yatık — dikleştirin", GRI))
        panel_renk = (150, 30, 30, 210)
    else:
        yon = "sağ omuz yüksek" if aci > 0 else ("sol omuz yüksek" if aci < 0
                                                 else "simetrik")
        baslik = f"{abs(aci):.1f}°  {yon}"
        alt_satirlar.append((f"aksiller eğim · IMU roll {roll_derece:+.1f}° "
                             f"düzeltildi", (205, 205, 205)))
        alt_satirlar.append((dagilim_aciklama(aci),
                             (255, 160, 90) if belirgin_asimetri(aci)
                             else (170, 170, 170)))
        if sapma_der is not None:
            taraf = ("sağa" if sapma_px > 0 else "sola") if abs(sapma_px) >= 1 else "—"
            alt_satirlar.append(
                (f"koronal denge: {abs(sapma_der):.1f}° {taraf} "
                 f"({abs(sapma_px):.0f} px)", MOR))
            # Cobb tahmini ekle
            try:
                from importlib import import_module
                cobb_mod = import_module("48_cobb_kalibrasyon")
                c_res = cobb_mod.cobb_tahmin_et(aci, abs(sapma_der), 0.0, max(govde, 1.0))
                alt_satirlar.append((f"Cobb tahmini: ~{c_res.tahmin_derece:.1f}° (±{c_res.hata_payi_derece:.1f}°) — {c_res.kategori}", (255, 215, 0)))
            except Exception:
                pass
        else:
            try:
                from importlib import import_module
                cobb_mod = import_module("48_cobb_kalibrasyon")
                c_res = cobb_mod.cobb_tahmin_et(aci)
                alt_satirlar.append((f"Cobb tahmini: ~{c_res.tahmin_derece:.1f}° (±{c_res.hata_payi_derece:.1f}°) — {c_res.kategori}", (255, 215, 0)))
            except Exception:
                pass
        panel_renk = (0, 0, 0, 190)

    # panel yuksekligini gercek yazi metriginden hesapla, tahminle degil
    h1 = f_buyuk.getbbox("Ayg")[3] - f_buyuk.getbbox("Ayg")[1]
    h2 = f_kucuk.getbbox("Ayg")[3] - f_kucuk.getbbox("Ayg")[1]
    ic = int(6 * olcek)
    pw = int(max([d.textlength(baslik, font=f_buyuk)] +
                 [d.textlength(s, font=f_kucuk) for s, _ in alt_satirlar])) + pad * 2
    ph = h1 + len(alt_satirlar) * (h2 + ic) + ic * 2
    d.rounded_rectangle([pad, pad, pad + pw, pad + ph], radius=int(10 * olcek),
                        fill=panel_renk)
    d.text((pad + pad, pad + ic), baslik, font=f_buyuk, fill=BEYAZ)
    y = pad + ic * 2 + h1
    for metin, renk in alt_satirlar:
        d.text((pad + pad, y), metin, font=f_kucuk, fill=renk)
        y += h2 + ic

    # --- tibbi uyari ---
    f_uyari = _yazi(int(15 * olcek))
    uyari = "Tanı aracı değildir · tarama amaçlıdır"
    uw = d.textlength(uyari, font=f_uyari)
    d.rectangle([0, H - int(26 * olcek), W, H], fill=(0, 0, 0, 160))
    d.text(((W - uw) / 2, H - int(22 * olcek)), uyari, font=f_uyari, fill=(210, 210, 210))
    return im


# ------------------------------------------------------------------- ornek uretim

def _aag_noktalari():
    """aag export'undan {dosya_kok: (c7, sakrum)} tablosu. Yoksa bos."""
    import json
    from collections import defaultdict
    KOK = Path(__file__).resolve().parent.parent
    yol = KOK / "data" / "aag" / "train" / "_annotations.coco.json"
    if not yol.exists():
        return {}
    d = json.loads(yol.read_text(encoding="utf-8"))
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

    tablo = {}
    for img_id, anots in per.items():
        if sorted(a["category_id"] for a in anots) != [1, 2]:
            continue
        m = {a["category_id"]: (a["bbox"][0] + a["bbox"][2] / 2,
                                a["bbox"][1] + a["bbox"][3] / 2) for a in anots}
        tablo[kok(isim[img_id])] = (m[1], m[2])
    return tablo


def main():
    import json
    KOK = Path(__file__).resolve().parent.parent
    veri = KOK / "data" / "final"
    cikti = KOK / "runs" / "kilavuz"
    cikti.mkdir(parents=True, exist_ok=True)

    aag = _aag_noktalari()
    kayitlar = json.loads((veri / "kayitlar.json").read_text(encoding="utf-8"))
    # C7/sakrum'u olan ornekleri tercih et ki iki kilavuz da gorunsun
    ornekler = [r for r in kayitlar if Path(r["dosya"]).stem in aag][:6]
    if len(ornekler) < 6:
        ornekler += [r for r in kayitlar if r not in ornekler][:6 - len(ornekler)]

    # ucuncu ornekte telefonun 6 derece egik tutuldugu senaryoyu gosterelim
    senaryolar = [(0.0, 0.0), (0.0, 0.0), (6.0, 0.0), (0.0, 0.0), (-4.0, 0.0), (0.0, 12.0)]

    kareler = []
    for r, (roll, pitch) in zip(ornekler, senaryolar):
        im = Image.open(veri / r["split"] / r["dosya"])
        c7, sakrum = aag.get(Path(r["dosya"]).stem, (None, None))
        kareler.append(kilavuz_ciz(im, r["sag"], r["sol"], roll, pitch, c7, sakrum))

    K = 420
    tuval = Image.new("RGB", (3 * K, 2 * K), (16, 16, 16))
    for i, k in enumerate(kareler):
        tuval.paste(k.resize((K, K), Image.BILINEAR), ((i % 3) * K, (i // 3) * K))
    yol = cikti / "kilavuz_ornek.jpg"
    tuval.save(yol, quality=93)
    print(f"yazildi: {yol}")


if __name__ == "__main__":
    main()
