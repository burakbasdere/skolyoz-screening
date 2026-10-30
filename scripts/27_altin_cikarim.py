"""Cikarim zincirinin Kotlin portu icin ALTIN VERI uretir.

DIKKAT: .venv-tflite icinde calisir (26 numarali modulu import ediyor):

    .venv-tflite/Scripts/python.exe scripts/27_altin_cikarim.py

NEDEN 26'YI IMPORT EDIYORUZ
---------------------------
Beklenen degerleri bu dosyada yeniden yazsaydik iki ayri "referans" olurdu ve
zamanla sessizce ayrisirlardi. Cozme mantigini olcumu YAPAN modulden aliyoruz;
boylece Kotlin, gercekten dogruladigimiz davranisla karsilastiriliyor.

NE URETILIYOR
-------------
Kotlin tarafinda JVM birim testiyle dogrulanabilen, yani Bitmap/TFLite
gerektirmeyen adimlar:

  decode : YOLO ham ciktisindan nokta secimi (guven esigi dahil)
  kose   : kirpim penceresinin sol-ust kosesi (YUVARLAMA DAVRANISI)
  kirp   : sinir disina tasan kirpimda PIL'in siyah dolgusu
  norm1  : Asama 1 normalizasyonu (/255) + kanal sirasi
  norm2  : Asama 2 normalizasyonu (ImageNet) + kanal sirasi
  map    : 640 uzayindan orijinal cozunurluge geri mapleme

Cikti: android/app/src/main/test-resources yerine
       android/app/src/test/resources/cikarim_altin_veri.csv
"""

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
degerlendirme = import_module("26_tflite_degerlendirme")

KOK = Path(__file__).resolve().parent.parent
CIKTI = (KOK / "android" / "app" / "src" / "test" / "resources"
         / "cikarim_altin_veri.csv")

ORT = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
KIRPIM = 128
ADAY = 16          # testte tam 8400 aday tasimaya gerek yok


def decode_vakasi(satirlar, rng, esik_alti):
    """Sentetik (11, ADAY) cikti uretip 26'nin cozme fonksiyonuna verir."""
    a = rng.uniform(-1.0, 1.0, size=(11, ADAY)).astype(np.float32)
    # koordinat kanallarini gercekci araliga tasi (0-640)
    for k in (0, 1, 2, 3, 5, 6, 8, 9):
        a[k] = rng.uniform(0.0, 640.0, size=ADAY).astype(np.float32)
    a[4] = rng.uniform(0.0, 0.2 if esik_alti else 1.0, size=ADAY).astype(np.float32)
    if not esik_alti:
        a[4][7] = 0.93        # net bir kazanan olsun

    sonuc = degerlendirme.asama1_coz([a[None, ...]])
    duz = ",".join(f"{v:.6f}" for v in a.reshape(-1))
    if sonuc is None:
        satirlar.append(f"decode,{ADAY},YOK,0,0,0,0,{duz}")
    else:
        satirlar.append(
            f"decode,{ADAY},VAR,"
            f"{sonuc['sag'][0]:.6f},{sonuc['sag'][1]:.6f},"
            f"{sonuc['sol'][0]:.6f},{sonuc['sol'][1]:.6f},{duz}")


def main():
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    satirlar = ["# tip,...  — 27_altin_cikarim.py uretir, elle duzenleme"]

    # --- decode: esigi gecen ve gecmeyen iki vaka
    decode_vakasi(satirlar, rng, esik_alti=False)
    decode_vakasi(satirlar, rng, esik_alti=True)

    # --- kose: yuvarlama davranisi. Python round() YARIMLARI CIFTE yuvarlar.
    # Kotlin'de Math.round yukari yuvarlar; bu yuzden Math.rint kullaniliyor.
    # Asagidaki .5 vakalari tam olarak o farki yakalar.
    for cx, cy in [(100.0, 200.0), (63.5, 64.5), (64.5, 65.5),
                   (0.0, 0.0), (639.4, 12.6), (-3.2, 700.8),
                   (320.5, 320.5), (10.49, 10.51)]:
        sol = int(round(cx - KIRPIM / 2))
        ust = int(round(cy - KIRPIM / 2))
        satirlar.append(f"kose,{cx:.6f},{cy:.6f},{sol},{ust}")

    # --- kirp: PIL sinir disini SIYAHLA dolduruyor. Asama 2 bu dagilimla
    # egitildi; kenar yansitma veya pencereyi iceri cekme modeli gormedigi
    # bir dagilima sokar. Kisi karenin kenarinda dururken bu gercekten oluyor.
    G, Y = 40, 30
    im = Image.new("RGB", (G, Y))
    px = im.load()
    for y in range(Y):
        for x in range(G):
            px[x, y] = ((x * 7) % 256, (y * 11) % 256, (x + y) % 256)
    for solX, ustY in [(0, 0), (-20, -10), (30, 20), (-140, -140)]:
        k = im.crop((solX, ustY, solX + KIRPIM, ustY + KIRPIM))
        a = np.asarray(k, dtype=np.int64)
        # bir kac ornek nokta yeter; tam 128x128 tasimaya gerek yok
        ornekler = [(0, 0), (5, 3), (64, 64), (127, 127), (20, 100)]
        deger = ";".join(
            f"{sx}:{sy}:{a[sy, sx, 0]}:{a[sy, sx, 1]}:{a[sy, sx, 2]}"
            for sx, sy in ornekler)
        satirlar.append(f"kirp,{G},{Y},{solX},{ustY},{deger}")

    # --- normalizasyon + kanal sirasi. RGB/BGR karisikligi keypoint hatasini
    # 24 px'den 40 px'e cikarmisti (MOBIL_YAPILACAKLAR.md 5.4); bu satirlar
    # kanal sirasi ters cevrilirse aninda kirilir.
    for r, g, b in [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0),
                    (0, 0, 255), (12, 200, 77), (128, 64, 32)]:
        argb = (0xFF << 24) | (r << 16) | (g << 8) | b
        v = np.array([r, g, b], dtype=np.float32) / 255.0
        n2 = (v - ORT) / STD
        satirlar.append(
            f"norm1,{argb},{v[0]:.8f},{v[1]:.8f},{v[2]:.8f}")
        satirlar.append(
            f"norm2,{argb},{n2[0]:.8f},{n2[1]:.8f},{n2[2]:.8f}")

    # --- 640 uzayindan orijinale mapleme. Bu adim atlanirsa 4:3 bir
    # fotografta ~%25 aci hatasi olusuyor (MOBIL_YAPILACAKLAR.md 3).
    for x, y, W, H in [(320.0, 320.0, 640, 640), (100.0, 500.0, 1536, 2048),
                       (0.0, 0.0, 3000, 4000), (639.0, 1.0, 1200, 1600),
                       (200.5, 450.25, 1536, 2048)]:
        satirlar.append(
            f"map,{x:.6f},{y:.6f},{W},{H},"
            f"{x * W / 640.0:.8f},{y * H / 640.0:.8f}")

    # --- heatmap: soft-argmax + entropi (bkz. heatmap_vakalari)
    satirlar += heatmap_vakalari()

    CIKTI.write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    boyut = CIKTI.stat().st_size / 1e3
    print(f"yazildi: {CIKTI}  ({boyut:.0f} KB)")
    sayim = {}
    for s in satirlar[1:]:
        sayim[s.split(",", 1)[0]] = sayim.get(s.split(",", 1)[0], 0) + 1
    for t, n in sayim.items():
        print(f"  {t:<8} {n} vaka")


# ---------------------------------------------------------------- ek: heatmap
# Bu script ilk yazildiginda Asama 2 koordinat varyantiydi ve soft-argmax
# modelin ICINDEYDI. Entropi kapisi eklenince heatmap varyantina gecildi ve
# soft-argmax Kotlin'e tasindi — dolayisiyla artik o da Python'a karsi
# dogrulanmali. Entropi de oyle: kapinin karari ona dayaniyor.
def heatmap_vakalari():
    """Sentetik heatmap'lerden soft-argmax + entropi altin verisi."""
    rng = np.random.default_rng(21)
    satirlar = []
    N = 64

    def gauss(cx, cy, sigma, tepe=6.0):
        ax = np.arange(N, dtype=np.float64)
        gx = np.exp(-((ax - cx) ** 2) / (2 * sigma ** 2))
        gy = np.exp(-((ax - cy) ** 2) / (2 * sigma ** 2))
        return (tepe * gy[:, None] * gx[None, :]).astype(np.float32)

    vakalar = [
        ("keskin tek tepe", gauss(31.5, 31.5, 1.2)),
        ("kenarda tepe", gauss(3.0, 60.0, 1.5)),
        ("genis tepe", gauss(20.0, 40.0, 6.0)),
        # IKI TEPE: entropi kapisinin var olma sebebi. soft-argmax ikisinin
        # ORTASINI verir ve o nokta hicbir tepenin ustunde degildir.
        ("iki tepe", gauss(12.0, 32.0, 2.0) + gauss(52.0, 32.0, 2.0)),
        ("neredeyse duz", rng.normal(0, 0.02, (N, N)).astype(np.float32)),
    ]
    for _ad, h in vakalar:
        duz = h.reshape(-1).astype(np.float64) * 10.0
        duz = duz - duz.max()
        p = np.exp(duz)
        p /= p.sum()
        p2 = p.reshape(N, N)
        izgara = np.linspace(0, 1, N)
        x = float((p2 * izgara[None, :]).sum()) * 127.0
        y = float((p2 * izgara[:, None]).sum()) * 127.0
        ent = float(-(p * np.log(p + 1e-12)).sum())
        deger = ";".join(f"{v:.6f}" for v in h.reshape(-1))
        satirlar.append(f"heatmap,{N},{x:.8f},{y:.8f},{ent:.8f},{deger}")
    return satirlar


if __name__ == "__main__":
    main()
