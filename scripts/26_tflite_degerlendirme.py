"""TFLite boru hattini test setinde degerlendirir — quantization kaybini olcer.

DIKKAT: .venv-tflite icinde calisir:

    .venv-tflite/Scripts/python.exe scripts/26_tflite_degerlendirme.py

NE ISE YARAR
------------
20_tam_degerlendirme.py PyTorch boru hattini olcuyordu. Bu script AYNI test
setinde, AYNI metriklerle TFLite'i olcuyor. Aradaki fark quantization kaybidir.

MOBIL_YAPILACAKLAR.md Faz 1 kabul kriteri:
    aci hatasi medyani <= 0.50 derece   (float PyTorch surumunde 0.380)
Asilirsa float16'ya duseriz.

NEDEN ULTRALYTICS YOK
---------------------
Bu ortamda torch da ultralytics de yok (bilerek — tensorflow ile catisiyorlar).
Yani Asama 1'in ham ciktisini KENDIMIZ cozmek zorundayiz. Bu bir yuk degil,
gereklilik: mobil tarafta Kotlin de tam olarak bunu yapacak. Buradaki cozme
mantigi Kotlin portunun referansi.
"""

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
kilavuz = import_module("19_kilavuz_cizim")

try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ImportError:
    from tensorflow.lite import Interpreter
    OpResolverType = None

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "final"
EXPORT = KOK / "runs" / "export"

MODEL_BOYUT = 640
KIRPIM = 128
GUVEN_ESIGI = 0.25
ORT = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIMETRI_BANDI = 1.0
TARAMA_ESIKLERI = (3.0, 5.0, 7.0)

# 20_tam_degerlendirme.py'nin PyTorch sonuclari — karsilastirma tabani
TABAN = {"px_medyan": 3.55, "aci_medyan": 0.380, "aci_p90": 1.706}


class TFLite:
    """Girdi/cikti quantization'ini kendisi halleden ince sarmalayici.

    INT8 modellerde girdi ve cikti tam sayidir; olcek ve sifir noktasi
    modelin icinde yazilidir. Bunlari elle uygulamak yerine tensor
    tanimlarindan okuyoruz — aksi halde ayni kodu her varyant icin
    degistirmek gerekirdi ve karsilastirma kirlenirdi.
    """

    def __init__(self, yol):
        self.yol = Path(yol)
        self.xnnpack = True
        try:
            self.ip = Interpreter(model_path=str(yol))
            self.ip.allocate_tensors()
        except RuntimeError as e:
            # INT8 modellerde XNNPACK bazi dugumleri hazirlayamiyor
            # ("failed to create XNNPACK runtime"). Delegesiz referans
            # cekirdekle acmak dogrulugu degistirmez, yalnizca yavaslatir —
            # burada olctugumuz sey hiz degil dogruluk.
            if OpResolverType is None or "XNNPACK" not in str(e):
                raise
            self.xnnpack = False
            self.ip = Interpreter(
                model_path=str(yol),
                experimental_op_resolver_type=(
                    OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES),
            )
            self.ip.allocate_tensors()
        self.girdi = self.ip.get_input_details()[0]
        self.ciktilar = self.ip.get_output_details()

    @property
    def boyut_mb(self):
        return self.yol.stat().st_size / 1e6

    def __call__(self, x):
        g = self.girdi
        olcek, sifir = g["quantization"]
        if olcek != 0:
            x = np.clip(np.round(x / olcek + sifir),
                        np.iinfo(g["dtype"]).min,
                        np.iinfo(g["dtype"]).max)
        self.ip.set_tensor(g["index"], x.astype(g["dtype"]))
        self.ip.invoke()
        sonuc = []
        for c in self.ciktilar:
            v = self.ip.get_tensor(c["index"])
            olcek, sifir = c["quantization"]
            if olcek != 0:
                v = (v.astype(np.float32) - sifir) * olcek
            sonuc.append(v.astype(np.float32))
        return sonuc


def asama1_coz(cikti):
    """YOLO-pose ham ciktisindan iki noktayi cikarir.

    Kanal duzeni: [cx, cy, w, h, conf, kx0, ky0, kv0, kx1, ky1, kv1]
    Koordinatlar 640x640 piksel uzayinda (ultralytics head'de zaten cozuluyor).

    NMS YOK: karede tek kisi bekliyoruz, en yuksek guvenli aday yeterli.
    Ayni sadelestirme mobil tarafta da gecerli olacak.
    """
    a = cikti[0]
    if a.ndim == 3:
        a = a[0]
    # (11, 8400) mi (8400, 11) mi — kanal eksenini boyutundan buluyoruz,
    # varsaymiyoruz. onnx2tf eksen sirasini degistirebiliyor.
    if a.shape[0] != 11 and a.shape[1] == 11:
        a = a.T
    if a.shape[0] != 11:
        raise ValueError(f"beklenmeyen asama1 cikti sekli: {cikti[0].shape}")

    guven = a[4]
    i = int(np.argmax(guven))
    if float(guven[i]) < GUVEN_ESIGI:
        return None
    return {
        "sag": (float(a[5][i]), float(a[6][i])),   # index 0 = SAG
        "sol": (float(a[8][i]), float(a[9][i])),   # index 1 = SOL
    }


def yumusak_argmax(heatmap, sicaklik=10.0):
    """Heatmap -> alt-piksel koordinat. 17_rafinaj_model.py'nin birebir esi.

    Heatmap varyantini degerlendirmek icin gerekli: o modelde soft-argmax
    grafigin disinda kaliyor.
    """
    h = np.squeeze(heatmap)
    if h.ndim != 2:
        raise ValueError(f"beklenmeyen heatmap sekli: {heatmap.shape}")
    n = h.shape[0]
    duz = h.reshape(-1).astype(np.float64) * sicaklik
    duz -= duz.max()                       # sayisal tasma korumasi
    p = np.exp(duz)
    p /= p.sum()
    p = p.reshape(n, n)
    izgara = np.linspace(0, 1, n)
    x = float((p * izgara[None, :]).sum())
    y = float((p * izgara[:, None]).sum())
    return x * (KIRPIM - 1), y * (KIRPIM - 1)


def rafine_et(model, im, kaba, heatmap_mi):
    """Kaba nokta etrafindan kirpip alt-piksel nokta dondurur.

    20_tam_degerlendirme.py::rafine_et ile birebir ayni geometri — yuvarlama
    dahil. Ufak bir kayma bile karsilastirmayi bozardi.
    """
    sol = kaba[0] - KIRPIM / 2
    ust = kaba[1] - KIRPIM / 2
    kirpim = im.crop((int(round(sol)), int(round(ust)),
                      int(round(sol)) + KIRPIM, int(round(ust)) + KIRPIM))
    a = np.asarray(kirpim.convert("RGB"), dtype=np.float32) / 255.0
    a = (a - ORT) / STD
    cikti = model(a[None, ...])            # NHWC
    if heatmap_mi:
        kx, ky = yumusak_argmax(cikti[0])
    else:
        k = np.squeeze(cikti[0])
        kx, ky = float(k[0]), float(k[1])
    return (round(sol) + kx, round(ust) + ky)


def yuzdelik(v, p):
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    a, u = math.floor(k), math.ceil(k)
    return s[int(k)] if a == u else s[a] * (u - k) + s[u] * (k - a)


def sinif_metrikleri(gercek, tahmin, etiketler):
    n = len(gercek)
    acc = sum(1 for g, t in zip(gercek, tahmin) if g == t) / n if n else 0.0
    f1ler = []
    for e in etiketler:
        tp = sum(1 for g, t in zip(gercek, tahmin) if g == e and t == e)
        fp = sum(1 for g, t in zip(gercek, tahmin) if g != e and t == e)
        fn = sum(1 for g, t in zip(gercek, tahmin) if g == e and t != e)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1ler.append(2 * p * r / (p + r) if p + r else 0.0)
    return acc, sum(f1ler) / len(etiketler)


def degerlendir(m1, m2, heatmap_mi, kayitlar):
    px, aci_hatasi, g_aci, t_aci = [], [], [], []
    atlanan = 0
    for r in kayitlar:
        yol = VERI / r["split"] / r["dosya"]
        im = Image.open(yol).convert("RGB")
        g = {"sag": tuple(r["sag"]), "sol": tuple(r["sol"])}

        # STRETCH — letterbox degil. Egitimde de stretch kullanilmisti.
        x = np.asarray(im.resize((MODEL_BOYUT, MODEL_BOYUT), Image.BILINEAR),
                       dtype=np.float32) / 255.0
        kaba = asama1_coz(m1(x[None, ...]))
        if kaba is None:
            atlanan += 1
            continue

        # Test goruntuleri zaten 640x640; olcek 1. Gercek telefonda burada
        # orijinal cozunurluge geri mapleme yapilacak.
        ince = {k: rafine_et(m2, im, v, heatmap_mi) for k, v in kaba.items()}
        for k in ("sag", "sol"):
            px.append(math.dist(ince[k], g[k]))

        ga = kilavuz.aksiller_aci(g["sag"], g["sol"])
        ta = kilavuz.aksiller_aci(ince["sag"], ince["sol"])
        aci_hatasi.append(abs(ta - ga))
        g_aci.append(ga)
        t_aci.append(ta)

    # Tespit hic olmayabilir (INT8'de goruldu: guven kanali quantize edilince
    # tamamen eziliyor). Cokmek yerine bunu bir SONUC olarak raporluyoruz —
    # "olculemedi" ile "kotu olctu" farkli seylerdir.
    if not px:
        return {"n": 0, "atlanan": atlanan, "tespit_yok": True}

    def sinifla(a):
        if abs(a) < SIMETRI_BANDI:
            return "simetrik"
        return "sag" if a > 0 else "sol"

    omuz_acc, omuz_f1 = sinif_metrikleri(
        [sinifla(a) for a in g_aci], [sinifla(a) for a in t_aci],
        ["sag", "sol", "simetrik"])
    tarama = {}
    for e in TARAMA_ESIKLERI:
        tarama[e] = sinif_metrikleri(
            ["poz" if abs(a) >= e else "neg" for a in g_aci],
            ["poz" if abs(a) >= e else "neg" for a in t_aci],
            ["poz", "neg"])

    return {
        "n": len(g_aci), "atlanan": atlanan, "tespit_yok": False,
        "px_medyan": statistics.median(px), "px_ort": float(np.mean(px)),
        "px_p90": yuzdelik(px, 90),
        "pck5": 100.0 * sum(1 for v in px if v <= 5) / len(px),
        "pck10": 100.0 * sum(1 for v in px if v <= 10) / len(px),
        "aci_medyan": statistics.median(aci_hatasi),
        "aci_ort": float(np.mean(aci_hatasi)),
        "aci_p90": yuzdelik(aci_hatasi, 90),
        "omuz_acc": omuz_acc, "omuz_f1": omuz_f1, "tarama": tarama,
    }


def varyantlari_bul(klasor, onek):
    """Uretilen tflite dosyalarini varyant adina gore esler.

    Dosya adlari "<onek>_<varyant>.tflite" bicimindedir, ornegin
    "asama2_koord_float16.tflite". Oneki UZUNLUGUNA gore kirpiyoruz; ilk
    alt cizgiye gore bolmek "koord_float16" gibi anahtarlar uretip iki
    asamanin varyantlarinin eslesmesini engelliyordu.

    Hangi varyantlarin uretildigini varsaymiyoruz — onnx2tf surumune ve
    modelin quantize edilebilirligine gore degisiyor.
    """
    bulunan = {}
    for f in sorted(Path(klasor).glob("*.tflite")):
        if not f.stem.startswith(onek + "_"):
            continue
        bulunan[f.stem[len(onek) + 1:]] = f
    return bulunan


def main():
    kayitlar = [r for r in json.loads((VERI / "kayitlar.json")
                                      .read_text(encoding="utf-8"))
                if r["split"] == "test"]
    print(f"Test goruntusu: {len(kayitlar)}")

    a1 = varyantlari_bul(EXPORT / "tflite_asama1", "asama1")
    a2k = varyantlari_bul(EXPORT / "tflite_asama2_koord", "asama2_koord")
    a2h = varyantlari_bul(EXPORT / "tflite_asama2_heatmap", "asama2_heatmap")
    if not a1 or not a2k:
        sys.exit("tflite dosyalari yok — once 25_tflite_donusum.py calistir")

    print("\nUretilen varyantlar:")
    for ad, d in (("asama1", a1), ("asama2_koord", a2k),
                  ("asama2_heatmap", a2h)):
        print(f"  {ad}: {', '.join(d) or '(yok)'}")

    # Ayni quantization seviyesini iki asamada da kullanan kombinasyonlar.
    # Karisik kombinasyon (ornegin float32 + int8) mobilde anlamsiz olurdu.
    ortak = [v for v in a1 if v in a2k]
    print(f"\nDegerlendirilecek kombinasyon: {len(ortak)}")

    basliklar = ("varyant", "MB", "px med", "aci med", "aci p90",
                 "PCK5", "PCK10", "omuz acc")
    print()
    print("=" * 88)
    print(f"{basliklar[0]:<28}{basliklar[1]:>7}{basliklar[2]:>8}"
          f"{basliklar[3]:>9}{basliklar[4]:>9}{basliklar[5]:>7}"
          f"{basliklar[6]:>7}{basliklar[7]:>10}")
    print("=" * 88)

    sonuclar = {}
    for v in ortak:
        for etiket, a2, hm in (("koord", a2k, False), ("heatmap", a2h, True)):
            if v not in a2:
                continue
            m1, m2 = TFLite(a1[v]), TFLite(a2[v])
            s = degerlendir(m1, m2, hm, kayitlar)
            boyut = m1.boyut_mb + m2.boyut_mb
            ad = f"{v} / {etiket}"
            sonuclar[ad] = (s, boyut)
            if s["tespit_yok"]:
                print(f"{ad:<28}{boyut:>7.2f}    TESPIT YOK "
                      f"({s['atlanan']}/{len(kayitlar)} goruntude aday "
                      f"guven esigini gecemedi)")
                continue
            print(f"{ad:<28}{boyut:>7.2f}{s['px_medyan']:>8.2f}"
                  f"{s['aci_medyan']:>9.3f}{s['aci_p90']:>9.3f}"
                  f"{s['pck5']:>6.0f}%{s['pck10']:>6.0f}%"
                  f"{s['omuz_acc']:>10.3f}")

    print("-" * 88)
    print(f"{'PyTorch (taban)':<28}{'—':>7}{TABAN['px_medyan']:>8.2f}"
          f"{TABAN['aci_medyan']:>9.3f}{TABAN['aci_p90']:>9.3f}")
    print("=" * 88)

    # TESPIT YOK cikan varyantlar icin sebebi göster. Hipotez: YOLO'nun tek
    # cikti tensoru hem kutu koordinatlarini (0-640) hem guveni (0-1)
    # tasiyor. Tek bir INT8 olcegi ikisini birden temsil edemiyor: olcek
    # 640/127 ~ 5 secilince 0-1 araligindaki guven tamamen sifira eziliyor.
    olusuz = [a for a, (s, _) in sonuclar.items() if s["tespit_yok"]]
    if olusuz:
        print("\nTESPIT YOK cikan varyantlarin cikti quantization'i:")
        for v in ortak:
            if not any(a.startswith(v + " ") for a in olusuz):
                continue
            m = TFLite(a1[v])
            for c in m.ciktilar:
                olcek, sifir = c["quantization"]
                print(f"  {v:<36} olcek={olcek:.4f} sifir={sifir}")
                if olcek > 0.02:
                    print(f"    -> 0-1 arasindaki guven degeri bu olcekle "
                          f"en fazla {1.0 / olcek:.1f} adima dusuyor; "
                          f"esik 0.25 asilamiyor")

    print("\nKABUL KRITERI: aci hatasi medyani <= 0.500 derece")
    gecen = [(a, s, b) for a, (s, b) in sonuclar.items()
             if not s["tespit_yok"] and s["aci_medyan"] <= 0.500]
    if not gecen:
        print("  HICBIRI GECMEDI — float16'ya dus veya donusumu incele")
    for a, s, b in sorted(gecen, key=lambda t: t[2]):
        print(f"  GECTI  {a:<24} {s['aci_medyan']:.3f} derece, {b:.2f} MB")

    print("\nBOYUT HEDEFI: toplam < 10 MB")
    for a, (s, b) in sorted(sonuclar.items(), key=lambda t: t[1][1]):
        if s["tespit_yok"]:
            continue
        print(f"  {'TAMAM' if b < 10 else 'ASIYOR':<7}{a:<24} {b:.2f} MB")


if __name__ == "__main__":
    main()
