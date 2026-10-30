"""Asama 1g - Kopya kumelerini ORTALAYARAK etiket gurultusunu dusur.

Kacirilan firsat: bir kopya kumesindeki her uye, ayni fotografin Roboflow'da
AYRI goruntu olarak durup AYRI etiketlenmis hali. Yani elimizde 2-10 tane
BAGIMSIZ etiket var. Onceki boru hatti her kumeden birini alip gerisini
atiyordu.

Iki kazanc:
  1) Olcum : uyeler arasi sacilim = etiket gurultusu (baska veri gerekmez)
  2) Iyilestirme : ortalama alarak gurultu ~sqrt(n) kat duser

Geometri notu: goruntuyu dondurup germek, sonra geri dondurmek aci icin
TAM BIR NO-OP'tur (vektor (dx*S/W, dy*S/H) her iki yolda da ayni cikar).
Dolayisiyla uyeler arasi fark stretch artefakti degil, saf etiket gurultusu.
"""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

KOK = Path(__file__).resolve().parent.parent
KAYNAK = KOK / "data" / "coco" / "train"
CLEAN2 = KOK / "data" / "clean2"
CIKTI = CLEAN2 / "ortalanmis.json"

SAG, SOL = 1, 2
IMZA, ESIK = 64, 0.98


def ileri_map(x, y, k, s):
    for _ in range(k % 4):
        x, y = y, s - 1 - x
    return x, y


def yuzdelik(v, p):
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    a, u = math.floor(k), math.ceil(k)
    return s[int(k)] if a == u else s[a] * (u - k) + s[u] * (k - a)


def main():
    d = json.loads(COCO_JSON.read_text(encoding="utf-8"))
    isim = {g["id"]: g["file_name"] for g in d["images"]}
    per = defaultdict(list)
    for a in d["annotations"]:
        per[a["image_id"]].append(a)
    iyi = {}
    for img_id, anots in per.items():
        if sorted(a["category_id"] for a in anots) != [SAG, SOL]:
            continue
        iyi[isim[img_id]] = {
            a["category_id"]: [a["bbox"][0] + a["bbox"][2] / 2,
                               a["bbox"][1] + a["bbox"][3] / 2]
            for a in anots}

    adlar = sorted(iyi)
    ham = np.zeros((len(adlar), IMZA, IMZA), dtype=np.float32)
    for i, ad in enumerate(adlar):
        ham[i] = np.asarray(Image.open(KAYNAK / ad).convert("L")
                            .resize((IMZA, IMZA), Image.BILINEAR), dtype=np.float32)

    def nrm(a):
        v = a.reshape(a.shape[0], -1)
        v = v - v.mean(axis=1, keepdims=True)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n < 1e-6] = 1.0
        return v / n

    taban = nrm(ham)
    ebeveyn = list(range(len(adlar)))

    def bul(x):
        while ebeveyn[x] != x:
            ebeveyn[x] = ebeveyn[ebeveyn[x]]
            x = ebeveyn[x]
        return x

    # i -> j iliskisi ve rotasyon farki
    rot_iliski = defaultdict(dict)
    for k in range(4):
        kor = taban @ nrm(np.rot90(ham, k, axes=(1, 2)).copy()).T
        for i, j in zip(*np.where(kor > ESIK)):
            i, j = int(i), int(j)
            if i == j:
                continue
            rot_iliski[i][j] = k
            ra, rb = bul(i), bul(j)
            if ra != rb:
                ebeveyn[max(ra, rb)] = min(ra, rb)

    kume = defaultdict(list)
    for i in range(len(adlar)):
        kume[bul(i)].append(i)

    # clean2'deki temsilcinin kanonik rotasyonu
    c2 = json.loads((CLEAN2 / "kayitlar.json").read_text(encoding="utf-8"))
    kanonik = {r["kaynak"]: r["rotasyon"] // 90 for r in c2}

    sacilim_px, sacilim_deg, kazanc = [], [], []
    ortalanmis = []

    for kok, uyeler in kume.items():
        temsilci = None
        for i in uyeler:
            if adlar[i] in kanonik:
                temsilci = i
                break
        if temsilci is None:
            continue
        k0 = kanonik[adlar[temsilci]]

        noktalar = []
        for i in uyeler:
            # i'nin temsilciye gore rotasyonu
            if i == temsilci:
                dk = 0
            elif i in rot_iliski.get(temsilci, {}):
                dk = rot_iliski[temsilci][i]
            else:
                continue
            s = 640
            for c in (SAG, SOL):
                pass
            p1 = ileri_map(*iyi[adlar[i]][SAG], (dk + k0) % 4, s)
            p2 = ileri_map(*iyi[adlar[i]][SOL], (dk + k0) % 4, s)
            # konum kurali
            p_sag, p_sol = (p1, p2) if p1[0] > p2[0] else (p2, p1)
            noktalar.append((p_sag, p_sol))

        if len(noktalar) < 2:
            if noktalar:
                (ps, pl) = noktalar[0]
                ortalanmis.append({"kaynak": adlar[temsilci], "n": 1,
                                   "sacilim_px": None, "sacilim_deg": None,
                                   "sag": list(ps), "sol": list(pl)})
            continue

        sag_arr = np.array([p[0] for p in noktalar], dtype=np.float64)
        sol_arr = np.array([p[1] for p in noktalar], dtype=np.float64)

        # sacilim: her noktanin ortalamadan sapmasinin RMS'i
        for arr in (sag_arr, sol_arr):
            merkez = arr.mean(axis=0)
            sacilim_px.append(float(np.sqrt(((arr - merkez) ** 2).sum(axis=1).mean())))

        acilar = []
        for ps, pl in noktalar:
            dx, dy = pl[0] - ps[0], pl[1] - ps[1]
            acilar.append(math.degrees(math.atan2(dy, abs(dx))))
        if len(acilar) >= 2:
            sacilim_deg.append(float(np.std(acilar, ddof=1)))
            kazanc.append((float(np.std(acilar, ddof=1)),
                           float(np.std(acilar, ddof=1)) / math.sqrt(len(acilar))))

        # sacilim kayda giriyor: yuksekse kumede bozuk etiket var, elenmeli
        sac_px = max(
            float(np.sqrt(((sag_arr - sag_arr.mean(axis=0)) ** 2).sum(axis=1).mean())),
            float(np.sqrt(((sol_arr - sol_arr.mean(axis=0)) ** 2).sum(axis=1).mean())))
        ortalanmis.append({
            "kaynak": adlar[temsilci], "n": len(noktalar),
            "sacilim_px": round(sac_px, 2),
            "sacilim_deg": round(float(np.std(acilar, ddof=1)), 3),
            "sag": [round(float(sag_arr[:, 0].mean()), 2),
                    round(float(sag_arr[:, 1].mean()), 2)],
            "sol": [round(float(sol_arr[:, 0].mean()), 2),
                    round(float(sol_arr[:, 1].mean()), 2)],
        })

    print("=" * 66)
    print("ETIKET GURULTUSU  (kopya kumeleri icindeki bagimsiz etiketlerden)")
    print("=" * 66)
    print(f"  Olculebilen kume : {len(sacilim_deg)}")
    print()
    print("  Nokta konumu sacilimi (RMS, piksel):")
    print(f"    medyan {statistics.median(sacilim_px):5.2f} px   "
          f"p90 {yuzdelik(sacilim_px,90):5.2f} px   max {max(sacilim_px):6.2f} px")
    print()
    print("  ACI sacilimi (std, derece)  <-- ASIL SAYI:")
    print(f"    medyan {statistics.median(sacilim_deg):5.3f}°   "
          f"p90 {yuzdelik(sacilim_deg,90):5.3f}°   max {max(sacilim_deg):6.3f}°")

    tek = statistics.median(sacilim_deg)
    ort = statistics.median([g[1] for g in kazanc])
    print()
    print("=" * 66)
    print("ORTALAMANIN KAZANCI")
    print("=" * 66)
    print(f"  Tek etiket kullaninca  : {tek:.3f}° gurultu")
    print(f"  Ortalama kullaninca    : {ort:.3f}° gurultu")
    print(f"  Iyilesme               : {tek/ort:.2f}x")
    print()
    print(f"  Karsilastirma: veri setinin medyan |aci|'si 2.19°")
    print(f"    tek etiket  -> sinyal/gurultu = {2.19/tek:.1f}")
    print(f"    ortalama    -> sinyal/gurultu = {2.19/ort:.1f}")

    CIKTI.write_text(json.dumps(ortalanmis, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    coklu = sum(1 for r in ortalanmis if r["n"] > 1)
    print()
    print(f"  Yazildi: {CIKTI}")
    print(f"    {len(ortalanmis)} kayit, {coklu} tanesi ortalanmis (n>1)")


if __name__ == "__main__":
    main()
