"""Asama 1f - Nihai temiz set: filtre + sizintisiz bolme.

Girdi : data/clean2  (384 kanonik goruntu, pose ile dik, konum kuraliyla R/L)
Cikti : data/final   (filtrelenmis, gruplanmis, bolunmus)

Filtreler (ikisi de ayni copu isaret ediyor, birbirini dogruluyor):
  pose_guven >= 0.50   -> yatan/egilmis poz, stok foto, illustrasyon eler
  |aci|      <= 25     -> ayakta durmayan vakalari eler

Bolme guvenligi:
  Kopya kumelemesi 0.98 esigiyle yapilmisti; bu sadece neredeyse AYNI
  kareyi yakalar. Ayni kisinin ayri cekilmis fotografi ayri kume sayilir
  ve train/val'e dagilabilir -> sizinti. Bolmeden once daha GEVSEK esikle
  (0.90) yeniden gruplayip, grubu bolunmeden tek split'e veriyoruz.
"""

import json
import math
import random
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

KOK = Path(__file__).resolve().parent.parent
KAYNAK = KOK / "data" / "clean2"
HEDEF = KOK / "data" / "final"

POSE_ESIK = 0.50
ACI_ESIK = 25.0
# Bolme guvenligi icin gevsek benzerlik esigi.
# Benzerlik dagiliminda dogal bir bosluk YOK (p99=0.66, p99.9=0.93), yani bu
# bir yargi karari. 0.85-0.90 bandinda 42 cift var; 0.90 secilirse bunlar
# train/val'e dagilir. 337 goruntuluk sette bu ~%12 sizinti riski demek.
# Ucuz sigorta olarak muhafazakar tarafi seciyoruz.
GRUP_ESIK = 0.85
SACILIM_ESIK = 15.0    # kume ici etiket sacilimi (px); ustu = bozuk etiket var
IMZA = 64
TOHUM = 1337
ORAN = (0.70, 0.15, 0.15)


def grupla(yollar, esik):
    ham = np.zeros((len(yollar), IMZA, IMZA), dtype=np.float32)
    for i, p in enumerate(yollar):
        ham[i] = np.asarray(Image.open(p).convert("L")
                            .resize((IMZA, IMZA), Image.BILINEAR), dtype=np.float32)
    v = ham.reshape(len(yollar), -1)
    v = v - v.mean(axis=1, keepdims=True)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-6] = 1.0
    v = v / n

    ebeveyn = list(range(len(yollar)))

    def bul(x):
        while ebeveyn[x] != x:
            ebeveyn[x] = ebeveyn[ebeveyn[x]]
            x = ebeveyn[x]
        return x

    kor = v @ v.T
    for i, j in zip(*np.where(kor > esik)):
        i, j = int(i), int(j)
        if i < j:
            ra, rb = bul(i), bul(j)
            if ra != rb:
                ebeveyn[max(ra, rb)] = min(ra, rb)

    gruplar = defaultdict(list)
    for i in range(len(yollar)):
        gruplar[bul(i)].append(i)
    return list(gruplar.values())


def main():
    random.seed(TOHUM)
    kayitlar = json.loads((KAYNAK / "kayitlar.json").read_text(encoding="utf-8"))
    print(f"[1] Girdi : {len(kayitlar)}")

    # --- ORTALANMIS etiketleri uygula (kopya kumelerindeki bagimsiz etiketler) ---
    ort = {r["kaynak"]: r
           for r in json.loads((KAYNAK / "ortalanmis.json").read_text(encoding="utf-8"))}
    uygulanan, elenen_sacilim = 0, 0
    temiz = []
    for r in kayitlar:
        o = ort.get(r["kaynak"])
        if o is None:
            temiz.append(r)
            continue
        if o["sacilim_px"] is not None and o["sacilim_px"] > SACILIM_ESIK:
            elenen_sacilim += 1          # kumede bozuk etiket var, guvenilmez
            continue
        if o["n"] > 1:
            r = dict(r)
            r["sag"], r["sol"] = o["sag"], o["sol"]
            r["etiket_n"] = o["n"]
            dx, dy = r["sol"][0] - r["sag"][0], r["sol"][1] - r["sag"][1]
            r["aci"] = round(math.degrees(math.atan2(dy, abs(dx))), 3)
            r["D"] = round(math.hypot(dx, dy), 2)
            uygulanan += 1
        temiz.append(r)
    print(f"[1b] Ortalanmis etiket uygulanan : {uygulanan}")
    print(f"     Sacilim >{SACILIM_ESIK} px oldugu icin elenen : {elenen_sacilim}")

    tutulan = [r for r in temiz
               if r["pose_guven"] >= POSE_ESIK and abs(r["aci"]) <= ACI_ESIK]
    print(f"[2] Filtre sonrasi : {len(tutulan)}   (elenen {len(temiz)-len(tutulan)})")

    yollar = [KAYNAK / "images" / r["dosya"] for r in tutulan]
    gruplar = grupla(yollar, GRUP_ESIK)
    print(f"[3] Bolme grubu (esik {GRUP_ESIK}) : {len(gruplar)}")
    coklu = [g for g in gruplar if len(g) > 1]
    print(f"    Birden fazla goruntulu grup : {len(coklu)}"
          f"  (en buyuk {max((len(g) for g in gruplar), default=0)})")
    print(f"    -> bu goruntuler ayni kisi olabilir, ayni split'e gidecek")

    # gruplari karistir, hedef oranlara gore dagit
    random.shuffle(gruplar)
    n = len(tutulan)
    hedef = {"train": ORAN[0] * n, "val": ORAN[1] * n, "test": ORAN[2] * n}
    dolu = {"train": 0, "val": 0, "test": 0}
    atama = {}
    for g in sorted(gruplar, key=len, reverse=True):
        # en cok acigi olan split'e ver
        s = max(dolu, key=lambda k: hedef[k] - dolu[k])
        for i in g:
            atama[i] = s
        dolu[s] += len(g)

    if HEDEF.exists():
        shutil.rmtree(HEDEF)
    for s in ("train", "val", "test"):
        (HEDEF / s).mkdir(parents=True)

    cikti = []
    for i, r in enumerate(tutulan):
        s = atama[i]
        yeni_ad = f"{s}_{i:04d}.jpg"
        shutil.copy2(yollar[i], HEDEF / s / yeni_ad)
        kayit = dict(r)
        kayit["dosya"] = yeni_ad
        kayit["split"] = s
        cikti.append(kayit)

    (HEDEF / "kayitlar.json").write_text(
        json.dumps(cikti, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[4] Bolme : " + "  ".join(f"{k}={v}" for k, v in sorted(dolu.items())))

    print()
    print("=" * 62)
    print("NIHAI SET OZETI")
    print("=" * 62)
    acilar = [r["aci"] for r in cikti]
    mutlak = sorted(abs(a) for a in acilar)
    Dler = sorted(r["D"] for r in cikti)
    renkli = sum(1 for r in cikti if r["renkli"])

    print(f"  Goruntu           : {len(cikti)}")
    print(f"  Renkli            : {renkli}  ({100.0*renkli/len(cikti):.0f}%)")
    print(f"  Aci  medyan|aci|  : {statistics.median(mutlak):.2f}°"
          f"   p90 {mutlak[int(len(mutlak)*0.9)]:.2f}°   max {mutlak[-1]:.2f}°")
    print(f"  D    medyan       : {statistics.median(Dler):.0f} px"
          f"   p10 {Dler[len(Dler)//10]:.0f} px")
    print(f"  -> medyan D'de 1 px hata = "
          f"{math.degrees(math.atan(1.0/statistics.median(Dler))):.3f}°")

    print()
    print("  |aci| dagilimi:")
    for alt, ust in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 25)]:
        adet = sum(1 for x in mutlak if alt <= x < ust)
        oran = 100.0 * adet / len(mutlak)
        print(f"    {alt:2d}-{ust:2d}° : {adet:4d}  ({oran:5.1f}%) {'#'*int(oran/2)}")

    print(f"\nCikti: {HEDEF}")


if __name__ == "__main__":
    main()
