"""Asama 0b - Gurultu tabani ve etiket tutarliligi.

Kullaniciya ek is yaptirmadan, mevcut veriden cikarilabilecek iki kritik sayi:

  1) TEST-RETEST: 93 hastanin tam 4 fotografi var. Ayni hastanin iki ayri
     fotografindaki aci farki -> bu projenin gurultu tabani. Model bunun
     altina inemez, dolayisiyla hedefimizi bu belirler.

  2) ETIKET OZENI: ayni fotograftaki sol ve sag kutunun boyut farki.
     Ayni kisinin iki koltuk alti benzer buyuklukte olmali; buyuk fark
     = ozensiz kutu = kaymis merkez = aci hatasi.

Not: 90 derece donuk goruntuler (-2 / -4 sonekli) once duzeltilir.
"""

import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

COCO_JSON = Path(__file__).resolve().parent.parent / "data" / "coco" / "train" / "_annotations.coco.json"
SAG, SOL = 1, 2


def yuzdelik(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    a, u = math.floor(k), math.ceil(k)
    return s[int(k)] if a == u else s[a] * (u - k) + s[u] * (k - a)


def ozet(ad, v, birim=""):
    if not v:
        print(f"  {ad}: veri yok")
        return
    print(f"  {ad:34s} medyan {statistics.median(v):6.2f}{birim}   "
          f"p90 {yuzdelik(v,90):6.2f}{birim}   p95 {yuzdelik(v,95):6.2f}{birim}   "
          f"max {max(v):6.2f}{birim}")


def main():
    d = json.loads(COCO_JSON.read_text(encoding="utf-8"))
    isim = {g["id"]: g["file_name"] for g in d["images"]}
    per = defaultdict(list)
    for a in d["annotations"]:
        per[a["image_id"]].append(a)

    kayitlar = {}  # file_name -> dict
    for img_id, anots in per.items():
        if sorted(a["category_id"] for a in anots) != [SAG, SOL]:
            continue
        kutu = {a["category_id"]: a["bbox"] for a in anots}
        merkez = {c: (b[0] + b[2] / 2, b[1] + b[3] / 2) for c, b in kutu.items()}
        dx = merkez[SOL][0] - merkez[SAG][0]
        dy = merkez[SOL][1] - merkez[SAG][1]
        ham = math.degrees(math.atan2(dy, abs(dx)))

        # 90 derece donukse duzelt: donuk karede aci ~ +-90, isaret yonu verir
        donuk = abs(ham) > 45
        aci = (ham - 90.0) if ham > 45 else ((ham + 90.0) if ham < -45 else ham)

        kayitlar[isim[img_id]] = {
            "aci": aci,
            "donuk": donuk,
            "D": math.hypot(dx, dy),
            "kutu": kutu,
        }

    print("=" * 70)
    print("1. ETIKET OZENI  -  ayni fotoftaki sol/sag kutu simetrisi")
    print("=" * 70)
    alan_orani, kenar_farki, merkez_kaymasi = [], [], []
    for k in kayitlar.values():
        bs, bl = k["kutu"][SAG], k["kutu"][SOL]
        alan_s, alan_l = bs[2] * bs[3], bl[2] * bl[3]
        if min(alan_s, alan_l) > 0:
            alan_orani.append(max(alan_s, alan_l) / min(alan_s, alan_l))
        kenar_s = (bs[2] + bs[3]) / 2
        kenar_l = (bl[2] + bl[3]) / 2
        kenar_farki.append(abs(kenar_s - kenar_l))
        # kutu yarim kenari, merkez belirsizliginin kaba ust siniri
        merkez_kaymasi.extend([kenar_s / 2, kenar_l / 2])

    ozet("Sol/sag kutu ALAN orani (1.0 ideal)", alan_orani, "x")
    ozet("Sol/sag ort. kenar farki", kenar_farki, " px")
    print()
    print("  Kutu yari-kenari (merkez belirsizliginin kaba olcegi):")
    ozet("  yari-kenar", merkez_kaymasi, " px")

    print()
    print("=" * 70)
    print("2. TEST-RETEST  -  ayni hastanin farkli fotograflari")
    print("=" * 70)

    hasta = defaultdict(dict)
    for ad, k in kayitlar.items():
        m = re.match(r"^(\d+)-(\d+)_", ad)
        if m:
            hasta[m.group(1)][m.group(2)] = k

    tam = {h: v for h, v in hasta.items() if len(v) >= 2}
    print(f"  En az 2 fotografi olan hasta: {len(tam)}")
    print()

    # hangi cekim ciftleri karsilastirilabilir?
    ciftler = [("1", "3"), ("2", "4"), ("1", "2"), ("3", "4"), ("1", "4"), ("2", "3")]
    for a, b in ciftler:
        farklar = []
        for h, v in tam.items():
            if a in v and b in v:
                farklar.append(abs(v[a]["aci"] - v[b]["aci"]))
        if not farklar:
            continue
        print(f"  Cekim {a} vs {b}   (n={len(farklar):3d})   "
              f"medyan fark {statistics.median(farklar):5.2f}°   "
              f"p90 {yuzdelik(farklar,90):5.2f}°   max {max(farklar):6.2f}°")

    print()
    print("  YORUM: '1 vs 3' ve '2 vs 4' ayni tip cekimin tekrari ise, oradaki")
    print("         medyan fark projenin GURULTU TABANI'dir. Farkli cift")
    print("         kombinasyonlari cok daha buyukse, cekimler farkli poz/")
    print("         goruntu tipidir ve birlikte egitilemez.")

    print()
    print("=" * 70)
    print("3. CEKIM TIPI KONTROLU  -  her cekim numarasinin aci profili")
    print("=" * 70)
    per_cekim = defaultdict(list)
    for ad, k in kayitlar.items():
        m = re.match(r"^(\d+)-(\d+)_", ad)
        per_cekim[m.group(2) if m else "IMG"].append(k)

    print(f"  {'cekim':8s} {'n':>5s} {'medyan|aci|':>12s} {'medyan D':>10s} {'donuk%':>8s}")
    for c in sorted(per_cekim, key=lambda x: (x == "IMG", x)):
        v = per_cekim[c]
        mut = [abs(x["aci"]) for x in v]
        ds = [x["D"] for x in v]
        dr = 100.0 * sum(1 for x in v if x["donuk"]) / len(v)
        print(f"  {c:8s} {len(v):5d} {statistics.median(mut):11.2f}° "
              f"{statistics.median(ds):9.1f} {dr:7.1f}%")


if __name__ == "__main__":
    main()
