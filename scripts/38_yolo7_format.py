"""7 landmark'lı YOLO-pose veri seti — görünürlük maskeli.

    python scripts/38_yolo7_format.py --export data/aag7/train/_annotations.coco.json

NOKTA SIRASI (değiştirme — model, Android ve değerlendirme buna bağlı)
    0 sag koltuk    1 sol koltuk    2 c7
    3 omurga_1      4 omurga_2      5 omurga_3      6 sakrum

0 ve 1 mevcut sırayı korur; eski etiketler ve Android tarafı bozulmaz.

GÖRÜNÜRLÜK MASKESİ
------------------
Her görüntüde 7 noktanın hepsi olmak zorunda değil. Etiketlenmemiş nokta
v=0 yazılır; Ultralytics'in pose kaybı bunları maskeler
(utils/loss.py: kpt_mask = gt_kpt[..., 2] != 0).

Böylece:
  - 337 görüntünün TAMAMI koltuk altı için kullanılır
  - ~183 görüntü C7/omurga/sakrum için kullanılır
  - belden kesilmiş görüntüler sakrum hariç her şeye katkı verir

DİKKAT — kobj=0 ŞART
--------------------
Ultralytics ayrıca bir görünürlük başlığı eğitir (kpts_obj_loss). Etiketsiz
ama aslında GÖRÜNEN noktaları v=0 yaptığımız için bu başlık yanlış öğrenir
("bu nokta yok" der). Eğitimde kobj=0.0 verilmeli. scripts/40 bunu yapıyor.
"""

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

KOK = Path(__file__).resolve().parent.parent
FINAL = KOK / "data" / "final"
HEDEF = KOK / "data" / "yolo7"

NOKTA_SIRASI = ["sag koltuk", "sol koltuk", "c7",
                "omurga_1", "omurga_2", "omurga_3", "sakrum"]
N = len(NOKTA_SIRASI)

# Roboflow sinif adi -> nokta indeksi. Kucuk harfe indirilmis, bosluk/alt
# cizgi normalize edilmis halleriyle eslesir.
TAKMA_AD = {
    "sag koltuk": 0, "sagkoltuk": 0, "sag_koltuk": 0, "right armpit": 0,
    "sol koltuk": 1, "solkoltuk": 1, "sol_koltuk": 1, "left armpit": 1,
    "c7": 2,
    "omurga_1": 3, "omurga1": 3, "spine_1": 3, "o1": 3,
    "omurga_2": 4, "omurga2": 4, "spine_2": 4, "o2": 4,
    "omurga_3": 5, "omurga3": 5, "spine_3": 5, "o3": 5,
    "sakrum": 6, "sacrum": 6, "bel_sokumu": 6, "belsokumu": 6,
}

KENAR_PAY = 0.06


def normalize(ad):
    return " ".join(str(ad).lower().replace("_", " ").split())


def sinif_indeksi(ad):
    n = normalize(ad)
    if n in TAKMA_AD:
        return TAKMA_AD[n]
    return TAKMA_AD.get(n.replace(" ", ""), None)


def kok_ad(ad):
    """Roboflow dosya adindan orijinal koku cikar."""
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


def _keypoint_export_oku(d):
    """COCO Keypoints formati: goruntu basina tek nesne, noktalar dizide."""
    kat_kp = {}
    for c in d["categories"]:
        if c.get("keypoints"):
            kat_kp[c["id"]] = [sinif_indeksi(ad) for ad in c["keypoints"]]
    if not kat_kp:
        return None
    bilinmeyen = set()
    for c in d["categories"]:
        for ad in c.get("keypoints", []):
            if sinif_indeksi(ad) is None:
                bilinmeyen.add(ad)
    if bilinmeyen:
        print(f"  UYARI: taninmayan keypoint adlari: {sorted(bilinmeyen)}")
        print(f"         TAKMA_AD sozlugune ekle, yoksa o noktalar atlanir.")

    isim = {g["id"]: g["file_name"] for g in d["images"]}
    per = defaultdict(dict)
    for a in d["annotations"]:
        harita = kat_kp.get(a["category_id"])
        kp = a.get("keypoints")
        if harita is None or not kp:
            continue
        kok = kok_ad(isim[a["image_id"]])
        for j, idx in enumerate(harita):
            if idx is None:
                continue
            x, y, v = kp[j * 3], kp[j * 3 + 1], kp[j * 3 + 2]
            if v == 0:                       # etiketlenmemis
                continue
            per[kok][idx] = (float(x), float(y))
    return dict(per)


def export_oku(yol):
    """Roboflow COCO export -> {dosya_kok: {nokta_idx: (x, y)}}

    Hem COCO Keypoints (nokta dizisi) hem sinif-basina-bbox formatini okur;
    hangisi oldugunu kategorilerdeki 'keypoints' alanindan anlar.
    """
    if yol is None or not Path(yol).exists():
        return {}, Counter()
    d = json.loads(Path(yol).read_text(encoding="utf-8"))

    kp = _keypoint_export_oku(d)
    if kp is not None:
        print("  format: COCO Keypoints")
        return kp, Counter()

    print("  format: sinif basina bounding box")
    kat = {c["id"]: sinif_indeksi(c.get("name")) for c in d["categories"]}
    bilinmeyen = {c["name"] for c in d["categories"] if sinif_indeksi(c.get("name")) is None
                  and normalize(c.get("name")) not in ("aag", "bladeco", "none")}
    if bilinmeyen:
        print(f"  UYARI: taninmayan sinif adlari: {sorted(bilinmeyen)}")
        print(f"         TAKMA_AD sozlugune ekle, yoksa o noktalar atlanir.")

    isim = {g["id"]: g["file_name"] for g in d["images"]}
    per = defaultdict(dict)
    coklu = Counter()
    for a in d["annotations"]:
        idx = kat.get(a["category_id"])
        if idx is None:
            continue
        kok = kok_ad(isim[a["image_id"]])
        x, y, w, h = a["bbox"]
        if idx in per[kok]:
            coklu[kok] += 1          # ayni sinif iki kez -> supheli, ilkini tut
            continue
        per[kok][idx] = (x + w / 2.0, y + h / 2.0)
    return dict(per), coklu


def yeniden_bol(kayitlar, oran_str, esik=0.85, tohum=1337):
    """Split'i yeniden ata. Katlara GORUNTU degil GRUP verilir.

    data/final 0.98 esigiyle tekillestirilmis ama 0.85-0.98 arasi benzer
    kareler kalmis olabilir; ayni kisinin iki karesi farkli split'e duserse
    skor siser. Bu yuzden once benzerlik gruplamasi yapiliyor.
    """
    import random
    from PIL import Image as _Im

    oran = [float(x) for x in oran_str.split("/")]
    oran = [x / sum(oran) for x in oran]
    print(f"[0] Yeniden bolme: train {oran[0]:.0%} / val {oran[1]:.0%} / "
          f"test {oran[2]:.0%}")

    yollar, gecerli = [], []
    for r in kayitlar:
        p = FINAL / r["split"] / r["dosya"]
        if p.exists():
            yollar.append(p)
            gecerli.append(r)

    N, S = len(yollar), 64
    ham = np.zeros((N, S, S), dtype=np.float32)
    for i, p in enumerate(yollar):
        ham[i] = np.asarray(_Im.open(p).convert("L").resize((S, S), _Im.BILINEAR),
                            dtype=np.float32)
    v = ham.reshape(N, -1)
    v = v - v.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    nrm[nrm < 1e-6] = 1.0
    v = v / nrm

    ebeveyn = list(range(N))

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
    gr = defaultdict(list)
    for i in range(N):
        gr[bul(i)].append(i)
    grup = list(gr.values())

    random.Random(tohum).shuffle(grup)
    hedef = [o * N for o in oran]
    dolu = [0, 0, 0]
    adlar = ["train", "val", "test"]
    for g in sorted(grup, key=len, reverse=True):
        k = max(range(3), key=lambda i: hedef[i] - dolu[i])
        for i in g:
            # kaynak dosya HALA eski split klasorunde; okuma icin sakla
            gecerli[i].setdefault("_kaynak_split", gecerli[i]["split"])
            gecerli[i]["split"] = adlar[k]
        dolu[k] += len(g)

    print(f"    grup: {len(grup)}  ->  " +
          "  ".join(f"{adlar[i]}={dolu[i]}" for i in range(3)))
    return gecerli


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None,
                    help="Roboflow COCO export json (c7/omurga/sakrum)")
    ap.add_argument("--export-koltuk", action="store_true",
                    help="Koltuk altini da export'tan al. VARSAYILAN: ALMA.")
    ap.add_argument("--bol", default=None, metavar="TR/VA/TE",
                    help="Yeniden bol, orn. 80/10/10. Verilmezse data/final'in "
                         "split'i aynen kullanilir.")
    a = ap.parse_args()

    kayitlar = json.loads((FINAL / "kayitlar.json").read_text(encoding="utf-8"))

    if a.bol:
        kayitlar = yeniden_bol(kayitlar, a.bol)

    yeni, coklu = export_oku(a.export)
    if a.export:
        print(f"export: {len(yeni)} goruntude yeni nokta bulundu")
        if coklu:
            print(f"  mukerrer sinifli goruntu: {len(coklu)} (ilki kullanildi)")
    else:
        print("export verilmedi -> sadece koltuk alti noktalari yazilir")

    if HEDEF.exists():
        shutil.rmtree(HEDEF)
    for s in ("train", "val", "test"):
        (HEDEF / s / "images").mkdir(parents=True)
        (HEDEF / s / "labels").mkdir(parents=True)

    from ultralytics import YOLO
    pose = YOLO("yolo11n-pose.pt")

    kapsam = Counter()
    per_goruntu = Counter()
    yazilan = 0

    for r in kayitlar:
        # yeniden bolme yapildiysa dosya HALA eski klasorde duruyor
        kaynak = FINAL / r.get("_kaynak_split", r["split"]) / r["dosya"]
        if not kaynak.exists():
            continue
        kok = Path(r["dosya"]).stem
        im = Image.open(kaynak).convert("RGB")
        W, H = im.size

        noktalar = {}
        if r.get("sag"):
            noktalar[0] = tuple(r["sag"])
        if r.get("sol"):
            noktalar[1] = tuple(r["sol"])
        for idx, p in yeni.get(kok, {}).items():
            # Koltuk alti (0,1) VARSAYILAN OLARAK export'tan ALINMAZ.
            # data/final'dekiler ORTALANMIS etiketler: kopya kumelerindeki
            # bagimsiz etiketlerin ortalamasi, gurultusu 0.435 yerine 0.232 derece.
            # Yeni bir turda tek tek cizilen koltuk alti bunlardan kotudur ve
            # uzerine yazarsa kazanim kaybolur (olculdu: 3.55 -> 4.31 px).
            if idx in (0, 1) and not a.export_koltuk:
                continue
            noktalar[idx] = p

        if not noktalar:
            continue
        for i in noktalar:
            kapsam[i] += 1
        per_goruntu[len(noktalar)] += 1

        im.save(HEDEF / r["split"] / "images" / r["dosya"], quality=95)

        # --- kisi kutusu: pose modelinden, tum etiketli noktalari kapsayacak sekilde
        # ultralytics numpy diziyi BGR varsayar; dosya yolu vermek dogrusu
        s = pose.predict(str(kaynak), verbose=False, conf=0.25)[0]
        if len(s.boxes) > 0:
            i = int(np.argmax(s.boxes.conf.cpu().numpy()))
            x1, y1, x2, y2 = map(float, s.boxes.xyxy.cpu().numpy()[i])
        else:
            xs = [p[0] for p in noktalar.values()]
            ys = [p[1] for p in noktalar.values()]
            m = 40.0
            x1, y1, x2, y2 = min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m

        for p in noktalar.values():
            x1, x2 = min(x1, p[0]), max(x2, p[0])
            y1, y2 = min(y1, p[1]), max(y2, p[1])
        pw, ph = (x2 - x1) * KENAR_PAY, (y2 - y1) * KENAR_PAY
        x1 = max(0.0, x1 - pw); y1 = max(0.0, y1 - ph)
        x2 = min(float(W), x2 + pw); y2 = min(float(H), y2 + ph)

        parca = [f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                 f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"]
        for i in range(N):
            if i in noktalar:
                x, y = noktalar[i]
                parca.append(f"{x/W:.6f} {y/H:.6f} 2")
            else:
                parca.append("0.000000 0.000000 0")   # v=0 -> kayipta maskelenir
        (HEDEF / r["split"] / "labels" / f"{kok}.txt").write_text(
            " ".join(parca) + "\n", encoding="utf-8")
        yazilan += 1

    # flip_idx: yatay flip 0<->1 takas eder, orta hat noktalari yerinde kalir.
    # Egitimde fliplr=0 kullaniyoruz ama format dogru olsun.
    yaml = f"""# Skolyoz - 7 landmark
# 0 sag koltuk  1 sol koltuk  2 c7  3 omurga_1  4 omurga_2  5 omurga_3  6 sakrum
path: {HEDEF.as_posix()}
train: train/images
val: val/images
test: test/images

kpt_shape: [{N}, 3]
flip_idx: [1, 0, 2, 3, 4, 5, 6]

names:
  0: kisi
"""
    (HEDEF / "skolyoz7.yaml").write_text(yaml, encoding="utf-8")

    print()
    print("=" * 62)
    print("NOKTA KAPSAMI")
    print("=" * 62)
    for i, ad in enumerate(NOKTA_SIRASI):
        n = kapsam[i]
        print(f"  {i} {ad:<14} {n:4d} goruntu  ({100.0*n/max(yazilan,1):5.1f}%)")
    print()
    print("  goruntu basina etiketli nokta sayisi:")
    for k in sorted(per_goruntu):
        print(f"    {k} nokta -> {per_goruntu[k]:4d} goruntu")
    print()
    for s in ("train", "val", "test"):
        print(f"  {s:6s}: {len(list((HEDEF/s/'images').glob('*.jpg')))}")
    print(f"\n  yazildi: {HEDEF / 'skolyoz7.yaml'}")
    print("  EGITIMDE kobj=0.0 KULLAN (bkz. dosya basligi)")


if __name__ == "__main__":
    main()
