"""Roboflow'a yüklenecek paketi hazırlar: işlenmiş görüntüler + MEVCUT etiketler.

    python scripts/41_roboflow_yukleme.py                 # grup 1-3 (183 görüntü)
    python scripts/41_roboflow_yukleme.py --gruplar 1,2,3,4

NEDEN
-----
Roboflow, görüntüleri COCO anotasyon dosyasıyla BİRLİKTE kabul eder. Böylece
koltuk altı / C7 / sakrum hazır gelir; kullanıcı sadece omurga_1..3 çizer.
7 nokta yerine 3 nokta = üçte bir iş.

NE NEREDEN GELİYOR
------------------
  sag koltuk, sol koltuk : data/final/kayitlar.json
      Bunlar ORTALANMIŞ etiketler — kopya kümelerindeki bağımsız etiketlerin
      ortalaması, gürültüsü 0.435° yerine 0.232°. Yeniden çizilirse bu
      kazanç kaybolur. DOKUNMA.
  c7, sakrum             : aag export'u (varsa)
      Orijinal kutu boyutlarıyla korunur.

ÇIKTI (git'e girmez)
  data/roboflow_yukleme/*.jpg
  data/roboflow_yukleme/_annotations.coco.json
"""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

KOK = Path(__file__).resolve().parent.parent
SET = KOK / "data" / "etiket_seti"
FINAL = KOK / "data" / "final"
AAG = KOK / "data" / "aag" / "train" / "_annotations.coco.json"
CIKTI = KOK / "data" / "roboflow_yukleme"

# Roboflow'da gorunecek sinif adlari. Sira 38_yolo7_format.py ile AYNI olmali.
SINIFLAR = ["sag koltuk", "sol koltuk", "c7", "omurga_1", "omurga_2",
            "omurga_3", "sakrum"]
KOLTUK_KUTU = 28.0     # ortalanmis merkez etrafina sentetik kutu (merkez onemli)

GRUP_ADI = {
    "1": "1_oncelik_kanitli",
    "2": "2_duzeltilecek",
    "3": "3_yeni_iyi",
    "4": "4_sinirda",
}


def kok_ad(ad):
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


def aag_kutulari():
    """{dosya_kok: {'c7': bbox, 'sakrum': bbox}} — orijinal kutu boyutlariyla."""
    if not AAG.exists():
        return {}
    d = json.loads(AAG.read_text(encoding="utf-8"))
    ad = {c["id"]: str(c.get("name", "")).lower() for c in d["categories"]}
    isim = {g["id"]: g["file_name"] for g in d["images"]}
    out = defaultdict(dict)
    for a in d["annotations"]:
        s = ad.get(a["category_id"])
        if s not in ("c7", "sakrum"):
            continue
        k = kok_ad(isim[a["image_id"]])
        if s in out[k]:
            continue           # mukerrer: ilkini tut, kullanici duzeltecek
        out[k][s] = list(a["bbox"])
    return dict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gruplar", default="1,2,3")
    ap.add_argument("--format", choices=["keypoint", "kutu"], default="keypoint",
                    help="keypoint: COCO Keypoints (ONERILEN). kutu: sinif basina bbox")
    a = ap.parse_args()
    gruplar = [GRUP_ADI[g.strip()] for g in a.gruplar.split(",") if g.strip() in GRUP_ADI]

    kayitlar = {r["dosya"]: r for r in
                json.loads((FINAL / "kayitlar.json").read_text(encoding="utf-8"))}
    aag = aag_kutulari()

    if CIKTI.exists():
        shutil.rmtree(CIKTI)
    CIKTI.mkdir(parents=True)

    images, annotations = [], []
    anot_id = 1
    sayim = Counter()

    for grup in gruplar:
        klasor = SET / grup
        if not klasor.exists():
            print(f"  UYARI: {klasor} yok, atlandi")
            continue
        for p in sorted(klasor.glob("*.jpg")):
            r = kayitlar.get(p.name)
            if r is None:
                continue
            W, H = Image.open(p).size
            shutil.copy2(p, CIKTI / p.name)
            img_id = len(images)
            images.append({"id": img_id, "file_name": p.name,
                           "width": W, "height": H, "license": 1})

            def ekle(sinif, bbox):
                nonlocal anot_id
                x, y, w, h = bbox
                # kadraj disina tasmasin
                x = max(0.0, min(x, W - 1)); y = max(0.0, min(y, H - 1))
                w = max(4.0, min(w, W - x)); h = max(4.0, min(h, H - y))
                annotations.append({
                    "id": anot_id, "image_id": img_id,
                    "category_id": SINIFLAR.index(sinif) + 1,
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2), "iscrowd": 0, "segmentation": [],
                })
                anot_id += 1
                sayim[sinif] += 1

            for sinif, anahtar in (("sag koltuk", "sag"), ("sol koltuk", "sol")):
                p_ = r.get(anahtar)
                if p_:
                    ekle(sinif, [p_[0] - KOLTUK_KUTU / 2, p_[1] - KOLTUK_KUTU / 2,
                                 KOLTUK_KUTU, KOLTUK_KUTU])

            for sinif in ("c7", "sakrum"):
                b = aag.get(p.stem, {}).get(sinif)
                if b:
                    ekle(sinif, b)

    if a.format == "keypoint":
        # --- COCO Keypoints: goruntu basina TEK nesne, 7 noktali dizi ---
        # Iskelet: koltuk alti cizgisi + C7'den sakruma omurga zinciri (1-indeksli)
        iskelet = [[1, 2], [3, 4], [4, 5], [5, 6], [6, 7]]
        kp_anot = []
        per_img = defaultdict(dict)
        for an in annotations:
            sinif = SINIFLAR[an["category_id"] - 1]
            x, y, w, h = an["bbox"]
            per_img[an["image_id"]][sinif] = (x + w / 2.0, y + h / 2.0)

        for g in images:
            noktalar = per_img.get(g["id"], {})
            kps, n_gorunur = [], 0
            xs, ys = [], []
            for s in SINIFLAR:
                if s in noktalar:
                    x, y = noktalar[s]
                    kps += [round(x, 2), round(y, 2), 2]
                    xs.append(x); ys.append(y); n_gorunur += 1
                else:
                    kps += [0, 0, 0]          # v=0: etiketlenmemis
            if not xs:
                continue
            # kisi kutusu: tum noktalari kapsayan genis kutu (Roboflow ister)
            m = 60.0
            x1 = max(0.0, min(xs) - m); y1 = max(0.0, min(ys) - m)
            x2 = min(float(g["width"]), max(xs) + m)
            y2 = min(float(g["height"]), max(ys) + m)
            kp_anot.append({
                "id": len(kp_anot) + 1, "image_id": g["id"], "category_id": 1,
                "bbox": [round(x1, 2), round(y1, 2),
                         round(x2 - x1, 2), round(y2 - y1, 2)],
                "area": round((x2 - x1) * (y2 - y1), 2), "iscrowd": 0,
                "num_keypoints": n_gorunur, "keypoints": kps, "segmentation": [],
            })

        coco = {
            "info": {"description": "skolyoz 7 landmark - on doldurulmus",
                     "version": "1"},
            "licenses": [{"id": 1, "name": "CC BY 4.0"}],
            "categories": [{"id": 1, "name": "kisi", "supercategory": "none",
                            "keypoints": SINIFLAR, "skeleton": iskelet}],
            "images": images,
            "annotations": kp_anot,
        }
        anot_sayisi = len(kp_anot)
        proje_tipi = "Keypoint Detection"
    else:
        coco = {
            "info": {"description": "skolyoz 7 landmark - on doldurulmus",
                     "version": "1"},
            "licenses": [{"id": 1, "name": "CC BY 4.0"}],
            "categories": ([{"id": 0, "name": "skolyoz", "supercategory": "none"}] +
                           [{"id": i + 1, "name": s, "supercategory": "skolyoz"}
                            for i, s in enumerate(SINIFLAR)]),
            "images": images,
            "annotations": annotations,
        }
        anot_sayisi = len(annotations)
        proje_tipi = "Object Detection"

    (CIKTI / "_annotations.coco.json").write_text(
        json.dumps(coco, ensure_ascii=False), encoding="utf-8")

    print("=" * 62)
    print(f"ROBOFLOW YUKLEME PAKETI HAZIR  ({a.format})")
    print("=" * 62)
    print(f"  goruntu    : {len(images)}")
    print(f"  anotasyon  : {anot_sayisi}")
    for s in SINIFLAR:
        n = sayim[s]
        durum = "" if n else "   <-- SEN CIZECEKSIN"
        print(f"    {s:<14} {n:4d}{durum}")
    print(f"\n  klasor: {CIKTI}")
    print(f"\n  Roboflow'da: yeni proje -> **{proje_tipi}** -> Upload ->")
    print("  bu klasordeki TUM dosyalari (jpg'ler + _annotations.coco.json)")
    print("  birlikte surukle. Etiketler otomatik yuklenir.")
    if a.format == "keypoint":
        print("\n  Iskelet siniflari sirayla tanimlanacak:")
        print("    " + "  ->  ".join(SINIFLAR))


if __name__ == "__main__":
    main()
