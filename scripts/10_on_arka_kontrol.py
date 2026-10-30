"""Asama 1e - Pose modeli sol/sag'i dogru mu atiyor?

Gozlem: kanoniklestirilmis sayfada R ve L yaklasik yari yariya karisik.
Hipotez: COCO pose modelleri agirlikli olarak ONE BAKAN insanla egitilir.
Arkadan cekimde model kisiyi onden saniyor ve sol/sag'i ters atiyor.

Geometri:
  ARKADAN cekim -> kisinin sag omzu karede SAGDA  (sag_x > sol_x)
  ONDEN cekim   -> kisinin sag omzu karede SOLDA  (sag_x < sol_x)

Yuz noktalari (burun/goz/kulak) guveni on/arka ayrimini verir:
  yuz gorunuyor  -> onden cekim
  yuz yok        -> arkadan cekim

Bu ikisini capraz kontrol edip modelin sol/sag atamasina guvenilip
guvenilemeyecegine karar veriyoruz.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

KOK = Path(__file__).resolve().parent.parent / "data" / "clean2"
BURUN, SOL_GOZ, SAG_GOZ, SOL_KULAK, SAG_KULAK = 0, 1, 2, 3, 4
SOL_OMUZ, SAG_OMUZ = 5, 6


def main():
    from ultralytics import YOLO
    model = YOLO("yolo11n-pose.pt")

    kayitlar = json.loads((KOK / "kayitlar.json").read_text(encoding="utf-8"))
    print(f"Incelenen: {len(kayitlar)}")

    sayim = Counter()
    detay = []

    for r in kayitlar:
        # ultralytics numpy diziyi BGR varsayar; RGB vermek tespiti bozar
        s = model.predict(str(KOK / "images" / r["dosya"]), verbose=False, conf=0.25)[0]
        if s.keypoints is None or len(s.boxes) == 0:
            sayim["pose_yok"] += 1
            continue
        i = int(np.argmax(s.boxes.conf.cpu().numpy()))
        kp = s.keypoints.data.cpu().numpy()[i]

        yuz_guven = float(max(kp[BURUN][2], kp[SOL_GOZ][2], kp[SAG_GOZ][2],
                              kp[SOL_KULAK][2], kp[SAG_KULAK][2]))
        if min(kp[SOL_OMUZ][2], kp[SAG_OMUZ][2]) < 0.3:
            sayim["omuz_zayif"] += 1
            continue

        # model kisinin sag omzunu karede saga mi koydu?
        model_sag_sagda = bool(kp[SAG_OMUZ][0] > kp[SOL_OMUZ][0])
        yuz_var = yuz_guven >= 0.5

        # yuz varsa ONDEN cekim: dogru olan sag_x < sol_x  (yani sag SOLDA)
        # yuz yoksa ARKADAN cekim: dogru olan sag_x > sol_x (yani sag SAGDA)
        beklenen = (not yuz_var)
        dogru = (model_sag_sagda == beklenen)

        anahtar = ("yuz_var" if yuz_var else "yuz_yok") + ("_dogru" if dogru else "_TERS")
        sayim[anahtar] += 1
        detay.append({"dosya": r["dosya"], "yuz_guven": round(yuz_guven, 3),
                      "model_sag_sagda": model_sag_sagda, "yuz_var": yuz_var})

    print()
    print("=" * 62)
    print("SONUC")
    print("=" * 62)
    for k, v in sorted(sayim.items()):
        print(f"  {k:18s}: {v}")

    yv = sayim["yuz_var_dogru"] + sayim["yuz_var_TERS"]
    yy = sayim["yuz_yok_dogru"] + sayim["yuz_yok_TERS"]
    print()
    print(f"  YUZ GORUNEN (onden cekim)  : {yv}")
    if yv:
        print(f"    model dogru : {100.0*sayim['yuz_var_dogru']/yv:.1f}%")
    print(f"  YUZ GORUNMEYEN (arkadan)   : {yy}")
    if yy:
        print(f"    model dogru : {100.0*sayim['yuz_yok_dogru']/yy:.1f}%")

    print()
    print("  Yorum: arkadan cekimde oran ~%50 ise model sol/sag'i tahmin")
    print("         ediyor demektir; guvenilmez. O zaman sol/sag'i pose")
    print("         modelinden DEGIL, konum kuralindan alacagiz.")

    (KOK / "on_arka.json").write_text(
        json.dumps(detay, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
