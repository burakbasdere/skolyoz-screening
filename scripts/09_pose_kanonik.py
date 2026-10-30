"""Asama 1d - Pose modeli ile kanoniklestirme.

El yapimi sezgiseller basarisiz oldu (ten rengi: gri goruntude kor,
siluet genisligi: %50 uyum = sinyal yok). Onun yerine COCO uzerinde
egitilmis hazir bir insan pose modeli kullaniyoruz.

Model her karede 17 nokta verir; bize gereken dordu:
    5  sol omuz     6  sag omuz        <- ANATOMIK sol/sag
   11  sol kalca   12  sag kalca

Bu uc sorunu birden cozer:
  1) Oryantasyon : omuzlar kalcalarin ustundeyse goruntu dik
  2) Sag/sol     : model anatomik tarafi dogrudan verir, konum tahmini yok
  3) Gri goruntu : renkten bagimsiz calisir

Ek olarak her goruntu icin domain metadatasi cikarir (renkli mi, tam govde
gorunuyor mu, egilmis mi) - egitim setini daraltmak gerekirse diye.
"""

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

KOK = Path(__file__).resolve().parent.parent
KAYNAK = KOK / "data" / "coco" / "train"
HEDEF = KOK / "data" / "clean2"

SAG, SOL = 1, 2
IMZA, ESIK = 64, 0.98

SOL_OMUZ, SAG_OMUZ, SOL_KALCA, SAG_KALCA = 5, 6, 11, 12
BURUN = 0


def ileri_map(x, y, k, s):
    for _ in range(k % 4):
        x, y = y, s - 1 - x
    return x, y


def kayitlari_yukle():
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
    return iyi


def kumele(adlar):
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

    for k in range(4):
        kor = taban @ nrm(np.rot90(ham, k, axes=(1, 2)).copy()).T
        for i, j in zip(*np.where(kor > ESIK)):
            i, j = int(i), int(j)
            if i != j:
                ra, rb = bul(i), bul(j)
                if ra != rb:
                    ebeveyn[max(ra, rb)] = min(ra, rb)

    kume = defaultdict(list)
    for i in range(len(adlar)):
        kume[bul(i)].append(adlar[i])
    return list(kume.values())


def en_iyi_kisi(sonuc):
    """En yuksek guvenli kisiyi dondur: (keypoints Nx3, kutu guveni)."""
    if sonuc.keypoints is None or len(sonuc.boxes) == 0:
        return None, 0.0
    guvenler = sonuc.boxes.conf.cpu().numpy()
    i = int(np.argmax(guvenler))
    kp = sonuc.keypoints.data.cpu().numpy()[i]  # (17,3): x,y,conf
    return kp, float(guvenler[i])


def rotasyon_skoru(kp, guven):
    """Bu rotasyon ne kadar 'dik'? Yuksek skor = daha dik."""
    if kp is None:
        return -1e9, {}
    om = [kp[SOL_OMUZ], kp[SAG_OMUZ]]
    ka = [kp[SOL_KALCA], kp[SAG_KALCA]]
    om_g = min(om[0][2], om[1][2])
    ka_g = min(ka[0][2], ka[1][2])
    if om_g < 0.3:
        return -1e6 + guven, {}

    om_y = (om[0][1] + om[1][1]) / 2.0
    # pose modeli np.float32 dondurur; JSON icin Python float'a cevir
    bilgi = {"omuz_guven": round(float(om_g), 3), "kalca_guven": round(float(ka_g), 3)}

    if ka_g >= 0.3:
        ka_y = (ka[0][1] + ka[1][1]) / 2.0
        # dik ise omuzlar kalcalarin USTUNDE => om_y < ka_y
        govde = ka_y - om_y
        omuz_gen = math.hypot(om[0][0] - om[1][0], om[0][1] - om[1][1])
        bilgi["govde_boy_oran"] = round(float(govde / omuz_gen), 3) if omuz_gen > 1 else 0.0
        return govde + 200.0 * (om_g + ka_g), bilgi

    # kalca yoksa burun ipucu: burun omuzlarin ustunde olmali
    if kp[BURUN][2] >= 0.3:
        return (om_y - kp[BURUN][1]) + 100.0 * om_g, bilgi
    return 100.0 * om_g, bilgi


def main():
    from ultralytics import YOLO

    model = YOLO("yolo11n-pose.pt")
    iyi = kayitlari_yukle()
    adlar = sorted(iyi)
    print(f"[1] Kullanilabilir goruntu : {len(adlar)}")

    kumeler = kumele(adlar)
    print(f"[2] Benzersiz kume : {len(kumeler)}")

    if HEDEF.exists():
        shutil.rmtree(HEDEF)
    (HEDEF / "images").mkdir(parents=True)

    kayitlar, pose_yok = [], []

    for idx, kume in enumerate(kumeler):
        ad = kume[0]
        rgb = np.asarray(Image.open(KAYNAK / ad).convert("RGB"))
        s = rgb.shape[0]

        adaylar = []
        for k in range(4):
            # ultralytics numpy diziyi BGR varsayar; RGB vermek tespiti bozar
            dondurulmus = np.rot90(rgb, k)[:, :, ::-1].copy()
            r = model.predict(dondurulmus, verbose=False, conf=0.25)[0]
            kp, guven = en_iyi_kisi(r)
            skor, bilgi = rotasyon_skoru(kp, guven)
            adaylar.append((skor, k, kp, guven, bilgi))

        skor, k, kp, guven, bilgi = max(adaylar, key=lambda t: t[0])
        if kp is None or skor <= -1e5:
            pose_yok.append(ad)
            continue

        arr = np.rot90(rgb, k).copy()
        p1 = ileri_map(*iyi[ad][SAG], k, s)
        p2 = ileri_map(*iyi[ad][SOL], k, s)

        # --- sag/sol atamasi: KONUM KURALI ---
        # Pose modelinin sol/sag atamasi arkadan cekimde sadece %10 dogru
        # (bkz. 10_on_arka_kontrol.py): model sirti gorunce kisiyi onden
        # saniyor ve taraflari ters ceviriyor. O yuzden modele guvenmiyoruz.
        # Arkadan cekimde kisinin sag omzu karede SAGDADIR - deterministik.
        p_sag, p_sol = (p1, p2) if p1[0] > p2[0] else (p2, p1)
        # modelin gorusu kullanilmiyor, sadece kayit icin saklaniyor
        pose_sag_sagda = bool(kp[SAG_OMUZ][0] > kp[SOL_OMUZ][0])

        p_sag = [float(p_sag[0]), float(p_sag[1])]
        p_sol = [float(p_sol[0]), float(p_sol[1])]
        dx, dy = p_sol[0] - p_sag[0], p_sol[1] - p_sag[1]
        aci = math.degrees(math.atan2(dy, abs(dx)))
        D = math.hypot(dx, dy)

        cikti = f"{idx:04d}.jpg"
        Image.fromarray(arr).save(HEDEF / "images" / cikti, quality=95)

        # domain metadatasi
        hsv = np.asarray(Image.fromarray(arr).convert("HSV"))
        doygunluk = float(hsv[:, :, 1].mean())

        kayitlar.append({
            "dosya": cikti, "kaynak": ad, "kume_boyu": len(kume),
            "rotasyon": k * 90,
            "sag": [round(p_sag[0], 2), round(p_sag[1], 2)],
            "sol": [round(p_sol[0], 2), round(p_sol[1], 2)],
            "aci": round(aci, 3), "D": round(D, 2),
            "pose_guven": round(guven, 3),
            "pose_sag_sagda": pose_sag_sagda,   # kullanilmiyor, kayit icin
            "renkli": doygunluk > 25, "doygunluk": round(doygunluk, 1),
            **bilgi,
        })

        if (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{len(kumeler)} ...")

    print(f"[3] Kanoniklestirildi : {len(kayitlar)}")
    print(f"    Pose bulunamadi   : {len(pose_yok)}  (illustrasyon/manken olabilir)")

    (HEDEF / "kayitlar.json").write_text(
        json.dumps(kayitlar, ensure_ascii=False, indent=2), encoding="utf-8")
    (HEDEF / "pose_yok.json").write_text(
        json.dumps(pose_yok, ensure_ascii=False, indent=2), encoding="utf-8")

    renkli = sum(1 for r in kayitlar if r["renkli"])
    tam = sum(1 for r in kayitlar if r.get("kalca_guven", 0) >= 0.3)
    print()
    print(f"    Renkli goruntu     : {renkli} / {len(kayitlar)}")
    print(f"    Kalca gorunen (tam govde): {tam} / {len(kayitlar)}")
    print(f"\nCikti: {HEDEF}")


if __name__ == "__main__":
    main()
