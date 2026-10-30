"""Aşama 2 için k-katlı çapraz doğrulama.

    python scripts/45_rafinaj7_cv.py --kat 5

NEDEN
-----
Omurgalı test seti sadece 22 görüntü. "Belirgin eğrilikte %81.8" gibi sayılar
n=11'den geliyordu; güven aralığı devasa. CV ile her görüntü bir turda ölçülür,
yani ölçüm örneklemi 22 yerine 180 olur.

Ayrıca eğitim verisi artar: sabit split'te 136, 5 katta 144, 10 katta 162.

SIZINTI KORUMASI
----------------
Katlara GÖRÜNTÜ değil GRUP atanır. data/final zaten 0.98 eşiğiyle
tekilleştirilmiş ama 0.85-0.98 arası benzer kareler kalmış olabilir; aynı
kişinin iki karesi farklı katlara düşerse skor şişer. Bu yüzden katlamadan
önce 0.85 benzerlikle yeniden gruplanıyor.

MODEL SEÇİMİ
------------
Her katın kendi doğrulama parçası var (o katın ölçüm parçası), bu yüzden
"en iyi epoch" seçimi dürüst kalıyor — test'e bakarak seçim yok.
"""

import argparse
import json
import math
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
_m = import_module("39_rafinaj7_model")
Rafinaj7, Rafinaj7Kayip = _m.Rafinaj7, _m.Rafinaj7Kayip
TIP_ADI = _m.TIP_ADI
_e = import_module("40_rafinaj7_egit")
KirpimVeri = _e.KirpimVeri

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "yolo7"
CIKTI = KOK / "runs" / "rafinaj7_cv"
IMZA = 64


def gruplar(yollar, esik=0.85):
    """Benzerlik tabanli gruplama — ayni kisi ayni kata dussun."""
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
    g = defaultdict(list)
    for i in range(len(yollar)):
        g[bul(i)].append(i)
    return list(g.values())


class Havuz(KirpimVeri):
    """train+val+test tek havuzda. Modul seviyesinde olmali: Windows'ta
    DataLoader isci sureclerini spawn ediyor ve sinifi pickle'lamasi gerekiyor;
    fonksiyon icinde tanimlanirsa num_workers>0 calismaz."""

    def __init__(self):
        self.ornekler = []
        for split in ("train", "val", "test"):
            self.ornekler += KirpimVeri(split, True).ornekler
        self.egitim = True


def egit_bir_kat(tr_ix, va_ix, tum, epochs, batch, lr, cihaz, kat, isci):
    ek = dict(num_workers=isci, persistent_workers=isci > 0,
              pin_memory=(cihaz == "cuda"))
    trl = DataLoader(Subset(tum, tr_ix), batch_size=batch, shuffle=True,
                     drop_last=True, **ek)
    val = DataLoader(Subset(tum, va_ix), batch_size=64, shuffle=False, **ek)

    model = Rafinaj7().to(cihaz)
    kayip_fn = Rafinaj7Kayip().to(cihaz)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    plan = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(len(trl), 1), pct_start=0.2)

    en_iyi, en_iyi_durum = 1e9, None
    for ep in range(1, epochs + 1):
        model.train()
        for x, y, t in trl:
            x, y, t = x.to(cihaz), y.to(cihaz), t.to(cihaz)
            koord, hm = model(x)
            kayip, _ = kayip_fn(koord, hm, y, t)
            opt.zero_grad(); kayip.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); plan.step()
        if ep % 5 == 0 or ep == epochs:
            med, _ = _e.dogrula(model, val, cihaz)
            if med < en_iyi:
                en_iyi = med
                en_iyi_durum = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(en_iyi_durum)
    return model, en_iyi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kat", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--isci", type=int, default=4, help="DataLoader isci sayisi")
    a = ap.parse_args()

    torch.manual_seed(1337); random.seed(1337); np.random.seed(1337)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    CIKTI.mkdir(parents=True, exist_ok=True)

    tum = Havuz()
    print(f"toplam ornek: {len(tum)}")
    print(f"tip dagilimi: {dict(tum.tip_dagilimi())}")

    # benzersiz goruntuler ve gruplari
    goruntuler = sorted({p for p, _, _ in tum.ornekler})
    print(f"benzersiz goruntu: {len(goruntuler)}")
    grup = gruplar(goruntuler)
    print(f"benzerlik grubu  : {len(grup)}  (en buyuk {max(len(g) for g in grup)})")

    yol_ix = {p: i for i, p in enumerate(goruntuler)}
    goruntu_kat = {}
    random.shuffle(grup)
    dolu = [0] * a.kat
    for g in sorted(grup, key=len, reverse=True):
        k = dolu.index(min(dolu))
        for i in g:
            goruntu_kat[goruntuler[i]] = k
        dolu[k] += len(g)
    print(f"kat buyuklukleri : {dolu}")

    ornek_kat = np.array([goruntu_kat[p] for p, _, _ in tum.ornekler])

    per_tip = defaultdict(list)
    hepsi = []
    for k in range(a.kat):
        va_ix = np.where(ornek_kat == k)[0].tolist()
        tr_ix = np.where(ornek_kat != k)[0].tolist()
        print(f"\n--- kat {k+1}/{a.kat}: train {len(tr_ix)}  eval {len(va_ix)} ---",
              flush=True)
        model, en_iyi = egit_bir_kat(tr_ix, va_ix, tum, a.epochs, a.batch,
                                     a.lr, cihaz, k, a.isci)
        # bu katin ORNEKLERINDE olcum (model onlari hic gormedi)
        val = DataLoader(Subset(tum, va_ix), batch_size=64, shuffle=False,
                         num_workers=a.isci, persistent_workers=a.isci > 0)
        model.eval()
        with torch.no_grad():
            for x, y, t in val:
                x, y, t = x.to(cihaz), y.to(cihaz), t.to(cihaz)
                koord, _ = model(x)
                sec = koord[torch.arange(len(t), device=cihaz), t]
                d = torch.sqrt(((sec - y) ** 2).sum(dim=1))
                for vv, tt in zip(d.cpu().tolist(), t.cpu().tolist()):
                    hepsi.append(vv); per_tip[tt].append(vv)
        torch.save({"model": model.state_dict(), "kat": k, "val": en_iyi},
                   CIKTI / f"kat{k}.pt")
        print(f"    kat {k+1} en iyi val: {en_iyi:.3f} px", flush=True)

    print("\n" + "=" * 60)
    print(f"CAPRAZ DOGRULAMA SONUCU  ({a.kat} kat, tum veri uzerinde)")
    print("=" * 60)
    print(f"  olcum ornegi: {len(hepsi)}   (sabit split'te sadece val vardi)")
    print(f"  GENEL medyan: {st.median(hepsi):.3f} px")
    print()
    print(f"  {'tip':<14}{'n':>6}{'medyan':>9}{'p90':>9}")
    for t in sorted(per_tip):
        v = sorted(per_tip[t])
        print(f"  {TIP_ADI[t]:<14}{len(v):>6}{st.median(v):>9.3f}"
              f"{v[int(len(v)*0.9)]:>9.3f}")
    (CIKTI / "sonuc.json").write_text(json.dumps(
        {TIP_ADI[t]: {"n": len(v), "medyan": st.median(v)}
         for t, v in per_tip.items()}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

