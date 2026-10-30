"""Asama 3b - Rafinaj agini egit.

Veri: data/final goruntuleri + ortalanmis etiketler.
Her (goruntu, nokta) cifti icin, GERCEK noktanin etrafinda JITTER'li bir
128x128 kirpim alinir. Jitter, Asama 1'in (~24 px) hatasini taklit eder;
boylece ag, kaba tahminin etrafindaki her konumdan dogru noktayi bulmayi
ogrenir.

Augmentasyon
------------
  jitter      : +-35 px (asama 1 hatasini temsil eder, asil regularizasyon)
  hflip       : serbest! Kirpim seviyesinde "hangi taraf" bilgisi tasinmiyor,
                sadece "kose nerede" soruluyor. Veriyi 2x yapar.
                (Tam goruntude flip YASAKTI - orada sag/sol anlam tasiyordu.)
  olcek       : 0.85-1.20, farkli govde buyukluklerini temsil eder
  renk        : parlaklik/kontrast/doygunluk, isik degisimi
"""

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
_m = import_module("17_rafinaj_model")
RafinajAgi, RafinajKayip = _m.RafinajAgi, _m.RafinajKayip

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "final"
CIKTI = KOK / "runs" / "rafinaj"

KIRPIM = 128
JITTER = 35
ORT = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class KirpimVeri(Dataset):
    def __init__(self, split, egitim):
        kayitlar = json.loads((VERI / "kayitlar.json").read_text(encoding="utf-8"))
        self.ornekler = []
        for r in kayitlar:
            if r["split"] != split:
                continue
            yol = VERI / split / r["dosya"]
            for anahtar in ("sag", "sol"):
                self.ornekler.append((yol, tuple(r[anahtar])))
        self.egitim = egitim

    def __len__(self):
        return len(self.ornekler)

    def __getitem__(self, i):
        yol, (gx, gy) = self.ornekler[i]
        im = Image.open(yol).convert("RGB")

        if self.egitim:
            jx = random.uniform(-JITTER, JITTER)
            jy = random.uniform(-JITTER, JITTER)
            olcek = random.uniform(0.85, 1.20)
        else:
            # dogrulamada deterministik ol: sabit tohumla tekrarlanabilir jitter
            rng = random.Random(i * 7919)
            jx = rng.uniform(-JITTER, JITTER)
            jy = rng.uniform(-JITTER, JITTER)
            olcek = 1.0

        pencere = KIRPIM * olcek
        # kirpim merkezi = gercek nokta + jitter (asama 1 hatasini taklit)
        cx, cy = gx + jx, gy + jy
        sol = cx - pencere / 2
        ust = cy - pencere / 2
        kirpim = im.crop((int(round(sol)), int(round(ust)),
                          int(round(sol + pencere)), int(round(ust + pencere))))
        kirpim = kirpim.resize((KIRPIM, KIRPIM), Image.BILINEAR)

        # gercek noktanin kirpim koordinati
        hx = (gx - round(sol)) * KIRPIM / pencere
        hy = (gy - round(ust)) * KIRPIM / pencere

        if self.egitim:
            if random.random() < 0.5:
                kirpim = kirpim.transpose(Image.FLIP_LEFT_RIGHT)
                hx = KIRPIM - 1 - hx
            for Sinif, alt, ust_ in ((ImageEnhance.Brightness, 0.7, 1.3),
                                     (ImageEnhance.Contrast, 0.7, 1.3),
                                     (ImageEnhance.Color, 0.6, 1.4)):
                kirpim = Sinif(kirpim).enhance(random.uniform(alt, ust_))

        a = np.asarray(kirpim, dtype=np.float32) / 255.0
        a = (a - ORT) / STD
        return (torch.from_numpy(a.transpose(2, 0, 1)),
                torch.tensor([hx, hy], dtype=torch.float32))


def dogrula(model, yukleyici, cihaz):
    model.eval()
    hatalar = []
    with torch.no_grad():
        for x, y in yukleyici:
            x, y = x.to(cihaz), y.to(cihaz)
            koord, _ = model(x)
            d = torch.sqrt(((koord - y) ** 2).sum(dim=1))
            hatalar.extend(d.cpu().tolist())
    return statistics.median(hatalar), float(np.mean(hatalar))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()

    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    CIKTI.mkdir(parents=True, exist_ok=True)

    tr = KirpimVeri("train", True)
    va = KirpimVeri("val", False)
    te = KirpimVeri("test", False)
    print(f"ornek: train {len(tr)}  val {len(va)}  test {len(te)}")

    trl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True)
    val = DataLoader(va, batch_size=64, shuffle=False, num_workers=0)
    tel = DataLoader(te, batch_size=64, shuffle=False, num_workers=0)

    model = RafinajAgi().to(cihaz)
    kayip_fn = RafinajKayip().to(cihaz)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    plan = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(trl), pct_start=0.2)

    en_iyi = 1e9
    print(f"\n{'epoch':>6} {'kayip':>9} {'koord':>8} {'val medyan':>12} {'val ort':>9}")
    for ep in range(1, a.epochs + 1):
        model.train()
        toplam, koord_top, n = 0.0, 0.0, 0
        for x, y in trl:
            x, y = x.to(cihaz), y.to(cihaz)
            koord, hm = model(x)
            kayip, koord_kayip = kayip_fn(koord, hm, y)
            opt.zero_grad()
            kayip.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            plan.step()
            toplam += kayip.item(); koord_top += koord_kayip.item(); n += 1

        if ep % 5 == 0 or ep == 1 or ep == a.epochs:
            med, ort = dogrula(model, val, cihaz)
            yildiz = ""
            if med < en_iyi:
                en_iyi = med
                torch.save({"model": model.state_dict(), "epoch": ep,
                            "val_medyan": med}, CIKTI / "best.pt")
                yildiz = "  *"
            print(f"{ep:6d} {toplam/n:9.4f} {koord_top/n:8.3f} "
                  f"{med:12.3f} {ort:9.3f}{yildiz}")

    d = torch.load(CIKTI / "best.pt", weights_only=False)
    model.load_state_dict(d["model"])
    for ad, yl in (("val", val), ("test", tel)):
        med, ort = dogrula(model, yl, cihaz)
        print(f"\nEN IYI ({d['epoch']}. epoch) {ad}: medyan {med:.3f} px   ort {ort:.3f} px")


if __name__ == "__main__":
    main()
