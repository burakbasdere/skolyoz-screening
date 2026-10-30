"""Çok tipli rafinaj ağını eğitir.

    python scripts/40_rafinaj7_egit.py --epochs 200 --batch 32

Veri: data/yolo7 etiketlerinden GÖRÜNÜR (v=2) noktalar. Her (görüntü, nokta)
çifti bir örnek; kırpım o noktanın etrafından JITTER'lı alınır (Aşama 1'in
hatasını taklit eder).

Kayıp yalnızca örneğin TİPİNE ait kanaldan alınır (bkz. 39). Böylece dört
tip tek gövdeyi paylaşır ama birbirinin heatmap'ini bozmaz.
"""

import argparse
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
_m = import_module("39_rafinaj7_model")
Rafinaj7, Rafinaj7Kayip = _m.Rafinaj7, _m.Rafinaj7Kayip
NOKTA_TIPI, TIP_ADI, TIP_SAYISI = _m.NOKTA_TIPI, _m.TIP_ADI, _m.TIP_SAYISI

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "yolo7"
CIKTI = KOK / "runs" / "rafinaj7"

KIRPIM = 128
JITTER = 35
ORT = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def etiket_oku(yol, W, H):
    """yolo7 etiket satiri -> {nokta_idx: (x, y)} (sadece v!=0)."""
    t = yol.read_text(encoding="utf-8").split()
    out = {}
    for i in range(7):
        x, y, v = float(t[5 + i * 3]), float(t[6 + i * 3]), float(t[7 + i * 3])
        if v != 0:
            out[i] = (x * W, y * H)
    return out


class KirpimVeri(Dataset):
    def __init__(self, split, egitim):
        self.ornekler = []
        img_dir = VERI / split / "images"
        lbl_dir = VERI / split / "labels"
        for p in sorted(img_dir.glob("*.jpg")):
            l = lbl_dir / (p.stem + ".txt")
            if not l.exists():
                continue
            W, H = Image.open(p).size
            for idx, (x, y) in etiket_oku(l, W, H).items():
                self.ornekler.append((p, (x, y), NOKTA_TIPI[idx]))
        self.egitim = egitim

    def tip_dagilimi(self):
        return Counter(TIP_ADI[t] for _, _, t in self.ornekler)

    def __len__(self):
        return len(self.ornekler)

    def __getitem__(self, i):
        yol, (gx, gy), tip = self.ornekler[i]
        im = Image.open(yol).convert("RGB")

        if self.egitim:
            jx = random.uniform(-JITTER, JITTER)
            jy = random.uniform(-JITTER, JITTER)
            olcek = random.uniform(0.85, 1.20)
        else:
            rng = random.Random(i * 7919)      # dogrulama tekrarlanabilir olsun
            jx = rng.uniform(-JITTER, JITTER)
            jy = rng.uniform(-JITTER, JITTER)
            olcek = 1.0

        pencere = KIRPIM * olcek
        sol = int(round(gx + jx - pencere / 2))
        ust = int(round(gy + jy - pencere / 2))
        kirpim = im.crop((sol, ust, sol + int(round(pencere)),
                          ust + int(round(pencere)))).resize((KIRPIM, KIRPIM),
                                                             Image.BILINEAR)
        hx = (gx - sol) * KIRPIM / pencere
        hy = (gy - ust) * KIRPIM / pencere

        if self.egitim:
            if random.random() < 0.5:
                kirpim = kirpim.transpose(Image.FLIP_LEFT_RIGHT)
                hx = KIRPIM - 1 - hx
            for Sinif, a, b in ((ImageEnhance.Brightness, 0.7, 1.3),
                                (ImageEnhance.Contrast, 0.7, 1.3),
                                (ImageEnhance.Color, 0.6, 1.4)):
                kirpim = Sinif(kirpim).enhance(random.uniform(a, b))

        arr = (np.asarray(kirpim, dtype=np.float32) / 255.0 - ORT) / STD
        return (torch.from_numpy(arr.transpose(2, 0, 1)),
                torch.tensor([hx, hy], dtype=torch.float32),
                torch.tensor(tip, dtype=torch.long))


def dogrula(model, yukleyici, cihaz):
    """Genel medyan + tip bazinda medyan piksel hatasi."""
    model.eval()
    hepsi, per_tip = [], defaultdict(list)
    with torch.no_grad():
        for x, y, t in yukleyici:
            x, y, t = x.to(cihaz), y.to(cihaz), t.to(cihaz)
            koord, _ = model(x)
            sec = koord[torch.arange(len(t), device=cihaz), t]
            d = torch.sqrt(((sec - y) ** 2).sum(dim=1))
            for v, tt in zip(d.cpu().tolist(), t.cpu().tolist()):
                hepsi.append(v)
                per_tip[tt].append(v)
    return statistics.median(hepsi), {TIP_ADI[k]: statistics.median(v)
                                      for k, v in sorted(per_tip.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()

    torch.manual_seed(1337); random.seed(1337); np.random.seed(1337)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    CIKTI.mkdir(parents=True, exist_ok=True)

    tr, va = KirpimVeri("train", True), KirpimVeri("val", False)
    print(f"ornek: train {len(tr)}  val {len(va)}")
    print(f"  train tip dagilimi: {dict(tr.tip_dagilimi())}")
    print(f"  val   tip dagilimi: {dict(va.tip_dagilimi())}")
    eksik = [ad for ad in TIP_ADI if ad not in tr.tip_dagilimi()]
    if eksik:
        print(f"  UYARI: su tipler icin egitim verisi YOK: {eksik}")
        print(f"         o kanallar ogrenilmez; etiketler gelince tekrar egit.")

    trl = DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=True)
    val = DataLoader(va, batch_size=64, shuffle=False)

    model = Rafinaj7().to(cihaz)
    kayip_fn = Rafinaj7Kayip().to(cihaz)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    plan = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(trl), pct_start=0.2)

    en_iyi = 1e9
    print(f"\n{'epoch':>6} {'kayip':>9} {'val medyan':>12}   tip bazinda")
    for ep in range(1, a.epochs + 1):
        model.train()
        top, n = 0.0, 0
        for x, y, t in trl:
            x, y, t = x.to(cihaz), y.to(cihaz), t.to(cihaz)
            koord, hm = model(x)
            kayip, _ = kayip_fn(koord, hm, y, t)
            opt.zero_grad(); kayip.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); plan.step()
            top += kayip.item(); n += 1

        if ep % 5 == 0 or ep == 1 or ep == a.epochs:
            med, per = dogrula(model, val, cihaz)
            yildiz = ""
            if med < en_iyi:
                en_iyi = med
                torch.save({"model": model.state_dict(), "epoch": ep,
                            "val_medyan": med, "tip": per}, CIKTI / "best.pt")
                yildiz = "  *"
            detay = "  ".join(f"{k} {v:.2f}" for k, v in per.items())
            print(f"{ep:6d} {top/n:9.4f} {med:12.3f}   {detay}{yildiz}")

    d = torch.load(CIKTI / "best.pt", weights_only=False)
    print(f"\nEN IYI: epoch {d['epoch']}  val medyan {d['val_medyan']:.3f} px")
    for k, v in d["tip"].items():
        print(f"  {k:<14} {v:.3f} px")


if __name__ == "__main__":
    main()
