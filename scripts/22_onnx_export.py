"""Faz 1 adim 1 - Her iki asamayi ONNX'e cevirir.

NEDEN AYRI ADIM
---------------
TFLite'a dogrudan gitmek yerine once ONNX uretiyoruz, cunku:
  - ONNX asamasinda cikti PyTorch ile birebir karsilastirilabilir
    (onnxruntime ile). Dogruluk kaybi olursa hangi adimda oldugu belli olur:
    torch -> onnx mi, onnx -> tflite mi?
  - ONNX -> TFLite donusumu (onnx2tf + tensorflow) torch ile ayni ortamda
    kurulunca bagimlilik catismasi cikariyor. Ayri venv'de yapiyoruz;
    o venv'in torch'a ihtiyaci olmasin diye ara format ONNX.

ASAMA 2 ICIN IKI VARYANT
------------------------
  a) tam    : soft-argmax dahil, cikti (1,2) koordinat
  b) heatmap: cikti (1,1,64,64) ham heatmap, argmax cagirana birakilir

MOBIL_YAPILACAKLAR.md Faz 1 sunu soyluyordu: soft-argmax donusumu
desteklenmezse heatmap'i disari ver. Ikisini de uretip TFLite tarafinda
hangisinin tuttuguna ve hangisinin daha az quantization kaybi verdigine
bakacagiz.
"""

import shutil
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
RafinajAgi = import_module("17_rafinaj_model").RafinajAgi

KOK = Path(__file__).resolve().parent.parent
YOLO_AGIRLIK = KOK / "runs" / "baseline" / "weights" / "best.pt"
RAF_AGIRLIK = KOK / "runs" / "rafinaj" / "best.pt"
CIKTI = KOK / "runs" / "export"

OPSET = 13          # onnx2tf'in en genis destekledigi araliktan
KIRPIM = 128


class RafinajKoordinat(nn.Module):
    """Sadece koordinat donduren sarmalayici (heatmap ciktisi atilir).

    Cok ciktili modelde onnx2tf bazen cikti sirasini degistiriyor; tek cikti
    birakmak mobil taraftaki indeks hatasi riskini sifirliyor.
    """

    def __init__(self, ag):
        super().__init__()
        self.ag = ag

    def forward(self, x):
        koord, _ = self.ag(x)
        return koord


class RafinajHeatmap(nn.Module):
    """Sadece ham heatmap donduren sarmalayici (soft-argmax disarida)."""

    def __init__(self, ag):
        super().__init__()
        self.ag = ag

    def forward(self, x):
        return self.ag.head(self.ag.govde(x))


def asama1_export():
    """YOLO11n-pose -> ONNX. Ultralytics kendi export'unu kullaniyor."""
    from ultralytics import YOLO

    print("=" * 70)
    print("ASAMA 1 - YOLO11n-pose -> ONNX")
    print("=" * 70)
    m = YOLO(str(YOLO_AGIRLIK))
    yol = m.export(format="onnx", imgsz=640, opset=OPSET, simplify=True,
                   nms=False, dynamic=False, batch=1, device="cpu")
    hedef = CIKTI / "asama1.onnx"
    shutil.move(str(yol), hedef)
    print(f"  -> {hedef}  ({hedef.stat().st_size / 1e6:.2f} MB)")
    return hedef


def asama2_export():
    """Rafinaj agi -> ONNX (iki varyant)."""
    print()
    print("=" * 70)
    print("ASAMA 2 - Rafinaj agi -> ONNX")
    print("=" * 70)
    d = torch.load(RAF_AGIRLIK, weights_only=False, map_location="cpu")
    ag = RafinajAgi(pretrained=False)
    ag.load_state_dict(d["model"])
    ag.eval()
    print(f"  agirlik: epoch {d['epoch']}, val medyan {d['val_medyan']:.3f} px")

    ornek = torch.randn(1, 3, KIRPIM, KIRPIM)
    yollar = {}
    for ad, sarmal, cikti_ad in (
        ("asama2_koord", RafinajKoordinat(ag), "koordinat"),
        ("asama2_heatmap", RafinajHeatmap(ag), "heatmap"),
    ):
        sarmal.eval()
        hedef = CIKTI / f"{ad}.onnx"
        torch.onnx.export(
            sarmal, (ornek,), str(hedef),
            input_names=["girdi"], output_names=[cikti_ad],
            opset_version=OPSET, dynamo=False,
            do_constant_folding=True,
        )
        print(f"  -> {hedef}  ({hedef.stat().st_size / 1e6:.2f} MB)")
        yollar[ad] = hedef
    return ag, yollar


def onnx_dogrula(ag, yollar):
    """ONNX ciktisini PyTorch ciktisiyla karsilastir.

    Bu adim atlanirsa, ileride TFLite'ta hata gorunce sucu quantization'a
    atariz; oysa hata torch->onnx adiminda olusmus olabilir.
    """
    import onnxruntime as ort

    print()
    print("=" * 70)
    print("DOGRULAMA - ONNX vs PyTorch (20 rastgele girdi)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 3, KIRPIM, KIRPIM), dtype=np.float32)

    with torch.no_grad():
        t_koord, t_hm = ag(torch.from_numpy(x))
    t_koord = t_koord.numpy()
    t_hm = t_hm.numpy()

    for ad, gercek in (("asama2_koord", t_koord), ("asama2_heatmap", t_hm)):
        s = ort.InferenceSession(str(yollar[ad]), providers=["CPUExecutionProvider"])
        ad_girdi = s.get_inputs()[0].name
        cikti = np.concatenate([s.run(None, {ad_girdi: x[i:i + 1]})[0]
                                for i in range(len(x))], axis=0)
        fark = np.abs(cikti - gercek)
        birim = "px" if ad.endswith("koord") else "birim"
        print(f"  {ad:<16} max fark {fark.max():.3e} {birim}   "
              f"ort {fark.mean():.3e}")
        if ad.endswith("koord") and fark.max() > 1e-3:
            print("    !! koordinat farki 1e-3 px'i asti, incele")

    s = ort.InferenceSession(str(CIKTI / "asama1.onnx"),
                             providers=["CPUExecutionProvider"])
    g = s.get_inputs()[0]
    c = s.get_outputs()[0]
    print(f"\n  asama1  girdi  {g.name} {g.shape} {g.type}")
    print(f"          cikti  {c.name} {c.shape}")
    print("          kanal duzeni: [x, y, w, h, conf, kx0, ky0, kv0, kx1, ky1, kv1]")


def main():
    CIKTI.mkdir(parents=True, exist_ok=True)
    asama1_export()
    ag, yollar = asama2_export()
    onnx_dogrula(ag, yollar)
    print(f"\nTum ciktilar: {CIKTI}")


if __name__ == "__main__":
    main()
