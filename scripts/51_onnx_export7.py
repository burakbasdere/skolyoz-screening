"""7 Landmark Modellerini ONNX ve TFLite'a Dönüştürür.

YOLO11n-pose (7 landmark: baseline7) ve Rafinaj7 (4 tip heatmap: rafinaj7)
modellerini ONNX formatına aktarır.
"""

import shutil
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
Rafinaj7 = import_module("39_rafinaj7_model").Rafinaj7

KOK = Path(__file__).resolve().parent.parent
YOLO_AGIRLIK = KOK / "runs" / "baseline7" / "weights" / "best.pt"
RAF_AGIRLIK = KOK / "runs" / "rafinaj7" / "best.pt"
CIKTI = KOK / "runs" / "export7"

OPSET = 13
KIRPIM = 128


class Rafinaj7Koordinat(nn.Module):
    def __init__(self, ag):
        super().__init__()
        self.ag = ag

    def forward(self, x):
        koord, _ = self.ag(x)
        return koord


class Rafinaj7Heatmap(nn.Module):
    def __init__(self, ag):
        super().__init__()
        self.ag = ag

    def forward(self, x):
        return self.ag.head(self.ag.govde(x))


def export7():
    CIKTI.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    print("=" * 70)
    print("ASAMA 1 (7 Landmark) - YOLO11n-pose -> ONNX")
    print("=" * 70)
    m = YOLO(str(YOLO_AGIRLIK))
    yol = m.export(format="onnx", imgsz=640, opset=OPSET, simplify=True,
                   nms=False, dynamic=False, batch=1, device="cpu")
    hedef1 = CIKTI / "asama1_7.onnx"
    shutil.move(str(yol), hedef1)
    print(f"  -> {hedef1}  ({hedef1.stat().st_size / 1e6:.2f} MB)")

    print()
    print("=" * 70)
    print("ASAMA 2 (7 Landmark / 4 Tip) - Rafinaj7 -> ONNX")
    print("=" * 70)
    d = torch.load(RAF_AGIRLIK, weights_only=False, map_location="cpu")
    ag = Rafinaj7(pretrained=False)
    ag.load_state_dict(d["model"])
    ag.eval()
    print(f"  agirlik: epoch {d['epoch']}, val medyan {d['val_medyan']:.3f} px")

    ornek = torch.randn(1, 3, KIRPIM, KIRPIM)
    for ad, sarmal, cikti_ad in (
        ("asama2_7_koord", Rafinaj7Koordinat(ag), "koordinat"),
        ("asama2_7_heatmap", Rafinaj7Heatmap(ag), "heatmap"),
    ):
        sarmal.eval()
        hedef2 = CIKTI / f"{ad}.onnx"
        torch.onnx.export(
            sarmal, (ornek,), str(hedef2),
            input_names=["girdi"], output_names=[cikti_ad],
            opset_version=OPSET, dynamo=False,
            do_constant_folding=True,
        )
        print(f"  -> {hedef2}  ({hedef2.stat().st_size / 1e6:.2f} MB)")

    print(f"\nONNX Dönüşümleri Tamamlandı: {CIKTI}")


if __name__ == "__main__":
    export7()
