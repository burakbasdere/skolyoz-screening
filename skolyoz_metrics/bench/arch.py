"""PyTorch benchmark of the two network architectures, with random weights.

The latency of a fixed-shape network does not depend on its weight values, so
the real architectures are timed on synthetic tensors. No trained weights and
no patient images are needed, which makes the numbers reproducible by anyone.

What is timed is the raw model forward pass in eager PyTorch: no image
decoding, no resizing, no NMS, no Python-side post-processing. It is an upper
bound on what a tuned runtime achieves and NOT a prediction of Android speed;
for the deployed format use ``tflite`` (and ultimately a real device).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from ..stats import summarize
from . import N_LANDMARKS, STAGE1_INPUT, STAGE2_INPUT
from .timing import time_calls


def load_stage2_class(root: Path):
    """Import ``Rafinaj7`` from the research scripts (file name starts with a digit)."""
    path = root / "scripts" / "39_rafinaj7_model.py"
    spec = importlib.util.spec_from_file_location("rafinaj7_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Rafinaj7


def build_stage1():
    """YOLO11n-pose head-configured for the 7 project landmarks, untrained."""
    from ultralytics.nn.tasks import PoseModel

    return PoseModel(
        "yolo11n-pose.yaml", ch=3, nc=1, data_kpt_shape=(N_LANDMARKS, 3), verbose=False
    )


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_gflops(model, example) -> float:
    """Forward FLOPs (2 x multiply-accumulates) from torch's FlopCounterMode.

    Uses ``no_grad`` rather than ``inference_mode``: under inference mode the
    counter silently records zero operations.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(example)
    return counter.get_total_flops() / 1e9


def run(root: Path, *, devices: list[str], threads: int, warmup: int, repeat: int) -> dict:
    import torch

    torch.set_num_threads(threads)
    stage1 = build_stage1().eval()
    stage2 = load_stage2_class(root)(pretrained=False).eval()

    probe1, probe2 = torch.rand(STAGE1_INPUT), torch.rand(STAGE2_INPUT)
    params1, params2 = count_params(stage1), count_params(stage2)
    gflops1, gflops2 = count_gflops(stage1, probe1), count_gflops(stage2, probe2)

    results: list[dict] = []
    for device in devices:
        s1, s2 = stage1.to(device), stage2.to(device)
        x1, x2 = probe1.to(device), probe2.to(device)
        sync = torch.cuda.synchronize if device.startswith("cuda") else None

        def pipeline(s1=s1, s2=s2, x1=x1, x2=x2):
            s1(x1)
            for _ in range(N_LANDMARKS):
                s2(x2)

        cases = [
            ("stage1", lambda s1=s1, x1=x1: s1(x1), STAGE1_INPUT, params1, gflops1),
            ("stage2", lambda s2=s2, x2=x2: s2(x2), STAGE2_INPUT, params2, gflops2),
            (
                "pipeline",
                pipeline,
                None,
                params1 + params2,
                gflops1 + N_LANDMARKS * gflops2,
            ),
        ]
        with torch.inference_mode():
            for name, fn, shape, params, gflops in cases:
                samples = time_calls(fn, warmup=warmup, repeat=repeat, sync=sync)
                results.append(
                    {
                        "name": name,
                        "device": device,
                        "input_shape": list(shape) if shape else None,
                        "params": params,
                        "gflops": round(gflops, 4),
                        "latency_ms": summarize(samples),
                    }
                )
    return {
        "results": results,
        "notes": [
            "Rastgele ağırlık, sentetik girdi; yalnızca eager PyTorch ileri geçişi "
            "(ön/son işleme ve NMS yok).",
            f"pipeline = 1 x stage1 + {N_LANDMARKS} x stage2; OlcumZinciri.kt'yi yansıtacak "
            "şekilde batch boyutu 1 ile sırayla çalıştırılır.",
            "params her öğrenilebilir parametreyi sayar; gflops = 2 x çarpma-toplama sayısı.",
        ],
    }
