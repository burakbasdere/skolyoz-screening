"""Latency benchmark of the deployed ``.tflite`` models on synthetic input.

This is the representative benchmark: it times the same flatbuffers the
Android app ships, through the same TFLite kernels (XNNPACK on float models).
The input is random, so no patient image is involved. The numbers still come
from a desktop/CI CPU - only a run on a physical phone predicts phone latency.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

from ..stats import summarize
from . import N_LANDMARKS
from .timing import time_calls

_INTERPRETER_MODULES = ("ai_edge_litert.interpreter", "tflite_runtime.interpreter")


def load_interpreter_class():
    """First available TFLite Interpreter: ai-edge-litert, tflite-runtime or TensorFlow."""
    for name in _INTERPRETER_MODULES:
        try:
            return importlib.import_module(name).Interpreter
        except ImportError:
            continue
    try:
        import tensorflow as tf

        return tf.lite.Interpreter
    except ImportError as exc:
        raise RuntimeError(
            "no TFLite runtime found; install one with: pip install ai-edge-litert"
        ) from exc


def synthetic_input(shape, dtype, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.floating):
        return rng.random(tuple(int(d) for d in shape)).astype(dtype)
    return np.zeros(tuple(int(d) for d in shape), dtype=dtype)


class _Model:
    """One loaded interpreter with a fixed synthetic input."""

    def __init__(self, path: Path, interpreter_cls, threads: int):
        self.path = path
        self.interp = interpreter_cls(model_path=str(path), num_threads=threads)
        self.interp.allocate_tensors()
        spec = self.interp.get_input_details()[0]
        self._index = spec["index"]
        self.input_shape = [int(d) for d in spec["shape"]]
        self.input_dtype = str(spec["dtype"].__name__)
        self._x = synthetic_input(spec["shape"], spec["dtype"])

    def invoke(self) -> None:
        self.interp.set_tensor(self._index, self._x)
        self.interp.invoke()

    @property
    def output_shapes(self) -> list[list[int]]:
        return [[int(d) for d in o["shape"]] for o in self.interp.get_output_details()]

    def describe(self) -> dict:
        data = self.path.read_bytes()
        return {
            "model_file": self.path.name,
            "model_bytes": len(data),
            "model_sha256": hashlib.sha256(data).hexdigest(),
            "input_shape": self.input_shape,
            "input_dtype": self.input_dtype,
            "output_shapes": self.output_shapes,
        }


def run(
    stage1_path: Path,
    stage2_path: Path,
    *,
    threads: int,
    warmup: int,
    repeat: int,
    interpreter_cls=None,
) -> dict:
    cls = interpreter_cls or load_interpreter_class()
    stage1 = _Model(stage1_path, cls, threads)
    stage2 = _Model(stage2_path, cls, threads)

    def pipeline() -> None:
        stage1.invoke()
        for _ in range(N_LANDMARKS):
            stage2.invoke()

    results = []
    for name, fn, info in (
        ("stage1", stage1.invoke, stage1.describe()),
        ("stage2", stage2.invoke, stage2.describe()),
        ("pipeline", pipeline, None),
    ):
        samples = time_calls(fn, warmup=warmup, repeat=repeat)
        entry = {"name": name, "device": f"cpu-{threads}t", "latency_ms": summarize(samples)}
        if info:
            entry.update(info)
        results.append(entry)
    return {
        "results": results,
        "notes": [
            "Synthetic random input; Interpreter.invoke() plus set_tensor only "
            "(no image decoding, resizing or post-processing).",
            f"pipeline = 1 x stage1 + {N_LANDMARKS} x stage2, run sequentially, "
            "mirroring OlcumZinciri.kt.",
            "Measured on a desktop/CI CPU through the Python TFLite runtime; this is not "
            "Android latency.",
        ],
    }
