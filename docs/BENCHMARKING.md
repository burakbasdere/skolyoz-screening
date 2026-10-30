# Benchmarking

Latency is measured **without any patient image and without trained weights**.
A fixed-shape network takes the same time whatever its weights are, so the real
architectures are timed on random tensors. That is what makes the numbers
reproducible by anyone, on any machine.

## What exists

| Command | Times | Needs |
|---|---|---|
| `python -m skolyoz_metrics bench tflite --stage1 asama1.tflite --stage2 asama2.tflite` | The **deployed** `.tflite` files through the TFLite runtime (XNNPACK, 4 threads by default — the app uses 4). The most representative desktop number. | a TFLite runtime (`ai-edge-litert`) and the two model files |
| `python -m skolyoz_metrics bench arch` | The two PyTorch architectures with random weights, on CPU and (if present) CUDA; also records parameters and GFLOPs. Needs no model files at all, so CI can run it. | `requirements/bench.txt` |
| `./gradlew :app:testDebugUnitTest --tests '*PerformansTest*' -i` | Kotlin pre-processing (downscale, crops, tensors, soft-argmax) on a desktop JVM. | JDK 17 |

Each `bench` run writes `metrics/benchmarks/<date>_<kind>_<cpu>.json`
(validated by `metrics/benchmark.schema.json`) with the CPU, OS, library
versions, thread count, warm-up and repeat counts, and the full latency summary
(n, min, mean, std, p50, p90, p95, p99, max). Then:

```bash
python -m skolyoz_metrics render    # refreshes docs/METRICS.md and the README tables
python -m skolyoz_metrics check     # what CI runs
```

The README headline shows the **newest** `tflite` run; older runs stay in the
folder as history. Commit a new file rather than editing an old one.

## What a "pipeline" number means

`pipeline` = one Stage 1 call + **seven** sequential Stage 2 calls at batch
size 1, mirroring `OlcumZinciri.kt`, which refines every landmark separately.
(`scripts/31` still prints a "~100–200 ms/device" target that was written when
only two points were refined.) A batched Stage 2 might be faster; the app does
not do that today.

## What is *not* measured

- **Android latency.** No benchmark here ran on a phone. Desktop CPUs and the
  Python TFLite wrapper are not a prediction of a low-end ARM device. This is the
  first entry in the [known gaps](METRICS.md#8-known-gaps).
- **Everything around the model:** JPEG decoding, resizing, drawing the overlay,
  IMU interpolation. Only the JVM resize/crop/tensor part is covered, separately.
- **Memory, energy, heat.** No peak RAM, battery or thermal-throttling data.
- **Accelerators.** No GPU or NNAPI delegate was tried.

## Reading the numbers honestly

- Use **p50 for typical behaviour and p90/p99 for the tail**; never the minimum.
- Expect run-to-run noise on a laptop. Two consecutive PyTorch-CPU runs on the
  development machine differed by about 13 % at the median for Stage 1
  (96.8 ms vs 85.4 ms). Compare runs on one machine, and treat gaps smaller
  than that as noise. GitHub-hosted CI runners are noisier still, which is why CI
  runs the benchmark as a smoke test and uploads the result as an artifact
  instead of committing it.
- Eager PyTorch is slower than TFLite on the same CPU (see METRICS §3.1);
  quote the TFLite number when talking about the product.

## Protocol for an on-device run (not yet executed)

When someone measures on a phone, record at least:

1. Device model, SoC, RAM, Android version, battery level, thermal state.
2. A **release** build (the debug build is universal and unoptimised).
3. Airplane mode, screen on, charger unplugged, device at rest for a few minutes.
4. 10 warm-up inferences, then ≥ 100 timed ones; report p50 / p90 / p99 of
   Stage 1, Stage 2 (per call) and the whole chain.
5. Peak RSS during a measurement, and the p50 after 5 minutes of repeated runs to
   expose throttling.

Until an automated harness exists, put such results under `device_reported` in
`metrics/results.json` with `status: reported` and the exact device — and close
the `on_device_inference_latency` gap only when that entry exists.
