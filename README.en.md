# Skolyoz — radiation-free back-asymmetry screening on a phone

**A single back photograph, analysed entirely on the phone, gives a screening
indicator for trunk asymmetry.** Seven anatomical landmarks are detected; the
**axillary tilt** and the **coronal balance** are computed from them, and the
phone's own IMU removes the error caused by holding the camera crooked.

🇹🇷 [Türkçe README](README.tr.md)

![Android](https://img.shields.io/badge/Android-min%20SDK%2026-3DDC84)
![TFLite](https://img.shields.io/badge/TFLite-float16-FF6F00)
![On-device](https://img.shields.io/badge/inference-on--device-0A7EA4)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)

> [!WARNING]
> **This is research software and a screening aid, not a diagnostic tool.** It
> measures *surface asymmetry*, which is **not** the radiographic Cobb angle,
> and it has not been validated on patients or cleared as a medical device.
> Do not use it to make clinical decisions.

## Headline metrics

<!-- headline:begin -->
| Metric | Value | Evidence |
|---|---|---|
| Axillary-tilt error, median / p90 | 0.271° / 1.237° | reported |
| Coronal-balance error, median / p90 | 1.292° / 3.385° | reported |
| Axillary landmark error, median (label noise floor) | 3.045 px (2.45 px) | reported |
| Screening recall at ≥ 3° tilt (held-out split, n = 50) | 93.8 % (15/16; 95 % CI 71.7–98.9 %) | reported |
| Model size, both stages, float16 | 6.44 MB | measured |
| Parameters, Stage 1 / Stage 2 | 2.69 M / 0.41 M | measured |
| Model forward, 1 × Stage 1 + 7 × Stage 2 (TFLite, CPU, median / p90) | 70.8 ms / 75.7 ms — 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz | measured |
| Kotlin ↔ Python parity (golden data) | 42 JVM + 4 device tests; reported diff 0.000 px / 0.0000° | reported |
| Open gaps | 5 listed — see [docs/METRICS.md](docs/METRICS.md#8-known-gaps) | — |
<!-- headline:end -->

Read [docs/METRICS.md](docs/METRICS.md) before quoting any of this. It states
what each number means, how it was obtained, how uncertain it is, and — in the
*Known gaps* section — what has **not** been measured. In short: the accuracy
figures are offline experiments on a small web-sourced image set (337 images),
the on-device accuracy and phone-photo accuracy are open, and the Cobb estimate
shown in the app is unvalidated.

Every number in the table above is generated from
[`metrics/results.json`](metrics/results.json) and
[`metrics/benchmarks/`](metrics/benchmarks); CI fails if they drift apart.

## How it works

```
Photo (CameraX) + simultaneous IMU gravity sample
   │  resize to 640×640 (training-time stretch, reproduced exactly)
   ▼
Stage 1 — YOLO11n-pose, fine-tuned: coarse position of 7 landmarks
   ▼
Stage 2 — MobileNetV3-small + heatmap + soft-argmax, once per landmark:
          128×128 crop around the coarse point → sub-pixel position
   ▼
Map back to the original resolution; measurement in pure geometry
   ├─ axillary tilt  = atan2(Δy, |Δx|) of the two armpit points − IMU roll
   └─ coronal balance = lateral offset of C7 from the plumb line through the sacrum
   ▼
Gates: |pitch| > 8° rejects the shot · low-confidence heatmaps are rejected
```

Why two stages: a single network regressing coordinates directly plateaued at
the same ≈24 px error on training, validation and test data, so the limit was
the representation, not the amount of data. Details and the measured effect:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Technology

| Layer | Stack |
|---|---|
| Mobile app | Kotlin 2.0, Jetpack Compose + Material 3, CameraX 1.4, TensorFlow Lite 2.16 (float16, 4 threads), Android min SDK 26 |
| Models | Ultralytics YOLO11n-pose (Stage 1), torchvision MobileNetV3-small + custom heatmap head (Stage 2), trained in PyTorch |
| Export | PyTorch → ONNX → `onnx2tf` → TFLite float16 |
| Metrics & benchmarks | Python package [`skolyoz_metrics`](skolyoz_metrics): JSON-schema-checked results, generated docs, data-free latency benchmarks |
| Quality gates | GitHub Actions: Ruff, pytest (incl. Python↔Kotlin golden-data parity), metrics drift check, Gradle unit tests |

Versions and where each is pinned: [docs/TECH_STACK.md](docs/TECH_STACK.md).

## Quick start

**Check the metrics and run the tests** — no GPU, no private data:

```bash
pip install -r requirements/dev.txt
python -m skolyoz_metrics check     # schema + consistency + generated docs in sync
pytest
ruff check .
```

**Benchmark the model architectures** — random weights, synthetic input:

```bash
pip install -r requirements/bench.txt
python -m skolyoz_metrics bench arch                                   # PyTorch
python -m skolyoz_metrics bench tflite --stage1 asama1.tflite --stage2 asama2.tflite
python -m skolyoz_metrics render                                       # refresh docs
```

See [docs/BENCHMARKING.md](docs/BENCHMARKING.md) for what is and is not measured.

**Build the Android app** (JDK 17; set `JAVA_HOME` to Android Studio's JBR):

```bash
cd android
./gradlew :app:testDebugUnitTest     # JVM tests, no device and no model files needed
./gradlew :app:assembleDebug
```

The trained model files are **not** distributed (see
[docs/DATA_AND_LICENSING.md](docs/DATA_AND_LICENSING.md)). To run the app, put
`asama1.tflite` and `asama2.tflite` in `android/app/src/main/assets/`.

**Reproduce the research** (`scripts/00`–`52`) needs the private image set and
a CUDA GPU; see [requirements/train.txt](requirements/train.txt) and the
Turkish research notes in [docs/tr/](docs/tr).

## Repository layout

```
android/            Kotlin app (+ golden-data unit tests)
scripts/            numbered research pipeline, 00 → 52 (frozen record of the experiments)
skolyoz_metrics/    metrics tooling and benchmarks (the maintained Python code)
metrics/            results.json (+ schemas) and recorded benchmark runs
tests/              pytest suite
docs/               METRICS.md (generated), ARCHITECTURE, TECH_STACK, BENCHMARKING, MODEL_CARD, ...
docs/tr/            original Turkish technical report and working notes
requirements/       dependency sets: dev, bench, train, export
.github/            CI workflows, Dependabot, issue / PR templates
```

## Known limitations

1. **Domain gap.** Training images are web-sourced; phone photos come from a
   different distribution. Three consecutive shots of one scene gave 1.25°,
   3.37° and 6.73°. A confidence gate now rejects unstable measurements instead
   of showing a wrong angle, but it does not fix the cause — phone-captured
   training data does.
2. **No spine-curve profile.** The surface midline deviation (≈1.4 % of trunk
   height) is below the label noise of the data; this was shown by measurement,
   not assumed.
3. **No Cobb validation.** There is no radiograph-paired data, so nothing here
   estimates a Cobb angle. The app's Cobb estimate uses uncalibrated
   coefficients — see the `cobb_estimate` entry in
   [docs/METRICS.md](docs/METRICS.md#8-known-gaps).

## Privacy, data and licence

Inference runs on the device; the app's manifest requests only the `CAMERA`
permission (no `INTERNET`) and disables backups. Patient photographs are never
committed (`.gitignore`). Training data provenance, third-party licences and the
AGPL-3.0 implications of Ultralytics are in
[docs/DATA_AND_LICENSING.md](docs/DATA_AND_LICENSING.md).

**Licence: not yet chosen.** Until a `LICENSE` file is added the code is "all
rights reserved". The choice is constrained by the Ultralytics dependency — see
that document.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The one rule that matters: a change that
touches a number must go through `metrics/results.json` with a status and a
source.
