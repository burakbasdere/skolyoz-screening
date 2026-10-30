# Technology stack

Versions are taken from the build files named in the last column, not from
memory. If a pin changes there, change it here (or better, change it only there).

## Android app

| Concern | Choice | Pinned in |
|---|---|---|
| Language | Kotlin 2.0.21, JVM target 17 | `android/gradle/libs.versions.toml`, `app/build.gradle.kts` |
| Build | Android Gradle Plugin 8.7.3, Gradle 8.10.2 (wrapper) | `libs.versions.toml`, `gradle-wrapper.properties` |
| SDK levels | min 26 (Android 8.0), compile / target 35 | `app/build.gradle.kts` |
| UI | Jetpack Compose (BOM 2024.12.01), Material 3, Activity-Compose 1.9.3, Lifecycle 2.8.7 | `libs.versions.toml` |
| Camera | CameraX 1.4.1 (`core`, `camera2`, `lifecycle`, `view`); zero-shutter-lag capture; portrait locked | `libs.versions.toml`, `CekimEkrani.kt` |
| Orientation sensing | `SensorManager` `TYPE_GRAVITY`, ring buffer + linear interpolation at the frame's hardware timestamp | `sensor/` |
| Inference | TensorFlow Lite 2.16.1, 4 CPU threads, models memory-mapped (`noCompress += tflite`) | `libs.versions.toml`, `OlcumZinciri.kt` |
| Release ABIs | `arm64-v8a`, `armeabi-v7a` (debug stays universal for emulators) | `app/build.gradle.kts` |
| Unit tests | JUnit 4.13.2; golden-data CSVs; a JVM pre-processing timing test | `app/src/test` |
| Device tests | `androidx.test` (ext-junit 1.2.1, runner 1.6.2) — run on a phone, not in CI | `app/src/androidTest` |
| Permissions | `CAMERA` only, no `INTERNET` (the merged manifest also carries AndroidX's app-private `DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION`); camera and accelerometer declared as required features; backups off | `AndroidManifest.xml` |

## Models and training

| Concern | Choice |
|---|---|
| Stage 1 | Ultralytics YOLO11n-pose, fine-tuned for 7 keypoints at 640 × 640 |
| Stage 2 | torchvision MobileNetV3-small (first 9 feature blocks, stride 16) + 3 transposed convolutions + a 64 × 64 heatmap per landmark type, soft-argmax readout |
| Framework | PyTorch (CUDA; RTX 50xx / Blackwell needs the `cu128` wheels) |
| Annotation | Roboflow (keypoint export), cleaned and de-duplicated by `scripts/00`–`12` |
| Validation | Group-aware 5-fold cross-validation (`scripts/45`, `46`) |
| Export | PyTorch → ONNX (`scripts/22`, `51`) → `onnx2tf` → TFLite float16 (`scripts/25`, `52`) |
| Quantisation | float16 shipped; full-integer INT8 was tried and fails for Stage 1 (see METRICS §2.1) |

Python dependency sets: [`requirements/`](../requirements) — `dev`, `bench`,
`train`, `export`. Export uses its own virtual environment because TensorFlow
and `onnx2tf` pin NumPy / protobuf versions that conflict with training.

## Metrics, benchmarks and quality gates

| Concern | Choice |
|---|---|
| Package | `skolyoz_metrics` — standard library only at runtime (`bench` extras bring torch, ultralytics, a TFLite runtime) |
| Source of truth | `metrics/results.json` validated by `metrics/results.schema.json`; benchmark runs by `metrics/benchmark.schema.json` |
| Generated docs | `docs/METRICS.md` and the headline table in both READMEs (`python -m skolyoz_metrics render`) |
| Statistics | linear-interpolation percentiles; Wilson score intervals for proportions |
| Lint / test | Ruff (E, F, I, UP, B) and pytest; research scripts are excluded from lint |
| CI | GitHub Actions: `ci.yml` (lint, tests, metrics check, benchmark smoke run), `android.yml` (wrapper validation, Gradle tests, debug build) |
| Dependency updates | Dependabot for Actions, pip and Gradle |

## Tooling the benchmark numbers were produced with

`torch 2.5.1`, `ultralytics 8.4.38`, `ai-edge-litert 2.3.0`, Python 3.11 on
Windows 11; the exact machine and thread settings are recorded inside every
file in `metrics/benchmarks/`.
