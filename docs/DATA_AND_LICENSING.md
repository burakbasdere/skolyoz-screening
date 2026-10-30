# Data, privacy and licensing

> This page records what is known and what must be decided before the project
> is published. It is engineering notes, not legal advice.

## 1. Patient data never enters the repository

Back photographs are personal health data. `.gitignore` excludes `data/`,
`datasets/`, all raster image types and DICOM files, `runs/`, and every model
format (`*.pt`, `*.pth`, `*.onnx`, `*.tflite`) and package (`*.apk`, `*.aab`).
Documentation figures are the one exception: PNGs under `docs/` are allowed, so
**check any image you add there contains no real person**.

If a patient image is ever committed, treat it as a leak: deleting it in a later
commit does not remove it from history. Rewrite history and rotate any links.
See [SECURITY.md](../SECURITY.md).

## 2. Training data provenance

- Source: a Roboflow export (`education-ovbua/bladeco v1`), 1,119 files. The
  images are web-sourced: clinical photographs, stock images and greyscale
  publication figures ([tr/TEKNIK_DOKUMAN.md §7](tr/TEKNIK_DOKUMAN.md)).
- **The licence of the individual images is not recorded anywhere in the
  project.** Do not redistribute the images, and assume any weights trained on
  them may inherit restrictions until that is clarified.
- The original photographs are gone; only the stretched 640 × 640 versions
  remain, so provenance cannot be reconstructed from the files.

## 3. Third-party components

| Component | Where | Licence (verify against the upstream before relying on it) |
|---|---|---|
| Ultralytics YOLO11 (code **and** pretrained weights) | Stage-1 training, many `scripts/`, `skolyoz_metrics bench arch` | **AGPL-3.0**; an Ultralytics Enterprise licence exists for closed use |
| PyTorch, torchvision | Stage-2 training and benchmarks | BSD-style |
| MobileNetV3-small ImageNet weights (torchvision) | Stage-2 initialisation | Code BSD-style; the weights were trained on ImageNet, whose terms restrict commercial use — check |
| TensorFlow Lite, AndroidX (Compose, CameraX, …), Kotlin | Android app | Apache-2.0 |
| onnx2tf | Export | MIT |
| ai-edge-litert | TFLite benchmark | Apache-2.0 |

## 4. The AGPL question — decide before publishing

The fine-tuned Stage-1 model is derived from Ultralytics YOLO11, and the Android
app embeds it. Under Ultralytics' terms that makes the model, and an app that
ships it, subject to AGPL-3.0: users who receive the app must be able to obtain
the corresponding source. The research scripts that `import ultralytics` are in
the same position.

| Option | Consequence |
|---|---|
| **Release everything under AGPL-3.0** | Simplest and consistent. Anyone may use and modify it, but derivative works (including network services) must stay open. Fits an open research project. |
| Split: permissive licence for original code (`android/` sources, `skolyoz_metrics/`, docs), AGPL for anything importing Ultralytics and for the weights | More precise, but the shipped APK still contains the AGPL weights, so the *distributed app* is AGPL anyway. |
| Buy an Ultralytics Enterprise licence, or retrain Stage 1 with a permissively licensed detector | Frees the licence, at a cost or an engineering effort. Stage 1 would need re-evaluation. |

**Recommendation:** release under AGPL-3.0 unless a commercial product is
planned, and add the `LICENSE` file in the same commit that publishes the
repository. No licence has been chosen yet; until one is, the code is "all rights
reserved" and others cannot legally reuse it.

## 5. Weights and APKs

The earlier repository committed a debug APK and an unsigned release APK.
Those contain `asama1.tflite` and `asama2.tflite`, so the weights were
effectively published although the README said they were not. This repository
leaves binaries out. If you want people to try the app, attach a build to a
GitHub Release **after** settling sections 2 and 4, and record its SHA-256 in
`metrics/results.json` (`artifacts`).

## 6. Regulation

Software intended for a medical purpose can be a regulated medical device (for
example under the EU MDR, FDA rules or the Turkish TİTCK framework) even when it
is free. This project positions itself as research software and a screening aid
and says so everywhere the user sees output. If you distribute it to the public
for health use, get regulatory advice first.
