# Model card

Numbers are not repeated here; see [METRICS.md](METRICS.md). Every claim below
is limited to what that page and the research notes support.

## Models

| | Stage 1 | Stage 2 |
|---|---|---|
| Role | Coarse localisation of 7 landmarks | Sub-pixel refinement of each landmark |
| Architecture | Ultralytics YOLO11n-pose, fine-tuned (7 keypoints) | MobileNetV3-small trunk + transposed-convolution head, heatmap + soft-argmax |
| Input | 640 × 640 RGB, `/255` | 128 × 128 RGB crop, ImageNet mean/std |
| Shipped as | TFLite float16 | TFLite float16 |
| Weights | not distributed (see [DATA_AND_LICENSING.md](DATA_AND_LICENSING.md)) | not distributed |

## Intended use

- **Screening aid for trunk asymmetry** in research, education and pilot studies:
  a person standing upright, photographed **from behind**, phone held upright, to
  suggest whether a referral to a clinician may be worthwhile.
- Outputs are *indicators* (axillary tilt, coronal balance) shown with a
  mandatory "not a diagnostic tool" notice.

## Out of scope — do not use for

- **Diagnosis, severity grading or treatment decisions.**
- **Estimating a Cobb angle.** The project has no radiograph-paired data. The
  Cobb figure the Android overlay currently shows is an uncalibrated
  placeholder (see the `cobb_estimate` gap).
- Front or side views, the forward-bend (Adams) test, video, or a front-facing camera
  (it mirrors the image, which swaps left and right).
- Any population that may be absent from the training data. Its age, sex and skin-tone
  composition is undocumented (see below), so no population is known to be covered.

## Training and evaluation data

- 337 cleaned images from a web-sourced Roboflow export (1,119 files →
  441 unique → 337 usable), resized to 640 × 640 *without* preserving aspect
  ratio. The mix includes clinical photographs, stock images and greyscale
  figures from publications. The originals are unavailable.
- Duplicates (rotated copies of one photograph) and left/right label conflicts
  were removed or repaired; independent labels were averaged to cut label noise.
- Evaluation: group-aware 5-fold cross-validation of Stage 2 with simulated
  coarse seeds, plus a single held-out split of 50 images. Both protocols and
  their limits are in METRICS §1.

## Known limitations

- **Domain gap.** The target input — colour phone photographs under the intended
  capture protocol — is not what the model was trained or evaluated on. In an
  informal check, three consecutive shots of one scene gave widely different
  angles, which is why a confidence gate exists.
- **No end-to-end or on-device accuracy** for the 7-landmark system.
- **Spine landmarks are weak** and no curve profile is delivered; the limit is
  label noise, shown by measurement.
- **Small data.** Spine labels exist for 182 images; screening-decision
  statistics rest on 4–16 positive cases.
- **No subgroup analysis.** The sources record no demographics, skin tone,
  body-habitus or age information, so performance across such groups is
  unknown. Treat any claim of fairness as unsupported.

## Risks and mitigations

| Risk | Mitigation in the project | Still open |
|---|---|---|
| False reassurance ("looks normal") delaying care | Mandatory notice; recall treated as the priority metric | Not validated on patients; recall CI is wide |
| False alarms | Output labelled as screening indicator with a position in the dataset distribution, never a clinical grade | Prevalence in intended population unknown |
| Silently wrong angle from an unstable heatmap | Two-criterion confidence gate rejects the shot | Gate not yet tested on a real capture campaign |
| Wrong angle from a tilted phone | IMU roll correction, pitch gate at 8° | Depends on sensor quality |
| Exposure of health data | On-device inference, no `INTERNET` permission, backups off, `.gitignore` for images | Debug builds write diagnostic images to disk (`BuildConfig.DEBUG` only) — never use a debug build with real subjects |
| Over-claiming | Generated metrics page with statuses and a gaps register | — |
