# Security and privacy policy

## Reporting

Please report vulnerabilities, and especially **any personal health data found
in the repository or its history**, privately — use GitHub's "Report a
vulnerability" (private advisory) on this repository, not a public issue.
Include what you found and where, but do not paste the data itself.

You can expect an acknowledgement within a few days. This is a small research
project, so there is no bug bounty and no formal SLA.

## Scope

- The Android app and everything under `android/`.
- The Python tooling under `skolyoz_metrics/`.
- Accidentally published patient images, labels or model weights.

## Design choices that matter for privacy

- Inference is fully on-device; the app requests no `INTERNET` permission.
- Backups are disabled (`allowBackup="false"`).
- Debug builds write diagnostic images to storage. Do **not** run a debug build
  on real subjects.
- Image files and model weights are excluded by `.gitignore`.

## Medical disclaimer

This is research software and a screening aid. It is not a diagnostic tool and
has not been validated on patients or cleared as a medical device.
