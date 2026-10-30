## What and why

<!-- One or two sentences. -->

## Checklist

- [ ] No patient images, labels derived from them, or trained weights are included
- [ ] `python -m skolyoz_metrics check`, `pytest` and `ruff check .` pass
- [ ] If a number changed: it is edited in `metrics/results.json` (with `status` and `source`) or in a new `metrics/benchmarks/` file, and `python -m skolyoz_metrics render` output is committed
- [ ] If measurement maths changed: Python reference, golden CSVs and Kotlin port all updated; Gradle tests pass
- [ ] The change does not introduce a Cobb-angle claim in UI or docs

## Metrics affected

<!-- Which metric ids / sections? "None" is fine. -->
