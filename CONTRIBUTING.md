# Contributing

Thanks for looking. This is research software with a medical flavour, so the
rules below exist to keep it honest and safe, not to slow you down.

## Ground rules

1. **No patient data, ever** — no photographs, no labels derived from them, no
   weights trained on them. See [docs/DATA_AND_LICENSING.md](docs/DATA_AND_LICENSING.md).
2. **Numbers live in `metrics/`.** If your change adds or alters a metric, edit
   `metrics/results.json` (every section needs a `status` and a `source`) or add a
   benchmark file, then run `python -m skolyoz_metrics render`. Never type a
   result into a README by hand: CI regenerates it and fails on any difference.
3. **State what a number is not.** A held-out result is not a cross-validation
   result; a desktop timing is not a phone timing. If you can't measure it, add a
   `gaps` entry instead of omitting it.
4. **Don't state a Cobb angle** in new UI or docs. The project has no
   radiograph-paired data.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements/dev.txt
python -m skolyoz_metrics check && pytest && ruff check .
```

Heavier, optional: `requirements/bench.txt` for benchmarks and the architecture
tests (they skip themselves when torch / ultralytics are absent).

Android (JDK 17; Android SDK with platform 35):

```bash
cd android && ./gradlew :app:testDebugUnitTest
```

## Changing the measurement maths

The Kotlin code is a port of `scripts/19_kilavuz_cizim.py`. Change the Python
reference first, regenerate the golden data (`scripts/23_altin_veri.py`, `27`,
`29`), then make the Kotlin tests pass. Both `tests/test_golden_parity.py` and the
Gradle tests must be green.

## Style

- Python: Ruff (`ruff check .`), line length 100, type hints on public functions.
  `scripts/` is a frozen record of the experiments and is not linted; fix bugs
  there, but don't reformat it.
- Kotlin: official style (`kotlin.code.style=official`).
- Commits: imperative, one logical change each ("Add p99 to latency summary").
- Pull requests: say which metrics changed and attach the regenerated files.

## Reporting a problem

Use the issue templates. For anything involving patient data, **don't** attach
it — see [SECURITY.md](SECURITY.md).
