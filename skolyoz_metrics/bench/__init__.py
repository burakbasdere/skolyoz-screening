"""Data-free latency benchmarks (synthetic inputs, no patient images needed)."""

# Number of landmarks the app refines per measurement: one Stage-2 call each
# (android/.../cikarim/OlcumZinciri.kt iterates over all seven coarse points).
N_LANDMARKS = 7

STAGE1_INPUT = (1, 3, 640, 640)
STAGE2_INPUT = (1, 3, 128, 128)
