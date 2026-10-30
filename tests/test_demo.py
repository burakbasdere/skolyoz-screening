"""The demo's pure helpers (no model files needed)."""

import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

from skolyoz_metrics import demo  # noqa: E402


def test_soft_argmax_finds_a_sharp_peak():
    heat = np.full((64, 64), -5.0)
    heat[20, 32] = 5.0
    x, y, entropy, peak_distance = demo.soft_argmax(heat)
    assert x == pytest.approx(32 / 63 * 127, abs=0.1)
    assert y == pytest.approx(20 / 63 * 127, abs=0.1)
    assert entropy < 0.1 and peak_distance < 0.5


def test_uniform_heatmap_is_maximally_uncertain():
    _, _, entropy, peak_distance = demo.soft_argmax(np.zeros((64, 64)))
    assert entropy == pytest.approx(math.log(64 * 64), abs=1e-6)
    assert entropy > demo.ENTROPY_MAX and peak_distance > demo.PEAK_DISTANCE_MAX


def test_two_sharp_peaks_pass_the_entropy_gate_but_trip_the_peak_gate():
    # The blind spot documented in OnIsleme.kt: low entropy, yet the soft-argmax lands between
    # the two peaks, on nothing. Only the peak-distance criterion catches it.
    heat = np.full((64, 64), -5.0)
    heat[10, 10] = heat[10, 50] = 5.0
    _, _, entropy, peak_distance = demo.soft_argmax(heat)
    assert entropy < demo.ENTROPY_MAX
    assert peak_distance > demo.PEAK_DISTANCE_MAX


def test_soft_argmax_rejects_non_square_input():
    with pytest.raises(ValueError):
        demo.soft_argmax(np.zeros(4095))


def test_decode_picks_the_most_confident_candidate():
    out = np.zeros((26, 8400))
    out[4, 100], out[4, 50] = 0.9, 0.5
    for k in range(7):
        out[5 + 3 * k, 100], out[6 + 3 * k, 100] = 10 * k, 20 * k
    points, confidence = demo.decode_stage1(out)
    assert confidence == pytest.approx(0.9)
    assert len(points) == 7 and points[3] == (30.0, 60.0)


def test_decode_returns_none_below_the_confidence_threshold():
    out = np.zeros((26, 8400))
    out[4, 7] = demo.CONFIDENCE_MIN - 0.01
    assert demo.decode_stage1(out) is None


def test_only_the_two_axillary_points_decide_reliability():
    ok = [(2.0, 5.0), (2.4, 14.9), (9.0, 99.0)]           # the third point is ignored
    bad_entropy = [(2.6, 1.0), (1.0, 1.0)]
    bad_peak = [(1.0, 1.0), (1.0, 15.1)]
    base = {"size": (10, 10), "confidence": 0.9, "found": True}
    assert demo.Measurement(**base, gates=ok).reliable
    assert not demo.Measurement(**base, gates=bad_entropy).reliable
    assert not demo.Measurement(**base, gates=bad_peak).reliable
    assert not demo.Measurement((10, 10), 0.1, False).reliable


def test_figure_is_deterministic_and_the_tilt_changes_it():
    a, b, tilted = demo.figure(0.0), demo.figure(0.0), demo.figure(5.0)
    assert a.size == (900, 1200)
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != tilted.tobytes()


def test_overlay_draws_guides_and_banner_without_models():
    img = demo.figure()
    pts = [(250 + 10 * k, 400 + 60 * k) for k in range(7)]
    m = demo.Measurement(img.size, 0.9, True, pts, pts, [(1.0, 1.0)] * 7)
    drawn = demo.overlay(img, m, ["satır"], 0.0, "ÖLÇÜM GEÇERSİZ")
    assert drawn.size == img.size and drawn.tobytes() != img.tobytes()
    hollow = demo.overlay(img, m, ["satır"], guides=False)
    assert hollow.tobytes() != drawn.tobytes()


def test_the_examples_cover_every_rejection_path():
    assert {e.kind for e in demo.EXAMPLES} == {"figure", "blur", "pitch", "blank"}
    assert len({e.name for e in demo.EXAMPLES}) == len(demo.EXAMPLES)


def test_pitch_example_is_rejected_by_the_reference_gate(root):
    ref = demo.reference(root)
    pitch = next(e.pitch for e in demo.EXAMPLES if e.kind == "pitch")
    assert ref.guven_kontrolu(pitch, demo.PITCH_MAX) is False
