"""The TFLite benchmark wiring, exercised with a fake interpreter (no model file needed)."""

import pytest

np = pytest.importorskip("numpy")

from skolyoz_metrics.bench import N_LANDMARKS, tflite  # noqa: E402


class FakeInterpreter:
    instances: list["FakeInterpreter"] = []

    def __init__(self, model_path, num_threads):
        self.model_path, self.num_threads = model_path, num_threads
        self.invocations = 0
        self.shape = (1, 640, 640, 3) if "one" in model_path else (1, 128, 128, 3)
        FakeInterpreter.instances.append(self)

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"index": 0, "shape": np.array(self.shape), "dtype": np.float32}]

    def get_output_details(self):
        return [{"shape": np.array((1, 26, 8400))}]

    def set_tensor(self, index, value):
        assert value.shape == self.shape and value.dtype == np.float32

    def invoke(self):
        self.invocations += 1


def test_pipeline_runs_one_stage1_and_seven_stage2_per_measurement(tmp_path):
    FakeInterpreter.instances.clear()
    one, two = tmp_path / "one.tflite", tmp_path / "two.tflite"
    one.write_bytes(b"stage1")
    two.write_bytes(b"stage2-bytes")

    out = tflite.run(one, two, threads=3, warmup=2, repeat=4, interpreter_cls=FakeInterpreter)

    s1, s2 = FakeInterpreter.instances
    assert (s1.num_threads, s2.num_threads) == (3, 3)
    timed = 2 + 4  # warm-up + timed runs of each case
    assert s1.invocations == timed + timed  # its own case + the pipeline case
    assert s2.invocations == timed + timed * N_LANDMARKS

    by_name = {r["name"]: r for r in out["results"]}
    assert set(by_name) == {"stage1", "stage2", "pipeline"}
    assert by_name["stage1"]["model_bytes"] == 6 and by_name["stage2"]["model_bytes"] == 12
    assert len(by_name["stage1"]["model_sha256"]) == 64
    assert by_name["stage1"]["input_shape"] == [1, 640, 640, 3]
    assert by_name["pipeline"]["latency_ms"]["n"] == 4
    assert by_name["stage1"]["device"] == "cpu-3t"


def test_synthetic_input_is_deterministic_and_float_in_unit_range():
    a = tflite.synthetic_input((1, 4, 4, 3), np.float32, seed=7)
    b = tflite.synthetic_input((1, 4, 4, 3), np.float32, seed=7)
    assert np.array_equal(a, b) and a.dtype == np.float32
    assert 0.0 <= a.min() and a.max() < 1.0
    assert not tflite.synthetic_input((1, 2), np.int8).any()
