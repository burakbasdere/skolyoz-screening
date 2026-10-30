"""Architecture facts that must stay true. Skipped when torch / ultralytics are absent."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")
pytest.importorskip("torchvision")

from skolyoz_metrics.bench import STAGE1_INPUT, STAGE2_INPUT, arch  # noqa: E402


@pytest.fixture(scope="module")
def models(root):
    stage1 = arch.build_stage1().eval()
    stage2 = arch.load_stage2_class(root)(pretrained=False).eval()
    return stage1, stage2


def test_parameter_counts_match_the_published_metrics(models, results):
    stage1, stage2 = models
    published = {r["model"]: r for r in results["model_complexity"]["rows"]}
    assert arch.count_params(stage1) == published["stage1_yolo11n_pose_7kpt"]["params"]
    assert arch.count_params(stage2) == published["stage2_rafinaj7"]["params"]


def test_flop_counter_actually_counts(models, results):
    # Guards against FlopCounterMode silently returning 0 (it does under inference_mode).
    _, stage2 = models
    gflops = arch.count_gflops(stage2, torch.rand(STAGE2_INPUT))
    published = next(r for r in results["model_complexity"]["rows"]
                     if r["model"] == "stage2_rafinaj7")["gflops"]
    assert gflops == pytest.approx(published, rel=0.01)


def test_output_shapes_match_what_the_android_decoder_expects(models):
    stage1, stage2 = models
    with torch.no_grad():
        out1 = stage1(torch.zeros(STAGE1_INPUT))
        coords, heatmaps = stage2(torch.zeros(STAGE2_INPUT))
    pred = out1[0] if isinstance(out1, (list, tuple)) else out1
    # Asama1Coz.kt: (1, 26, 8400) = 4 box + 1 conf + 7 landmarks x (x, y, visibility)
    assert tuple(pred.shape) == (1, 26, 8400)
    assert tuple(coords.shape) == (1, 4, 2)  # 4 visual types
    assert tuple(heatmaps.shape) == (1, 4, 64, 64)


def test_run_reports_every_stage_on_cpu(root):
    out = arch.run(root, devices=["cpu"], threads=1, warmup=0, repeat=2)
    assert [r["name"] for r in out["results"]] == ["stage1", "stage2", "pipeline"]
    pipeline = out["results"][2]
    assert pipeline["gflops"] > out["results"][0]["gflops"]
    assert pipeline["latency_ms"]["n"] == 2
