"""Skolyoz metrik araçları: ``python -m skolyoz_metrics {check,render,bench}``."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .bench import env as bench_env
from .render import normalise, render_all
from .results import (
    consistency_errors,
    load_benchmarks,
    load_results,
    repo_root,
    schema_errors,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _drift(root: Path, results: dict, benchmarks: list[dict]) -> list[str]:
    """Names of generated files whose content on disk differs from the JSON."""
    stale = []
    for path, wanted in render_all(root, results, benchmarks).items():
        current = normalise(path.read_text(encoding="utf-8")) if path.exists() else ""
        if current != wanted:
            diff = difflib.unified_diff(
                current.splitlines(), wanted.splitlines(), "diskteki", "JSON'dan", lineterm="", n=0
            )
            stale.append(f"{path.relative_to(root)} güncel değil:\n    " + "\n    ".join(
                list(diff)[:12]))
    return stale


def cmd_render(args: argparse.Namespace) -> int:
    root = args.root
    results, benchmarks = load_results(root), load_benchmarks(root)
    if args.check:
        stale = _drift(root, results, benchmarks)
        for line in stale:
            print(line)
        if stale:
            print("\n`python -m skolyoz_metrics render` çalıştırıp sonucu commit edin.")
        return 1 if stale else 0
    for path, text in render_all(root, results, benchmarks).items():
        _write_text(path, text)
        print(f"yazıldı: {path.relative_to(root)}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    root = args.root
    problems: list[str] = []
    results = load_results(root)
    problems += [f"results.json: {m}"
                 for m in schema_errors(results, root / "metrics" / "results.schema.json")]
    benchmarks = load_benchmarks(root)
    for run in benchmarks:
        problems += [f"benchmarks/{run['_file']}: {m}" for m in schema_errors(
            {k: v for k, v in run.items() if k != "_file"},
            root / "metrics" / "benchmark.schema.json")]
    if not problems:  # consistency checks assume a schema-valid document
        problems += [f"results.json: {m}" for m in consistency_errors(results)]
        problems += _drift(root, results, benchmarks)
    for p in problems:
        print(f"HATA {p}")
    if not problems:
        print(f"TAMAM: results.json, {len(benchmarks)} benchmark dosyası, "
              "üretilen dokümanlar senkron")
    return 1 if problems else 0


def _slug(text: str, limit: int = 40) -> str:
    """File-name friendly CPU label.

    '11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz' -> '11th-gen-intel-core-i7-11800h'
    """
    text = re.sub(r"\((?:r|tm)\)|\bcpu\b|@.*$", "", text, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit] or "unknown"


def _finish_bench(args: argparse.Namespace, kind: str, env: dict, payload: dict) -> int:
    now = datetime.now(timezone.utc)
    run = {
        "schema_version": 1,
        "kind": kind,
        "created_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": env,
        "config": {"threads": args.threads, "warmup": args.warmup, "repeat": args.repeat,
                   "input": "synthetic"},
        **payload,
    }
    out = args.out or (args.root / "metrics" / "benchmarks"
                       / f"{now:%Y-%m-%d}_{kind}_{_slug(env['cpu'])}.json")
    _write_text(out, json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    print(f"yazıldı: {out}")
    for r in run["results"]:
        lat = r["latency_ms"]
        print(f"  {r['name']:<9} {r['device']:<8} p50 {lat['p50']:8.2f} ms   "
              f"p90 {lat['p90']:8.2f} ms   p99 {lat['p99']:8.2f} ms")
    print("docs/METRICS.md ve README tablolarını yenilemek için "
          "`python -m skolyoz_metrics render` çalıştırın.")
    return 0


def cmd_bench_arch(args: argparse.Namespace) -> int:
    import torch

    from .bench import arch

    devices = args.devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    payload = arch.run(args.root, devices=devices, threads=args.threads,
                       warmup=args.warmup, repeat=args.repeat)
    return _finish_bench(args, "arch", bench_env.describe(args.root, torch_module=torch), payload)


def cmd_bench_tflite(args: argparse.Namespace) -> int:
    from .bench import tflite

    payload = tflite.run(args.stage1, args.stage2, threads=args.threads,
                         warmup=args.warmup, repeat=args.repeat)
    env = bench_env.describe(args.root)
    for module in ("ai_edge_litert", "tflite_runtime", "tensorflow"):
        try:
            env["tflite_runtime"] = f"{module} {__import__(module).__version__}"
            break
        except (ImportError, AttributeError):
            continue
    return _finish_bench(args, "tflite", env, payload)


def cmd_demo(args: argparse.Namespace) -> int:
    from . import demo

    out_dir = args.out or (args.root / "docs" / "img" / "ornekler")
    for ex in demo.run(args.root, args.stage1, args.stage2, out_dir, args.threads):
        what = (f"reddedildi: {ex.rejected}" if ex.rejected
                else f"eğim {ex.angle:+.2f}°")
        print(f"  {ex.name:<10} {what}")
    print(f"yazıldı: {out_dir} ve docs/ORNEKLER.md")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skolyoz-metrics", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--root", type=Path, default=repo_root(), help="depo kök dizini")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="metrik dosyalarını doğrula, üretilen dokümanların güncel "
                   "olduğunu denetle").set_defaults(func=cmd_check)

    render = sub.add_parser("render", help="docs/METRICS.md ve README tablolarını yeniden üret")
    render.add_argument("--check", action="store_true",
                        help="yazmak yerine, güncel değilse hata ver")
    render.set_defaults(func=cmd_render)

    bench = sub.add_parser("bench", help="veri gerektirmeyen gecikme ölçümleri")
    bsub = bench.add_subparsers(dest="kind", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--threads", type=int, default=4,
                       help="CPU iş parçacığı sayısı (Android uygulaması 4 kullanır)")
        p.add_argument("--warmup", type=int, default=10, help="ölçülmeyen ısınma koşusu")
        p.add_argument("--repeat", type=int, default=100, help="ölçülen koşu sayısı")
        p.add_argument("--out", type=Path,
                       help="çıktı dosyası (varsayılan: metrics/benchmarks/...)")

    arch = bsub.add_parser("arch", help="rastgele ağırlıklı PyTorch mimarileri")
    arch.add_argument("--devices", nargs="+",
                      help="örn. cpu cuda (varsayılan: cpu + varsa cuda)")
    common(arch)
    arch.set_defaults(func=cmd_bench_arch)

    tfl = bsub.add_parser("tflite", help="dağıtılan .tflite dosyaları")
    tfl.add_argument("--stage1", type=Path, required=True, help="asama1.tflite")
    tfl.add_argument("--stage2", type=Path, required=True, help="asama2.tflite")
    common(tfl)
    tfl.set_defaults(func=cmd_bench_tflite)

    demo_p = sub.add_parser("demo", help="sentetik örnek figürde gerçek modellerle örnek çıktılar")
    demo_p.add_argument("--stage1", type=Path, required=True, help="asama1.tflite")
    demo_p.add_argument("--stage2", type=Path, required=True, help="asama2.tflite")
    demo_p.add_argument("--threads", type=int, default=4)
    demo_p.add_argument("--out", type=Path, help="görüntü klasörü (varsayılan: docs/img/ornekler)")
    demo_p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
