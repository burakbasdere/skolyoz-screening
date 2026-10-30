"""Describe the machine a benchmark ran on (no hostname, no username)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def _cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True,
            )
            return out.stdout.strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError, ImportError):
        pass
    return platform.processor() or "unknown"


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def describe(root: Path, *, torch_module=None) -> dict:
    env: dict = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "logical_cores": os.cpu_count(),
        "python": platform.python_version(),
        "commit": git_commit(root),
    }
    if torch_module is not None:
        env["torch"] = torch_module.__version__
        env["cuda_device"] = (
            torch_module.cuda.get_device_name(0) if torch_module.cuda.is_available() else None
        )
    return env
