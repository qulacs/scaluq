"""Generate current bindings' stubs on the build host without AVX512 emulation."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

destination = Path(os.environ["SCALUQ_STUB_DIR"])
env = dict(
    os.environ,
    SCALUQ_CPU_NATIVE="OFF",
    SCALUQ_CPU_ARCH="",
    CMAKE_CROSSCOMPILING_EMULATOR="",
)
env.pop("SCALUQ_STUB_DIR")
with tempfile.TemporaryDirectory(prefix="scaluq-stubs-") as temporary:
    build = Path(temporary) / "build"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            sys.argv[1],
            "--no-deps",
            "--wheel-dir",
            temporary,
            "--config-settings",
            f"build-dir={build}",
        ],
        env=env,
        check=True,
    )
    destination.mkdir(parents=True, exist_ok=True)
    for stub in ("root", "main", "gate"):
        shutil.copyfile(
            build / "python/generated-stubs" / f"{stub}.pyi",
            destination / f"{stub}.pyi",
        )
