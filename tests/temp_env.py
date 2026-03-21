from __future__ import annotations

import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path


ENV_TEMP_KEYS = ("TMPDIR", "TEMP", "TMP")
WORKSPACE_TEMP_DIR = Path(__file__).resolve().parent.parent / ".tmp-tests"


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_dir = path / f".tmp-write-probe-{os.getpid()}-{uuid.uuid4().hex}"
        probe_dir.mkdir()
        probe_path = probe_dir / "probe.txt"
        probe_path.write_text("", encoding="utf-8")
        shutil.rmtree(probe_dir)
        return True
    except OSError:
        return False


def get_test_temp_root() -> Path:
    for key in ENV_TEMP_KEYS:
        value = os.environ.get(key)
        if value:
            candidate = Path(value)
            if _is_writable_dir(candidate):
                return candidate
    return WORKSPACE_TEMP_DIR


@contextmanager
def temporary_directory():
    temp_root = get_test_temp_root()
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"tmp-{uuid.uuid4().hex}"
    temp_dir.mkdir()
    try:
        yield str(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
