"""Guards for the BLAS/OpenMP thread caps.

The failure mode this file exists to catch is silent. OpenBLAS reads
OPENBLAS_NUM_THREADS once, when its shared library is first loaded. Move
``apply_thread_caps()`` below an import that pulls in numpy and it still runs, still
sets the variables, still returns them -- and has no effect whatsoever on the thread
pool. Nothing raises. The scan just quietly goes back to being 3.5x slower.

So the load-bearing tests here are the source-level ones: they parse the entry points
and assert the call happens before any import that could reach numpy. A runtime test
cannot check this, because by the time pytest imports anything, numpy is already
loaded in the test process.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys

import pytest

import thread_limits

REPO = pathlib.Path(__file__).resolve().parent.parent

# Entry points that own a process and fan out to workers. Both must cap before numpy.
ENTRY_POINTS = ("app.py", "scheduled_scan.py")

# Modules known to pull numpy in transitively. Not exhaustive by design -- the test
# below treats *any* non-stdlib import before the cap call as a failure, and this list
# only sharpens the error message.
NUMPY_REACHING = {
    "numpy", "pandas", "scipy", "sklearn", "hmmlearn", "yfinance", "ta",
    "screener", "hmm_engine", "data_loader", "backtester", "walk_forward",
    "api", "options_picker", "gex_engine", "strategy_v2", "strategy_leaps",
}

STDLIB_SAFE = {
    "os", "sys", "json", "time", "datetime", "pathlib", "typing", "argparse",
    "logging", "threading", "collections", "itertools", "functools", "math",
    "dataclasses", "enum", "warnings", "traceback", "subprocess", "smtplib",
    "email", "csv", "io", "re", "copy", "random", "concurrent", "asyncio",
    "contextlib", "__future__", "abc", "uuid", "hashlib", "textwrap",
}


def _toplevel_statements(filename: str) -> list[ast.stmt]:
    return ast.parse((REPO / filename).read_text()).body


def _cap_call_index(body: list[ast.stmt]) -> int | None:
    """Index of the statement that calls apply_thread_caps() at module level."""
    for i, node in enumerate(body):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name == "apply_thread_caps":
                return i
    return None


@pytest.mark.parametrize("filename", ENTRY_POINTS)
def test_entry_point_applies_thread_caps(filename):
    """Each entry point must actually call apply_thread_caps() at import time."""
    body = _toplevel_statements(filename)
    assert _cap_call_index(body) is not None, (
        f"{filename} does not call apply_thread_caps() at module level. Without it the "
        f"process-pool scan runs ~3.5x slower (see thread_limits for the numbers)."
    )


@pytest.mark.parametrize("filename", ENTRY_POINTS)
def test_caps_are_applied_before_anything_imports_numpy(filename):
    """The cap call must precede every import that could load numpy.

    This is the whole point. Setting the variables after OpenBLAS has loaded is a
    no-op that leaves every assertion about the variables themselves passing.
    """
    body = _toplevel_statements(filename)
    cap_at = _cap_call_index(body)
    assert cap_at is not None, f"{filename} never calls apply_thread_caps()"

    offenders = []
    for node in body[:cap_at]:
        if isinstance(node, ast.Import):
            roots = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            # A relative import has no module name; treat it as repo-local.
            roots = [(node.module or "").split(".")[0] or "<relative>"]
        else:
            continue
        for root in roots:
            if root == "thread_limits" or root in STDLIB_SAFE:
                continue
            offenders.append((root, node.lineno))

    assert not offenders, (
        f"{filename} imports {offenders} before calling apply_thread_caps(). "
        f"OpenBLAS reads its thread-count variables when the library first loads, so "
        f"capping after any of these silently does nothing. Move the call above them."
    )


def test_thread_limits_module_does_not_import_numpy():
    """thread_limits must stay importable without loading numpy itself.

    If it grows a numpy import, it defeats its own purpose: importing it to apply the
    caps would load OpenBLAS first.
    """
    tree = ast.parse((REPO / "thread_limits.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not (imported & NUMPY_REACHING), (
        f"thread_limits imports {imported & NUMPY_REACHING}, which loads numpy before "
        f"the caps can take effect."
    )


def test_apply_thread_caps_sets_the_documented_variables(monkeypatch):
    for var in thread_limits.THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    applied = thread_limits.apply_thread_caps()
    assert set(applied) == set(thread_limits.THREAD_ENV_VARS)
    for var in thread_limits.THREAD_ENV_VARS:
        assert os.environ[var] == "1"


def test_an_existing_setting_is_never_overruled(monkeypatch):
    """An exported variable is the operator's channel. We must not overwrite it.

    Someone who has deliberately set OMP_NUM_THREADS=4 for a big backfill should not
    have it silently reset to 1 by importing the app.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    for var in thread_limits.THREAD_ENV_VARS:
        if var != "OMP_NUM_THREADS":
            monkeypatch.delenv(var, raising=False)

    applied = thread_limits.apply_thread_caps()

    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert "OMP_NUM_THREADS" not in applied
    assert applied["OPENBLAS_NUM_THREADS"] == "1"


def test_the_caps_actually_reach_openblas_in_a_child_process():
    """End-to-end: importing the app must leave OpenBLAS at one thread, in the parent
    and in a pool worker.

    Runs in a subprocess because this test process has already imported numpy, so the
    thread pool here is fixed no matter what the environment says. This is the test
    that would catch the caps being set too late; the AST checks above catch it
    earlier and with a better message.
    """
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "import app\n"
        "import numpy as np, threadpoolctl\n"
        "np.random.rand(32, 32) @ np.random.rand(32, 32)\n"
        "counts = sorted(p['num_threads'] for p in threadpoolctl.threadpool_info())\n"
        "print('COUNTS', counts)\n"
    ) % str(REPO)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(REPO), timeout=180,
        env={k: v for k, v in os.environ.items() if k not in thread_limits.THREAD_ENV_VARS},
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr[-2000:]}"

    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("COUNTS")]
    assert line, f"no COUNTS line in output:\n{proc.stdout[-2000:]}"
    counts = ast.literal_eval(line[-1][len("COUNTS"):].strip())

    assert counts, "threadpoolctl reported no pools at all -- did the matmul run?"
    assert set(counts) == {1}, (
        f"expected every native thread pool capped to 1, got {counts}. The caps are "
        f"being applied after OpenBLAS loaded, or not at all."
    )
