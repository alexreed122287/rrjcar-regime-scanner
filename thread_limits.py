"""Cap the BLAS/OpenMP thread pools before numpy is imported.

Why this exists
---------------
The scan fans out across a ``ProcessPoolExecutor`` (6 workers locally). Each worker
process loads its own OpenBLAS, which sizes its thread pool from the core count, so
6 workers x 2 threads contend for 2 cores. Nothing in the repo set the cap.

The matrices here are tiny -- ``GaussianHMM(n_components=7, covariance_type="full")``
on 3 features, so the per-iteration work is on 3x3 covariances. There is no useful
parallelism inside a 3x3 Cholesky; what the extra threads add is contention and
OpenBLAS spin-waiting. Measured with ``tools/bench_blas_threads.py`` (2 cores, 6
workers, 6 fits, medians over repetitions, config order shuffled per repetition):

    bars   uncapped   capped=1   effect
    1500     21.76s      4.92s   4.4x faster
    4745     27.24s      7.71s   3.5x faster   <- 730d hourly, the serving default

The thread-pool path (``screener.py``, and the cloud branch of the scan) is within
noise either way: 5.52s vs 5.45s. That is expected -- one process means one OpenBLAS
pool, so there was never any oversubscription there to remove.

Confirmed end to end on a live server, 10 tickers, period_days=730, warm data cache,
medians of 3 runs after a discarded warm-up:

    endpoint                pool            uncapped   capped   effect
    /api/scan/stream        6 processes       19.32s    6.69s   2.9x faster
    /api/scan               6 threads          3.53s    3.44s   noise (~2.5%)

So the win is real but narrow: it lands entirely on the streaming endpoint, which is
the one the dashboard uses. Anyone measuring via /api/scan will see nothing and
conclude this change does nothing.

The caps do not change results. Same data, same seed, capped vs uncapped: identical
state assignments, identical state counts, means equal to 10 decimals, and
log-likelihood agreeing to 2e-10 (floating-point summation order, not a decision
change). This is a throughput change only.

Ordering requirement
--------------------
OpenBLAS reads these variables once, when the shared library is first loaded. Setting
them after ``import numpy`` is a no-op that looks like it worked. So this module must
not import numpy, and entry points must call ``apply_thread_caps()`` before importing
anything that pulls numpy in. ``tests/test_thread_limits.py`` enforces both properties
at the source level, because the failure mode is silent.
"""

from __future__ import annotations

import os

# OMP_NUM_THREADS covers the OpenMP runtime that hmmlearn/scikit-learn use;
# the other three cover the BLAS backends that may sit under numpy/scipy.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

DEFAULT_LIMIT = "1"


def apply_thread_caps(limit: str = DEFAULT_LIMIT) -> dict[str, str]:
    """Set each thread-count variable that is not already set.

    Returns the variables this call actually set, so a caller can log it.

    Anything already present in the environment is left alone. If you have exported
    OMP_NUM_THREADS=4 deliberately, this must not silently overrule you -- an
    environment variable is the operator's channel, not ours.
    """
    applied: dict[str, str] = {}
    for var in THREAD_ENV_VARS:
        if os.environ.get(var) is None:
            os.environ[var] = limit
            applied[var] = limit
    return applied
