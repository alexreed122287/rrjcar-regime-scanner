"""Measure whether BLAS thread caps change scan throughput.

The claim under test, carried in docs/validation-findings.md as "found, not fixed":
concurrent HMM fits oversubscribe OpenBLAS against 2 cores because nothing sets
OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS, and capping them would help.

That was asserted from reading the code. This measures it.

The shapes matter and argue against the claim before a single timing is taken: the
serving path fits GaussianHMM(n_components=7, covariance_type="full") on 3 features.
The per-iteration linear algebra is on 3x3 covariance matrices. Threading a 3x3
Cholesky is overhead, not speedup. Two effects therefore pull in opposite directions
and only measurement settles which dominates:

  - Process pool: each of the 6 workers loads its own OpenBLAS with its own 2-thread
    pool, so 12 threads contend for 2 cores. Capping should help.
  - Any pool: OpenBLAS spin-waits before parking its threads, so on matrices this
    small the cap may simply remove work that was never being done in parallel anyway.

Run:
    python tools/bench_blas_threads.py
    python tools/bench_blas_threads.py --reps 5 --bars 4745
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import random
import subprocess
import sys
import time

# Must be importable in a fresh subprocess with caps applied via the environment,
# so the fit body is a module-level function and imports numpy lazily inside it.


def _fit_once(seed: int, bars: int, n_components: int) -> float:
    import numpy as np
    from hmmlearn.hmm import GaussianHMM

    rng = np.random.default_rng(seed)
    # Three features shaped like the real ones: returns, range, volume_change.
    # Regime structure is present so EM has something to find and does not
    # degenerate into an unrealistically fast convergence.
    n_feat = 3
    X = np.empty((bars, n_feat))
    state = 0
    for i in range(bars):
        if rng.random() < 0.02:
            state = int(rng.integers(0, 3))
        scale = (0.5, 1.0, 2.5)[state]
        X[i, 0] = rng.normal(0, 0.01 * scale)
        X[i, 1] = abs(rng.normal(0.01 * scale, 0.004))
        X[i, 2] = rng.normal(0, 0.3 * scale)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    t0 = time.perf_counter()
    GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=100,
        tol=1e-4,
        random_state=seed,
        verbose=False,
    ).fit(X)
    return time.perf_counter() - t0


def _worker(args):
    return _fit_once(*args)


def run_pool(kind: str, workers: int, tickers: int, bars: int, n_components: int) -> float:
    """Wall-clock for one full 'scan' of `tickers` fits across `workers` workers."""
    jobs = [(i, bars, n_components) for i in range(tickers)]
    Pool = (
        concurrent.futures.ProcessPoolExecutor
        if kind == "process"
        else concurrent.futures.ThreadPoolExecutor
    )
    t0 = time.perf_counter()
    with Pool(max_workers=workers) as ex:
        list(ex.map(_worker, jobs))
    return time.perf_counter() - t0


CAP_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def child_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--kind", default="process")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tickers", type=int, default=12)
    ap.add_argument("--bars", type=int, default=4745)
    ap.add_argument("--regimes", type=int, default=7)
    a = ap.parse_args()

    import threadpoolctl
    import numpy as np

    np.random.rand(64, 64) @ np.random.rand(64, 64)
    pools = [
        {"api": p["internal_api"], "threads": p["num_threads"]}
        for p in threadpoolctl.threadpool_info()
    ]
    elapsed = run_pool(a.kind, a.workers, a.tickers, a.bars, a.regimes)
    print(json.dumps({"elapsed": elapsed, "pools": pools}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--bars", type=int, default=4745, help="730d hourly ~= 4745")
    ap.add_argument("--tickers", type=int, default=12)
    ap.add_argument("--regimes", type=int, default=7)
    args = ap.parse_args()

    here = os.path.abspath(__file__)
    results: dict[str, list[float]] = {}
    pools_seen: dict[str, object] = {}

    # process pool with 6 workers is the local default (_DEFAULT_WORKERS);
    # thread pool with 10 is what screener.py uses.
    configs = [
        ("process-6 uncapped", "process", 6, None),
        ("process-6 capped=1", "process", 6, "1"),
        ("thread-10 uncapped", "thread", 10, None),
        ("thread-10 capped=1", "thread", 10, "1"),
    ]

    print(
        f"bars={args.bars} tickers={args.tickers} regimes={args.regimes} "
        f"reps={args.reps} cores={os.cpu_count()}\n"
    )

    for rep in range(args.reps):
        # Shuffle within each rep. Run in fixed order, the first config eats all the
        # cold-start cost (imports, page cache, CPU frequency ramp) and looks slower
        # for a reason that has nothing to do with BLAS threads.
        order = configs[:]
        random.shuffle(order)
        for label, kind, workers, cap in order:
            env = dict(os.environ)
            for v in CAP_VARS:
                env.pop(v, None)
            if cap:
                for v in CAP_VARS:
                    env[v] = cap
            cmd = [
                sys.executable, here, "--child",
                "--kind", kind, "--workers", str(workers),
                "--tickers", str(args.tickers), "--bars", str(args.bars),
                "--regimes", str(args.regimes),
            ]
            out = subprocess.run(
                cmd, env=env, capture_output=True, text=True, check=True
            ).stdout.strip().splitlines()[-1]
            payload = json.loads(out)
            results.setdefault(label, []).append(payload["elapsed"])
            pools_seen[label] = payload["pools"]
            print(f"  rep{rep + 1} {label:22s} {payload['elapsed']:7.2f}s")
        print()

    print("=" * 64)
    print(f"{'config':22s} {'median':>9s} {'min':>8s} {'max':>8s}   BLAS pool")
    for label, _, _, _ in configs:
        ts = results[label]
        pool = pools_seen[label]
        pstr = ",".join(f"{p['api']}={p['threads']}" for p in pool) or "none"
        print(
            f"{label:22s} {statistics.median(ts):8.2f}s {min(ts):7.2f}s "
            f"{max(ts):7.2f}s   {pstr}"
        )

    print()
    for kind, workers in (("process", 6), ("thread", 10)):
        unc = statistics.median(results[f"{kind}-{workers} uncapped"])
        cap = statistics.median(results[f"{kind}-{workers} capped=1"])
        delta = (unc - cap) / unc * 100
        verdict = (
            f"capping is {delta:.1f}% FASTER" if delta > 0
            else f"capping is {-delta:.1f}% SLOWER"
        )
        print(f"{kind}-{workers}: uncapped {unc:.2f}s vs capped {cap:.2f}s -> {verdict}")


if __name__ == "__main__":
    if "--child" in sys.argv:
        child_main()
    else:
        main()
