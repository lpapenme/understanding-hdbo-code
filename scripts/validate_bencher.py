#!/usr/bin/env python
"""End-to-end validation against the bencher gRPC server.

This script:

1. Connects to the bencher service running on ``localhost:50051`` (assumed
   already running in Docker - see the original repo for the
   ``container.sdef`` / Docker setup).
2. Runs ``--n-iter`` BO iterations on the requested benchmark with MSR.
3. Writes the trajectory to ``--out`` for later comparison against the
   paper's published curves.

The script is intentionally short so it can serve as a smoke test that the
published package and the bencher service speak to each other end-to-end.

Example::

    python scripts/validate_bencher.py --benchmark mopta08 --n-iter 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from understanding_hdbo import bayesian_optimization
from understanding_hdbo.objectives import bencher_objective, benchmark_dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", default="mopta08",
        choices=["mopta08", "lasso-dna", "svm", "mujoco-ant", "mujoco-humanoid", "rover"],
    )
    parser.add_argument("--method", default="msr", choices=[
        "msr", "mle_scaled", "mle_ln2", "mle_ln2_raasp", "dsp",
    ])
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--n-init", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--refit-every", type=int, default=1,
        help="Refit GP hyperparameters every N iterations (default: 1).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/validate_bencher"),
        help="Directory to write the trajectory and metadata to.",
    )
    parser.add_argument("--hostname", default="localhost")
    args = parser.parse_args()

    dim = benchmark_dim(args.benchmark)
    print(f"running {args.method} on {args.benchmark} (d={dim}) for {args.n_iter} iters")

    objective = bencher_objective(args.benchmark, hostname=args.hostname)

    t0 = time.time()
    result = bayesian_optimization(
        objective=objective,
        dim=dim,
        method=args.method,
        n_init=args.n_init,
        n_iter=args.n_iter,
        noiseless=False,         # real-world benchmarks may have noise
        init_seed=args.seed,
        refit_every=args.refit_every,
        verbose=True,
    )
    elapsed = time.time() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "x_train.npy", result.x_train.numpy())
    np.save(args.out / "y_train.npy", result.y_train.numpy())
    np.save(args.out / "best_so_far.npy", result.best_so_far)
    np.save(args.out / "mean_lengthscale.npy", result.mean_lengthscale)

    metadata = {
        **vars(args),
        "out": str(args.out),
        "dim": dim,
        "elapsed_sec": elapsed,
        "final_best": float(result.best_so_far[-1]),
        "n_observations": int(result.x_train.shape[0]),
    }
    with open(args.out / "metadata.json", "w") as fp:
        json.dump(metadata, fp, default=str, indent=2)

    print()
    print(f"finished in {elapsed:.1f}s")
    print(f"final best value (maximization sign): {result.best_so_far[-1]:.5f}")
    print(f"trajectory written to {args.out}")


if __name__ == "__main__":
    main()
