"""Command-line entry point for the publication code.

Usage examples
--------------

Synthetic GP-sample sanity check (no external dependencies)::

    hdbo-run --objective gp-sample --dim 50 --method msr --n-iter 30

Reproduce one MSR trajectory on Mopta08 (requires the bencher server running
on ``localhost:50051``)::

    hdbo-run --objective bencher --benchmark mopta08 --method msr \\
             --n-iter 1000 --seed 1 --out runs/mopta08_msr_1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from understanding_hdbo.bo import bayesian_optimization
from understanding_hdbo.methods import METHODS
from understanding_hdbo.objectives import (
    benchmark_dim,
    bencher_objective,
    gp_sample_objective,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run high-dimensional BO with one of the methods from the paper.")
    parser.add_argument("--method", required=True, choices=sorted(METHODS), help="Which method to run.")
    parser.add_argument(
        "--objective", required=True, choices=["gp-sample", "bencher"],
        help="Where to source function evaluations from.",
    )
    parser.add_argument("--benchmark", default=None, help="Bencher benchmark name (when --objective bencher).")
    parser.add_argument("--dim", type=int, default=None, help="Dim of GP-sample objective.")
    parser.add_argument("--gp-sample-ls", type=float, default=0.5, help="Length scale of the synthetic GP sample.")
    parser.add_argument("--matheron-seed", type=int, default=0, help="Seed for the GP-sample Matheron path.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the Sobol initial design.")
    parser.add_argument("--n-init", type=int, default=None, help="Initial design size (default: floor(3*sqrt(d))).")
    parser.add_argument("--n-iter", type=int, default=100, help="BO iterations after initial design.")
    parser.add_argument(
        "--refit-every", type=int, default=1,
        help="Refit GP hyperparameters every N iterations (default: 1 = every "
             "iteration, as in the paper). Larger values trade modelling fidelity "
             "for runtime.",
    )
    parser.add_argument(
        "--noiseless", action=argparse.BooleanOptionalAction, default=True,
        help="Freeze GP noise to 1e-4 (default). Pass --no-noiseless to learn "
             "the noise instead, e.g. for real-world benchmarks.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional directory to write trajectory to.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.objective == "gp-sample":
        if args.dim is None:
            raise SystemExit("--dim is required when --objective gp-sample")
        objective = gp_sample_objective(
            dim=args.dim,
            lengthscale=args.gp_sample_ls,
            seed=args.matheron_seed,
        )
        dim = args.dim
    else:  # bencher
        if args.benchmark is None:
            raise SystemExit("--benchmark is required when --objective bencher")
        dim = benchmark_dim(args.benchmark)
        if args.dim is not None and args.dim != dim:
            raise SystemExit(
                f"--dim={args.dim} does not match {args.benchmark!r}'s "
                f"canonical dim {dim}"
            )
        objective = bencher_objective(args.benchmark, expected_dim=dim)

    result = bayesian_optimization(
        objective=objective,
        dim=dim,
        method=args.method,
        n_init=args.n_init,
        n_iter=args.n_iter,
        noiseless=args.noiseless,
        init_seed=args.seed,
        refit_every=args.refit_every,
    )

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        np.save(args.out / "x_train.npy", result.x_train.numpy())
        np.save(args.out / "y_train.npy", result.y_train.numpy())
        np.save(args.out / "best_so_far.npy", result.best_so_far)
        np.save(args.out / "mean_lengthscale.npy", result.mean_lengthscale)
        with open(args.out / "args.json", "w") as fp:
            json.dump({**vars(args), "out": str(args.out)}, fp, default=str, indent=2)
        print(f"wrote trajectory to {args.out}")


if __name__ == "__main__":
    main()
