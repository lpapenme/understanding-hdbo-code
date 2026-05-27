"""Objective-function adapters.

Two kinds of objectives are provided:

1. :func:`gp_sample_objective` - a Matheron-sampled realisation of an isotropic
   RBF GP, used for the controlled experiments in the paper and for the
   reproducibility test (synthetic, noise-free, fast).
2. :func:`bencher_objective` - a thin wrapper around the ``bencherscaffold``
   gRPC client used by the original code to evaluate Mopta08, Lasso-DNA, SVM,
   Mujoco-Ant, Mujoco-Humanoid and Rover.  The wrapper assumes a server is
   running on ``localhost:50051`` (e.g. the ``bencher`` Docker image bundled
   with the original code).

Both functions return a callable ``f(x: Tensor) -> Tensor`` that the BO loop
treats as a *maximisation* target.  The bencher wrapper negates the raw value
returned by the server because the underlying benchmarks are defined as
*minimisation* problems.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Callable

import torch
from botorch.models import SingleTaskGP
from botorch.sampling.pathwise import draw_matheron_paths
from gpytorch.kernels import RBFKernel

ObjectiveFn = Callable[[torch.Tensor], torch.Tensor]


# ----------------------------- synthetic --------------------------------- #

def gp_sample_objective(
    dim: int,
    *,
    lengthscale: float = 0.5,
    seed: int = 0,
    dtype: torch.dtype = torch.double,
    bell_taper: bool = True,
) -> ObjectiveFn:
    """Return a callable that evaluates a fixed sample drawn from an isotropic
    RBF Gaussian-process prior with the given length scale.

    The ``bell_taper`` flag reproduces a multiplicative bell-shaped envelope
    used in the paper's Section 3 experiments to localise the optimiser away
    from the boundary.  Set it to ``False`` if you want the raw GP sample.
    """
    gp = SingleTaskGP(
        torch.empty(0, dim, dtype=dtype),
        torch.empty(0, 1, dtype=dtype),
        covar_module=RBFKernel(ard_num_dims=dim),
    ).cpu()
    gp.covar_module.lengthscale = torch.tensor([lengthscale])
    gp.likelihood.noise = torch.tensor(1e-4)

    torch_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        path = draw_matheron_paths(model=deepcopy(gp), sample_shape=torch.Size([1])).cpu()
    finally:
        torch.random.set_rng_state(torch_state)

    inv_sqrt_dim = 1.0 / math.sqrt(dim)

    def f(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu().reshape(1, -1).to(dtype)
        with torch.no_grad():
            val = path(x).squeeze()
            if bell_taper:
                val = val * torch.exp(-inv_sqrt_dim * (x - 0.8) ** 2).prod()
        return val.detach()

    return f


# ----------------------------- bencher ----------------------------------- #

_BENCHER_BENCHMARKS_AND_DIMS = {
    "lasso-dna": 180,
    "mopta08": 124,
    "svm": 388,
    "mujoco-ant": 888,
    "mujoco-humanoid": 6392,
    "rover": 60,
}


def bencher_objective(
    benchmark_name: str,
    *,
    expected_dim: int | None = None,
    hostname: str = "localhost",
    max_retries: int = 5,
    retry_sleep_s: float = 5.0,
    dtype: torch.dtype = torch.double,
) -> ObjectiveFn:
    """Return a callable that evaluates a benchmark via the bencher gRPC server.

    The server must be reachable at ``{hostname}:50051`` and must expose
    ``benchmark_name``.  See the README for instructions on starting the
    Docker container that ships with the bencher project.

    Imports of :mod:`bencherscaffold` are deferred to call time so that the
    rest of the package can be used without bencher installed.
    """
    if benchmark_name not in _BENCHER_BENCHMARKS_AND_DIMS:
        raise ValueError(
            f"unknown benchmark {benchmark_name!r}; "
            f"choose one of {sorted(_BENCHER_BENCHMARKS_AND_DIMS)}"
        )
    if expected_dim is not None:
        assert expected_dim == _BENCHER_BENCHMARKS_AND_DIMS[benchmark_name], (
            f"dim mismatch: passed {expected_dim} but {benchmark_name!r} is "
            f"{_BENCHER_BENCHMARKS_AND_DIMS[benchmark_name]}-dimensional"
        )

    try:
        from bencherscaffold.client import BencherClient
        from bencherscaffold.protoclasses.bencher_pb2 import Value, ValueType
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "bencher_objective requires the optional 'bencherscaffold' dependency; "
            "install with `pip install understanding-hdbo[bencher]`"
        ) from exc

    client = BencherClient(address=hostname, max_retries=max_retries)

    def f(x: torch.Tensor) -> torch.Tensor:
        arr = x.detach().cpu().reshape(-1).to(torch.double).tolist()
        values = [Value(type=ValueType.CONTINUOUS, value=float(v)) for v in arr]

        for attempt in range(max_retries):
            try:
                res = client.evaluate_point(
                    benchmark_name=benchmark_name, point=values,
                )
                # Bencher returns the raw benchmark value; we maximise -f.
                return torch.tensor(-res, dtype=dtype)
            except Exception as exc:  # noqa: BLE001
                print(f"bencher evaluation failed (attempt {attempt + 1}): {exc!r}")
                if attempt + 1 == max_retries:
                    raise
                time.sleep(retry_sleep_s)
        raise RuntimeError("unreachable")

    return f


def benchmark_dim(benchmark_name: str) -> int:
    """Look up the canonical dimensionality of a bencher benchmark."""
    return _BENCHER_BENCHMARKS_AND_DIMS[benchmark_name]
