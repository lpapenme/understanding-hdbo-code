"""Behavioural equivalence test between the published code and the original
research code.

The test runs both implementations side-by-side on the same synthetic objective
and verifies that they produce *numerically identical* trajectories for the
five methods studied in the paper.

The original code is imported from the ``sample_from_high_dim_gp`` package - the
test is skipped if that package is not on the path.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest
import torch

# Make the original code importable.  Adjust ``OG_REPO`` if it lives elsewhere.
OG_REPO = pathlib.Path("/Users/lpapenme/IdeaProjects/sample-from-high-dim-GP")
if OG_REPO.exists():
    sys.path.insert(0, str(OG_REPO))

og = pytest.importorskip("sample_from_high_dim_gp")

import gpytorch  # noqa: E402
from botorch.acquisition import LogExpectedImprovement  # noqa: E402
from botorch.optim import gen_batch_initial_conditions, optimize_acqf  # noqa: E402
from torch.quasirandom import SobolEngine  # noqa: E402

from sample_from_high_dim_gp.objective_functions import get_model, get_objective  # noqa: E402

from understanding_hdbo import bayesian_optimization  # noqa: E402
from understanding_hdbo.methods import METHODS  # noqa: E402
from understanding_hdbo.objectives import gp_sample_objective  # noqa: E402


DIM = 25          # small enough to run a few iterations fast
N_INIT = 8        # ~ 3*sqrt(25) / 2
N_ITER = 4
DTYPE = torch.double
DEVICE = torch.device("cpu")


def _og_one_step(
    *,
    method,
    x_train,
    y_train,
    objective,
):
    """One BO iteration via the original ``get_model`` + BoTorch ``optimize_acqf``.

    Mirrors the relevant slice of ``acqs_and_priors_on_gp_sample.py``.
    """
    y_norm = (y_train - y_train.mean()) / y_train.std()

    init_ls = method.initial_lengthscale(DIM)

    # Map our method to the original code's CLI flags.
    if method.ls_prior == "none":
        og_prior = "none"
    elif method.ls_prior == "dsp":
        og_prior = "carl"
    else:
        raise ValueError(method.ls_prior)

    model = get_model(
        x_train=x_train,
        y_train=y_norm,
        fit_model=True,
        use_ard=True,
        benchmark_gp=None,
        gp_lengthscale=None,
        dim=DIM,
        fully_bayesian=False,
        fully_bayesian_warmup_steps=0,
        fully_bayesian_num_samples=0,
        fully_bayesian_thinning=0,
        prior=og_prior,
        device=DEVICE,
        noiseless_model_fit=True,
        initial_lengtscale=init_ls,
        fitting_grads=None,
    ).to(DEVICE)

    bounds = torch.stack(
        [torch.zeros(DIM, dtype=DTYPE), torch.ones(DIM, dtype=DTYPE)]
    ).to(DEVICE)

    with gpytorch.settings.cholesky_max_tries(9):
        acq = LogExpectedImprovement(model=model, best_f=y_norm.max().item())
        options = {"batch_limit": 1}
        if method.raasp:
            options["sample_around_best"] = True
            options["sample_around_best_sigma"] = 0.001
        kwargs = dict(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=512,
            options=options,
        )
        initial_conditions = gen_batch_initial_conditions(**kwargs)
        x_next, acq_val_next = optimize_acqf(
            **kwargs,
            batch_initial_conditions=initial_conditions,
            return_best_only=False,
        )
    x_next = x_next.detach().cpu()[acq_val_next.argmax()].reshape(1, -1)
    y_next = torch.tensor(
        [float(objective(x_next).reshape(()))], dtype=DTYPE,
    )

    ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
    return x_next, y_next, ls


def _run_og_reference(method, n_iter):
    """Reproduce a BO trajectory using the original code's functions."""
    # NB: pass a per-dim list so that the OG code's ``use_ard`` heuristic flips
    # to True and avoids the ``RBFKernel(ard_num_dims=1)`` debug assertion in
    # current gpytorch.  Numerically equivalent to a single scalar 0.5.
    objective, _ = get_objective(
        dim=DIM,
        seed=0,
        d_type=DTYPE,
        noiseless_model=True,
        fully_bayesian=False,
        lengthscale=[0.5] * DIM,
        load_likelihood=None,
        device=DEVICE,
        bencher_benchmark_name=None,
    )

    sobol = SobolEngine(dimension=DIM, scramble=True, seed=0)
    x_train = sobol.draw(n=N_INIT).to(dtype=DTYPE)
    y_train = torch.tensor([float(objective(x).reshape(())) for x in x_train], dtype=DTYPE)
    ls_hist: list[float] = []

    for _ in range(n_iter):
        x_next, y_next, ls = _og_one_step(
            method=method,
            x_train=x_train,
            y_train=y_train,
            objective=objective,
        )
        x_train = torch.cat([x_train, x_next])
        y_train = torch.cat([y_train, y_next])
        ls_hist.append(float(np.mean(ls)))

    return x_train, y_train, np.asarray(ls_hist)


def _run_new(method, dim, n_init, n_iter):
    torch.manual_seed(0)
    obj = gp_sample_objective(dim=dim, lengthscale=0.5, seed=0)
    return bayesian_optimization(
        objective=obj,
        dim=dim,
        method=method,
        n_init=n_init,
        n_iter=n_iter,
        noiseless=True,
        init_seed=0,
        device=DEVICE,
        dtype=DTYPE,
        verbose=False,
    )


@pytest.mark.parametrize("method_key", list(METHODS))
def test_equivalence(method_key):
    """For every method, OG and new code must produce identical trajectories."""
    method = METHODS[method_key]
    new_res = _run_new(method, dim=DIM, n_init=N_INIT, n_iter=N_ITER)

    # ---- OG reference ------------------------------------------------------
    torch.manual_seed(0)
    og_x, og_y, og_ls = _run_og_reference(method, n_iter=N_ITER)

    # Compare exactly (with a tiny tolerance for floating-point drift in
    # the optimiser/Cholesky path).
    assert torch.allclose(new_res.x_train, og_x, atol=1e-6, rtol=1e-6), (
        f"x_train differs for {method.name}:\n"
        f"  new[-1]={new_res.x_train[-1]}\n"
        f"  og[-1]={og_x[-1]}"
    )
    assert torch.allclose(new_res.y_train, og_y, atol=1e-6, rtol=1e-6), (
        f"y_train differs for {method.name}:\n"
        f"  new={new_res.y_train.tolist()}\n"
        f"  og ={og_y.tolist()}"
    )
    np.testing.assert_allclose(
        new_res.mean_lengthscale, og_ls, atol=1e-4, rtol=1e-4,
        err_msg=f"mean lengthscale differs for {method.name}",
    )
