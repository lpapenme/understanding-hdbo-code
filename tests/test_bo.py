"""Unit tests for understanding_hdbo.bo."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from understanding_hdbo.bo import BOResult, _initial_design, bayesian_optimization
from understanding_hdbo.methods import METHODS


# ------------------------------ _initial_design -------------------------- #

def test_initial_design_shape_dtype_and_range():
    x = _initial_design(n_init=8, dim=5, seed=0, dtype=torch.double)
    assert x.shape == (8, 5)
    assert x.dtype == torch.double
    assert (x >= 0).all() and (x <= 1).all()


def test_initial_design_is_seeded():
    a = _initial_design(n_init=8, dim=5, seed=0, dtype=torch.double)
    b = _initial_design(n_init=8, dim=5, seed=0, dtype=torch.double)
    c = _initial_design(n_init=8, dim=5, seed=1, dtype=torch.double)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# --------------------------------- BOResult ------------------------------ #

def test_boresult_defaults():
    x = torch.zeros(3, 2)
    y = torch.zeros(3)
    best = np.zeros(3)
    r = BOResult(x_train=x, y_train=y, best_so_far=best)
    assert r.mean_lengthscale.shape == (0,)
    assert r.method_name == ""


# ----------------------- bayesian_optimization guards -------------------- #

def test_bo_rejects_bad_refit_every():
    with pytest.raises(ValueError, match="refit_every"):
        bayesian_optimization(
            objective=lambda x: torch.tensor(0.0),
            dim=3,
            method="msr",
            refit_every=0,
        )


def test_bo_rejects_unknown_method():
    with pytest.raises(KeyError):
        bayesian_optimization(
            objective=lambda x: torch.tensor(0.0),
            dim=3,
            method="does-not-exist",
        )


# ------------------------------- smoke test ------------------------------ #

@pytest.mark.slow
def test_bayesian_optimization_smoke(tiny_gp_objective, cpu_device):
    n_init, n_iter, dim = 4, 2, 3
    result = bayesian_optimization(
        objective=tiny_gp_objective,
        dim=dim,
        method="mle_ln2",
        n_init=n_init,
        n_iter=n_iter,
        init_seed=0,
        acq_raw_samples=32,
        acq_num_restarts=2,
        verbose=False,
        device=cpu_device,
    )
    n = n_init + n_iter
    assert isinstance(result, BOResult)
    assert result.x_train.shape == (n, dim)
    assert result.y_train.shape == (n,)
    assert result.best_so_far.shape == (n,)
    assert result.mean_lengthscale.shape == (n_iter,)
    assert result.method_name == METHODS["mle_ln2"].name
    # best-so-far must be monotonically non-decreasing.
    assert (np.diff(result.best_so_far) >= 0).all()


@pytest.mark.slow
def test_bo_string_and_object_method_match(tiny_gp_objective, cpu_device):
    def run(method):
        torch.manual_seed(0)
        return bayesian_optimization(
            objective=tiny_gp_objective,
            dim=3,
            method=method,
            n_init=4,
            n_iter=2,
            init_seed=0,
            acq_raw_samples=32,
            acq_num_restarts=2,
            verbose=False,
            device=cpu_device,
        )

    by_name = run("mle_ln2")
    by_object = run(METHODS["mle_ln2"])
    assert torch.allclose(by_name.x_train, by_object.x_train, atol=1e-6)
    assert torch.allclose(by_name.y_train, by_object.y_train, atol=1e-6)
