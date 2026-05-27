"""Shared fixtures for the unit test suite.

All tests run on CPU with float64 to stay deterministic and independent of the
machine's CUDA availability (the BO loop otherwise defaults to CUDA).  Torch's
global RNG is reseeded before every test.
"""

from __future__ import annotations

import gpytorch
import pytest
import torch

from understanding_hdbo.methods import METHODS
from understanding_hdbo.model import build_model
from understanding_hdbo.objectives import gp_sample_objective


@pytest.fixture
def cpu_device() -> torch.device:
    """The device every test pins to."""
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def _seed_torch():
    """Reseed torch's global RNG before each test for reproducibility."""
    torch.manual_seed(0)
    yield


@pytest.fixture
def tiny_gp_objective():
    """A small, fast, deterministic synthetic objective (3-dim GP sample)."""
    return gp_sample_objective(dim=3, seed=0)


@pytest.fixture
def tiny_mll(cpu_device):
    """A SingleTaskGP + ExactMarginalLogLikelihood on a few well-conditioned
    3-dim points, ready to hand to :func:`understanding_hdbo.fit.fit_mll`."""
    dim = 3
    x_train = torch.rand(5, dim, dtype=torch.double)
    y_train = torch.rand(5, dtype=torch.double)
    model = build_model(
        method=METHODS["mle_ln2"],
        x_train=x_train,
        y_train=y_train,
        dim=dim,
        device=cpu_device,
        dtype=torch.double,
    )
    return gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
