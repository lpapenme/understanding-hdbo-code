"""Unit tests for understanding_hdbo.objectives."""

from __future__ import annotations

import pytest
import torch

from understanding_hdbo.objectives import (
    _BENCHER_BENCHMARKS_AND_DIMS,
    benchmark_dim,
    bencher_objective,
    gp_sample_objective,
)


# ----------------------------- benchmark_dim ----------------------------- #

@pytest.mark.parametrize("name,dim", list(_BENCHER_BENCHMARKS_AND_DIMS.items()))
def test_benchmark_dim_known(name, dim):
    assert benchmark_dim(name) == dim


def test_benchmark_dim_unknown_raises():
    with pytest.raises(KeyError):
        benchmark_dim("not-a-benchmark")


# --------------------- bencher_objective validation ---------------------- #
# These paths raise before the deferred bencherscaffold import, so they need
# neither the optional dependency nor a running gRPC server.

def test_bencher_objective_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown benchmark"):
        bencher_objective("not-a-benchmark")


def test_bencher_objective_dim_mismatch_raises():
    with pytest.raises(AssertionError, match="dim mismatch"):
        bencher_objective("mopta08", expected_dim=999)


# --------------------------- gp_sample_objective ------------------------- #

def test_gp_sample_objective_returns_scalar():
    f = gp_sample_objective(dim=3, seed=0)
    val = f(torch.full((3,), 0.3, dtype=torch.double))
    assert isinstance(val, torch.Tensor)
    assert val.shape == torch.Size([])


def test_gp_sample_objective_is_deterministic_in_seed():
    x = torch.full((3,), 0.3, dtype=torch.double)
    f0a = gp_sample_objective(dim=3, seed=0)
    f0b = gp_sample_objective(dim=3, seed=0)
    f1 = gp_sample_objective(dim=3, seed=1)
    assert f0a(x) == f0b(x)
    assert f0a(x) != f1(x)


def test_gp_sample_objective_accepts_flat_and_batched_shapes():
    f = gp_sample_objective(dim=3, seed=0)
    flat = f(torch.full((3,), 0.3, dtype=torch.double))
    batched = f(torch.full((1, 3), 0.3, dtype=torch.double))
    assert flat == batched


def test_gp_sample_objective_bell_taper_changes_value():
    x = torch.full((3,), 0.2, dtype=torch.double)  # != 0.8, so taper != 1
    f_taper = gp_sample_objective(dim=3, seed=0, bell_taper=True)
    f_raw = gp_sample_objective(dim=3, seed=0, bell_taper=False)
    assert f_taper(x) != f_raw(x)


def test_gp_sample_objective_restores_global_rng_state():
    before = torch.random.get_rng_state()
    gp_sample_objective(dim=3, seed=123)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
