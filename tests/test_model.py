"""Unit tests for understanding_hdbo.model (construction only, no fitting)."""

from __future__ import annotations

import math

import pytest
import torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import LogNormalPrior

from understanding_hdbo.methods import METHODS
from understanding_hdbo.model import _length_scale_prior, build_model


# --------------------------- _length_scale_prior ------------------------- #

def test_length_scale_prior_none(cpu_device):
    assert _length_scale_prior("none", dim=10, device=cpu_device) is None


@pytest.mark.parametrize("dim", [1, 10, 100])
def test_length_scale_prior_dsp(cpu_device, dim):
    prior = _length_scale_prior("dsp", dim=dim, device=cpu_device)
    assert isinstance(prior, LogNormalPrior)
    assert prior.loc.item() == pytest.approx(1.41 + math.log(dim) / 2.0, rel=1e-5)
    assert prior.scale.item() == pytest.approx(math.sqrt(3.0), rel=1e-5)


def test_length_scale_prior_unknown_raises(cpu_device):
    with pytest.raises(ValueError, match="unknown ls_prior"):
        _length_scale_prior("bogus", dim=10, device=cpu_device)


# ------------------------------- build_model ----------------------------- #

def _build(method_key, *, dim=4, noiseless=True, device):
    x_train = torch.rand(5, dim, dtype=torch.double)
    y_train = torch.rand(5, dtype=torch.double)
    return build_model(
        method=METHODS[method_key],
        x_train=x_train,
        y_train=y_train,
        dim=dim,
        noiseless=noiseless,
        device=device,
    )


def test_build_model_kernel_structure(cpu_device):
    dim = 4
    model = _build("mle_ln2", dim=dim, device=cpu_device)
    assert isinstance(model, SingleTaskGP)
    assert isinstance(model.covar_module, ScaleKernel)
    base = model.covar_module.base_kernel
    assert isinstance(base, MaternKernel)
    assert base.nu == 2.5
    assert base.ard_num_dims == dim


@pytest.mark.parametrize("key", list(METHODS))
def test_build_model_seeds_initial_lengthscale(cpu_device, key):
    dim = 4
    model = _build(key, dim=dim, device=cpu_device)
    expected = METHODS[key].initial_lengthscale(dim)
    ls = model.covar_module.base_kernel.lengthscale.detach().flatten()
    assert ls.shape[0] == dim
    for v in ls.tolist():
        assert v == pytest.approx(expected, rel=1e-5)


def test_build_model_noiseless_freezes_noise(cpu_device):
    model = _build("mle_ln2", noiseless=True, device=cpu_device)
    assert model.likelihood.noise.item() == pytest.approx(1e-4)
    assert model.likelihood.raw_noise.requires_grad is False


def test_build_model_noiseless_default_is_frozen(cpu_device):
    # Regression guard: noiseless now defaults to True.
    x_train = torch.rand(5, 4, dtype=torch.double)
    y_train = torch.rand(5, dtype=torch.double)
    model = build_model(
        method=METHODS["mle_ln2"],
        x_train=x_train,
        y_train=y_train,
        dim=4,
        device=cpu_device,
    )
    assert model.likelihood.raw_noise.requires_grad is False


def test_build_model_noisy_learns_noise(cpu_device):
    model = _build("mle_ln2", noiseless=False, device=cpu_device)
    assert model.likelihood.raw_noise.requires_grad is True


def test_build_model_dsp_attaches_lengthscale_prior(cpu_device):
    model = _build("dsp", device=cpu_device)
    base = model.covar_module.base_kernel
    assert "lengthscale_prior" in base._priors
    assert isinstance(base._priors["lengthscale_prior"][0], LogNormalPrior)


def test_build_model_mle_has_no_lengthscale_prior(cpu_device):
    model = _build("mle_ln2", device=cpu_device)
    assert "lengthscale_prior" not in model.covar_module.base_kernel._priors
