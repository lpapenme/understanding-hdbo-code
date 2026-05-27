"""GP surrogate model construction for high-dimensional BO.

The exact construction is identical to ``objective_functions.get_model`` in the
original code release: ARD Matern-5/2 covariance wrapped in a ``ScaleKernel``
with the BoTorch-default ``Gamma(2, 0.15)`` output scale prior and a
``GaussianLikelihood`` carrying BoTorch's default ``Gamma(1.1, 0.05)`` noise
prior.  By default (``noiseless=True``) the likelihood noise is clamped to
``1e-4`` and frozen, which is the right behaviour for synthetic GP-sample
objectives and matches ``--noiseless-model true`` in the original code.  Pass
``noiseless=False`` to let the noise be learned (e.g. for real-world
benchmarks), matching the original ``--noiseless-model false`` setting.

The two priors that distinguish the methods in the paper are placed by this
module:

* ``ls_prior="none"``  - no length-scale prior is set (pure MLE).
* ``ls_prior="dsp"``   - the dimensionality-scaled log-normal length-scale
                         prior of Hvarfner et al. (2024).
"""

from __future__ import annotations

import math

import torch
from botorch.models import SingleTaskGP
from botorch.models.utils.gpytorch_modules import (
    get_gaussian_likelihood_with_gamma_prior,
)
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior, LogNormalPrior

from understanding_hdbo.methods import DSP_LOC_CONST, DSP_SCALE, Method


def _length_scale_prior(ls_prior: str, dim: int, device: torch.device):
    if ls_prior == "none":
        return None
    if ls_prior == "dsp":
        # Build the prior's location and scale at PyTorch's default float dtype
        # (typically float32) to reproduce the construction in the original
        # research code:
        #     LogNormalPrior(torch.tensor(1.41 + log(d)/2),
        #                    torch.sqrt(torch.tensor(3.0)))
        # These get upcast to the model's float64 dtype on first use, but
        # constructing them in float32 introduces a tiny (~1e-7) difference
        # from a pure-float64 build.  We keep the float32 construction to
        # preserve numerical behaviour of the published experiments.
        loc = torch.tensor(DSP_LOC_CONST + math.log(dim) / 2.0, device=device)
        scale = torch.sqrt(torch.tensor(DSP_SCALE ** 2, device=device))
        return LogNormalPrior(loc, scale)
    raise ValueError(f"unknown ls_prior {ls_prior!r}")


def build_model(
    *,
    method: Method,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    dim: int,
    noiseless: bool = True,
    device: torch.device,
    dtype: torch.dtype = torch.double,
) -> SingleTaskGP:
    """Construct an un-fitted SingleTaskGP whose hyperparameters reflect
    ``method`` and whose initial length scale is seeded according to
    :meth:`Method.initial_lengthscale`.

    The returned model is *not yet fitted* - call :func:`understanding_hdbo.fit.fit_mll`
    on its ``ExactMarginalLogLikelihood`` to optimise the (penalised) MLL.
    """

    prior = _length_scale_prior(method.ls_prior, dim=dim, device=device)

    base_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=dim,
        lengthscale_prior=prior,
    )
    covar_module = ScaleKernel(
        base_kernel=base_kernel,
        outputscale_prior=GammaPrior(2.0, 0.15),
    )

    # Seed the length scale with the method-specific initial value.
    init_ls = method.initial_lengthscale(dim)
    covar_module.base_kernel.lengthscale = torch.full(
        (dim,), init_ls, dtype=dtype, device=device,
    )

    likelihood = get_gaussian_likelihood_with_gamma_prior()
    if noiseless:
        # Matches old code: fix noise to 1e-4 for noise-free GP-sample objectives.
        likelihood.noise = torch.tensor(1e-4)
        likelihood.noise.requires_grad = False

    model = SingleTaskGP(
        x_train.to(device=device, dtype=dtype),
        y_train.to(device=device, dtype=dtype).unsqueeze(-1),
        covar_module=covar_module,
        likelihood=likelihood,
    ).to(device=device, dtype=dtype)

    return model
