"""Minimal high-dimensional Bayesian optimisation loop.

This module exposes :func:`bayesian_optimization`, a single function that runs a
BO loop for one of the methods defined in :mod:`understanding_hdbo.methods`.

The loop mirrors the behaviour of ``acqs_and_priors_on_gp_sample.py`` in the
original code release for the configurations used to produce Figures 8-10 of
the paper.  Concretely:

* The initial design is drawn from a scrambled Sobol sequence.
* The surrogate is a SingleTaskGP with a Matern-5/2 ARD kernel and a
  Gamma(2, 0.15) output-scale prior.  The length-scale prior is either omitted
  (MLE-based methods) or the DSP prior from Hvarfner et al. (2024).
* The acquisition function is LogEI.
* The acquisition function is maximised with 5 restarts and 512 raw samples.
* When the method enables RAASP, ``optimize_acqf`` is invoked with
  ``options={"sample_around_best": True}``.  We use the value
  ``sample_around_best_sigma=0.001`` to match the BoTorch default used in the
  original runs (the ``--scale-acq-distr`` flag in the old code was left at its
  default ``False``).
* On a Cholesky / fitting failure the next candidate is drawn uniformly at
  random, again matching the original behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import gpytorch
import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine

from understanding_hdbo.fit import fit_mll
from understanding_hdbo.methods import Method, METHODS
from understanding_hdbo.model import build_model


ObjectiveFn = Callable[[torch.Tensor], torch.Tensor]
"""An objective takes a tensor of shape ``(d,)`` or ``(1, d)`` whose entries are
in ``[0, 1]`` and returns a scalar tensor with the value to *maximise*."""


@dataclass
class BOResult:
    """Container for the trajectory of a BO run."""

    x_train: torch.Tensor   # (n, d)
    y_train: torch.Tensor   # (n,)
    best_so_far: np.ndarray  # (n,)
    mean_lengthscale: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    method_name: str = ""


def _initial_design(n_init: int, dim: int, seed: Optional[int], dtype: torch.dtype) -> torch.Tensor:
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    return sobol.draw(n=n_init).to(dtype=dtype)


def bayesian_optimization(
    *,
    objective: ObjectiveFn,
    dim: int,
    method: Method | str,
    n_init: Optional[int] = None,
    n_iter: int = 100,
    noiseless: bool = True,
    init_seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.double,
    acq_raw_samples: int = 512,
    acq_num_restarts: int = 5,
    refit_every: int = 1,
    verbose: bool = True,
) -> BOResult:
    """Run Bayesian optimisation on ``objective`` using the given ``method``.

    Args:
        objective: Function mapping a point in ``[0, 1]^d`` to a scalar tensor
            (we maximise; pass ``-f`` if you want to minimise).
        dim: Dimensionality ``d`` of the input space.
        method: One of the keys in :data:`METHODS` or a :class:`Method` value.
        n_init: Size of the initial design.  Defaults to ``floor(3 * sqrt(d))``,
            matching the paper's evaluation protocol.
        n_iter: Number of acquisition iterations after the initial design.
        noiseless: If ``True`` (the default), freeze the GP likelihood noise to
            ``1e-4``. This is the right choice for synthetic GP-sample
            objectives. Pass ``False`` to let the noise be learned (e.g. for the
            real-world benchmarks Mopta08, Lasso-DNA, Ant, Humanoid, SVM, Rover),
            which matches the original ``--noiseless-model false`` setting.
        init_seed: Seed for the Sobol initial design.
        device: Torch device; defaults to CUDA if available, else CPU.
        dtype: Torch dtype; the original code used ``torch.double``.
        acq_raw_samples: Raw samples passed to ``optimize_acqf``.
        acq_num_restarts: Restarts for the multi-start acquisition optimiser.
        refit_every: Refit GP hyperparameters every ``refit_every`` iterations.
            When ``> 1``, iterations whose index is not a multiple of
            ``refit_every`` skip the (potentially expensive) MLE/MAP fit and
            reuse the most recent fitted hyperparameters - only the training
            set on the GP grows.  ``refit_every=1`` (default) reproduces the
            behaviour of the paper's experiments, in which the GP is refitted
            from scratch after every new observation.
        verbose: Whether to print per-iteration status lines.

    Returns:
        :class:`BOResult` containing the full ``(x, y)`` trajectory.
    """
    if refit_every < 1:
        raise ValueError(f"refit_every must be >= 1, got {refit_every!r}")
    if isinstance(method, str):
        try:
            method = METHODS[method]
        except KeyError as exc:
            raise KeyError(
                f"unknown method {method!r}; choose from {list(METHODS)}"
            ) from exc

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if n_init is None:
        n_init = int(math.floor(3.0 * math.sqrt(dim)))

    bounds = torch.stack(
        [torch.zeros(dim, dtype=dtype), torch.ones(dim, dtype=dtype)]
    ).to(device)

    # ---- initial design ----------------------------------------------------
    x_train = _initial_design(n_init, dim, seed=init_seed, dtype=dtype)
    y_train = torch.tensor(
        [float(objective(x).reshape(())) for x in x_train], dtype=dtype,
    )

    mean_ls_hist: list[float] = []
    last_fitted_state: dict | None = None

    # ---- BO loop -----------------------------------------------------------
    for it in range(n_iter):
        y_norm = (y_train - y_train.mean()) / y_train.std()

        # A fresh SingleTaskGP must be built every iteration so that the GP
        # carries the current training set; we additionally re-fit the
        # hyperparameters whenever ``it`` lines up with the ``refit_every``
        # cadence, and otherwise restore the most recent fitted hyperparameters
        # via load_state_dict.
        model = build_model(
            method=method,
            x_train=x_train,
            y_train=y_norm,
            dim=dim,
            noiseless=noiseless,
            device=device,
            dtype=dtype,
        )
        if (it % refit_every) == 0 or last_fitted_state is None:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            fit_mll(mll)
            last_fitted_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            model.load_state_dict(last_fitted_state)
            model.eval()

        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
        mean_ls_hist.append(float(np.mean(ls)))

        # ---- maximise acquisition --------------------------------------
        with gpytorch.settings.cholesky_max_tries(9):
            try:
                acq = LogExpectedImprovement(model=model, best_f=y_norm.max().item())
                options = {"batch_limit": 1}
                if method.raasp:
                    # RAASP / "sample around best" in BoTorch optimize_acqf.
                    # sigma=0.001 matches the BoTorch default used in the
                    # original runs (the legacy --scale-acq-distr flag was
                    # False).  prob_perturb is computed by BoTorch as
                    # min(1, 20/d) which is exactly the RAASP probability.
                    options["sample_around_best"] = True
                    options["sample_around_best_sigma"] = 0.001
                x_next, _ = optimize_acqf(
                    acq_function=acq,
                    bounds=bounds,
                    q=1,
                    num_restarts=acq_num_restarts,
                    raw_samples=acq_raw_samples,
                    options=options,
                )
                x_next = x_next.detach().cpu().to(dtype)
            except Exception as exc:  # noqa: BLE001 - mirror old behaviour
                print(f"acquisition optimisation failed ({exc!r}); sampling at random.")
                x_next = torch.rand(1, dim, dtype=dtype)

        y_next = torch.tensor(
            [float(objective(x_next).reshape(()))], dtype=dtype,
        )

        x_train = torch.cat([x_train, x_next])
        y_train = torch.cat([y_train, y_next])

        if verbose:
            print(
                f"[{method.name}] iter {it + 1}/{n_iter} "
                f"best={y_train.max().item():.4f} next={y_next.item():.4f} "
                f"mean_ls={mean_ls_hist[-1]:.3f}"
            )

    best_so_far = np.maximum.accumulate(y_train.cpu().numpy())
    return BOResult(
        x_train=x_train.cpu(),
        y_train=y_train.cpu(),
        best_so_far=best_so_far,
        mean_lengthscale=np.asarray(mean_ls_hist),
        method_name=method.name,
    )
