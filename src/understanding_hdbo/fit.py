"""GP hyperparameter fitting.

This file replicates the behaviour of ``util.fit_mll`` from the original code:

* Try the BoTorch default fitter (``fit_gpytorch_mll``) under a
  ``gpytorch.settings.cholesky_max_tries(9)`` context.
* On a :class:`botorch.exceptions.ModelFittingError`, fall back to Adam for
  500 iterations using lr=0.01.
* On a :class:`linear_operator.utils.errors.NotPSDError` *during* the fallback,
  give up silently and leave the model in eval mode (this matches the original
  code; in practice the failure happens at most once per run on degenerate
  Cholesky factorisations).

The diagnostic ``RollingAverageForwardBackwardClosure`` from the original code
is intentionally omitted - it only recorded fitting gradients for plotting and
did not affect the optimisation trajectory.
"""

from __future__ import annotations

import gpytorch
import torch
from botorch import fit_gpytorch_mll
from botorch.exceptions import ModelFittingError
from gpytorch.mlls import MarginalLogLikelihood
from linear_operator.utils.errors import NotPSDError


def fit_mll(mll: MarginalLogLikelihood, n_iter: int = 500) -> None:
    """Fit ``mll`` in place, with the original code's Adam fallback."""
    model = mll.model
    with gpytorch.settings.cholesky_max_tries(9):
        try:
            fit_gpytorch_mll(mll)
        except ModelFittingError as e:
            print(f"fit_gpytorch_mll failed: {e}; falling back to Adam.")
            try:
                train_x = model.train_inputs[0]
                train_y = model.train_targets
                model.train()
                optimizer = torch.optim.Adam(
                    [{"params": model.parameters()}], lr=0.01
                )
                for _ in range(n_iter):
                    optimizer.zero_grad()
                    output = model(train_x)
                    loss = -mll(output, train_y.flatten())
                    loss.backward()
                    optimizer.step()
                model.eval()
            except NotPSDError as nerr:
                print(f"Adam fallback also failed: {nerr}; skipping fit.")
                model.eval()
