"""Method definitions for the five BO methods compared in the paper.

Each :class:`Method` specifies the three knobs that distinguish the methods in
Table 1 of the paper:

1. ``ls_prior``  - whether and which length scale prior is placed on the GP.
2. ``ls_init``   - how the initial value of the length scale is chosen for the
                   gradient-based optimizer that maximises the (penalised)
                   marginal log likelihood.
3. ``raasp``     - whether random axis-aligned subspace perturbation (RAASP)
                   sampling is used to seed the acquisition-function optimiser.

The values below reproduce the configurations used to produce Figures 8, 9, and
10 in the paper (see ``test_acq_effect_real_world.txt`` in the original code
release).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

PriorName = Literal["none", "dsp"]
InitMode = Literal["ln2", "scaled", "prior_mode"]


@dataclass(frozen=True)
class Method:
    """Configuration of a single high-dimensional BO method."""

    name: str
    ls_prior: PriorName
    ls_init: InitMode
    raasp: bool
    description: str = ""

    def initial_lengthscale(self, dim: int) -> float:
        """Return the initial length scale used to seed MLE/MAP optimisation."""
        if self.ls_init == "ln2":
            # GPyTorch default initial value.
            return math.log(2.0)
        if self.ls_init == "scaled":
            # MSR / MLE-scaled: l_init = sqrt(d) / 10
            return math.sqrt(dim) / 10.0
        if self.ls_init == "prior_mode":
            # DSP: initial value = mode of the LogNormal length-scale prior
            # used by Hvarfner et al. (2024):  loc = DSP_LOC_CONST + log(d) / 2,
            # scale = sqrt(3); mode = exp(loc - scale**2).
            loc = DSP_LOC_CONST + math.log(dim) / 2.0
            return math.exp(loc - 3.0)
        raise ValueError(f"unknown ls_init {self.ls_init!r}")


# Note on the DSP prior parameters.
# Hvarfner et al. (2024) define the LogNormal length-scale prior as
# LogNormal(sqrt(2) + log(d)/2, sqrt(3)).  The original BoTorch / paper code
# released alongside our manuscript uses the truncated value ``1.41`` for
# sqrt(2).  We keep ``1.41`` here so that this reference implementation
# numerically reproduces the runs used in the paper.
DSP_LOC_CONST: float = 1.41
DSP_SCALE: float = math.sqrt(3.0)

METHODS: dict[str, Method] = {
    "msr": Method(
        name="MSR",
        ls_prior="none",
        ls_init="scaled",
        raasp=True,
        description="MLE with scaled initial length scale (sqrt(d)/10) and RAASP",
    ),
    "mle_scaled": Method(
        name="MLE (scaled)",
        ls_prior="none",
        ls_init="scaled",
        raasp=False,
        description="MLE with scaled initial length scale (sqrt(d)/10), no RAASP",
    ),
    "mle_ln2": Method(
        name="MLE (l=ln 2)",
        ls_prior="none",
        ls_init="ln2",
        raasp=False,
        description="MLE with GPyTorch default initial length scale ln(2), no RAASP",
    ),
    "mle_ln2_raasp": Method(
        name="MLE (l=ln 2, RAASP)",
        ls_prior="none",
        ls_init="ln2",
        raasp=True,
        description="MLE with default initial length scale ln(2) and RAASP",
    ),
    "dsp": Method(
        name="DSP",
        ls_prior="dsp",
        ls_init="prior_mode",
        raasp=True,
        description="MAP with the dimensionality-scaled log-normal prior of "
                    "Hvarfner et al. (2024) and RAASP",
    ),
}
