"""Reference implementation of MSR for high-dimensional Bayesian optimization.

Companion code for:

    Papenmeier, L., Poloczek, M., and Nardi, L.
    "Understanding High-Dimensional Bayesian Optimization."
    ICML 2025.

The package exposes a minimal BO loop that supports the five methods compared
in Table 1 / Figure 8 of the paper:

* ``MSR``                  - MLE with scaled initial length scale and RAASP.
* ``MLE_SCALED``           - MLE with scaled initial length scale, no RAASP.
* ``MLE_LN2``              - MLE with the GPyTorch default initial length scale
                             ln(2), no RAASP.
* ``MLE_LN2_RAASP``        - MLE with the default initial length scale ln(2)
                             and RAASP.
* ``DSP``                  - MAP with the dimensionality-scaled log-normal
                             prior of Hvarfner et al. (2024).
"""

from understanding_hdbo.methods import Method, METHODS
from understanding_hdbo.bo import bayesian_optimization

__all__ = ["Method", "METHODS", "bayesian_optimization"]
