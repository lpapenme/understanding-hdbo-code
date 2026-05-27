"""Unit tests for understanding_hdbo.fit."""

from __future__ import annotations

import pytest

from understanding_hdbo.fit import fit_mll


@pytest.mark.slow
def test_fit_mll_runs_and_leaves_model_in_eval(tiny_mll):
    result = fit_mll(tiny_mll)
    assert result is None
    assert tiny_mll.model.training is False
