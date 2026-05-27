"""Unit tests for understanding_hdbo.methods."""

from __future__ import annotations

import dataclasses
import math

import pytest

from understanding_hdbo.methods import (
    DSP_LOC_CONST,
    DSP_SCALE,
    METHODS,
    Method,
)


@pytest.mark.parametrize("dim", [1, 10, 100, 1000])
def test_initial_lengthscale_ln2_is_constant(dim):
    m = Method(name="x", ls_prior="none", ls_init="ln2", raasp=False)
    assert m.initial_lengthscale(dim) == math.log(2.0)


@pytest.mark.parametrize("dim", [1, 10, 100, 1000])
def test_initial_lengthscale_scaled(dim):
    m = Method(name="x", ls_prior="none", ls_init="scaled", raasp=False)
    assert m.initial_lengthscale(dim) == pytest.approx(math.sqrt(dim) / 10.0)


@pytest.mark.parametrize("dim", [1, 10, 100, 1000])
def test_initial_lengthscale_prior_mode(dim):
    m = Method(name="x", ls_prior="dsp", ls_init="prior_mode", raasp=False)
    expected = math.exp(DSP_LOC_CONST + math.log(dim) / 2.0 - 3.0)
    assert m.initial_lengthscale(dim) == pytest.approx(expected)


def test_initial_lengthscale_unknown_mode_raises():
    m = Method(name="x", ls_prior="none", ls_init="bogus", raasp=False)
    with pytest.raises(ValueError, match="unknown ls_init"):
        m.initial_lengthscale(10)


# (key, ls_prior, ls_init, raasp) for each registered method.
_EXPECTED = [
    ("msr", "none", "scaled", True),
    ("mle_scaled", "none", "scaled", False),
    ("mle_ln2", "none", "ln2", False),
    ("mle_ln2_raasp", "none", "ln2", True),
    ("dsp", "dsp", "prior_mode", True),
]


def test_methods_registry_keys():
    assert set(METHODS) == {key for key, *_ in _EXPECTED}


@pytest.mark.parametrize("key,ls_prior,ls_init,raasp", _EXPECTED)
def test_methods_registry_fields(key, ls_prior, ls_init, raasp):
    m = METHODS[key]
    assert m.ls_prior == ls_prior
    assert m.ls_init == ls_init
    assert m.raasp is raasp


def test_dsp_constants():
    assert DSP_LOC_CONST == 1.41
    assert DSP_SCALE == math.sqrt(3.0)


def test_method_is_frozen():
    m = METHODS["msr"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.raasp = False  # type: ignore[misc]
