"""Unit tests for understanding_hdbo.run argument parsing."""

from __future__ import annotations

import sys

from understanding_hdbo.run import _parse_args

_BASE_ARGV = ["hdbo-run", "--method", "msr", "--objective", "gp-sample", "--dim", "5"]


def _parse(monkeypatch, extra=()):
    monkeypatch.setattr(sys, "argv", _BASE_ARGV + list(extra))
    return _parse_args()


def test_parse_args_defaults(monkeypatch):
    args = _parse(monkeypatch)
    assert args.method == "msr"
    assert args.objective == "gp-sample"
    assert args.dim == 5
    assert args.n_iter == 100
    assert args.refit_every == 1
    assert args.seed == 0


def test_parse_args_noiseless_default_true(monkeypatch):
    # Regression guard: --noiseless now defaults on.
    args = _parse(monkeypatch)
    assert args.noiseless is True


def test_parse_args_no_noiseless(monkeypatch):
    args = _parse(monkeypatch, ["--no-noiseless"])
    assert args.noiseless is False


def test_parse_args_explicit_noiseless(monkeypatch):
    args = _parse(monkeypatch, ["--noiseless"])
    assert args.noiseless is True
