# Understanding High-Dimensional Bayesian Optimization

Companion code for

> Papenmeier, L., Poloczek, M., and Nardi, L. **Understanding High-Dimensional
> Bayesian Optimization.** *Proceedings of the 42nd International Conference on
> Machine Learning (ICML)*, 2025. [arXiv:2502.09198](https://arxiv.org/abs/2502.09198)

This repository is a minimal, publication-ready reference implementation of the
**MSR** (MLE Scaled with RAASP) algorithm proposed in the paper, together with
the four comparator BO setups used to produce Figures 8-10:

| Method               | Length-scale prior        | Initial length scale            | RAASP sampling |
|----------------------|---------------------------|---------------------------------|----------------|
| `msr` (ours)         | none (pure MLE)           | sqrt(d) / 10                    | yes            |
| `mle_scaled`         | none (pure MLE)           | sqrt(d) / 10                    | no             |
| `mle_ln2`            | none (pure MLE)           | ln 2 (GPyTorch default)         | no             |
| `mle_ln2_raasp`      | none (pure MLE)           | ln 2 (GPyTorch default)         | yes            |
| `dsp`                | dimensionality-scaled LogNormal | mode of the prior         | yes            |

The implementation is a faithful port of the original research code
([github.com/LeoIV/sample-from-high-dim-GP](https://github.com/LeoIV/sample-from-high-dim-GP),
hereafter "the original code") and has been verified to produce
**numerically identical BO trajectories** for all five methods on a fixed
synthetic GP-sample objective with fixed seeds; see `tests/test_equivalence.py`.

## Installation

```bash
pip install -e .                  # core
pip install -e .[bencher,dev]     # for benchmark evaluation and tests
```

The code targets Python >= 3.11 and is tested against the same dependency
versions used to produce the paper's results: `botorch==0.9.5`, `gpytorch==1.11`,
`torch==2.3.0`, `numpy==1.26.4`.

## Quick start

Run MSR on a synthetic 50-dimensional GP-sample objective:

```bash
hdbo-run --objective gp-sample --dim 50 --method msr --n-iter 30
```

Run MSR on a real-world benchmark via the bencher gRPC server (assumes a
bencher Docker container is listening on `localhost:50051`; see the original
repo's `container.sdef` for the server setup):

```bash
hdbo-run --objective bencher --benchmark mopta08 --method msr \
         --n-iter 1000 --seed 1 --out runs/mopta08_msr_1
```

The output directory contains `x_train.npy`, `y_train.npy`, `best_so_far.npy`,
`mean_lengthscale.npy`, and `args.json`.

## Refitting cadence (speed knob)

The default behaviour refits the GP hyperparameters from scratch after *every*
new observation, matching the paper's evaluation protocol. On the highest-
dimensional benchmarks (e.g. 888-d Ant, 6392-d Humanoid) the MLE/MAP fit is
the dominant cost. The `--refit-every N` flag (and the equivalent
`refit_every=N` argument of `bayesian_optimization`) lets you refit only every
`N` iterations and reuse the most recently fitted hyperparameters on the
intermediate iterations:

```bash
# Run MSR on Mopta08, refitting only every 5th iteration:
hdbo-run --objective bencher --benchmark mopta08 --method msr \
         --n-iter 1000 --refit-every 5 --out runs/mopta08_msr_refit5
```

`refit_every=1` (the default) reproduces the paper's experiments exactly; any
larger value trades modelling fidelity for runtime. The training data on the
GP still grows every iteration - only the (potentially expensive) MLL/MAP
optimisation is skipped. As a quick reference, on a 50-d GP-sample objective:

| `refit_every` | Time for 10 iters | Distinct fitted length scales |
|---------------|-------------------|-------------------------------|
| 1 (default)   | 15.7 s            | 10                            |
| 4             | 7.3 s             | 3 (iters 0, 4, 8)             |

If you publish results obtained with `refit_every > 1`, please report the
value alongside the trajectories.

## Reproducing Figures 8-10

```bash
bash scripts/reproduce_fig8.sh                       # all methods, all benchmarks
bash scripts/reproduce_fig8.sh msr mopta08 1         # one configuration
```

The script loops over the five methods and six benchmarks
(`mopta08`, `lasso-dna`, `svm`, `mujoco-ant`, `mujoco-humanoid`, `rover`) and
uses the iteration counts reported in the paper. Each individual run writes
to `runs/fig8/<benchmark>_<method>_<rep>/`.

## Benchmark setup

The real-world benchmarks are evaluated through the bencher gRPC service used
in the original research code. To start it:

```bash
# (in the original repo)
docker build -t bencher .          # or use container.sdef with Apptainer
docker run --rm -p 50051:50051 bencher
```

The `understanding_hdbo.objectives.bencher_objective` helper wraps the
`bencherscaffold` client and assumes the server is reachable at
`localhost:50051`. Pass `--hostname` to `validate_bencher.py` for a different
host.

## Validating against the bencher

A short end-to-end smoke test on Mopta08 (124-d, cheapest of the real-world
benchmarks):

```bash
python scripts/validate_bencher.py --benchmark mopta08 --n-iter 20
```

This evaluates 20 + floor(3 * sqrt(124)) = 53 points and writes the
trajectory to `runs/validate_bencher/`.

## Behavioural equivalence to the original code

`tests/test_equivalence.py` runs both the published BO loop and a reference
loop built directly from the original code's `objective_functions.get_model`
and `botorch.optim.optimize_acqf`, on the same synthetic GP-sample objective
with the same seeds, for each of the five methods. Trajectories agree to
`1e-6` absolute / relative tolerance:

```bash
pytest tests/test_equivalence.py -v
```

This test requires the original `sample-from-high-dim-GP` repo to be on the
Python path (edit `OG_REPO` at the top of the test file to point at your
local checkout).

## Package layout

```
src/understanding_hdbo/
    methods.py      # Method configurations (Table 1)
    model.py        # GP construction (ARD Matern-5/2, optional DSP prior)
    fit.py          # MLE/MAP fitting with Adam fallback
    bo.py           # Main BO loop with LogEI + optional RAASP
    objectives.py   # Objective adapters (GP sample + bencher client)
    run.py          # CLI entry point (hdbo-run)
scripts/
    reproduce_fig8.sh
    validate_bencher.py
tests/
    test_equivalence.py
```

## Notes on faithful reproduction

A small number of details in the GP construction matter for bit-level
equivalence to the original runs and are documented inline:

1. The dimensionality-scaled log-normal prior used by `dsp` is built with
   the literal constant `1.41` (rather than `sqrt(2)`) and uses PyTorch's
   default float dtype during construction, matching the original code.
2. The GP likelihood noise is frozen to `1e-4` by default (`--noiseless`),
   which is correct for synthetic GP-sample objectives. Pass `--no-noiseless`
   to let the noise be learned (e.g. for real-world benchmarks), matching the
   original `--noiseless-model false` setting.
3. RAASP is delegated to BoTorch's `optimize_acqf(..., options={
   "sample_around_best": True, "sample_around_best_sigma": 0.001})`, which
   reproduces the random axis-aligned subspace perturbation used in the
   paper.
4. The Adam fallback in `fit_mll` is preserved unchanged from the original
   code.

## Citation

```bibtex
@inproceedings{papenmeier2025understanding,
  title     = {Understanding High-Dimensional Bayesian Optimization},
  author    = {Papenmeier, Leonard and Poloczek, Matthias and Nardi, Luigi},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  year      = {2025}
}
```

## License

MIT.
