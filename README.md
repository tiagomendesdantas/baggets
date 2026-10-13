# baggets

[![CI](https://github.com/tiagomendesdantas/baggets/actions/workflows/ci.yml/badge.svg)](https://github.com/tiagomendesdantas/baggets/actions/workflows/ci.yml)

Python reimplementation of **Bagged.Cluster.ETS**, the forecasting method from:

> Dantas, T.M. & Cyrino Oliveira, F.L. (2018). *Improving time series forecasting: An approach
> combining bootstrap aggregation, clusters and exponential smoothing.* International Journal of
> Forecasting, 34(4), 748–761. https://doi.org/10.1016/j.ijforecast.2018.05.006

The R code released with the paper is at
[tiagomendesdantas/Bagged.Cluster.ETS](https://github.com/tiagomendesdantas/Bagged.Cluster.ETS).
A verbatim copy lives in `validation_r/original/`, where this package is tested against it.

The method generates bootstrapped versions of a time series (Box-Cox + STL + moving block
bootstrap), evaluates them on a validation window, and builds a deliberately *less-correlated*
ensemble of ETS forecasts by selecting series across PAM clusters. The aim is the covariance term
of the ensemble variance, which plain bagging ignores. The final forecast is the pointwise median.

## Install

Python 3.11 or newer; tested on 3.12. baggets is not on PyPI; install it from GitHub:

```bash
pip install "git+https://github.com/tiagomendesdantas/baggets"
```

Two extras are optional: `torch` for the N-BEATS base learner and `aws` for the S3 artifact store,
e.g. `pip install "baggets[torch] @ git+https://github.com/tiagomendesdantas/baggets"`. To work on
the code, clone the repository and use `uv sync` (see [Development](#development)).

## Quick start

```python
import numpy as np
from baggets import BaggedETS, ClusterSelection

# ten years of a trending monthly series with a yearly cycle
rng = np.random.default_rng(0)
t = np.arange(120)
y = (100 + 0.5 * t) * (1 + 0.2 * np.sin(2 * np.pi * t / 12)) + rng.normal(0, 2, t.size)

model = BaggedETS(season_length=12, n_bootstraps=200,
                  selection=ClusterSelection(n_clusters=5), random_state=42)
model.fit(y)
fc = model.predict(h=12)
print(fc.point.round(1))   # median of the member forecasts
print(fc.ensemble.shape)   # (members, horizon)
```

Output:

```
[159.  176.3 189.  192.9 189.1 180.2 163.9 148.7 136.6 131.2 136.6 150.4]
(100, 12)
```

This takes about a minute on one core. To stay short it uses 200 bootstraps and a fixed k = 5.
The paper's configuration is 1,000 bootstraps (the default) with k chosen by silhouette width
(`ClusterSelection(n_clusters="auto")`); `ClusterSelection.paper()` pins its exact constants.

## Results

Two experiments with different setups. Their numbers are not comparable with each other.

**Reproduction on M3 monthly**: all 1,428 series, 1,000 bootstraps, 100 members
([reports/validation_report.md](reports/validation_report.md)).

- Reproduced: bagging improves on plain ETS, by about 0.4 points of mean sMAPE, for every variant
  tested. Plain ETS scores 14.253 against the paper's 14.135, and plain bagging under Bergmeir et
  al.'s protocol 13.846 against 13.653, both inside their tolerance gates.
- Not reproduced: the advantage of cluster selection over plain bagging. The paper reports 13.617
  against 13.653. Here the 2018 configuration scores 13.847 against 13.846, which also misses its
  own gate (paper value ± 0.20) by 0.03. The eight selection variants other than random are
  statistically indistinguishable (Friedman + Hochberg, adjusted p = 0.72 against the best-ranked
  control). The paper's own Table 6 did not find that edge significant either.

The report's reading is that at 100 members on M3 monthly, which members are selected makes no
measurable difference. Follow-up sweeps over ensemble size and aggregation are in
[reports/m6_findings.md](reports/m6_findings.md).

**N-BEATS as the base learner**: the first 300 M3 monthly series by id, 100 bootstraps, plain
bagging with mean aggregation ([reports/m8_nbeats.md](reports/m8_nbeats.md)). Each N-BEATS member
is trained on its own bootstrapped series, so it sees tens of windows; the N-BEATS paper instead
trains one model across all series.

| base learner | mean sMAPE | mean MASE |
|---|---|---|
| ETS | 24.59 | 0.723 |
| N-BEATS, interpretable | 25.28 | 0.775 |
| N-BEATS, generic | 71.87 | 2.161 |

ETS is ahead of interpretable N-BEATS under a pairwise Wilcoxon test (p = 0.045) but not after the
Friedman + Hochberg correction (p = 0.072). Generic N-BEATS beats ETS on 4 of the 300 series. These
results describe N-BEATS trained per series on little data, not N-BEATS as published.

## What the package contains

1. **The published method**, with statsforecast's AutoETS as the ETS engine, run with a tighter
   optimizer tolerance and more iterations than its default (`src/baggets/engine.py` explains
   why). It is checked against the original R code piece by piece, as described in the next
   section.
2. **A research platform**: ensemble-selection strategies are pluggable, so the paper's
   cluster-based selection can be compared directly with covariance-direct alternatives and
   ablations:
   - `ClusterSelection`: the paper's method (`.paper()` pins the exact 2018 constants);
   - `GreedyCovarianceSelection`: Caruana-style forward selection minimizing the
     *aggregated* ensemble's validation error;
   - `PortfolioSelection`: greedy bias² + covariance minimization (Ledoit-Wolf-shrunk),
     the direct formalization of the paper's Eq. (4);
   - `TopKSelection` (accuracy without diversity, the ablation the paper lacked),
     `RandomSelection`, `NoSelection` (Bergmeir et al.'s protocol via `NoSelection(100)` + mean).

## Validation against the original R code

The scripts in `validation_r/` run R (4.3, with `forecast`, `cluster` and `Mcomp`), including the
original 2018 code, and commit the outputs as fixtures, so the tests compare against R without
needing it.
The gates are frozen in [tests/fixtures/TOLERANCES.md](tests/fixtures/TOLERANCES.md).

| what is compared | gate | series | where it runs |
|---|---|---|---|
| Guerrero Box-Cox λ, the Box-Cox transform, STL (or loess) components | λ within 1e-3; transform within 1e-9; STL within 1e-9 × sd, loess within 1e-3 × sd | 12 fixture series: 8 M3 monthly, 2 quarterly, 1 yearly, 1 synthetic | CI (`tests/test_r_fixtures.py`) |
| Moving block bootstrap draws | KS test p > 0.01 on per-draw sd and pooled values | 2 of the fixture series | CI (`tests/test_r_fixtures.py`) |
| PAM clustering against `cluster::pam` | identical partitions for k = 2, 5, 20, 50; silhouette widths within 1e-9; same best k | one fixed distance matrix | CI (`tests/test_pam_vs_r.py`) |
| AutoETS against `forecast::ets` | same model form on ≥ 60% of series; scaled forecast distance median ≤ 0.10, max ≤ 1.5 | 30 M3 monthly series | local only: needs the M3 export, skipped without it (`tests/test_engine.py`) |
| The whole pipeline against the original `baggedClusterETS.R` (seed 42, B = 1,000) | R's test sMAPE inside the range of 10 Python seeds (one miss allowed); per-series difference within max(1.5, 3 × seed sd) and mean difference within ± 0.5 sMAPE points | 8 fixture series with k = 5, 2 of them also with automatic k | by hand (`experiments/tier4_compare.py`) |

The two ETS engines search the same 30-model space with different optimizers, so they do not
always pick the same model, and the ETS check is statistical rather than exact. For the same
reason the whole-pipeline gates apply only to series where plain ETS from the two engines agrees
within 0.5 sMAPE points; on the others, the gap is compared with the engine gap instead.

**Speed.** On one core of an Apple M2, with B = 1,000, k = 5 and 100 members, baggets took 69 to
183 seconds per series and the original R code 224 to 262 seconds, on the same three series: 1.4
to 3.2 times less time. Conditions and scripts are in [benchmarks/](benchmarks/README.md).

## A neural base learner: N-BEATS

The base learner is pluggable (`baggets.engine.ForecastEngine`). `ETSEngine` is
the paper's and the default; `NBeatsEngine` swaps in a PyTorch implementation of
N-BEATS (Oreshkin et al., ICLR 2020) so the *same* bootstrap machinery, member
selection, metrics and M3 harness compare a statistical base learner against a
deep one on identical footing.

```python
from baggets import BaggedETS, NBeatsConfig, NBeatsEngine, NoSelection

# y as in the quick start; needs the torch extra.
# Trains 200 small networks (validation + final members): a few minutes.
model = BaggedETS(season_length=12, n_bootstraps=100, n_select=100,
                  selection=NoSelection(100), aggregator="mean",   # plain bagging
                  engine=NBeatsEngine(12, NBeatsConfig(interpretable=True)))
model.fit(y)
model.predict(h=18).point
```

Two departures from the published N-BEATS, both because it is a base learner
inside a per-series bootstrap ensemble rather than the paper's setup, and both
worth stating plainly rather than discovering in the numbers:

1. **Per series, not cross-learning.** `fit(y)` receives one series, so a model
   sees only that series' windows. N-BEATS's published M3/M4 results come from
   cross-learning over all series, with orders of magnitude more data. On an M3
   monthly series of 50 points this leaves 8 windows: 6 to train on and 2 for
   early stopping.
2. **Depth and width scaled to the data.** The paper's generic stack is 30
   blocks of width 512. The defaults here are far smaller; the paper's sizes are
   still reachable through `NBeatsConfig`.

The engine falls back to a naive forecast, and says so, on series too short to
yield training windows, the same discipline `ETSEngine` applies to a failed fit.

Training can be instrumented: pass `monitor_factory=` to `NBeatsEngine` and the
training loop reports each epoch and batch to your monitor (the interface is in
the `NBeatsEngine` docstring; baggets depends on no monitoring library).
`experiments/probe_nbeats_training.py` is a complete example that counts dead
ReLU units per layer. On M3 series N1402 (generic N-BEATS, 400 epochs with early
stopping off, seed 7) it finds a layer with 60% dead units, and the validation
loss bottoms at epoch 56 of 400.

```bash
uv sync --extra torch
Rscript validation_r/export_m3_mcomp.R      # M3 data from R's Mcomp, written to data/m3/
uv run python experiments/run_m3.py precompute --group monthly --limit 300 \
    --n-bootstraps 100 --engine nbeats-i --epochs 150 --n-jobs 6
```

**Cost, and why neural runs are matched-subset.** An ETS fit is milliseconds; an
N-BEATS fit is seconds. At the paper's B=1000 over 1,428 series that is the
difference between an overnight run and an infeasible one. Neural runs therefore
use a smaller B on a series subset, and the ETS arm is rerun under the *same* B
and the *same* series, because otherwise the comparison means nothing.

## An S3 artifact store

Stage A is the expensive half, and its per-series cache is worth moving off one
laptop. `baggets.aws.S3ArtifactStore` (extra: `aws`, uses boto3 and the standard
credential chain) reads and writes those `.npz` tensors, and model checkpoints as
state dicts. It stores weights only, because `torch.save` of a whole module pickles
code paths, and an artifact store is the wrong place for something whose load is
arbitrary code execution.

## Development

```bash
uv sync                     # core dependencies + pytest and ruff (what CI's core job installs)
uv run pytest -n auto -rs   # what CI runs; R is not needed
uv sync --all-extras        # add torch, aws and the experiment tools
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs the tests without PyTorch, and again
with the `torch` and `aws` extras in a separate job that is allowed to fail. Tests that need the
M3 export or R skip with a reason. The M3 data comes from R's `Mcomp` package
(`Rscript validation_r/export_m3_mcomp.R`, written to `data/m3/`).

Regenerating the R fixtures (R 4.3+ with the `forecast`, `cluster` and `Mcomp` packages) is manual:
see `validation_r/README.md` and `make fixtures`.

## Layout

- `src/baggets/`: the package (bootstrap core, ETS engine wrapper, selection strategies, estimator)
- `tests/`: unit tests + committed R-generated fixtures
- `validation_r/`: scripts that generate reference fixtures from the original R code, and that code
- `experiments/`: M3 benchmark runner (precompute / evaluate / report) and analysis scripts
- `reports/`: write-ups of the experiments, with the commands that reproduce them
- `benchmarks/`: timing of the full method against the original R code

## Citation

If you use baggets, please cite the 2018 paper above. [CITATION.cff](CITATION.cff) has the
metadata for the paper and for this software.
