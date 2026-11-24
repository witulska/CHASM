# CHASM: Significance of Changes

CHASM (Change-point Hypothesis Assessment for Significance in Multivariate data)
detects statistically significant changes between two multivariate data
segments.

This repository presents methodology of CHASM described in 'Unsupervised and explainable authorization of change point in multivariate industrial data for trustworthy system and process monitoring' by J. Witulska, T. Barszcz, A. Wyłomańska, I. Jabłoński.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,simulations,notebooks]"
```

Alternatively, install dependencies from `requirements.txt` and
`requirements/quality.txt`.

For the Jupyter example, also install `requirements/notebooks.txt`.

## Library usage

```python
import numpy as np
from chasm import chasm_main

rng = np.random.default_rng(42)
segment1 = rng.normal(size=(200, 2))
segment2 = rng.normal(loc=1.0, size=(200, 2))
results = chasm_main(segment1, segment2, n_resamples=100)
```

`chasm_main` returns one result dictionary per metric. Every result follows the
same schema:

```python
{
    "decision": 0 or 1,
    "upper_bound": float or None,
    "lower_bound": float or None,
    "stat_value": float,
    "imf": float,
}
```

The bootstrap reference distribution is built from the two halves of
`segment1`, while the observed statistic compares the complete `segment1` and
`segment2`. Lower-bound metrics use `stat_value < lower_bound`; correlations
and divergences use `stat_value > upper_bound`.

The package includes L-moments, robust and classical moments, Spearman
correlation, entropy, and Bhattacharyya, Jeffrey and KL divergences. The
`chasm_bounds_main` function returns the metrics used in the article by
default. It also accepts a `metrics` sequence, so a caller can select a
different subset.

## Single simulated Gaussian example

The following example reproduces one Gaussian mean-change scenario: both
segments have identity covariance, but the mean changes from `[0, 0]` to
`[1, 1]`.

```python
import numpy as np

from chasm import chasm_main

np.random.seed(2024)
n_observations = 100
covariance = np.eye(2)

segment1 = np.random.multivariate_normal(
    mean=[0.0, 0.0],
    cov=covariance,
    size=n_observations,
)
segment2 = np.random.multivariate_normal(
    mean=[1.0, 1.0],
    cov=covariance,
    size=n_observations,
)

results = chasm_main(
    segment1,
    segment2,
    confidence_level=0.95,
    n_resamples=50,
)

for metric_name, metric_result in results.items():
    print(
        metric_name,
        "decision=",
        metric_result["decision"],
        "imf=",
        metric_result["imf"],
    )
```

For a given metric, `decision=1` means that CHASM detected a significant
change. The complete result, including the confidence bound and observed
statistic, is available in `metric_result`.

## Using only the classical mean

The bootstrap helpers are public, so an application can use only one
statistic. The following example defines a classical mean-change score and
returns the same dictionary schema as the built-in metrics:

```python
import numpy as np

from chasm import bootstraping_one_dim_stats


def classical_mean_score(
    first: np.ndarray,
    second: np.ndarray,
    axis: int | None = None,
) -> np.ndarray:
    """Return a lower-is-more-significant score based on the mean difference."""
    difference = np.abs(
        np.mean(first, axis=axis) - np.mean(second, axis=axis)
    )
    return np.array([-np.log(difference + np.finfo(float).eps)])


mean_result = bootstraping_one_dim_stats(
    segment1,
    segment2,
    statistic=classical_mean_score,
    confidence_level=0.95,
    n_resamples=100,
)
print(mean_result)
```

`mean_result` is a dictionary with the keys `decision`, `upper_bound`,
`lower_bound`, `stat_value` and `imf`. A one-dimensional custom statistic
should follow the same convention as the built-in one-dimensional metrics:
the score is compared with a lower confidence bound and a smaller observed
score indicates a change.

## Defining custom statistics

Use the helper that matches the direction and input structure of the
statistic:

- `bootstraping_one_dim_stats` is for differences between statistics of
  one-dimensional signals. The callable receives `(first, second, axis=...)`.
- `bootstraping_divergences` is for differences between distributions. The
  callable receives the two sample arrays and should return a scalar-like
  value or a one-element array. Its observed value is compared with an upper
  confidence bound.
- `bootstraping_corr` is for correlation-like statistics. By default it uses
  Spearman correlation, but a custom callable with the same signature can be
  passed through the `statistic` argument.

For example, a Wasserstein-distance statistic can be bootstrapped as follows:

```python
from scipy.stats import wasserstein_distance

from chasm import bootstraping_divergences


def wasserstein_statistic(
    first: np.ndarray,
    second: np.ndarray,
    axis: int | None = None,
) -> np.ndarray:
    del axis
    distances = [
        wasserstein_distance(first[:, column], second[:, column])
        for column in range(first.shape[1])
    ]
    return np.array([max(distances)])


divergence_result = bootstraping_divergences(
    segment1,
    segment2,
    statistic=wasserstein_statistic,
    confidence_level=0.95,
    n_resamples=100,
)
```

`main.py` exposes `ARTICLE_METRICS` and `chasm_bounds_main`, whose default
selection is the set of statistics used for the article: L-mean, L-scale,
L-skewness, L-kurtosis, robust scale, Spearman correlation, Bhattacharyya
distance and Jeffrey distance. This is a default selection, not a limitation:
pass another sequence of metric names to `chasm_bounds_main`, or call a
bootstrap helper directly for a new statistic.

## Reproducing simulations

Simulation parameters are kept separate from execution code:

```bash
python -m experiments.run_simulation \
  --config experiments/configs/gaussian.json \
  --output results/gaussian.csv

python -m experiments.run_simulation \
  --config experiments/configs/pareto.json \
  --output results/pareto.csv
```

Results are generated locally and the output directory is ignored by Git. The
original archives contained large CSV snapshots, notebooks and figures; these
are not part of the source repository. The experiment configurations and
drivers are the reproducible part of the workflow.

## Layout

- `src/chasm/` — installable library and scoring modules.
- `experiments/` — parameter files and Monte Carlo simulation drivers.
- `notebooks/` — runnable examples of simulated data and custom statistics.
- `tests/` — automated API, utility and aggregation tests.
- `LICENSE` — license retained from the source archives.
