"""Run a configured CHASM Monte Carlo experiment.

Examples:
    python -m experiments.run_simulation --config experiments/configs/gaussian.json
    python -m experiments.run_simulation --config experiments/configs/pareto.json --output results/pareto.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chasm import chasm_main, generate_corr_matrix_for_sim, mdim_pareto_rv

from .common import aggregate_decisions


def _matrix(value: Any, dimension: int) -> np.ndarray:
    if value == "identity":
        return np.eye(dimension)
    if value == "default":
        return generate_corr_matrix_for_sim(dimension)
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (dimension, dimension):
        raise ValueError("Covariance matrix does not match dimension")
    return matrix


def _gaussian_segments(config: dict[str, Any], scenario: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    dimension = int(config["dimension"])
    sample_size = int(config["sample_size"])
    loc1 = np.asarray(scenario.get("loc1", [0.0] * dimension), dtype=float)
    loc2 = np.asarray(scenario.get("loc2", loc1), dtype=float)
    cov1 = _matrix(scenario.get("cov1", "identity"), dimension)
    cov2 = _matrix(scenario.get("cov2", "identity"), dimension)
    return (
        np.random.multivariate_normal(loc1, cov1, sample_size),
        np.random.multivariate_normal(loc2, cov2, sample_size),
    )


def _pareto_segments(config: dict[str, Any], scenario: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    dimension = int(config["dimension"])
    sample_size = int(config["sample_size"])
    return (
        mdim_pareto_rv(scenario["beta1"], scenario["alpha1"], dimension, sample_size),
        mdim_pareto_rv(scenario["beta2"], scenario["alpha2"], dimension, sample_size),
    )


def run(config: dict[str, Any]) -> pd.DataFrame:
    """Execute all scenarios from a JSON configuration."""
    np.random.seed(config.get("seed"))
    distribution = config["distribution"]
    generator = _gaussian_segments if distribution == "gaussian" else _pareto_segments
    rows = []
    for scenario in config["scenarios"]:
        results = []
        for _ in range(int(config["trials"])):
            segment1, segment2 = generator(config, scenario)
            results.append(
                chasm_main(
                    segment1,
                    segment2,
                    confidence_level=float(config.get("confidence_level", 0.95)),
                    n_resamples=int(config.get("n_resamples", 100)),
                )
            )
        rows.append({"scenario": scenario["name"], **aggregate_decisions(results)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CHASM experiment from a JSON file")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/simulation.csv"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    run(config).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
