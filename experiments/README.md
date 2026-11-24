# Paper-style simulations

The JSON files in `configs/` contain parameters, while the driver contains only
execution logic. Results are generated locally and intentionally excluded from
version control.

```bash
python -m experiments.run_simulation \
  --config experiments/configs/gaussian.json \
  --output results/gaussian.csv

python -m experiments.run_simulation \
  --config experiments/configs/pareto.json \
  --output results/pareto.csv
```

Each output row describes a scenario and each `sig_*` column contains the mean
of the nested result's `decision` field over all Monte Carlo trials.
