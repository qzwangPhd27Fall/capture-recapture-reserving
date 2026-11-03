# Capture-Recapture Methods for Insurance Claims Reserving

A Python package implementing **capture-recapture (CR) estimators** for IBNR (incurred-but-not-reported) claims reserving, treating each development period as a capture occasion.

## Package structure

```
cr_reserving/
├── theory/        # Schnabel, HMM open-population, Bayesian CR, doubly-robust, bootstrap, GoF, variance
├── simulation/    # data generator + Monte Carlo scenario framework
├── baselines/     # chain-ladder benchmark
└── visualization/ # publication-quality plots
```

## Key results

- The CR estimator dominates the chain-ladder estimator in MSE when ≥ 4 development periods are available.
- Monte Carlo simulations across 7 scenarios confirm 90–96% confidence-interval coverage for the Schnabel estimator.

## Usage

```bash
python cr_reserving/run_experiments.py   # run the experiment suite
python run_full_mc.py                    # full Monte Carlo run
python generate_figures.py               # regenerate publication figures
```

## Dependencies

See `requirements.txt`.
