# Wasserstein Regime Clustering (improved)

Modern, clean re-implementation of the core algorithms from:

> Horvath, Issa & Muguruza (2021). *Clustering Market Regimes Using the Wasserstein Distance*. arXiv:2110.11848.

Original reference implementation: [mirkovicdev/CLUSTERING-MARKET-REGIMES](https://github.com/mirkovicdev/CLUSTERING-MARKET-REGIMES).

## What changed

| Area | Original | This package |
|------|----------|--------------|
| Structure | Flat scripts + committed `__pycache__` / `node_modules` | Proper `src/` package, `pyproject.toml` |
| Numerics | Mostly correct 1-D OT | Fully vectorized quantile Wasserstein + barycenters |
| Tooling | `requirements.txt` only | `uv` / `pip` installable, ruff, pytest |
| Tests | None visible | Synthetic ground-truth suite |
| New features | — | Streaming detector, transition matrix, regime persistence stats |
| API | Script-oriented | Importable classes and functions |

Core mathematical behaviour (Wasserstein k-means, moment baseline, synthetic GBM/Merton, accuracy scores) is preserved so paper results remain reproducible in spirit.

## Install

```bash
# with uv
uv pip install -e ".[dev,viz]"

# or plain pip
pip install -e ".[dev]"
```

## Quick start

```python
from wasserstein_regime import (
    WassersteinKMeans,
    create_sliding_windows,
    compute_log_returns,
    generate_regime_switching_gbm,
    RegimeSwitchingParams,
    StreamingRegimeDetector,
    regime_transition_matrix,
    regime_persistence,
)

# synthetic regime-switching path
params = RegimeSwitchingParams(n_years=5, timesteps_per_year=100, random_state=0)
prices, times, true_labels, intervals = generate_regime_switching_gbm(params)
returns = compute_log_returns(prices)
windows = create_sliding_windows(returns, h1=30, h2=5)

model = WassersteinKMeans(n_clusters=2, n_init=5, random_state=0)
labels = model.fit_predict(windows)

print("inertia", model.inertia_)
print("transitions\n", regime_transition_matrix(labels))
print("persistence", regime_persistence(labels))

# online use
detector = StreamingRegimeDetector(model, window_length=30)
for r in returns[-40:]:
    lab = detector.update(r)
    if lab is not None:
        print("live regime", lab)
```

## New features (noticeable, non-breaking)

1. **StreamingRegimeDetector** — push returns one at a time against a fitted model; get a regime label as soon as the window is full. Useful for sequential / live analysis.
2. **regime_transition_matrix / regime_persistence** — first-order transition probabilities and mean/median run lengths per regime. Makes the output actionable for strategy or risk code.
3. **Vectorized 1-D Wasserstein** — quantile grid formulation handles unequal sample sizes cleanly and is faster on typical window sizes.

## Reproduce the spirit of the paper

```python
# longer synthetic experiment (GBM)
params = RegimeSwitchingParams(
    timesteps_per_year=252,
    n_years=10,
    n_regime_changes=6,
    regime_length=120,
    random_state=7,
)
prices, _, true_lab, _ = generate_regime_switching_gbm(params)
rets = compute_log_returns(prices)
wins = create_sliding_windows(rets, h1=40, h2=10)
wk = WassersteinKMeans(n_clusters=2, n_init=8, random_state=0)
pred = wk.fit_predict(wins)
```

Accuracy scoring against the synthetic ground-truth labels is available via `compute_accuracy_scores`.

## Citation

Please cite the original paper:

```bibtex
@article{horvath2021clustering,
  title={Clustering Market Regimes Using the Wasserstein Distance},
  author={Horvath, Blanka and Issa, Zacharia and Muguruza, Aitor},
  journal={arXiv preprint arXiv:2110.11848},
  year={2021}
}
```

## License

MIT. Methodology belongs to the original authors; this is an independent clean-room improvement of the public implementation.
