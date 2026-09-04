# Lacuna Evaluation

[![Tests](https://github.com/WMU-Herculaneum-Project/lacuna_eval/actions/workflows/tests.yml/badge.svg)](https://github.com/WMU-Herculaneum-Project/lacuna_eval/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/WMU-Herculaneum-Project/lacuna_eval)](LICENSE)

Code to evaluate the performance of Lacuna models on the task of predicting the filling of
lacunae in ancient text.

## Metrics

Pass model predictions as a ranked list, with the acceptable answers in a set:

```python
from lacuna_eval.metrics import dcg, in_top_k, ndcg, top_1

guesses = ["dog", "fox", "cat", "rabbit", "wolf"]
truth = {"fox", "cat"}

in_top_k(guesses, truth, k=2)  # 1.0
top_1(guesses, truth)          # 0.0
dcg(guesses, truth, k=2)
ndcg(guesses, truth)
```

The guesses must be ordered from most to least likely. Use `k` to limit `dcg`,
`ndcg`, or `in_top_k` to the first `k` guesses.

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)

## Setup

Create the project environment and install the runtime and development dependencies:

```sh
uv sync
```

## Tests

Run the test suite with:

```sh
uv run pytest
```

The project also includes Black, isort, and Flake8 for development checks:

```sh
uv run black --check .
uv run isort --check-only .
uv run flake8 .
```
