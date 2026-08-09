"""Benchmark skgrad against generic central finite differences.

Run from the repository root with::

    python benchmarks/numerical_gradients.py
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

import skgrad


FEATURE_COUNTS = (10, 100, 1_000)
EVALUATION_ROWS = 100
STEP = 1e-5


def numerical_gradient(model: object, X: np.ndarray, step: float = STEP) -> np.ndarray:
    """Return batched two-sided central differences of the first model output."""
    gradient = np.empty_like(X, dtype=float)
    for feature in range(X.shape[1]):
        plus = X.copy()
        minus = X.copy()
        plus[:, feature] += step
        minus[:, feature] -= step
        gradient[:, feature] = (
            skgrad.model_output(model, plus)[:, 0]
            - skgrad.model_output(model, minus)[:, 0]
        ) / (2.0 * step)
    return gradient


def fitted_model(kind: str, n_features: int) -> tuple[object, np.ndarray]:
    rng = np.random.default_rng(20260809 + n_features)
    n_training = max(200, min(500, 2 * n_features))
    X_train = rng.normal(size=(n_training, n_features))
    X_eval = rng.normal(size=(EVALUATION_ROWS, n_features))
    signal = X_train[:, : min(10, n_features)].sum(axis=1)

    if kind == "LogisticRegression":
        model = LogisticRegression(max_iter=200, random_state=0).fit(
            X_train, signal > np.median(signal)
        )
    elif kind == "MLPRegressor":
        model = MLPRegressor(
            hidden_layer_sizes=(64, 64), activation="tanh", solver="lbfgs",
            max_iter=20, random_state=0,
        ).fit(X_train, signal)
    elif kind == "MLPClassifier":
        labels = np.digitize(signal, np.quantile(signal, (1 / 3, 2 / 3)))
        model = MLPClassifier(
            hidden_layer_sizes=(32,), activation="tanh", solver="lbfgs",
            max_iter=20, random_state=0,
        ).fit(X_train, labels)
    else:
        raise ValueError(kind)
    return model, X_eval


def median_time(function, warmups: int, repetitions: int) -> float:
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def errors(model: object, X: np.ndarray) -> tuple[float, float]:
    analytic = skgrad.input_gradient(model, X, target=0)
    numerical = numerical_gradient(model, X)
    difference = np.abs(analytic - numerical)
    relative = difference.max() / max(np.abs(numerical).max(), np.finfo(float).eps)
    return float(difference.max()), float(relative)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    models = {
        kind: {p: fitted_model(kind, p) for p in FEATURE_COUNTS}
        for kind in ("LogisticRegression", "MLPRegressor")
    }

    print("Accuracy (100 rows, central difference step 1e-5)")
    for kind, p in (("LogisticRegression", 100), ("MLPRegressor", 100)):
        model, X = models[kind][p]
        maximum, relative = errors(model, X)
        print(f"{kind:20s} p={p:4d} max_abs={maximum:.3e} relative={relative:.3e}")
    classifier, classifier_X = fitted_model("MLPClassifier", 100)
    maximum, relative = errors(classifier, classifier_X)
    print(f"{'MLPClassifier':20s} p={100:4d} max_abs={maximum:.3e} relative={relative:.3e}")

    print("\nSpeed (100 rows; warm-run median seconds)")
    for kind, fitted in models.items():
        for p, (model, X) in fitted.items():
            analytic = lambda: skgrad.input_gradient(model, X, target=0)
            numerical = lambda: numerical_gradient(model, X)
            analytic_time = median_time(analytic, args.warmups, args.repetitions)
            numerical_time = median_time(numerical, args.warmups, args.repetitions)
            maximum, _ = errors(model, X)
            print(
                f"{kind:20s} p={p:4d} skgrad={analytic_time:.6f} "
                f"numerical={numerical_time:.6f} speedup={numerical_time / analytic_time:.1f} "
                f"max_abs={maximum:.3e}"
            )

    print("\nEnvironment")
    print(f"platform={platform.platform()}")
    print(f"processor={platform.processor() or platform.machine()}")
    print(f"python={sys.version.split()[0]}")
    print(f"numpy={np.__version__}")
    print(f"scikit-learn={sklearn.__version__}")
    print(f"skgrad={skgrad.__version__}")


if __name__ == "__main__":
    main()
