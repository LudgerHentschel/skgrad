"""Benchmark skgrad MLP gradients against PyTorch CPU autodiff.

Run from the repository root with::

    python benchmarks/pytorch_autodiff.py
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
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from torch import nn

import skgrad
from skgrad import _mlp


SAMPLE_ROWS = (1, 10, 100, 1_000, 5_000, 10_000, 100_000)
N_FEATURES = 20
HIDDEN_LAYERS = (64, 64)


def equivalent_models() -> tuple[MLPRegressor, nn.Sequential]:
    """Return sklearn and PyTorch models with identical float64 parameters."""

    rng = np.random.default_rng(20260810)
    X_fit = rng.normal(size=(32, N_FEATURES))
    y_fit = rng.normal(size=32)
    sklearn_model = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="tanh",
        solver="lbfgs",
        max_iter=1,
        random_state=0,
    ).fit(X_fit, y_fit)

    layer_sizes = (N_FEATURES,) + HIDDEN_LAYERS + (1,)
    sklearn_model.coefs_ = [
        rng.normal(scale=1.0 / np.sqrt(n_in), size=(n_in, n_out))
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
    ]
    sklearn_model.intercepts_ = [
        rng.normal(scale=0.1, size=n_out) for n_out in layer_sizes[1:]
    ]

    torch_model = nn.Sequential(
        nn.Linear(20, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    ).double()
    linear_layers = [layer for layer in torch_model if isinstance(layer, nn.Linear)]
    with torch.no_grad():
        for layer, weights, biases in zip(
            linear_layers, sklearn_model.coefs_, sklearn_model.intercepts_
        ):
            layer.weight.copy_(torch.from_numpy(weights.T))
            layer.bias.copy_(torch.from_numpy(biases))
    torch_model.eval()
    return sklearn_model, torch_model


def median_time(function, warmups: int, repetitions: int) -> float:
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    torch.set_num_threads(4)
    sklearn_model, torch_model = equivalent_models()
    rng = np.random.default_rng(20260811)

    print("Speed (warm-run median seconds)")
    for n_rows in SAMPLE_ROWS:
        X = np.ascontiguousarray(rng.normal(size=(n_rows, N_FEATURES)))
        torch_X = torch.from_numpy(X).requires_grad_(True)

        def skgrad_gradient():
            return skgrad.input_gradient(sklearn_model, X)

        def torch_gradient():
            return torch.autograd.grad(torch_model(torch_X).sum(), torch_X)[0]

        with torch.no_grad():
            torch_values = torch_model(torch_X).numpy()[:, 0]
        np.testing.assert_allclose(
            skgrad.model_output(sklearn_model, X)[:, 0],
            torch_values,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            skgrad_gradient(),
            torch_gradient().detach().numpy(),
            rtol=1e-12,
            atol=1e-12,
        )

        skgrad_time = median_time(skgrad_gradient, args.warmups, args.repetitions)
        torch_time = median_time(torch_gradient, args.warmups, args.repetitions)
        ratio = torch_time / skgrad_time
        relative = (
            f"skgrad {ratio:.2f}x faster"
            if ratio >= 1.0
            else f"PyTorch {1.0 / ratio:.2f}x faster"
        )
        print(
            f"rows={n_rows:6d} workers={_mlp._parallel_worker_count(n_rows)} "
            f"skgrad={skgrad_time:.6f} torch={torch_time:.6f} {relative}"
        )

    print("\nEnvironment")
    print(f"platform={platform.platform()}")
    print(f"processor={platform.processor() or platform.machine()}")
    print(f"python={sys.version.split()[0]}")
    print(f"numpy={np.__version__}")
    print(f"scikit-learn={sklearn.__version__}")
    print(f"pytorch={torch.__version__}")
    print(f"pytorch-intra-operation-threads={torch.get_num_threads()}")
    print(f"pytorch-inter-operation-threads={torch.get_num_interop_threads()}")


if __name__ == "__main__":
    main()
