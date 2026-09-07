"""Analytic composition through fitted scikit-learn MLPs."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
from threading import Lock
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_info

from ._dispatch import inherits_prediction
from ._inputs import validate_target


FloatArray = NDArray[np.floating]
_PARALLEL_MIN_SAMPLES = 5_000
_PARALLEL_MIN_SAMPLES_PER_WORKER = 2_500
_PARALLEL_MAX_WORKERS = 4
_PARALLEL_EXECUTOR: Optional[ThreadPoolExecutor] = None
_PARALLEL_EXECUTOR_PID: Optional[int] = None
_PARALLEL_EXECUTOR_LOCK = Lock()


def _reset_parallel_state_after_fork() -> None:
    global _PARALLEL_EXECUTOR, _PARALLEL_EXECUTOR_PID, _PARALLEL_EXECUTOR_LOCK
    _PARALLEL_EXECUTOR = None
    _PARALLEL_EXECUTOR_PID = None
    _PARALLEL_EXECUTOR_LOCK = Lock()
    _parallel_capacity.cache_clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_parallel_state_after_fork)


def mlp_supports(model: object) -> bool:
    return inherits_prediction(model, (MLPRegressor, MLPClassifier))


def mlp_value_and_jacobian(
    model: object,
    X: FloatArray,
) -> Tuple[FloatArray, FloatArray]:
    activation, hidden_outputs = _forward_hidden(model, X)
    output_weights = np.asarray(model.coefs_[-1])
    values = _output_values(model, activation)
    jacobian = np.broadcast_to(
        output_weights.T.astype(values.dtype, copy=False),
        (X.shape[0], output_weights.shape[1], output_weights.shape[0]),
    ).copy()
    if _has_exponential_output(model):
        jacobian *= values[:, :, None]

    for layer in range(len(hidden_outputs) - 1, -1, -1):
        derivative = _activation_derivative(hidden_outputs[layer], model.activation)
        jacobian *= derivative[:, None, :]
        jacobian = np.einsum(
            "sou,iu->soi", jacobian, np.asarray(model.coefs_[layer], dtype=X.dtype)
        )
    return values, jacobian


def mlp_model_output(model: object, X: FloatArray) -> FloatArray:
    """Return regression predictions or classification logits only."""

    activation, _ = _forward_hidden(model, X)
    return _output_values(model, activation)


def mlp_input_gradient(
    model: object,
    X: FloatArray,
    target: Optional[int] = None,
) -> FloatArray:
    """Return one selected output gradient without forming a full Jacobian."""

    workers = _parallel_worker_count(X.shape[0])
    if workers > 1:
        chunks = np.array_split(X, workers)
        gradients = _parallel_executor().map(
            lambda chunk: _mlp_input_gradient_serial(model, chunk, target),
            chunks,
        )
        return np.concatenate(tuple(gradients), axis=0)
    return _mlp_input_gradient_serial(model, X, target)


def _mlp_input_gradient_serial(
    model: object,
    X: FloatArray,
    target: Optional[int],
) -> FloatArray:
    activation, hidden_outputs = _forward_hidden(model, X)
    output_weights = np.asarray(model.coefs_[-1])
    target_index = validate_target(output_weights.shape[1], target)
    gradient_dtype = X.dtype
    output_gradient = output_weights[:, target_index].astype(
        gradient_dtype, copy=False
    )
    if _has_exponential_output(model):
        output_value = np.exp(
            activation @ output_gradient + np.asarray(model.intercepts_[-1], dtype=X.dtype)[target_index]
        )
        output_gradient = output_value[:, None] * output_gradient

    if not hidden_outputs:
        if output_gradient.ndim == 2:
            return output_gradient
        return np.broadcast_to(
            output_gradient,
            (X.shape[0], output_weights.shape[0]),
        ).copy()

    # Keeping the selected-output adjoint two-dimensional lets NumPy use its
    # optimized matrix multiplication path and avoids the full
    # (samples, outputs, hidden units) intermediate used by the general API.
    if output_gradient.ndim == 1:
        gradient = np.broadcast_to(output_gradient, hidden_outputs[-1].shape).copy()
    else:
        gradient = output_gradient
    for layer in range(len(hidden_outputs) - 1, -1, -1):
        gradient *= _activation_derivative(hidden_outputs[layer], model.activation)
        gradient = gradient @ np.asarray(model.coefs_[layer], dtype=X.dtype).T
    return gradient


def _parallel_worker_count(n_samples: int) -> int:
    """Choose bounded row parallelism without oversubscribing native kernels."""

    if n_samples < _PARALLEL_MIN_SAMPLES:
        return 1
    available_workers = _parallel_capacity()
    useful_workers = n_samples // _PARALLEL_MIN_SAMPLES_PER_WORKER
    return max(1, min(_PARALLEL_MAX_WORKERS, available_workers, useful_workers))


@lru_cache(maxsize=1)
def _parallel_capacity() -> int:
    """Estimate safe outer concurrency from CPUs and active BLAS threads."""

    available_cpus = os.cpu_count() or 1
    native_threads = max(
        (
            int(pool.get("num_threads", 1))
            for pool in threadpool_info()
            if pool.get("user_api") == "blas"
        ),
        default=1,
    )
    capacity = max(1, available_cpus // max(1, native_threads))
    # Tall, narrow MLP matrix multiplies often leave cores idle even when BLAS
    # advertises all CPUs. On machines with at least four CPUs, two outer row
    # workers is a conservative floor that captures that otherwise-unused
    # parallelism without allowing an unbounded oversubscription multiplier.
    if available_cpus >= 4:
        capacity = max(2, capacity)
    return capacity


def _parallel_executor() -> ThreadPoolExecutor:
    """Return a lazily created executor so repeated calls avoid startup cost."""

    global _PARALLEL_EXECUTOR, _PARALLEL_EXECUTOR_PID
    process_id = os.getpid()
    if _PARALLEL_EXECUTOR is None or _PARALLEL_EXECUTOR_PID != process_id:
        with _PARALLEL_EXECUTOR_LOCK:
            if _PARALLEL_EXECUTOR is None or _PARALLEL_EXECUTOR_PID != process_id:
                _PARALLEL_EXECUTOR = ThreadPoolExecutor(
                    max_workers=_PARALLEL_MAX_WORKERS,
                    thread_name_prefix="skgrad",
                )
                _PARALLEL_EXECUTOR_PID = process_id
    return _PARALLEL_EXECUTOR


def _forward_hidden(
    model: object,
    X: FloatArray,
) -> Tuple[FloatArray, List[FloatArray]]:
    check_is_fitted(model, attributes=["coefs_", "intercepts_"])
    n_features = int(model.n_features_in_)
    if X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} features; model expects {n_features}")
    if isinstance(model, MLPRegressor) and model.out_activation_ not in {
        "identity",
        "exp",
    }:
        raise ValueError(
            f"unsupported MLPRegressor output activation: {model.out_activation_}"
        )

    activation = X
    hidden_outputs: List[FloatArray] = []
    for weights, intercept in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        activation = _activate(activation @ np.asarray(weights, dtype=X.dtype) + np.asarray(intercept, dtype=X.dtype), model.activation)
        hidden_outputs.append(activation)
    return activation, hidden_outputs


def _output_values(model: object, activation: FloatArray) -> FloatArray:
    values = activation @ np.asarray(model.coefs_[-1], dtype=activation.dtype) + np.asarray(model.intercepts_[-1], dtype=activation.dtype)
    if _has_exponential_output(model):
        values = np.exp(values)
    return np.asarray(values).reshape(activation.shape[0], -1)


def _has_exponential_output(model: object) -> bool:
    return isinstance(model, MLPRegressor) and model.out_activation_ == "exp"


def _activate(values: FloatArray, activation: str) -> FloatArray:
    if activation == "identity":
        return values
    if activation == "relu":
        return np.maximum(values, 0.0)
    if activation == "tanh":
        return np.tanh(values)
    if activation == "logistic":
        result = np.empty_like(values)
        positive = values >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exponential = np.exp(values[~positive])
        result[~positive] = exponential / (1.0 + exponential)
        return result
    raise ValueError(f"unsupported MLP hidden activation: {activation}")


def _activation_derivative(activated: FloatArray, activation: str) -> FloatArray:
    if activation == "identity":
        return np.ones_like(activated)
    if activation == "relu":
        return (activated > 0.0).astype(activated.dtype)
    if activation == "tanh":
        return 1.0 - activated**2
    if activation == "logistic":
        return activated * (1.0 - activated)
    raise ValueError(f"unsupported MLP hidden activation: {activation}")
