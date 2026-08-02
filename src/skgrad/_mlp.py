"""Analytic composition through fitted scikit-learn MLPs."""

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils.validation import check_is_fitted


FloatArray = NDArray[np.floating]


def mlp_supports(model: object) -> bool:
    return isinstance(model, (MLPRegressor, MLPClassifier))


def mlp_value_and_jacobian(
    model: object,
    X: FloatArray,
) -> Tuple[FloatArray, FloatArray]:
    activation, hidden_outputs = _forward_hidden(model, X)
    output_weights = np.asarray(model.coefs_[-1], dtype=float)
    values = activation @ output_weights + model.intercepts_[-1]
    values = np.asarray(values, dtype=float).reshape(X.shape[0], -1)
    jacobian = np.broadcast_to(
        output_weights.T,
        (X.shape[0], output_weights.shape[1], output_weights.shape[0]),
    ).copy()

    for layer in range(len(hidden_outputs) - 1, -1, -1):
        derivative = _activation_derivative(hidden_outputs[layer], model.activation)
        jacobian *= derivative[:, None, :]
        jacobian = np.einsum(
            "sou,iu->soi", jacobian, np.asarray(model.coefs_[layer], dtype=float)
        )
    return values, jacobian


def mlp_model_output(model: object, X: FloatArray) -> FloatArray:
    """Return regression predictions or classification logits only."""

    activation, _ = _forward_hidden(model, X)
    values = activation @ model.coefs_[-1] + model.intercepts_[-1]
    return np.asarray(values, dtype=float).reshape(X.shape[0], -1)


def _forward_hidden(
    model: object,
    X: FloatArray,
) -> Tuple[FloatArray, List[FloatArray]]:
    check_is_fitted(model, attributes=["coefs_", "intercepts_"])
    n_features = int(model.n_features_in_)
    if X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} features; model expects {n_features}")
    if isinstance(model, MLPRegressor) and model.out_activation_ != "identity":
        raise ValueError("skgrad currently requires identity-output MLPRegressor models")

    activation = X
    hidden_outputs: List[FloatArray] = []
    for weights, intercept in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        activation = _activate(activation @ weights + intercept, model.activation)
        hidden_outputs.append(activation)
    return activation, hidden_outputs


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
        return (activated > 0.0).astype(float)
    if activation == "tanh":
        return 1.0 - activated**2
    if activation == "logistic":
        return activated * (1.0 - activated)
    raise ValueError(f"unsupported MLP hidden activation: {activation}")
