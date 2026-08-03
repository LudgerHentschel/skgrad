"""Public model-dispatch API."""

from typing import NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from ._affine import affine_model_output, affine_supports, affine_value_and_jacobian
from ._inputs import normalize_data
from ._mlp import mlp_model_output, mlp_supports, mlp_value_and_jacobian


FloatArray = NDArray[np.floating]


class GradientResult(NamedTuple):
    """Model values and their input Jacobians."""

    values: FloatArray
    jacobian: FloatArray


class GradientProperties(NamedTuple):
    """Properties that downstream gradient consumers may optimize around."""

    constant_jacobian: bool


def supports(model: object) -> bool:
    """Return whether skgrad has an analytic backend for ``model``."""

    return affine_supports(model) or mlp_supports(model)


def gradient_properties(model: object) -> GradientProperties:
    """Return computational properties of a supported model's Jacobian."""

    if affine_supports(model):
        return GradientProperties(constant_jacobian=True)
    if mlp_supports(model):
        return GradientProperties(constant_jacobian=False)
    raise TypeError(f"skgrad does not support {type(model).__name__}")


def value_and_jacobian(model: object, X: object) -> GradientResult:
    """Return model values and analytic input Jacobians.

    Values have shape ``(n_samples, n_outputs)``. Jacobians have shape
    ``(n_samples, n_outputs, n_features)``. Classifiers return raw decision
    scores or logits rather than probabilities.
    """

    data = normalize_data(X)
    if affine_supports(model):
        values, jacobian = affine_value_and_jacobian(model, data)
    elif mlp_supports(model):
        values, jacobian = mlp_value_and_jacobian(model, data)
    else:
        raise TypeError(f"skgrad does not support {type(model).__name__}")
    return GradientResult(values, jacobian)


def model_output(model: object, X: object) -> FloatArray:
    """Return predictions for regressors or raw scores/logits for classifiers."""

    data = normalize_data(X)
    if affine_supports(model):
        return affine_model_output(model, data)
    if mlp_supports(model):
        return mlp_model_output(model, data)
    raise TypeError(f"skgrad does not support {type(model).__name__}")


def input_jacobian(model: object, X: object) -> FloatArray:
    """Return input Jacobians with shape ``(samples, outputs, features)``."""

    return value_and_jacobian(model, X).jacobian


def input_gradient(
    model: object,
    X: object,
    target: Optional[int] = None,
) -> FloatArray:
    """Return gradients for one selected scalar model output."""

    jacobian = input_jacobian(model, X)
    n_outputs = jacobian.shape[1]
    if target is None:
        if n_outputs != 1:
            raise ValueError("target is required when the model has multiple outputs")
        target_index = 0
    else:
        if not isinstance(target, int) or isinstance(target, bool):
            raise TypeError("target must be an integer or None")
        if target < 0 or target >= n_outputs:
            raise ValueError(
                f"target must be between 0 and {n_outputs - 1}, got {target}"
            )
        target_index = target
    return jacobian[:, target_index, :]
