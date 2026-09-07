"""Public model-dispatch API."""

from typing import NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ._affine import affine_model_output, affine_supports, affine_value_and_jacobian
from ._inputs import normalize_data, validate_target
from ._mlp import (
    mlp_input_gradient,
    mlp_model_output,
    mlp_supports,
    mlp_value_and_jacobian,
)
from ._pipeline import (
    pipeline_properties,
    pipeline_input_gradient,
    pipeline_model_output,
    pipeline_supports,
    pipeline_value_and_jacobian,
)
from ._svm import (
    svm_constant_jacobian,
    svm_model_output,
    svm_supports,
    svm_value_and_jacobian,
)


FloatArray = NDArray[np.floating]


class GradientResult(NamedTuple):
    """Model values and their input Jacobians."""

    values: FloatArray
    jacobian: FloatArray


class GradientProperties(NamedTuple):
    """Properties that downstream gradient consumers may optimize around."""

    constant_jacobian: bool
    exact_quadrature_steps: Optional[int]


def supports(model: object) -> bool:
    """Return whether skgrad has an analytic backend for ``model``."""

    return _estimator_supports(model) or pipeline_supports(
        model, _estimator_supports
    )


def gradient_properties(model: object) -> GradientProperties:
    """Return computational properties of a supported model's Jacobian."""

    if affine_supports(model):
        return GradientProperties(constant_jacobian=True, exact_quadrature_steps=1)
    if svm_supports(model):
        constant = svm_constant_jacobian(model)
        return GradientProperties(
            constant_jacobian=constant,
            exact_quadrature_steps=1 if constant else None,
        )
    if mlp_supports(model):
        return GradientProperties(
            constant_jacobian=False, exact_quadrature_steps=None
        )
    if pipeline_supports(model, _estimator_supports):
        return GradientProperties(
            *pipeline_properties(model, _estimator_constant_jacobian)
        )
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
    elif svm_supports(model):
        values, jacobian = svm_value_and_jacobian(model, data)
    elif mlp_supports(model):
        values, jacobian = mlp_value_and_jacobian(model, data)
    elif pipeline_supports(model, _estimator_supports):
        values, jacobian = pipeline_value_and_jacobian(
            model, data, _estimator_value_and_jacobian
        )
    else:
        raise TypeError(f"skgrad does not support {type(model).__name__}")
    return GradientResult(values, jacobian)


def model_output(model: object, X: object) -> FloatArray:
    """Return predictions for regressors or raw scores/logits for classifiers."""

    data = normalize_data(X)
    if affine_supports(model):
        return affine_model_output(model, data)
    if svm_supports(model):
        return svm_model_output(model, data)
    if mlp_supports(model):
        return mlp_model_output(model, data)
    if pipeline_supports(model, _estimator_supports):
        return pipeline_model_output(model, data, _estimator_model_output)
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

    if mlp_supports(model):
        return mlp_input_gradient(model, normalize_data(X), target)

    if pipeline_supports(model, _estimator_supports):
        return pipeline_input_gradient(
            model, normalize_data(X), target, input_gradient
        )

    jacobian = input_jacobian(model, X)
    target_index = validate_target(jacobian.shape[1], target)
    return jacobian[:, target_index, :]


def _estimator_supports(model: object) -> bool:
    return affine_supports(model) or svm_supports(model) or mlp_supports(model)


def _estimator_constant_jacobian(model: object) -> bool:
    if affine_supports(model):
        return True
    if svm_supports(model):
        return svm_constant_jacobian(model)
    if mlp_supports(model):
        return False
    raise TypeError(f"skgrad does not support {type(model).__name__}")


def _estimator_value_and_jacobian(
    model: object,
    X: FloatArray,
) -> Tuple[FloatArray, FloatArray]:
    if affine_supports(model):
        return affine_value_and_jacobian(model, X)
    if svm_supports(model):
        return svm_value_and_jacobian(model, X)
    if mlp_supports(model):
        return mlp_value_and_jacobian(model, X)
    raise TypeError(f"skgrad does not support {type(model).__name__}")


def _estimator_model_output(model: object, X: FloatArray) -> FloatArray:
    if affine_supports(model):
        return affine_model_output(model, X)
    if svm_supports(model):
        return svm_model_output(model, X)
    if mlp_supports(model):
        return mlp_model_output(model, X)
    raise TypeError(f"skgrad does not support {type(model).__name__}")
