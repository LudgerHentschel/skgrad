import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

import skgrad


def test_one_sample_is_normalized_and_helpers_agree():
    model = LinearRegression().fit(np.eye(2), np.array([1.0, 2.0]))

    result = skgrad.value_and_jacobian(model, [0.2, 0.4])

    assert result.values.shape == (1, 1)
    assert result.jacobian.shape == (1, 1, 2)
    np.testing.assert_array_equal(skgrad.model_output(model, [0.2, 0.4]), result.values)
    np.testing.assert_array_equal(skgrad.input_jacobian(model, [0.2, 0.4]), result.jacobian)


def test_validation_and_unsupported_model_errors():
    fitted = LinearRegression().fit(np.eye(2), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        skgrad.input_jacobian(fitted, [[np.nan, 0.0]])
    with pytest.raises(ValueError, match="expects 2"):
        skgrad.input_jacobian(fitted, [[1.0, 2.0, 3.0]])
    with pytest.raises(NotFittedError):
        skgrad.input_jacobian(LinearRegression(), [[1.0, 2.0]])

    tree = DecisionTreeRegressor().fit(np.eye(2), np.array([1.0, 2.0]))
    assert not skgrad.supports(tree)
    with pytest.raises(TypeError, match="does not support"):
        skgrad.input_jacobian(tree, [[1.0, 2.0]])


def test_support_is_unified_and_gradient_properties_enable_optimization():
    affine = LinearRegression()
    mlp = MLPRegressor()
    tree = DecisionTreeRegressor()

    assert skgrad.supports(affine)
    assert skgrad.gradient_properties(affine).constant_jacobian
    assert skgrad.supports(mlp)
    assert not skgrad.gradient_properties(mlp).constant_jacobian
    assert not skgrad.supports(tree)
    with pytest.raises(TypeError, match="does not support"):
        skgrad.gradient_properties(tree)


def test_target_validation():
    model = LinearRegression().fit(np.eye(2), np.array([1.0, 2.0]))
    with pytest.raises(TypeError, match="integer"):
        skgrad.input_gradient(model, [[1.0, 2.0]], target=0.5)
    with pytest.raises(ValueError, match="between"):
        skgrad.input_gradient(model, [[1.0, 2.0]], target=1)
