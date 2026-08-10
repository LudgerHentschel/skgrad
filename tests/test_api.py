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


def test_target_accepts_numpy_integers_but_rejects_booleans():
    X = np.eye(2)
    targets = np.column_stack(([1.0, 2.0], [3.0, 4.0]))
    model = LinearRegression().fit(X, targets)

    for target in (np.int32(1), np.int64(1), np.intp(1)):
        np.testing.assert_allclose(
            skgrad.input_gradient(model, X, target=target),
            skgrad.input_gradient(model, X, target=1),
        )
    for target in (True, np.bool_(True)):
        with pytest.raises(TypeError, match="integer"):
            skgrad.input_gradient(model, X, target=target)


def test_sparse_input_has_clear_error():
    sparse = pytest.importorskip("scipy.sparse")
    model = LinearRegression().fit(np.eye(2), np.array([1.0, 2.0]))

    with pytest.raises(TypeError, match="dense.*sparse"):
        skgrad.model_output(model, sparse.csr_matrix(np.eye(2)))


@pytest.mark.parametrize(
    "data",
    [np.empty((0, 2)), np.empty((2, 0)), np.zeros((2, 2, 2))],
)
def test_degenerate_input_shapes_are_rejected(data):
    model = LinearRegression().fit(np.eye(2), np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="non-empty one- or two-dimensional"):
        skgrad.model_output(model, data)


def test_integer_input_is_coerced_to_float64():
    model = LinearRegression().fit(np.eye(2), np.array([1.0, 2.0]))

    result = skgrad.model_output(model, np.array([[1, 0], [0, 1]]))

    assert result.dtype == np.float64
    np.testing.assert_allclose(result[:, 0], [1.0, 2.0])
