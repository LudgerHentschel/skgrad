import numpy as np
import pytest
from sklearn.svm import NuSVC, NuSVR, SVC, SVR

import skgrad


KERNELS = ("linear", "poly", "rbf", "sigmoid")


def _data():
    rng = np.random.default_rng(30)
    X = rng.normal(size=(80, 4))
    score = np.sin(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] - X[:, 3]
    return X, score


def _finite_difference(function, X, step=1e-6):
    gradient = np.empty_like(X)
    for feature in range(X.shape[1]):
        plus = X.copy()
        minus = X.copy()
        plus[:, feature] += step
        minus[:, feature] -= step
        gradient[:, feature] = (function(plus) - function(minus)) / (2.0 * step)
    return gradient


@pytest.mark.parametrize("estimator", [SVR, NuSVR])
@pytest.mark.parametrize("kernel", KERNELS)
def test_svm_regression_matches_prediction_and_finite_difference(estimator, kernel):
    X, y = _data()
    model = estimator(kernel=kernel, gamma=0.4, coef0=0.2, degree=3).fit(X, y)
    data = X[:7]

    values, jacobian = skgrad.value_and_jacobian(model, data)

    np.testing.assert_allclose(values[:, 0], model.predict(data), atol=1e-12)
    np.testing.assert_allclose(
        skgrad.model_output(model, data)[:, 0], model.predict(data), atol=1e-10
    )
    np.testing.assert_allclose(
        jacobian[:, 0, :],
        _finite_difference(model.predict, data),
        rtol=2e-6,
        atol=2e-7,
    )


@pytest.mark.parametrize("estimator", [SVC, NuSVC])
@pytest.mark.parametrize("kernel", KERNELS)
def test_binary_svm_matches_decision_score_and_finite_difference(estimator, kernel):
    X, score = _data()
    y = score > np.median(score)
    model = estimator(kernel=kernel, gamma=0.4, coef0=0.2, degree=3).fit(X, y)
    data = X[:7]

    values, jacobian = skgrad.value_and_jacobian(model, data)

    np.testing.assert_allclose(values[:, 0], model.decision_function(data), atol=1e-12)
    np.testing.assert_allclose(
        skgrad.model_output(model, data)[:, 0],
        model.decision_function(data),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        jacobian[:, 0, :],
        _finite_difference(model.decision_function, data),
        rtol=2e-6,
        atol=2e-7,
    )


def test_kernel_gradient_properties():
    assert skgrad.gradient_properties(SVR(kernel="linear")).constant_jacobian
    assert skgrad.gradient_properties(SVR(kernel="poly", degree=1)).constant_jacobian
    assert not skgrad.gradient_properties(SVR(kernel="poly", degree=2)).constant_jacobian
    assert not skgrad.gradient_properties(SVR(kernel="rbf")).constant_jacobian


def test_multiclass_and_user_defined_kernels_are_not_supported():
    X, score = _data()
    y = np.digitize(score, np.quantile(score, (1 / 3, 2 / 3)))
    multiclass = SVC().fit(X, y)
    callable_kernel = SVC(kernel=lambda left, right: left @ right.T)
    precomputed = SVR(kernel="precomputed")

    assert not skgrad.supports(multiclass)
    assert not skgrad.supports(callable_kernel)
    assert not skgrad.supports(precomputed)
    with pytest.raises(TypeError, match="does not support"):
        skgrad.input_gradient(multiclass, X[:2])


def test_rbf_output_is_stable_with_large_common_offset():
    X = 1e8 + np.arange(12, dtype=float).reshape(-1, 1)
    model = SVR(kernel="rbf", gamma=1).fit(X, np.sin(np.arange(12)))
    values = skgrad.model_output(model, X)[:, 0]
    np.testing.assert_allclose(values, model.predict(X), atol=1e-12)
    np.testing.assert_allclose(
        values, skgrad.value_and_jacobian(model, X).values[:, 0], atol=1e-12
    )
