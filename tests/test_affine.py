import numpy as np
import pytest
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.svm import LinearSVC, LinearSVR

import skgrad


def _regression_data():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(80, 4))
    y = 1.5 * X[:, 0] - 0.7 * X[:, 1] + 0.2 * X[:, 2] + 0.1
    return X, y


@pytest.mark.parametrize(
    "model",
    [
        LinearRegression(),
        Ridge(alpha=0.7),
        Lasso(alpha=0.01, max_iter=5000),
        ElasticNet(alpha=0.01, l1_ratio=0.4, max_iter=5000),
        LinearSVR(max_iter=5000),
    ],
)
def test_affine_regression_values_and_jacobian(model):
    X, y = _regression_data()
    model.fit(X, y)
    data = X[:7]

    result = skgrad.value_and_jacobian(model, data)

    np.testing.assert_allclose(result.values[:, 0], model.predict(data))
    np.testing.assert_allclose(
        result.jacobian,
        np.broadcast_to(model.coef_, (data.shape[0], 1, data.shape[1])),
    )
    assert skgrad.supports(model)


@pytest.mark.parametrize(
    "model",
    [LogisticRegression(), RidgeClassifier(alpha=0.5), LinearSVC(max_iter=5000)],
)
def test_binary_classification_returns_decision_score(model):
    X, score = _regression_data()
    y = (score > np.median(score)).astype(int)
    model.fit(X, y)

    values, jacobian = skgrad.value_and_jacobian(model, X[:6])

    np.testing.assert_allclose(values[:, 0], model.decision_function(X[:6]))
    np.testing.assert_allclose(
        jacobian,
        np.broadcast_to(model.coef_, (6, 1, X.shape[1])),
    )


def test_multiclass_logistic_returns_all_logits_and_target_gradient():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(100, 3))
    scores = np.column_stack((X[:, 0], X[:, 1], -X[:, 0] - X[:, 1]))
    y = np.argmax(scores, axis=1)
    model = LogisticRegression(max_iter=1000).fit(X, y)

    values, jacobian = skgrad.value_and_jacobian(model, X[:5])

    np.testing.assert_allclose(values, model.decision_function(X[:5]))
    np.testing.assert_allclose(jacobian[0], model.coef_)
    np.testing.assert_allclose(
        skgrad.input_gradient(model, X[:5], target=2),
        np.broadcast_to(model.coef_[2], (5, X.shape[1])),
    )
    with pytest.raises(ValueError, match="target is required"):
        skgrad.input_gradient(model, X[:5])


def test_multioutput_ridge_shape_contract():
    X, y = _regression_data()
    targets = np.column_stack((y, -0.5 * y))
    model = Ridge(alpha=0.2).fit(X, targets)

    values, jacobian = skgrad.value_and_jacobian(model, X[:4])

    assert values.shape == (4, 2)
    assert jacobian.shape == (4, 2, 4)
    np.testing.assert_allclose(values, model.predict(X[:4]))
    np.testing.assert_allclose(jacobian[0], model.coef_)


@pytest.mark.parametrize("model_cls", [LinearRegression, Ridge])
def test_multioutput_without_intercept_broadcasts_scalar_intercept(model_cls):
    X, y = _regression_data()
    targets = np.column_stack((y, -0.5 * y))
    model = model_cls(fit_intercept=False).fit(X, targets)
    assert np.asarray(model.intercept_).size == 1

    values, jacobian = skgrad.value_and_jacobian(model, X[:4])

    assert values.shape == (4, 2)
    assert jacobian.shape == (4, 2, 4)
    np.testing.assert_allclose(values, model.predict(X[:4]))


def test_float32_affine_model_preserves_dtype():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(40, 3)).astype(np.float32)
    y = (1.5 * X[:, 0] - 0.7 * X[:, 1]).astype(np.float32)
    model = LinearRegression().fit(X, y)

    values, jacobian = skgrad.value_and_jacobian(model, X[:5])

    assert model.coef_.dtype == np.float32
    assert values.dtype == model.predict(X[:5]).dtype == np.float32
    assert jacobian.dtype == np.float32
