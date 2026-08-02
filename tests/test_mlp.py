import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier, MLPRegressor

import skgrad


def _one_hidden_mlp(activation):
    model = MLPRegressor(
        hidden_layer_sizes=(2,), activation=activation, solver="lbfgs",
        random_state=0, max_iter=1000
    ).fit(np.array([[-1.0], [0.0], [1.0]]), np.array([-1.0, 0.0, 1.0]))
    model.coefs_ = [np.array([[0.7, -0.4]]), np.array([[1.2], [-0.8]])]
    model.intercepts_ = [np.array([0.8, 0.9]), np.array([0.25])]
    return model


def _finite_difference(model, X, step=1e-6):
    result = np.empty((X.shape[0], skgrad.model_output(model, X).shape[1], X.shape[1]))
    for feature in range(X.shape[1]):
        plus = X.copy()
        minus = X.copy()
        plus[:, feature] += step
        minus[:, feature] -= step
        result[:, :, feature] = (
            skgrad.model_output(model, plus) - skgrad.model_output(model, minus)
        ) / (2.0 * step)
    return result


@pytest.mark.parametrize("activation", ["identity", "logistic", "tanh", "relu"])
def test_mlp_regressor_jacobian_matches_finite_difference(activation):
    model = _one_hidden_mlp(activation)
    X = np.array([[-0.3], [0.2], [0.7]])

    values, jacobian = skgrad.value_and_jacobian(model, X)

    np.testing.assert_allclose(values[:, 0], model.predict(X), atol=1e-12)
    np.testing.assert_allclose(jacobian, _finite_difference(model, X), atol=2e-6)
    np.testing.assert_allclose(skgrad.input_gradient(model, X), jacobian[:, 0, :])


def test_multiclass_mlp_returns_pre_softmax_logits():
    rng = np.random.default_rng(12)
    training = rng.normal(size=(60, 2))
    labels = np.argmax(
        np.column_stack((training[:, 0], training[:, 1], -training.sum(axis=1))),
        axis=1,
    )
    model = MLPClassifier(
        hidden_layer_sizes=(3,), activation="tanh", solver="lbfgs",
        random_state=2, max_iter=1000
    ).fit(training, labels)
    X = training[:5]

    values, jacobian = skgrad.value_and_jacobian(model, X)
    hidden = np.tanh(X @ model.coefs_[0] + model.intercepts_[0])
    expected_logits = hidden @ model.coefs_[1] + model.intercepts_[1]

    assert values.shape == (5, 3)
    assert jacobian.shape == (5, 3, 2)
    np.testing.assert_allclose(values, expected_logits)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probabilities, model.predict_proba(X))
    np.testing.assert_allclose(jacobian, _finite_difference(model, X), atol=2e-6)


def test_binary_mlp_classifier_returns_one_logit():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(50, 2))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    model = MLPClassifier(
        hidden_layer_sizes=(3,), activation="logistic", solver="lbfgs",
        random_state=3, max_iter=1000
    ).fit(X, y)

    values, jacobian = skgrad.value_and_jacobian(model, X[:4])

    assert values.shape == (4, 1)
    assert jacobian.shape == (4, 1, 2)
    probability = model.predict_proba(X[:4])[:, 1]
    np.testing.assert_allclose(values[:, 0], np.log(probability / (1.0 - probability)))
    np.testing.assert_allclose(jacobian, _finite_difference(model, X[:4]), atol=2e-6)
