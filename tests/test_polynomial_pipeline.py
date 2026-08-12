import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import skgrad


def _finite_difference(model, X, step=1e-5):
    gradient = np.empty_like(X, dtype=float)
    for feature in range(X.shape[1]):
        plus, minus = X.copy(), X.copy()
        plus[:, feature] += step
        minus[:, feature] -= step
        gradient[:, feature] = (
            model.predict(plus) - model.predict(minus)
        ) / (2.0 * step)
    return gradient


@pytest.mark.parametrize("include_bias", [False, True])
@pytest.mark.parametrize("interaction_only", [False, True])
def test_polynomial_pipeline_matches_finite_differences(
    include_bias, interaction_only
):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] ** 3 - 1.5 * X[:, 0] * X[:, 1] + 0.7 * X[:, 2] ** 2
    model = make_pipeline(
        PolynomialFeatures(
            degree=3,
            include_bias=include_bias,
            interaction_only=interaction_only,
        ),
        Ridge(alpha=0.1),
    ).fit(X, y)
    evaluation = np.vstack((np.zeros(3), X[:5]))

    result = skgrad.value_and_jacobian(model, evaluation)

    np.testing.assert_allclose(result.values[:, 0], model.predict(evaluation))
    np.testing.assert_allclose(
        result.jacobian[:, 0, :],
        _finite_difference(model, evaluation),
        rtol=2e-5,
        atol=2e-6,
    )


def test_post_expansion_scaling_is_included_in_chain_rule():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(300, 3))
    y = 2.0 * X[:, 0] ** 2 - X[:, 0] * X[:, 1] + 0.5 * X[:, 2] ** 3
    model = make_pipeline(
        PolynomialFeatures(degree=3),
        StandardScaler(),
        Ridge(alpha=0.5),
    ).fit(X, y)
    evaluation = np.vstack((np.zeros(3), X[:6]))

    result = skgrad.value_and_jacobian(model, evaluation)

    np.testing.assert_allclose(result.values[:, 0], model.predict(evaluation))
    np.testing.assert_allclose(
        result.jacobian[:, 0, :],
        _finite_difference(model, evaluation),
        rtol=2e-5,
        atol=2e-6,
    )


def test_known_polynomial_gradient_and_multioutput_shape():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(500, 2))
    targets = np.column_stack(
        (
            1.0 + 2.0 * X[:, 0] ** 2 - 3.0 * X[:, 0] * X[:, 1],
            -0.5 + 4.0 * X[:, 1] ** 3,
        )
    )
    model = make_pipeline(
        PolynomialFeatures(degree=3),
        LinearRegression(),
    ).fit(X, targets)
    evaluation = np.array([[0.0, 0.0], [0.5, -0.25], [-1.0, 2.0]])

    result = skgrad.value_and_jacobian(model, evaluation)

    expected = np.empty((len(evaluation), 2, 2))
    expected[:, 0, 0] = 4.0 * evaluation[:, 0] - 3.0 * evaluation[:, 1]
    expected[:, 0, 1] = -3.0 * evaluation[:, 0]
    expected[:, 1, 0] = 0.0
    expected[:, 1, 1] = 12.0 * evaluation[:, 1] ** 2
    np.testing.assert_allclose(result.jacobian, expected, atol=1e-12)
    np.testing.assert_allclose(
        skgrad.input_gradient(model, evaluation, target=1),
        expected[:, 1, :],
        atol=1e-12,
    )


def test_pipeline_support_is_deliberately_narrow():
    supported = make_pipeline(PolynomialFeatures(2), Ridge())
    scaled_terms = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge())
    scaled_inputs = make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge())
    unrelated = make_pipeline(PCA(2), Ridge())

    assert skgrad.supports(supported)
    assert skgrad.supports(scaled_terms)
    assert not skgrad.supports(scaled_inputs)
    assert not skgrad.supports(unrelated)
    assert not skgrad.gradient_properties(supported).constant_jacobian
    assert skgrad.gradient_properties(supported).exact_quadrature_steps == 1
    assert skgrad.gradient_properties(scaled_terms).exact_quadrature_steps == 1


def test_polynomial_pipeline_reports_exact_quadrature_order():
    cubic = make_pipeline(PolynomialFeatures(3), Ridge())
    quartic = make_pipeline(PolynomialFeatures(4), StandardScaler(), Ridge())
    nonlinear_downstream = make_pipeline(
        PolynomialFeatures(3), MLPRegressor(hidden_layer_sizes=(2,))
    )

    assert skgrad.gradient_properties(cubic).exact_quadrature_steps == 2
    assert skgrad.gradient_properties(quartic).exact_quadrature_steps == 2
    assert skgrad.gradient_properties(nonlinear_downstream).exact_quadrature_steps is None
