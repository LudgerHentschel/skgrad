"""Adversarial dispatch, precision, and quadrature release regressions."""
import numpy as np
import pytest
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                  LogisticRegression, LogisticRegressionCV, RidgeClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVC, NuSVC, SVR, NuSVR, LinearSVC, LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
import skgrad

BASES = [LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression,
         RidgeClassifier, LinearSVC, LinearSVR, SVC, NuSVC, SVR, NuSVR,
         MLPRegressor, MLPClassifier]

@pytest.mark.parametrize("base", BASES)
def test_prediction_overrides_rejected_through_mro_and_pipeline(base):
    method = "decision_function" if hasattr(base, "decision_function") else "predict"
    bad = type("ChangedPrediction", (base,), {method: lambda self, X: np.zeros(len(X))})
    indirect = type("IndirectOverride", (bad,), {})
    for cls in (bad, indirect):
        model = cls()
        for candidate in (model, make_pipeline(StandardScaler(), model)):
            assert not skgrad.supports(candidate)
            for operation in (skgrad.model_output, skgrad.input_gradient,
                              skgrad.input_jacobian, skgrad.value_and_jacobian):
                with pytest.raises(TypeError, match="does not support"):
                    operation(candidate, [[1., 2.]])
            with pytest.raises(TypeError):
                skgrad.gradient_properties(candidate)
    assert skgrad.supports(type("InheritedPrediction", (base,), {})())


def test_legitimate_sklearn_subclass_and_instance_override():
    X = np.arange(40.).reshape(20, 2)
    model = LogisticRegressionCV(cv=2).fit(X, np.arange(20) % 2)
    assert skgrad.supports(model)
    np.testing.assert_allclose(skgrad.model_output(model, X)[:, 0], model.decision_function(X))
    model.decision_function = lambda X: np.zeros(len(X))
    assert not skgrad.supports(model)
    changed = type("ChangedProbabilities", (MLPClassifier,), {"predict_proba": lambda self, X: X})
    assert not skgrad.supports(changed())


@pytest.mark.parametrize("kernel", ["poly", "rbf", "sigmoid", "linear"])
def test_missing_private_gamma(kernel):
    model = SVR(kernel=kernel).fit([[0.], [1.], [2.]], [0., 1., 4.])
    del model._gamma
    for operation in (skgrad.model_output, skgrad.value_and_jacobian, skgrad.input_gradient):
        if kernel == "linear":
            operation(model, [[.5]])
        else:
            with pytest.raises(RuntimeError, match="skgrad/scikit-learn incompatibility.*_gamma"):
                operation(model, [[.5]])


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("kind", ["affine", "mlp", "linear", "poly", "rbf", "sigmoid",
                                  "standard", "robust", "maxabs", "minmax", "pca", "polynomial"])
def test_input_precision_across_backends(dtype, kind):
    rng = np.random.default_rng(15)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] ** 2 + X[:, 1]
    transforms = dict(standard=StandardScaler, robust=RobustScaler, maxabs=MaxAbsScaler,
                      minmax=MinMaxScaler, pca=PCA, polynomial=PolynomialFeatures)
    if kind == "affine":
        model = Ridge()
    elif kind == "mlp":
        model = MLPRegressor(hidden_layer_sizes=(4,), activation="tanh", max_iter=5, random_state=0)
    elif kind in transforms:
        model = make_pipeline(transforms[kind](), Ridge())
    else:
        model = SVR(kernel=kind)
    model.fit(X, y)
    query = X[:4].astype(dtype)
    expected = np.float32 if dtype == np.float16 else dtype
    result = skgrad.value_and_jacobian(model, query)
    reference = skgrad.value_and_jacobian(model, query.astype(np.float64))
    for actual, accurate in zip(result, reference):
        assert actual.dtype == expected
        np.testing.assert_allclose(actual, accurate, rtol=2e-4, atol=2e-5)
    assert skgrad.model_output(model, query).dtype == expected
    assert skgrad.input_gradient(model, query).dtype == expected
    np.testing.assert_allclose(skgrad.model_output(model, query), result.values, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(skgrad.input_gradient(model, query), result.jacobian[:, 0], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("base", [SVR, NuSVR, SVC, NuSVC])
@pytest.mark.parametrize("degree", range(7))
def test_polynomial_svm_quadrature_exact_and_tight(base, degree):
    # A fitted one-dimensional expansion with a nonzero leading coefficient
    # avoids cancellation/degeneracy that can make a lower order sufficient.
    X = np.array([[.2], [.7], [1.3], [1.8]])
    y = [0, 0, 1, 1] if base in (SVC, NuSVC) else [.1, .4, 1.7, 3.]
    # LibSVM cannot fit NuSVC with a constant kernel (nonfinite duals).
    # Reuse finite fitted coefficients to test its degree-zero evaluation.
    fit_degree = 1 if base is NuSVC and degree == 0 else degree
    model = base(kernel="poly", degree=fit_degree, gamma=.8, coef0=.6).fit(X, y)
    model.degree = degree
    props = skgrad.gradient_properties(model)
    count = max(1, (degree + 1) // 2)
    assert props.exact_quadrature_steps == count
    assert props.constant_jacobian == (degree <= 1)
    start, end = .3, 1.4
    delta = skgrad.model_output(model, [[end]])[0, 0] - skgrad.model_output(model, [[start]])[0, 0]
    def integrate(n):
        nodes, weights = np.polynomial.legendre.leggauss(n)
        points = start + (nodes + 1) * (end - start) / 2
        return (end - start) / 2 * (weights @ skgrad.input_gradient(model, points[:, None])[:, 0])
    np.testing.assert_allclose(integrate(count), delta, rtol=2e-11, atol=2e-11)
    if count > 1:
        assert abs(integrate(count - 1) - delta) > 1e-7
