"""Pipeline tests use sklearn predictions, explicit derivatives, and IG completeness."""
import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (StandardScaler, RobustScaler, MaxAbsScaler,
                                   MinMaxScaler, PolynomialFeatures, FunctionTransformer)
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
import skgrad


def data(dtype=np.float64):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 4)).astype(dtype)
    y = np.column_stack((X[:, 0] + X[:, 1] ** 2, X[:, 2] - X[:, 3]))
    return X, y


def finite_difference(model, X, score=False, h=1e-5):
    predict = model.decision_function if score else model.predict
    columns = []
    for j in range(X.shape[1]):
        plus, minus = X.copy(), X.copy()
        plus[:, j] += h
        minus[:, j] -= h
        diff = (predict(plus) - predict(minus)) / (2*h)
        columns.append(diff.reshape(len(X), -1))
    return np.stack(columns, axis=-1)


@pytest.mark.parametrize("transformer", [
    StandardScaler(), StandardScaler(with_std=False), StandardScaler(with_mean=False),
    RobustScaler(), RobustScaler(with_scaling=False), RobustScaler(unit_variance=True),
    MaxAbsScaler(), MinMaxScaler(feature_range=(-2, 3)),
    PCA(n_components=3), PCA(n_components=3, whiten=True),
    SelectKBest(f_regression, k=2), VarianceThreshold(),
])
def test_transformer_gradient_matches_sklearn(transformer):
    X, y = data()
    # Supervised selectors need a scalar training target.
    model = make_pipeline(transformer, Ridge()).fit(X, y[:, 0])
    evaluation = X[:5]
    values, jacobian = skgrad.value_and_jacobian(model, evaluation)
    np.testing.assert_allclose(values[:, 0], model.predict(evaluation), atol=1e-12)
    np.testing.assert_allclose(skgrad.model_output(model, evaluation), values, atol=1e-12)
    np.testing.assert_allclose(jacobian, finite_difference(model, evaluation), atol=1e-8)
    np.testing.assert_allclose(skgrad.input_gradient(model, evaluation), jacobian[:, 0, :])
    assert skgrad.gradient_properties(model).constant_jacobian
    assert skgrad.gradient_properties(model).exact_quadrature_steps == 1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nested_scaling_polynomial_pca_multioutput_and_input_ownership(dtype):
    X, y = data(dtype)
    model = make_pipeline(
        make_pipeline(StandardScaler(copy=False), PolynomialFeatures(2)),
        RobustScaler(copy=False), PCA(5, whiten=True, copy=False), Ridge(),
    ).fit(X.copy(), y)
    evaluation = X[:4].copy()
    original = evaluation.copy()
    values, jac = skgrad.value_and_jacobian(model, evaluation)
    np.testing.assert_array_equal(evaluation, original)
    np.testing.assert_allclose(values, model.predict(evaluation.copy()), atol=2e-5)
    np.testing.assert_allclose(skgrad.model_output(model, evaluation), values, atol=2e-5)
    np.testing.assert_allclose(jac, finite_difference(model, evaluation.astype(float)), atol=2e-5)
    for target in range(2):
        np.testing.assert_allclose(skgrad.input_gradient(model, evaluation, target), jac[:, target], atol=2e-5)
    np.testing.assert_array_equal(evaluation, original)
    assert jac.shape == (4, 2, 4)


def test_clipping_boundary_convention_and_metadata():
    X = np.array([[0., 0.], [1., 1.], [0.5, 0.5]])
    model = make_pipeline(MinMaxScaler(clip=True), Ridge()).fit(X, X[:, 0]+X[:, 1])
    evaluation = np.array([[-1., 2.], [0., 1.], [0.3, 0.7]])
    gradient = skgrad.input_gradient(model, evaluation)
    np.testing.assert_array_equal(gradient[:2], 0.)
    np.testing.assert_allclose(gradient[2], model[-1].coef_)
    np.testing.assert_allclose(skgrad.model_output(model, evaluation)[:, 0], model.predict(evaluation))
    np.testing.assert_allclose(gradient[[0, 2]], finite_difference(model, evaluation[[0, 2]])[:, 0], atol=1e-8)
    assert skgrad.gradient_properties(model) == (False, None)


def test_multiple_polynomial_stages_metadata_and_raw_path_completeness():
    X, y = data()
    model = make_pipeline(StandardScaler(), PolynomialFeatures(2),
                          PolynomialFeatures(3), Ridge()).fit(X[:, :2], y[:, 0])
    props = skgrad.gradient_properties(model)
    assert props == (False, 3)
    baseline, point = np.array([-.2, .3]), np.array([.8, -.4])
    nodes, weights = np.polynomial.legendre.leggauss(props.exact_quadrature_steps)
    path = baseline + ((nodes+1)/2)[:, None]*(point-baseline)
    contributions = (point-baseline)*((weights/2) @ skgrad.input_gradient(model, path))
    difference = np.diff(model.predict(np.stack([baseline, point])))[0]
    np.testing.assert_allclose(contributions.sum(), difference, atol=1e-10)


def test_selected_mlp_uses_selected_backend(monkeypatch):
    X, y = data()
    model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(5,),
                          activation="tanh", solver="lbfgs", max_iter=2000, random_state=3)).fit(X, y)
    expected = finite_difference(model, X[:3])
    def fail(*args):
        raise AssertionError("full MLP Jacobian should not be constructed")
    monkeypatch.setattr("skgrad.api.mlp_value_and_jacobian", fail)
    for target in range(2):
        np.testing.assert_allclose(skgrad.input_gradient(model, X[:3], target), expected[:, target], atol=1e-7)
    assert skgrad.gradient_properties(model) == (False, None)


def test_classification_targets_and_passthrough():
    X, _ = data()
    y = np.argmax(X[:, :3], axis=1)
    model = Pipeline([("identity", "passthrough"), ("scale", StandardScaler()),
                      ("disabled", None), ("model", LogisticRegression())]).fit(X, y)
    values, jac = skgrad.value_and_jacobian(model, X[:3])
    np.testing.assert_allclose(values, model.decision_function(X[:3]))
    np.testing.assert_allclose(jac, finite_difference(model, X[:3], score=True), atol=1e-8)
    with pytest.raises(ValueError, match="target is required"):
        skgrad.input_gradient(model, X[:3])
    with pytest.raises(TypeError, match="integer"):
        skgrad.input_gradient(model, X[:3], True)
    with pytest.raises(ValueError, match="between"):
        skgrad.input_gradient(model, X[:3], 3)
    np.testing.assert_allclose(skgrad.input_gradient(model, X[0], np.int64(1)), jac[:1, 1])


def test_dropped_features_return_zero_gradient():
    X, y = data()
    model = make_pipeline(SelectKBest(f_regression, k=2), Ridge()).fit(X, y[:, 0])
    gradient = skgrad.input_gradient(model, X[:3])
    np.testing.assert_array_equal(gradient[:, ~model[0].get_support()], 0.)


def test_rejection_and_fitted_state():
    class CustomScaler(StandardScaler):
        def transform(self, X, copy=None):
            return X ** 2
    for transformer in [FunctionTransformer(np.sin), CustomScaler(),
                        ColumnTransformer([("scale", StandardScaler(), [0])])]:
        model = make_pipeline(transformer, Ridge())
        assert not skgrad.supports(model)
        with pytest.raises(TypeError, match="support"):
            skgrad.input_gradient(model, [[1., 2.]])
    assert not skgrad.supports(Pipeline([]))
    assert not skgrad.supports(Pipeline([("scale", StandardScaler()), ("end", None)]))
    model = make_pipeline(StandardScaler(), Ridge())
    assert skgrad.supports(model)
    with pytest.raises(NotFittedError):
        skgrad.input_gradient(model, [[1., 2.]])
    X, y = data()
    model.fit(X, y)
    with pytest.raises(ValueError):
        skgrad.input_gradient(model, [[1., 2.]])


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_degenerate_whitening_has_explicit_error():
    model = make_pipeline(PCA(2, whiten=True), Ridge()).fit(np.ones((5, 2)), np.arange(5))
    with pytest.raises(ValueError, match="non-degenerate"):
        skgrad.input_gradient(model, [[1., 1.]])


def test_invalid_nested_pipeline_does_not_silently_drop_steps():
    invalid = Pipeline([("nested", Pipeline([("model", Ridge()), ("end", None)]))])
    assert not skgrad.supports(invalid)
    empty = Pipeline([("empty", Pipeline([])), ("model", Ridge())])
    assert not skgrad.supports(empty)


@pytest.mark.parametrize("scaler", [StandardScaler(), RobustScaler(), MaxAbsScaler(), MinMaxScaler()])
def test_float32_scaler_pipeline_preserves_dtype_and_constant_columns(scaler):
    X, y = data(np.float32)
    X[:, 3] = 1
    model = make_pipeline(scaler, Ridge()).fit(X, y)
    values, jacobian = skgrad.value_and_jacobian(model, X[:3])
    assert values.dtype == np.float32
    assert jacobian.dtype == np.float32
    np.testing.assert_allclose(values, model.predict(X[:3]), atol=1e-6)
    np.testing.assert_allclose(jacobian, finite_difference(model, X[:3].astype(float)), atol=1e-6)
