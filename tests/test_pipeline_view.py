import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.exceptions import NotFittedError
import skgrad


def fitted():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(60, 3)) * [1, 4, 9]
    model = Pipeline([("preprocess", Pipeline([("scale", StandardScaler(copy=False)),
                    ("pca", PCA(2, whiten=True))])), ("regressor", Ridge())])
    model.fit(X.copy(), X[:, 0] - X[:, 1])
    return model, X


def test_views_preserve_outputs_and_select_gradient_coordinates():
    model, X = fitted()
    original = X.copy()
    for name in (None, "preprocess__scale", "preprocess__pca", "preprocess"):
        view = skgrad.pipeline_view(model, after=name)
        result = view.value_and_jacobian(X)
        np.testing.assert_allclose(result.values[:, 0], model.predict(X.copy()), atol=1e-12)
        np.testing.assert_allclose(view.model_output(X), result.values)
        np.testing.assert_allclose(view.input_jacobian(X), result.jacobian)
        np.testing.assert_allclose(view.input_gradient(X), result.jacobian[:, 0])
        assert view.input_gradient(X).shape[1] == (3 if name in (None, "preprocess__scale") else 2)
    np.testing.assert_array_equal(X, original)
    scale_view = skgrad.pipeline_view(model, after="preprocess__scale")
    np.testing.assert_allclose(scale_view.input_gradient(X) / model[0][0].scale_,
                               skgrad.input_gradient(model, X), atol=1e-12)
    assert list(scale_view.get_feature_names_out()) == ["x0", "x1", "x2"]
    assert list(skgrad.pipeline_view(model, after="preprocess").get_feature_names_out()) == ["pca0", "pca1"]


def test_feature_names_and_single_input():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 2))
    model = Pipeline([("poly", PolynomialFeatures(2)), ("reg", Ridge())]).fit(X, X[:, 0])
    view = skgrad.pipeline_view(model, after="poly")
    assert view.transform(X[0]).shape == (1, 6)
    assert list(view.get_feature_names_out(["a", "b"])) == ["1", "a", "b", "a^2", "a b", "b^2"]
    with pytest.raises(ValueError, match="align"):
        view.get_feature_names_out(["one"])
    with pytest.raises(ValueError, match="original"):
        view.transform(view.transform(X))


def test_invalid_selection_and_unfitted_models():
    model, X = fitted()
    for name in (False, 1, ""):
        with pytest.raises(TypeError, match="after"):
            skgrad.pipeline_view(model, after=name)
    with pytest.raises(ValueError, match="unknown"):
        skgrad.pipeline_view(model, after="missing")
    with pytest.raises(ValueError, match="final predictor"):
        skgrad.pipeline_view(model, after="regressor")
    with pytest.raises(TypeError, match="Pipeline"):
        skgrad.pipeline_view(model[-1], after="scale")
    with pytest.raises(NotFittedError):
        skgrad.pipeline_view(Pipeline([("scale", StandardScaler()), ("reg", Ridge())]), after="scale")
    unsupported = Pipeline([("custom", FunctionTransformer(np.sin)), ("reg", Ridge())]).fit(X, X[:, 0])
    with pytest.raises(TypeError, match="unsupported preprocessing"):
        skgrad.pipeline_view(unsupported, after="custom")


def test_dataframe_schema_validation():
    pd = pytest.importorskip("pandas")
    model, X = fitted()
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    model.fit(frame.copy(), X[:, 0])
    view = skgrad.pipeline_view(model, after="preprocess__scale")
    assert list(view.get_feature_names_out()) == list(frame.columns)
    view.transform(frame)
    with pytest.raises(ValueError, match="order"):
        view.transform(frame[["b", "a", "c"]])
