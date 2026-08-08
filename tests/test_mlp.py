import inspect
import multiprocessing as mp

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier, MLPRegressor

import skgrad
from skgrad import _mlp


MLP_REGRESSOR_SUPPORTS_LOSS = "loss" in inspect.signature(MLPRegressor).parameters
FORK_AVAILABLE = "fork" in mp.get_all_start_methods()


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


@pytest.mark.parametrize("activation", ["identity", "logistic", "tanh", "relu"])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_selected_mlp_gradient_matches_full_jacobian(activation):
    rng = np.random.default_rng(14)
    training = rng.normal(size=(80, 3))
    targets = rng.normal(size=(80, 2))
    model = MLPRegressor(
        hidden_layer_sizes=(5, 4), activation=activation, solver="lbfgs",
        random_state=4, max_iter=1000
    ).fit(training, targets)
    X = training[:7]
    jacobian = skgrad.input_jacobian(model, X)

    for target in range(2):
        np.testing.assert_allclose(
            skgrad.input_gradient(model, X, target=target),
            jacobian[:, target, :],
            rtol=1e-13,
            atol=1e-13,
        )


def test_selected_multiclass_mlp_gradient_matches_full_jacobian():
    rng = np.random.default_rng(15)
    training = rng.normal(size=(80, 3))
    labels = np.argmax(
        np.column_stack((training[:, 0], training[:, 1], -training.sum(axis=1))),
        axis=1,
    )
    model = MLPClassifier(
        hidden_layer_sizes=(5, 4), activation="tanh", solver="lbfgs",
        random_state=5, max_iter=1000
    ).fit(training, labels)
    X = training[:7]
    jacobian = skgrad.input_jacobian(model, X)

    for target in range(3):
        np.testing.assert_allclose(
            skgrad.input_gradient(model, X, target=target),
            jacobian[:, target, :],
            rtol=1e-13,
            atol=1e-13,
        )


def test_mlp_without_hidden_layers_uses_selected_output_weights():
    rng = np.random.default_rng(16)
    training = rng.normal(size=(40, 3))
    targets = rng.normal(size=(40, 2))
    model = MLPRegressor(
        hidden_layer_sizes=(), solver="lbfgs", random_state=6, max_iter=1000
    ).fit(training, targets)

    for target in range(2):
        expected = np.broadcast_to(model.coefs_[0][:, target], (4, 3))
        np.testing.assert_allclose(
            skgrad.input_gradient(model, training[:4], target=target), expected
        )


def test_fast_mlp_gradient_preserves_target_validation():
    rng = np.random.default_rng(17)
    training = rng.normal(size=(30, 2))
    targets = rng.normal(size=(30, 2))
    model = MLPRegressor(
        hidden_layer_sizes=(3,), solver="lbfgs", random_state=7, max_iter=1000
    ).fit(training, targets)

    with pytest.raises(ValueError, match="target is required"):
        skgrad.input_gradient(model, training[:2])
    with pytest.raises(TypeError, match="integer"):
        skgrad.input_gradient(model, training[:2], target=0.5)
    with pytest.raises(TypeError, match="integer"):
        skgrad.input_gradient(model, training[:2], target=True)
    with pytest.raises(ValueError, match="between"):
        skgrad.input_gradient(model, training[:2], target=2)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_large_mlp_gradient_parallel_path_matches_full_jacobian(monkeypatch):
    rng = np.random.default_rng(18)
    training = rng.normal(size=(80, 3))
    model = MLPRegressor(
        hidden_layer_sizes=(5, 4), activation="tanh", solver="lbfgs",
        random_state=8, max_iter=1000
    ).fit(training, rng.normal(size=80))
    X = rng.normal(size=(10_000, 3))
    monkeypatch.setattr(_mlp, "_parallel_worker_count", lambda n_samples: 4)

    np.testing.assert_allclose(
        skgrad.input_gradient(model, X),
        skgrad.input_jacobian(model, X)[:, 0, :],
        rtol=1e-13,
        atol=1e-13,
    )


def test_parallel_worker_heuristic_is_bounded(monkeypatch):
    monkeypatch.setattr(_mlp.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(
        _mlp,
        "threadpool_info",
        lambda: [{"user_api": "blas", "num_threads": 4}],
    )
    assert _mlp._parallel_worker_count(4_999) == 1
    assert _mlp._parallel_worker_count(5_000) == 2
    assert _mlp._parallel_worker_count(10_000) == 4
    assert _mlp._parallel_worker_count(1_000_000) == 4

    monkeypatch.setattr(
        _mlp,
        "threadpool_info",
        lambda: [{"user_api": "blas", "num_threads": 8}],
    )
    assert _mlp._parallel_worker_count(10_000) == 2

    monkeypatch.setattr(_mlp.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(
        _mlp,
        "threadpool_info",
        lambda: [{"user_api": "blas", "num_threads": 2}],
    )
    assert _mlp._parallel_worker_count(10_000) == 1


@pytest.mark.parametrize("activation", ["identity", "logistic", "tanh", "relu"])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_float32_mlp_preserves_values_and_gradient_dtype(activation):
    rng = np.random.default_rng(19)
    training = rng.normal(size=(60, 3)).astype(np.float32)
    targets = rng.normal(size=60).astype(np.float32)
    model = MLPRegressor(
        hidden_layer_sizes=(4, 3), activation=activation, max_iter=2,
        random_state=9
    ).fit(training, targets)
    X = training[:6]

    values, jacobian = skgrad.value_and_jacobian(model, X)
    gradient = skgrad.input_gradient(model, X)

    assert model.coefs_[0].dtype == np.float32
    assert model.predict(X).dtype == np.float32
    assert values.dtype == np.float32
    assert jacobian.dtype == np.float32
    assert gradient.dtype == np.float32
    np.testing.assert_allclose(values[:, 0], model.predict(X), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(gradient, jacobian[:, 0, :], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    ("fit_dtype", "input_dtype", "expected_dtype"),
    [
        (np.float32, np.float16, np.float32),
        (np.float32, np.float64, np.float64),
        (np.float64, np.float32, np.float64),
        (np.float64, np.float64, np.float64),
    ],
)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_mlp_mixed_dtype_matches_sklearn(fit_dtype, input_dtype, expected_dtype):
    rng = np.random.default_rng(20)
    training = rng.normal(size=(60, 3)).astype(fit_dtype)
    targets = rng.normal(size=60).astype(fit_dtype)
    model = MLPRegressor(
        hidden_layer_sizes=(4,), activation="tanh", max_iter=2, random_state=10
    ).fit(training, targets)
    X = training[:6].astype(input_dtype)

    values, jacobian = skgrad.value_and_jacobian(model, X)
    gradient = skgrad.input_gradient(model, X)

    assert values.dtype == expected_dtype
    assert jacobian.dtype == expected_dtype
    assert gradient.dtype == expected_dtype
    assert values.dtype == model.predict(X).dtype
    np.testing.assert_allclose(values[:, 0], model.predict(X), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(gradient, jacobian[:, 0, :], rtol=2e-6, atol=2e-6)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_float32_mlp_classifier_preserves_logit_and_gradient_dtype():
    rng = np.random.default_rng(21)
    training = rng.normal(size=(80, 3)).astype(np.float32)
    labels = np.argmax(
        np.column_stack((training[:, 0], training[:, 1], -training.sum(axis=1))),
        axis=1,
    )
    model = MLPClassifier(
        hidden_layer_sizes=(5,), activation="logistic", max_iter=2,
        random_state=11
    ).fit(training, labels)
    X = training[:6]

    values, jacobian = skgrad.value_and_jacobian(model, X)
    gradient = skgrad.input_gradient(model, X, target=1)

    assert values.dtype == np.float32
    assert jacobian.dtype == np.float32
    assert gradient.dtype == np.float32
    np.testing.assert_allclose(gradient, jacobian[:, 1, :], rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(
    not MLP_REGRESSOR_SUPPORTS_LOSS,
    reason="Poisson MLPRegressor was added in scikit-learn 1.7",
)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_poisson_mlp_values_and_gradients(dtype):
    rng = np.random.default_rng(22)
    training = rng.normal(size=(100, 3)).astype(dtype)
    targets = np.exp(
        0.3 * training[:, 0] - 0.2 * training[:, 1] + 0.1
    ).astype(dtype)
    model = MLPRegressor(
        hidden_layer_sizes=(5, 4), activation="tanh", loss="poisson",
        max_iter=5, random_state=12
    ).fit(training, targets)
    X = training[:7]

    values, jacobian = skgrad.value_and_jacobian(model, X)
    gradient = skgrad.input_gradient(model, X)

    assert model.out_activation_ == "exp"
    assert values.dtype == dtype
    assert jacobian.dtype == dtype
    assert gradient.dtype == dtype
    np.testing.assert_allclose(values[:, 0], model.predict(X), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(gradient, jacobian[:, 0, :], rtol=2e-6, atol=2e-6)
    finite_difference = _finite_difference(
        model, X.astype(np.float64), step=1e-4 if dtype == np.float32 else 1e-6
    )
    np.testing.assert_allclose(
        jacobian.astype(np.float64),
        finite_difference,
        rtol=2e-3 if dtype == np.float32 else 2e-6,
        atol=2e-3 if dtype == np.float32 else 2e-6,
    )


@pytest.mark.skipif(
    not MLP_REGRESSOR_SUPPORTS_LOSS,
    reason="Poisson MLPRegressor was added in scikit-learn 1.7",
)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_multioutput_poisson_mlp_selected_gradient():
    rng = np.random.default_rng(23)
    training = rng.normal(size=(100, 3))
    targets = np.column_stack(
        (
            np.exp(0.2 * training[:, 0] + 0.1),
            np.exp(-0.3 * training[:, 1] + 0.2),
        )
    )
    model = MLPRegressor(
        hidden_layer_sizes=(5,), activation="logistic", loss="poisson",
        max_iter=5, random_state=13
    ).fit(training, targets)
    X = training[:7]

    values, jacobian = skgrad.value_and_jacobian(model, X)

    np.testing.assert_allclose(values, model.predict(X), rtol=2e-12, atol=2e-12)
    for target in range(2):
        np.testing.assert_allclose(
            skgrad.input_gradient(model, X, target=target),
            jacobian[:, target, :],
            rtol=2e-12,
            atol=2e-12,
        )


@pytest.mark.skipif(not FORK_AVAILABLE, reason="requires the fork start method")
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_parallel_executor_is_recreated_after_fork():
    rng = np.random.default_rng(24)
    training = rng.normal(size=(40, 2))
    model = MLPRegressor(
        hidden_layer_sizes=(3,), max_iter=2, random_state=14
    ).fit(training, rng.normal(size=40))
    X = rng.normal(size=(5_000, 2))
    expected = skgrad.input_gradient(model, X)
    receive, send = mp.get_context("fork").Pipe(duplex=False)

    def calculate_in_child():
        gradient = skgrad.input_gradient(model, X)
        send.send((gradient.shape, float(np.max(np.abs(gradient - expected)))))
        send.close()

    process = mp.get_context("fork").Process(target=calculate_in_child)
    process.start()
    send.close()
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("parallel gradient calculation hung after fork")

    assert process.exitcode == 0
    shape, maximum_error = receive.recv()
    assert shape == expected.shape
    assert maximum_error == 0.0
    receive.close()
