"""Request one MLP output without constructing all gradients."""
import numpy as np
from sklearn.neural_network import MLPRegressor
import skgrad

rng = np.random.default_rng(7)
X = rng.normal(size=(120, 3))
y = np.column_stack((np.sin(X[:, 0]), X[:, 1] * X[:, 2]))
model = MLPRegressor(hidden_layer_sizes=(12,), activation="tanh",
                     solver="lbfgs", max_iter=2000, random_state=7).fit(X, y)
values, jacobian = skgrad.value_and_jacobian(model, X[:5])
selected = skgrad.input_gradient(model, X[:5], target=1)
np.testing.assert_allclose(values, model.predict(X[:5]))
np.testing.assert_allclose(selected, jacobian[:, 1, :], atol=1e-12)
print("Values, full Jacobian, selected gradient:", values.shape,
      jacobian.shape, selected.shape)
