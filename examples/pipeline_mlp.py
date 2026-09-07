"""Differentiate a complete scaled/PCA/MLP pipeline in original coordinates."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import skgrad

rng = np.random.default_rng(17)
X = rng.normal(size=(100, 4))
y = np.sin(X[:, 0]) + X[:, 1] - X[:, 2]
preprocessing = make_pipeline(StandardScaler(), PCA(3, whiten=True))
model = make_pipeline(preprocessing, MLPRegressor(
    hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs",
    max_iter=2000, tol=1e-3, alpha=0.1, random_state=17,
)).fit(X, y)
evaluation = X[:3]
gradient = skgrad.input_gradient(model, evaluation)
assert gradient.shape == (3, 4)  # original inputs, not three PCA components
step = 1e-5
for feature in range(4):
    plus, minus = evaluation.copy(), evaluation.copy()
    plus[:, feature] += step
    minus[:, feature] -= step
    numerical = (model.predict(plus) - model.predict(minus)) / (2 * step)
    np.testing.assert_allclose(gradient[:, feature], numerical, atol=1e-7)
print("Gradient in original feature coordinates:", gradient)
