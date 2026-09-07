"""Integrate an analytic polynomial gradient along one straight path."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
import skgrad

rng = np.random.default_rng(10)
X = rng.normal(size=(150, 2))
y = X[:, 0] ** 3 + X[:, 0] * X[:, 1] - 2 * X[:, 1]
model = make_pipeline(PolynomialFeatures(3), StandardScaler(), Ridge(alpha=0.01))
model.fit(X, y)
baseline = np.zeros(2)
point = np.array([0.8, -0.4])
order = skgrad.gradient_properties(model).exact_quadrature_steps
nodes, weights = np.polynomial.legendre.leggauss(order)
path = baseline + ((nodes + 1) / 2)[:, None] * (point - baseline)
gradients = skgrad.input_gradient(model, path)
attributions = (point - baseline) * ((weights / 2) @ gradients)
difference = model.predict(point[None, :])[0] - model.predict(baseline[None, :])[0]
np.testing.assert_allclose(attributions.sum(), difference, atol=1e-12)
print("Quadrature points:", order)
print("Feature contributions:", attributions)
print("Completeness residual:", attributions.sum() - difference)
