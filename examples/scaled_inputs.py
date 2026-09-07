"""Compose a scaler explicitly and return derivatives in original units."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import skgrad

rng = np.random.default_rng(9)
X = rng.normal(size=(100, 2)) * np.array([2.0, 20.0])
y = 3 * X[:, 0] - 0.5 * X[:, 1]
scaler = StandardScaler().fit(X)
model = Ridge().fit(scaler.transform(X), y)
evaluation = X[:3]
scaled_gradient = skgrad.input_gradient(model, scaler.transform(evaluation))
original_gradient = scaled_gradient / scaler.scale_[None, :]
pipeline = make_pipeline(scaler, model)
assert skgrad.supports(pipeline)
np.testing.assert_allclose(
    skgrad.input_gradient(pipeline, evaluation), original_gradient, atol=1e-12
)
step = 1e-4
for feature in range(2):
    plus, minus = evaluation.copy(), evaluation.copy()
    plus[:, feature] += step
    minus[:, feature] -= step
    numerical = (model.predict(scaler.transform(plus))
                 - model.predict(scaler.transform(minus))) / (2 * step)
    np.testing.assert_allclose(original_gradient[:, feature], numerical, atol=1e-8)
print("Gradient in original input units:", original_gradient[0])
