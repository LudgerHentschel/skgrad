"""Check an RBF SVR gradient against sklearn's own predictions."""
import numpy as np
from sklearn.svm import SVR
import skgrad

rng = np.random.default_rng(8)
X = rng.normal(size=(80, 3))
y = np.sin(X[:, 0]) + X[:, 1] * X[:, 2]
model = SVR(kernel="rbf", gamma=0.4).fit(X, y)
evaluation = X[:4]
analytic = skgrad.input_gradient(model, evaluation)
step = 1e-5
numerical = np.empty_like(evaluation)
for feature in range(evaluation.shape[1]):
    plus, minus = evaluation.copy(), evaluation.copy()
    plus[:, feature] += step
    minus[:, feature] -= step
    numerical[:, feature] = (model.predict(plus) - model.predict(minus)) / (2 * step)
np.testing.assert_allclose(analytic, numerical, atol=1e-8, rtol=1e-6)
print("Maximum absolute error:", np.max(np.abs(analytic - numerical)))
