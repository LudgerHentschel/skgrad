"""Differentiate each class score and verify softmax probabilities."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
import skgrad

rng = np.random.default_rng(12)
X = rng.normal(size=(150, 3))
y = np.argmax(np.column_stack((X[:, 0], X[:, 1], -X[:, 0])), axis=1)
model = LogisticRegression(max_iter=500).fit(X, y)
values, jacobian = skgrad.value_and_jacobian(model, X[:4])
np.testing.assert_allclose(values, model.decision_function(X[:4]))
np.testing.assert_allclose(softmax(values, axis=1), model.predict_proba(X[:4]))
for target, label in enumerate(model.classes_):
    gradient = skgrad.input_gradient(model, X[:4], target=target)
    np.testing.assert_allclose(gradient, jacobian[:, target, :])
    print(f"Class {label}: first input gradient {gradient[0]}")
