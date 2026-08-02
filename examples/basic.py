import numpy as np
from sklearn.linear_model import LogisticRegression

import skgrad


X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y = np.array([0, 1, 0, 1])
model = LogisticRegression().fit(X, y)

scores, jacobian = skgrad.value_and_jacobian(model, [[0.25, 0.75]])
print(scores)
print(jacobian)
