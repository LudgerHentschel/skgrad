"""Compare original and standardized gradients with one explicit boundary."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import skgrad

rng = np.random.default_rng(11)
X = rng.normal(size=(60, 3)) * [1, 4, 9]
model = Pipeline([("scale", StandardScaler()), ("regressor", Ridge())]).fit(
    X, X[:, 0] - X[:, 1]
)
original = skgrad.pipeline_view(model)
standardized = skgrad.pipeline_view(model, after="scale")
# Both view methods accept the same ORIGINAL inputs.
raw_gradient = original.input_gradient(X[:4])
z_gradient = standardized.input_gradient(X[:4])
np.testing.assert_allclose(z_gradient / model[0].scale_, raw_gradient)
np.testing.assert_allclose(original.model_output(X[:4]), standardized.model_output(X[:4]))
print("Original gradient:", raw_gradient[0])
print("Standardized gradient:", z_gradient[0])
print("Selected names:", standardized.get_feature_names_out())
