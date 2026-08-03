# skgrad

`skgrad` computes analytic input gradients for fitted scikit-learn models.
It provides one consistent interface for affine estimators and multilayer
perceptrons without numerical finite differences or automatic-differentiation
frameworks.

```python
import skgrad

values, jacobian = skgrad.value_and_jacobian(model, X)
```

`values` always has shape `(samples, outputs)` and `jacobian` has shape
`(samples, outputs, features)`. Regression values are predictions.
Classification values are decision scores or logits, never probabilities.

## Supported models

- `LinearRegression`, `Ridge`, `Lasso`, and `ElasticNet`
- `LogisticRegression` and `RidgeClassifier`
- `MLPRegressor` with identity output
- `MLPClassifier`, using pre-probability logits

Hidden MLP activations may be identity, logistic, tanh, or ReLU. ReLU uses a
zero derivative at its nondifferentiable origin, matching scikit-learn's
backpropagation convention.

## API

```python
skgrad.supports(model)
skgrad.model_output(model, X)
skgrad.input_jacobian(model, X)
skgrad.input_gradient(model, X, target=None)
skgrad.value_and_jacobian(model, X)
```

`input_gradient` is the scalar-output convenience API. A target is required
when the model has multiple outputs. `supports` is the single capability check;
users do not need to distinguish internal model families.

For downstream composition, `skgrad.gradient_properties(model)` reports
computational metadata such as whether the input Jacobian is constant. This
allows consumers to optimize integration without duplicating skgrad's estimator
registry or exposing separate family-specific support predicates.

Tree models, parameter gradients, numerical differentiation, Integrated
Gradients, and baseline handling are deliberately outside the package scope.

The `value_and_jacobian` primitive is designed for downstream composition.
For example, UnifiedIG integrates its Jacobians to explain model outputs, and
other consumers can combine them with analytic chain-rule transformations.
