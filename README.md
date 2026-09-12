# skgrad

[![Tests](https://github.com/LudgerHentschel/skgrad/actions/workflows/tests.yml/badge.svg)](https://github.com/LudgerHentschel/skgrad/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://ludgerhentschel.github.io/skgrad/)
[![PyPI version](https://img.shields.io/pypi/v/skgrad.svg)](https://pypi.org/project/skgrad/)
[![Python versions](https://img.shields.io/pypi/pyversions/skgrad.svg)](https://pypi.org/project/skgrad/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/LudgerHentschel/skgrad/blob/main/LICENSE)

**scikit-learn does not expose input derivatives. skgrad computes them in
closed form.**

For a supported fitted estimator `f` and input `x`, skgrad returns ∂f/∂x
analytically, to floating-point precision—without finite differences, automatic
differentiation, model conversion, or model approximation.

```python
gradient = skgrad.input_gradient(model, X)
```

Classification derivatives use decision scores or logits, not probabilities.
For one scalar output, `input_gradient` returns `(samples, features)`; the full
Jacobian has shape `(samples, outputs, features)`. Select `target` explicitly
for a multi-output model, or request `input_jacobian`. Unsupported models raise
an error instead of falling back to numerical differentiation.

For scalar-output models, `gradient[i, j]` is the derivative of prediction `i`
with respect to feature `j`. Multi-output models expose one gradient per target
or the complete input Jacobian.

Input derivatives support local sensitivity analysis, linearization around an
operating point, gradient-based optimization against a fitted model, and
gradient-based feature attribution, including Integrated Gradients.

Coverage includes linear models, kernel SVMs, multilayer perceptrons, and
pipelines composed of supported scalers, polynomial expansion, PCA, and fitted
feature selection. Unsupported estimators raise `TypeError` rather than
silently falling back to a numerical approximation.

skgrad also exposes derivative structure that downstream algorithms can exploit.
`gradient_properties` identifies constant Jacobians and, for polynomial
pipelines, the number of Gauss–Legendre nodes required for exact path
integration. Callers can therefore eliminate quadrature error rather than bound
it.

Read the **[skgrad documentation](https://ludgerhentschel.github.io/skgrad/)** for
worked examples, the API contract, model coverage, and numerical conventions.
For automated readers, [llms.txt](https://ludgerhentschel.github.io/skgrad/llms.txt)
links to the guides, complete example files, and rendered API reference.

## Installation

```console
pip install skgrad
```

`skgrad` requires Python 3.9 or later and installs NumPy, scikit-learn, and its
small runtime dependencies automatically.

## Quick start

```python
import numpy as np
from sklearn.neural_network import MLPRegressor

import skgrad

rng = np.random.default_rng(0)
X_train = rng.normal(size=(200, 4))
y_train = np.sin(X_train[:, 0]) + X_train[:, 1] * X_train[:, 2]

model = MLPRegressor(
    hidden_layer_sizes=(32, 32),
    activation="tanh",
    max_iter=1000,
    random_state=0,
).fit(X_train, y_train)

X_eval = X_train[:5]

values = skgrad.model_output(model, X_eval)       # (5, 1)
gradient = skgrad.input_gradient(model, X_eval)   # (5, 4)

values, jacobian = skgrad.value_and_jacobian(model, X_eval)
# values:   (samples, outputs)
# jacobian: (samples, outputs, features)
```

For a multi-output model, select one scalar output explicitly:

```python
gradient = skgrad.input_gradient(model, X_eval, target=1)
```

Use `skgrad.supports(model)` to check model coverage before calculation.

![A one-dimensional fitted ReLU network and its exact input gradients](https://raw.githubusercontent.com/LudgerHentschel/skgrad/main/docs/relu-analytic-gradients.png)

For a ReLU MLP, a forward pass identifies the active hidden units at each
input. Their known slopes and fitted weights then combine in a batched reverse
pass. With multiple features, the scalar slopes pictured above become input
gradient vectors computed by the same matrix operations.

## Supported models

| Family | Models | Differentiated output |
|---|---|---|
| Linear regression | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet` | Prediction |
| Linear classification | `LogisticRegression`, `RidgeClassifier`, `LinearSVC` | Decision score |
| Linear support-vector regression | `LinearSVR` | Prediction |
| Kernel support-vector regression | `SVR`, `NuSVR` | Prediction |
| Binary kernel classification | `SVC`, `NuSVC` | Decision score |
| Neural-network regression | `MLPRegressor` with squared-error or Poisson loss | Prediction, including the Poisson exponential output link |
| Neural-network classification | `MLPClassifier` | Binary or multiclass logits before logistic/softmax |
| Continuous pipelines | Supported scalers, polynomial expansion, PCA, and fitted feature selectors, then any supported estimator | Final estimator output, differentiated with respect to pipeline input features |

MLP hidden activations may be identity, logistic, tanh, or ReLU. At ReLU's
nondifferentiable origin, `skgrad` uses a zero derivative, matching
scikit-learn's backpropagation convention. Scalar and multi-output regression,
binary classification, and multiclass classification are supported.

Kernel SVMs support scikit-learn's `linear`, `poly`, `rbf`, and `sigmoid`
kernels. Multiclass kernel classifiers, callable kernels, and precomputed
kernels are not currently supported.

Tree models are intentionally excluded. Their predictions are piecewise
constant, so ordinary gradients are zero almost everywhere and undefined at
split boundaries. Use [TreeIG](https://github.com/LudgerHentschel/treeig),
which computes exact Integrated Gradients from the prediction jumps at tree
split crossings.

`skgrad` expects finite dense numeric inputs. Sequential and nested sklearn
pipelines may combine `StandardScaler`, `RobustScaler`, `MaxAbsScaler`,
`MinMaxScaler`, `PolynomialFeatures`, `PCA`, and supported fitted feature
selectors before a supported estimator. Gradients refer to the inputs of the
supplied pipeline, including every supported preprocessing chain rule.
Unknown transformers reject the entire analytic route; preprocessing is never
silently removed. See [pipeline conventions](https://ludgerhentschel.github.io/skgrad/pipelines.html)
for clipping, whitening, feature selection, and attribution coordinates.

## Output semantics

`skgrad` differentiates prediction functions with respect to input features,
not training losses with respect to fitted parameters.

- Regressors return their prediction output.
- Binary classifiers expose one score or logit for the positive class.
- Multiclass classifiers expose one score or logit per class in
  `model.classes_` order.
- Classification probabilities are deliberately not differentiated. Scores
  and logits compose cleanly with downstream attribution methods and avoid the
  redundant common direction of multiclass probabilities.
- Values and derivatives preserve normalized input precision across all backends:
  float32 stays float32, float16 promotes to float32, and float64 stays float64.
  Parameters are cast for evaluation; sklearn outputs may use a different dtype.

See [the shape and output semantics](https://ludgerhentschel.github.io/skgrad/semantics.html)
for the complete contract.

## Performance

Fast input gradients are the reason `skgrad` exists. Central finite differences
require two model evaluations per feature. `skgrad` instead reuses fitted
coefficients for affine models and computes MLP gradients with a forward and
reverse pass, obtaining all feature derivatives together. This matters especially
for Integrated Gradients, which evaluates gradients repeatedly along paths.

In the documented CPU benchmarks, logistic regression and MLP gradients were
roughly **14× faster at 10 features, 100–170× at 100 features, and 950–1,300× at
1,000 features** than central differences. The tested MLP also achieved speeds
comparable to PyTorch CPU autodiff, directly from the fitted scikit-learn model.
Results depend on the model, batch size, and runtime environment.

Use `input_gradient` when you need one output: it avoids constructing the full
multi-output Jacobian. See the **[performance guide](https://ludgerhentschel.github.io/skgrad/performance.html)**
for accuracy checks, full timing tables, benchmark methodology, and reproducible
scripts.

## API

```python
skgrad.supports(model)
skgrad.gradient_properties(model)
skgrad.model_output(model, X)
skgrad.input_gradient(model, X, target=None)
skgrad.input_jacobian(model, X)
skgrad.value_and_jacobian(model, X)
```

`value_and_jacobian` is the general composition primitive. Its result contains
`values` with shape `(samples, outputs)` and `jacobian` with shape
`(samples, outputs, features)`. `input_gradient` is the faster convenience API
when one scalar output is required.

`gradient_properties(model)` reports useful computational metadata. In
particular, downstream consumers can detect constant affine Jacobians and
avoid redundant evaluations. `exact_quadrature_steps` also reports when a
supported polynomial pipeline with an affine downstream estimator has a known
finite Gauss–Legendre order for exact straight-path gradient integration.

## Scope

`skgrad` deliberately provides input derivatives, not an explanation method.
It does not choose baselines, perform numerical differentiation, integrate
gradients, calculate parameter gradients, or produce attribution plots. This
narrow scope keeps it useful as a small computational backend that other
packages can compose.

The project is licensed under the
[BSD 3-Clause License](https://github.com/LudgerHentschel/skgrad/blob/main/LICENSE).

## Related projects

| Package | When to use it |
|---|---|
| [UnifiedIG](https://ludgerhentschel.github.io/unifiedig/) (`unifiedig`) | Compute feature attributions through a common Integrated Gradients interface; uses skgrad for supported smooth scikit-learn models. |
| [TreeIG](https://ludgerhentschel.github.io/treeig/) (`treeig`) | Compute tree-path attributions for supported tree models, whose ordinary gradients are zero almost everywhere. |
| [CBaseline](https://ludgerhentschel.github.io/cbaseline/) (`cbaseline`) | Construct empirical reference distributions for IG, SHAP, and other compatible attribution engines. |

Use skgrad directly for local sensitivity, input gradients, or Jacobians. See
[the Integrated Gradients stack](https://ludgerhentschel.github.io/skgrad/ig-stack.html)
for how the components compose. Keep background outputs on the same score scale
as the derivatives used in classification attribution.

Release maintainers: see [Publishing releases](docs/publishing.md).
