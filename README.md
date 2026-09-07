# skgrad

[![Tests](https://github.com/LudgerHentschel/skgrad/actions/workflows/tests.yml/badge.svg)](https://github.com/LudgerHentschel/skgrad/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://ludgerhentschel.github.io/skgrad/)
[![PyPI version](https://img.shields.io/pypi/v/skgrad.svg)](https://pypi.org/project/skgrad/)
[![Python versions](https://img.shields.io/pypi/pyversions/skgrad.svg)](https://pypi.org/project/skgrad/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/LudgerHentschel/skgrad/blob/main/LICENSE)

**Fast analytic input gradients for fitted scikit-learn models.**

`skgrad` differentiates a fitted model's prediction with respect to its input
features. It provides one NumPy-based interface for supported linear models,
classifiers, and multilayer perceptrons without finite differences, model
conversion, or an automatic-differentiation framework. Unsupported estimators raise `TypeError`; there is no numerical fallback.

```python
gradient = skgrad.input_gradient(model, X)
```

For scalar-output models, `gradient[i, j]` is the derivative of prediction `i`
with respect to feature `j`. Multi-output models expose one gradient per target
or the complete input Jacobian.

Read the **[skgrad documentation](https://ludgerhentschel.github.io/skgrad/)** for
worked examples, the API contract, model coverage, and numerical conventions.

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

See [the shape and output semantics](https://github.com/LudgerHentschel/skgrad/blob/main/docs/semantics.md)
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

## Unified IG

[Unified IG](https://github.com/LudgerHentschel/unifiedig) is an important
consumer of `skgrad`. It uses `skgrad` for analytic gradients of supported
smooth scikit-learn models, TreeIG for tree paths, and automatic or numerical
backends for other model families, presenting them through one Integrated
Gradients interface. Use Unified IG when the goal is feature attribution rather
than direct access to model input gradients.

[CBaseline](https://github.com/LudgerHentschel/cbaseline) constructs reference
baseline distributions; `skgrad` supplies analytic input derivatives; TreeIG
handles tree paths; UnifiedIG composes these components into attributions.

Release maintainers: see [Publishing releases](docs/publishing.md).
