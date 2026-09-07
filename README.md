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
- Affine models and MLPs follow NumPy/scikit-learn dtype promotion, preserving
  float32 when the input and fitted parameters are both float32. Scikit-learn's
  LibSVM estimators use float64 fitted parameters and outputs.

See [the shape and output semantics](https://github.com/LudgerHentschel/skgrad/blob/main/docs/semantics.md)
for the complete contract.

## Performance

For affine estimators, the input Jacobian is simply the fitted coefficient
matrix and is effectively free to reuse. MLP gradients require a forward pass
and reverse pass. The benchmarks below answer three separate questions:
whether the analytic gradients agree with numerical differentiation, how much
work generic numerical differentiation requires, and how the MLP implementation
compares with a highly optimized automatic-differentiation system.

### Numerical agreement

`skgrad` agrees closely with generic two-sided central differences. Each
numerical derivative below used
`(f(x + h e_j) - f(x - h e_j)) / (2h)` with `h = 1e-5`, applied to a batch of
100 rows and 100 features. Relative error is the maximum absolute error divided
by the largest absolute numerical-gradient entry:

| Model | Differentiated output | Maximum absolute error | Relative error |
|---|---|---:|---:|
| LogisticRegression | Decision score | 1.260e-10 | 7.511e-11 |
| MLPRegressor `(64, 64)`, tanh | Prediction | 1.216e-10 | 7.488e-11 |
| MLPClassifier `(32,)`, tanh | First class logit | 4.354e-10 | 1.090e-10 |

### Speed versus numerical differentiation

Central differences require two model evaluations for every input feature.
The following benchmark holds the evaluation batch at 100 rows while varying
the number of features. It compares a direct analytic gradient
(`LogisticRegression`) with a gradient composed by a forward and reverse pass
(`MLPRegressor`). Timings exclude fitting and are warm-run medians of seven
repetitions after two warmups:

| Model | Features | skgrad | Central differences | Speedup |
|---|---:|---:|---:|---:|
| LogisticRegression | 10 | 0.009 ms | 0.117 ms | 13.8× |
| LogisticRegression | 100 | 0.011 ms | 1.769 ms | 167.8× |
| LogisticRegression | 1,000 | 0.078 ms | 74.022 ms | 954.6× |
| MLPRegressor `(64, 64)`, tanh | 10 | 0.085 ms | 1.167 ms | 13.7× |
| MLPRegressor `(64, 64)`, tanh | 100 | 0.213 ms | 23.023 ms | 108.1× |
| MLPRegressor `(64, 64)`, tanh | 1,000 | 0.480 ms | 634.666 ms | 1,322.6× |

The numerical method perturbs one feature at a time but evaluates all 100 rows
in one model call, so it retains the estimator's batch efficiency. Its linear
growth in model evaluations with feature count is inherent to generic central
differences. The complete benchmark, including deterministic model generation
and environment reporting, is in
[`benchmarks/numerical_gradients.py`](https://github.com/LudgerHentschel/skgrad/blob/main/benchmarks/numerical_gradients.py). The speedup generally grows with feature count; its magnitude depends on the model and runtime environment.

### Speed versus PyTorch autodiff

PyTorch CPU autodiff provides a more demanding speed comparison for MLPs
because, like `skgrad`, it obtains all feature derivatives in one reverse pass.

The following controlled benchmark used the same 20-input, two-hidden-layer
`(64, 64)` tanh network, weights, biases, float64 inputs, and scalar output in
scikit-learn/skgrad and PyTorch. Predictions and gradients agreed to floating-
point precision. Timings are warm-run medians for gradient calculation only;
fitting, model conversion, and weight copying were excluded. Each median uses
25 repetitions after five warmups. PyTorch used four intra-operation CPU
threads. `skgrad` used its automatic row-parallel policy:

| Sample rows | skgrad workers | skgrad | PyTorch autodiff | Relative result |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.012 ms | 0.039 ms | skgrad 3.31× faster |
| 10 | 1 | 0.024 ms | 0.052 ms | skgrad 2.19× faster |
| 100 | 1 | 0.137 ms | 0.175 ms | skgrad 1.28× faster |
| 1,000 | 1 | 0.759 ms | 0.680 ms | PyTorch 1.12× faster |
| 5,000 | 2 | 3.398 ms | 2.679 ms | PyTorch 1.27× faster |
| 10,000 | 2 | 6.498 ms | 5.425 ms | PyTorch 1.20× faster |
| 100,000 | 2 | 54.462 ms | 51.697 ms | PyTorch 1.05× faster |

In this like-for-like comparison, `skgrad` has effectively the same speed as
PyTorch. The complete benchmark and deterministic model construction are in
[`benchmarks/pytorch_autodiff.py`](https://github.com/LudgerHentschel/skgrad/blob/main/benchmarks/pytorch_autodiff.py).

Both sets of benchmarks ran on an Apple-silicon macOS laptop with Python 3.13,
NumPy 2.4.6, and scikit-learn 1.9.0. The autodiff benchmark additionally used
PyTorch 2.12.0. Results will vary with network
shape, activation, dtype, CPU, BLAS implementation, and thread configuration;
the table is a transparent reference point rather than a universal performance
guarantee. It does not compare GPU execution.

For selected MLP outputs, batches below 5,000 rows use the low-overhead serial
path. Larger batches are split across a persistent, hardware-aware pool of up
to four workers. The pool is created lazily, so its first use includes a
one-time startup cost. Complete-Jacobian calls retain the general vectorized
path.

## How gradients are computed

`skgrad` uses the fitted estimator's known algebra rather than approximating
derivatives:

- **Affine models:** the fitted coefficient matrix is the constant input
  Jacobian.
- **MLPs:** a NumPy forward pass retains hidden activations, followed by the
  ordinary reverse chain rule through fitted weight matrices and activation
  derivatives.
- **Selected outputs:** `input_gradient` propagates only the requested scalar
  output as a two-dimensional batch, avoiding a full three-dimensional
  Jacobian.
- **Poisson MLPs:** the reverse pass includes the derivative of the exponential
  output link, `exp(z)`.
- **Large batches:** independent sample rows are divided among bounded workers,
  while dense matrix operations remain in optimized native numerical kernels.
  BLAS thread-capacity discovery is cached per process so repeated gradient
  calls, such as quadrature over many paths, do not rescan loaded libraries.

This is algorithmically the same backpropagation used by autodiff for an MLP,
but specialized to scikit-learn's fitted representation and requested output.
There is no computation graph, parameter-gradient bookkeeping, or framework
conversion in the gradient call.

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
