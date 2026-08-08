# skgrad

[![Tests](https://github.com/LudgerHentschel/skgrad/actions/workflows/tests.yml/badge.svg)](https://github.com/LudgerHentschel/skgrad/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/skgrad.svg)](https://pypi.org/project/skgrad/)
[![Python versions](https://img.shields.io/pypi/pyversions/skgrad.svg)](https://pypi.org/project/skgrad/)
[![License](https://img.shields.io/pypi/l/skgrad.svg)](https://github.com/LudgerHentschel/skgrad/blob/main/LICENSE)

**Fast analytic input gradients for fitted scikit-learn models.**

`skgrad` differentiates a fitted model's prediction with respect to its input
features. It provides one NumPy-based interface for supported linear models,
classifiers, and multilayer perceptrons without finite differences, model
conversion, or an automatic-differentiation framework.

```python
gradient = skgrad.input_gradient(model, X)
```

For scalar-output models, `gradient[i, j]` is the derivative of prediction `i`
with respect to feature `j`. Multi-output models expose one gradient per target
or the complete input Jacobian.

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

## Supported models

| Family | Models | Differentiated output |
|---|---|---|
| Linear regression | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet` | Prediction |
| Linear classification | `LogisticRegression`, `RidgeClassifier` | Decision score |
| Neural-network regression | `MLPRegressor` with squared-error or Poisson loss | Prediction, including the Poisson exponential output link |
| Neural-network classification | `MLPClassifier` | Binary or multiclass logits before logistic/softmax |

MLP hidden activations may be identity, logistic, tanh, or ReLU. At ReLU's
nondifferentiable origin, `skgrad` uses a zero derivative, matching
scikit-learn's backpropagation convention. Scalar and multi-output regression,
binary classification, and multiclass classification are supported.

Tree models are intentionally excluded. Their predictions are piecewise
constant, so ordinary gradients are zero almost everywhere and undefined at
split boundaries. Use [TreeIG](https://github.com/LudgerHentschel/treeig),
which computes exact Integrated Gradients from the prediction jumps at tree
split crossings.

`skgrad` currently expects finite dense numeric inputs. It differentiates the
supported fitted estimator itself; preprocessing pipelines and unsupported
estimators are not differentiated automatically.

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
- Float32 MLP inputs and weights remain float32. Mixed float32/float64
  calculations follow scikit-learn's promotion behavior.

See [the shape and output semantics](https://github.com/LudgerHentschel/skgrad/blob/main/docs/semantics.md)
for the complete contract.

## Performance

For affine estimators, the input Jacobian is simply the fitted coefficient
matrix and is effectively free to reuse. MLP gradients require a forward pass
and reverse pass, making PyTorch CPU autodiff a useful demanding comparison.

The following controlled benchmark used the same 20-input, two-hidden-layer
`(64, 64)` tanh network, weights, biases, float64 inputs, and scalar output in
scikit-learn/skgrad and PyTorch. Predictions and gradients agreed to floating-
point precision. Timings are warm-run medians for gradient calculation only;
fitting, model conversion, and weight copying were excluded. PyTorch used four
intra-operation CPU threads. `skgrad` used its automatic row-parallel policy:

| Sample rows | skgrad workers | skgrad | PyTorch autodiff | Relative result |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.011 ms | 0.033 ms | skgrad 3.00× faster |
| 10 | 1 | 0.024 ms | 0.048 ms | skgrad 1.99× faster |
| 100 | 1 | 0.122 ms | 0.163 ms | skgrad 1.33× faster |
| 1,000 | 1 | 1.166 ms | 0.741 ms | PyTorch 1.57× faster |
| 5,000 | 2 | 4.378 ms | 3.301 ms | PyTorch 1.33× faster |
| 10,000 | 4 | 5.562 ms | 6.909 ms | skgrad 1.24× faster |
| 100,000 | 4 | 51.016 ms | 60.631 ms | skgrad 1.19× faster |

The benchmark ran on an Apple-silicon macOS laptop with Python 3.13, NumPy
2.4.6, scikit-learn 1.9.0, and PyTorch 2.12.0. Results will vary with network
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
avoid redundant evaluations.

## Scope

`skgrad` deliberately provides input derivatives, not an explanation method.
It does not choose baselines, perform numerical differentiation, integrate
gradients, calculate parameter gradients, or produce attribution plots. This
narrow scope keeps it useful as a small computational backend that other
packages can compose.

The project is licensed under the
[MIT License](https://github.com/LudgerHentschel/skgrad/blob/main/LICENSE).

## Unified IG

[Unified IG](https://github.com/LudgerHentschel/unifiedig) is an important
consumer of `skgrad`. It uses `skgrad` for analytic gradients of supported
smooth scikit-learn models, TreeIG for tree paths, and automatic or numerical
backends for other model families, presenting them through one Integrated
Gradients interface. Use Unified IG when the goal is feature attribution rather
than direct access to model input gradients.
