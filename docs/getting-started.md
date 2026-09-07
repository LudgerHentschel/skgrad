# Getting started

Install the package in your Python environment:

```console
pip install skgrad
```

The runtime requires Python 3.9 or later, NumPy, SciPy, scikit-learn, and
threadpoolctl. Poisson-loss MLP regression requires scikit-learn 1.7 or later.
The documentation toolchain uses Python 3.12 in CI.

## Your first input gradient

This binary classifier differentiates its decision score, not its probability.
A positive derivative means increasing that input increases the score for
`model.classes_[1]`. Each row corresponds to an observation, each column to an
input feature.

```{literalinclude} ../examples/basic.py
:language: python
```

Run this file from an installed checkout with `python examples/basic.py`.
The gradient of this affine score is the same at every input, although the
score itself changes.

## Choose the right result

| Function | Result |
|---|---|
| `model_output(model, X)` | Values with shape `(samples, outputs)` |
| `input_gradient(model, X, target=...)` | One output's gradient, `(samples, features)` |
| `input_jacobian(model, X)` | All output derivatives, `(samples, outputs, features)` |
| `value_and_jacobian(model, X)` | Named pair with `.values` and `.jacobian` |

Omit `target` only for a single output. For multiclass and multioutput models,
select an integer output position or request the full Jacobian.

`supports(model)` checks whether an analytic backend exists. It does not certify
that the model is fitted. Fit the estimator first, then pass finite dense inputs
in the same feature order and units used at fitting time.
