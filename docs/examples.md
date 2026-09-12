---
myst:
  html_meta:
    description: "Run complete skgrad examples for input derivatives, preprocessing pipelines, and integration of polynomial gradients."
---

# Worked examples

Each example is a standalone script using only skgrad's runtime dependencies.
Install the checkout, then run `python scripts/check_examples.py` to execute all
examples and their numerical assertions. These checks verify derivative semantics,
not predictive quality of the small fitted demonstration models.

## Binary decision scores

The quick-start example shows a constant affine gradient. Binary classification
returns the score for the positive class, not a probability derivative.

```{literalinclude} ../examples/basic.py
:language: python
```

## Multiclass score gradients

A target is an output position in `classes_`. This multinomial logistic example
checks the score outputs and their softmax relationship to sklearn probabilities.
The gradients themselves remain on the score scale.

```{literalinclude} ../examples/multiclass.py
:language: python
```

## Multiple neural-network outputs

Use `target=1` for the second regression output. Selected MLP gradients avoid
forming every output's Jacobian; here the complete result is computed only to
verify equivalence. Expected shapes are `(5, 2)`, `(5, 2, 3)`, and `(5, 3)`.

```{literalinclude} ../examples/multioutput_mlp.py
:language: python
```

## Checking a kernel gradient

Central differences evaluate sklearn's own prediction function, providing an
independent numerical comparison. The check uses smooth RBF regression in
float64, with a fixed step and explicit tolerances.

```{literalinclude} ../examples/kernel_gradient.py
:language: python
```

## Scaling inputs explicitly

For `z_j = (x_j - mean_j) / scale_j`, the original-coordinate derivative is
`df/dx_j = (df/dz_j) / scale_j`. The pipeline now performs this chain rule automatically; the example checks
its result against manual scaling and independent finite differences.
It assumes StandardScaler's default `with_std=True` and continuous inputs.

```{literalinclude} ../examples/scaled_inputs.py
:language: python
```

## Integrating a polynomial gradient

A cubic polynomial has quadratic derivatives along a straight path. Two
Gauss–Legendre points integrate those derivatives exactly up to floating-point
error. Post-expansion scaling is already included by skgrad. Multiplication by
the input displacement gives feature contributions whose sum equals the fitted
prediction difference. The zero baseline here is illustrative, not a universal
choice of meaningful reference.

```{literalinclude} ../examples/polynomial_ig.py
:language: python
```

## Scaling and PCA before an MLP

This nested pipeline reduces four features to three PCA components. Its returned
gradient still has four columns in original input order. The script independently
checks those derivatives against perturbations of the complete sklearn pipeline.

```{literalinclude} ../examples/pipeline_mlp.py
:language: python
```

## Choosing original or standardized gradients

Both views accept original inputs, but return derivatives in the explicitly
selected coordinates. Their output values agree; their gradient units differ.

```{literalinclude} ../examples/feature_spaces.py
:language: python
```
