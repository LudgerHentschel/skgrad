---
myst:
  html_meta:
    description: "Compute skgrad input derivatives through supported fitted preprocessing pipelines while preserving feature coordinates."
---

# Pipeline gradients

Pass the complete fitted pipeline and its input observations. skgrad evaluates
its preprocessing and applies the chain rule back to those input coordinates.
Nested sklearn Pipelines and intermediate `None`/`passthrough` steps are accepted.
The final step must be a supported predictor.

## Supported transformations

| Transformation | Derivative and conventions |
|---|---|
| StandardScaler | Divide by fitted scale when `with_std=True`; centering contributes no derivative |
| RobustScaler | Divide by fitted scale when `with_scaling=True`, including `unit_variance` settings |
| MaxAbsScaler | Divide by fitted scale |
| MinMaxScaler | Multiply by fitted scale; optionally apply the clipping mask |
| PolynomialFeatures | Differentiate fitted monomials, including at zero inputs; multiple expansion stages are supported |
| PCA | Multiply by fitted components; account for whitening when enabled |
| Fitted feature selectors | Return selected-coordinate derivatives to their input positions; dropped positions receive zero |

Supported selector classes are `SelectKBest`, `SelectPercentile`, `SelectFpr`,
`SelectFdr`, `SelectFwe`, `GenericUnivariateSelect`, `VarianceThreshold`,
`SelectFromModel`, `RFE`, `RFECV`, and `SequentialFeatureSelector`. Selection is
held fixed after fitting; gradients do not differentiate the fitting procedure.

Recognition uses exact built-in transformer classes. Custom subclasses,
ColumnTransformer, arbitrary FunctionTransformer functions, one-hot encoders,
imputers, and other transformations reject the whole analytic route. Calling
`supports()` checks structure; fitted-state and input checks happen at evaluation.

## Original versus processed features

For a pipeline `F(x) = f(T(x))`, skgrad returns the derivative of `F` with respect
to `x`. PCA may change the number of internal features, but the output gradient
still has one column per input feature. The original input order is retained.
Data must be finite dense numeric inputs, including when selecting columns.

For IG, construct the straight path between baseline and observation in the
pipeline input space, and call skgrad on those path points. Preprocessing is
evaluated at each point. A straight path between transformed endpoints generally
describes a different path when preprocessing is nonlinear. The supplied model
object defines the attribution space; transformations done outside it are not
recoverable automatically.

For featurewise affine scaling, scale factors cancel between gradients and
input displacements when the baseline is transformed consistently. Mixing
features with PCA or expanding polynomial terms requires the full chain rule;
renaming processed-feature attributions cannot generally recover original-feature
attributions. See [worked examples](examples.md).

## Boundaries and whitening

With `MinMaxScaler(clip=True)`, derivatives are zero outside the fitted output
range and exactly at its boundaries. The boundary convention selects zero where
the ordinary derivative is not unique. Such pipelines do not report a global
constant Jacobian or a finite exact polynomial quadrature order: integration may
cross clipping boundaries.

Whitened PCA requires non-degenerate retained explained variances. skgrad raises
a clear error when their square roots are at or below machine epsilon, avoiding
version-dependent handling of near-zero whitening scales. Ordinary unwhitened
PCA has no such restriction.

## Metadata and performance

Affine preprocessing followed by an affine predictor reports a constant Jacobian
and a one-point exact quadrature order. Polynomial degrees multiply through
successive expansions; a total degree `d` permits `max(1, ceil(d/2))` Gauss–Legendre
points for straight-path gradient integration with an affine predictor. These
are conservative guarantees; special fitted coefficients can lower the degree.
Clipping and nonlinear downstream predictors disable these polynomial guarantees.

Scaler pullbacks use elementwise multiplication, PCA uses matrix multiplication,
and selectors scatter gradients to their original positions. Selected-output MLP
calls retain the two-dimensional reverse path. Polynomial derivatives still need
intermediate arrays that can grow with expansion size; batch rows for large
problems. Transformations configured with `copy=False` do not mutate caller data
inside skgrad.

## Choosing a feature space explicitly

For standalone gradients, use a fitted pipeline view:

```python
view = skgrad.pipeline_view(pipeline, after="scale")
gradient = view.input_gradient(X)  # X is still in ORIGINAL pipeline coordinates
names = view.get_feature_names_out()
```

The returned derivative is with respect to features after `scale`, evaluated at
those transformed observations. `view.model_output`, `view.input_jacobian`, and
`view.value_and_jacobian` also accept original inputs. Use `after=None` for
original-feature derivatives. Nested names such as `"preprocess__scale"` select
an intermediate boundary; a parent name selects that whole sub-pipeline's output.

For explicit composition, `view.model` accepts `view.transform(X)`. Do not pass
already transformed data to the view's convenience methods. Evaluate backend
eligibility with `skgrad.supports(view.model)`. Prefix transformations must be
supported continuous transforms; the remaining predictor is checked when using
its gradient API. Fitted steps are shared without refitting. Do not refit the
source pipeline while a view is in use.

In UnifiedIG, `Explainer(pipeline, baseline, attribute_after="scale")` performs
the equivalent selection and transforms original observations and baseline rows
together. The result records the selected feature space. IG attributions are
unchanged by featurewise affine scaling with consistently transformed baselines,
although gradient units and displayed feature values change.
