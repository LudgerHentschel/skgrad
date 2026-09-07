# Output and shape semantics

skgrad differentiates fitted prediction functions with respect to input
features. It does not differentiate training losses or fitted parameters.

## Shapes and targets

Inputs are finite dense numeric arrays with shape `(samples, features)`.
A one-dimensional array becomes one sample. Empty inputs, NaN, infinity, and
sparse matrices are rejected. Integer inputs become float64; float16 inputs
become float32. Feature names do not change positional feature ordering.

Values always have shape `(samples, outputs)`, including scalar predictions.
Jacobians have shape `(samples, outputs, features)`. A selected gradient has
shape `(samples, features)`. An explicit target indexes the output axis, not a
class label. Python and NumPy integers are accepted; booleans, negative indexes,
and out-of-range indexes are rejected.

## Output scale

- Regressors expose predictions, including the exponential link for Poisson MLPs.
- Binary classifiers expose one decision score or pre-logistic logit for
  `classes_[1]`.
- Multiclass classifiers expose scores or pre-softmax logits in `classes_` order.
- MLP multilabel outputs are independent pre-logistic logits in indicator-column
  order; they are not a multiclass softmax. Multilabel behavior is not covered by
  the current examples and should be validated for downstream attribution.
- Probabilities are not differentiated. If you compose a probability transform,
  its Jacobian must also appear in the chain rule.

Affine models and MLPs preserve float32 when fitted parameters and inputs are
both float32, following NumPy dtype promotion for mixed inputs. LibSVM models
use float64 fitted parameters and outputs. Computations can still overflow for
extreme inputs or fitted parameters; finite input validation does not guarantee
finite model outputs.

## Preprocessing coordinates

Gradients always refer to the input features of the supplied model object.
Supply the full supported pipeline and original inputs for original-feature
derivatives. Supply only the final estimator and transformed inputs to obtain
transformed-feature derivatives. Data and IG baselines must use the same input
space. Preprocessing done outside the supplied object cannot be inferred.

Supported pipeline transformations are differentiated in their fitted state;
feature selection is fixed and dropped input coordinates receive zero gradients.
See [pipeline conventions](pipelines.md) for clipping and whitening restrictions.

## Metadata and errors

`GradientResult` is a named tuple with `values` and `jacobian`.
`GradientProperties` has `constant_jacobian` and `exact_quadrature_steps`.
Use named attributes for forward-compatible access. A reported constant Jacobian
is an optimization guarantee; false can conservatively include constant special
cases. A finite quadrature order applies to straight-path integration of the
model gradient, not an arbitrary nonlinear loss composed afterward.

Unsupported estimators raise `TypeError`. Unfitted supported models fail fitted
state checks. Malformed inputs, mismatched feature counts, and invalid targets
raise validation errors. `supports()` does not silently invoke another backend.
