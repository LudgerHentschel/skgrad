---
myst:
  html_meta:
    description: "Understand skgrad numerical conventions and why unsupported estimators do not silently use finite differences."
---

# Numerical conventions and fallback policy

## Analytic derivatives

Affine estimators reuse fitted coefficients. MLPs propagate the chain rule
through fitted weights and hidden activations. SVMs differentiate built-in
kernels. Polynomial pipelines compose derivatives through expanded monomials,
including at zero-valued coordinates.

ReLU has no unique ordinary derivative at zero. skgrad chooses zero there.
Floating-point evaluation remains approximate even when the derivative formula
is analytic. For a smooth model, central differences are useful for checking
that formula, but errors depend on step size, input scale, and precision.

The [kernel example](examples.md#checking-a-kernel-gradient) checks an analytic
result against central differences of sklearn's own predictions. Test several
step sizes when diagnosing disagreement; very small steps amplify subtraction
error, while large steps approximate a wider neighborhood.

## Should skgrad have a numerical fallback?

The current API deliberately provides analytic derivatives only. Coverage is
strong for its listed families, but [important sklearn gaps remain](models.md).
A hidden fallback would make cost and accuracy depend on an implicit dispatch
choice and could approximate a different classification output scale.

[UnifiedIG](https://github.com/LudgerHentschel/unifiedig) already implements a
finite-difference backend for eligible unsupported sklearn models. Keep that
choice visible at the attribution layer, with explicit output semantics and
numerical settings. Standalone numerical checks can use a clearly chosen
prediction callable, as in the worked example.

Numerical gradients are useful for smooth unsupported functions. They do not
repair discontinuities: a tree's ordinary gradient misses its prediction jumps,
and a perturbation of a categorical feature may not describe a valid input.
Use [TreeIG](https://github.com/LudgerHentschel/treeig) for supported tree paths.

## Priorities for broader analytic coverage

Continuous preprocessing composition is now supported for the transformations
listed in [pipeline gradients](pipelines.md). ColumnTransformer composition,
additional affine estimator families, and generalized linear output links are
natural next steps. Kernel ridge and supported Gaussian process kernels are
further candidates. These remain future directions for 0.1.6.
