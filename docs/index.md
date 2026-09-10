# skgrad documentation

**scikit-learn does not expose input derivatives. skgrad computes them in
closed form.**

For a supported fitted estimator `f` and input `x`, skgrad returns ∂f/∂x
analytically, to floating-point precision—without finite differences, automatic
differentiation, model conversion, or model approximation. The same NumPy
interface handles supported affine models, neural networks, kernel SVMs, and
continuous preprocessing pipelines.

![A fitted ReLU network and its analytic derivative](relu-analytic-gradients.svg)

A ReLU network is piecewise linear. Its fitted weights and active units give
its input derivative directly; no finite-difference step or framework conversion
is needed. At a kink, skgrad uses the zero derivative convention for ReLU.

Start with [installation and your first gradient](getting-started.md), then
explore [worked examples](examples.md). Check [model coverage](models.md)
before applying the interface to a different estimator or preprocessing chain.

```{toctree}
:maxdepth: 1
:caption: User guide

getting-started
examples
models
pipelines
semantics
numerical
performance
ig-stack
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
releases
building
publishing
```
