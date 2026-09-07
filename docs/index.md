# skgrad documentation

**Analytic input gradients for fitted scikit-learn models.**

Use skgrad to ask how a prediction or classification score changes when an
input feature changes. The same NumPy interface handles supported affine
models, neural networks, kernel SVMs, and continuous preprocessing pipelines.

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
