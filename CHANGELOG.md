# Changelog

## Unreleased

## 0.1.6

- Reject estimator subclasses overriding differentiated prediction methods across
  affine, MLP, and kernel SVM backends; retain inherited sklearn subclasses.
- Guard the private sklearn fitted SVM gamma dependency and add pre-release CI.
- Preserve normalized input precision in values, gradients, and Jacobians across
  all backends, including float32 inputs with float64 fitted parameters.
- Report exact polynomial-kernel SVM quadrature as max(1, ceil(degree / 2)).
- Keep ColumnTransformer composition on the roadmap for a later release.

## 0.1.5

### Pipeline gradients and feature spaces

- Compose analytic gradients through sequential and nested pipelines containing
  supported scalers, polynomial expansion, PCA/whitening, and fitted selectors.
  Gradients refer to the supplied pipeline's original input coordinates.
- Add `pipeline_view(..., after="step")` for explicit transformed-feature
  gradients, including nested boundaries and transformed feature names.
- Preserve selected-output MLP execution and report conservative constant-Jacobian
  and exact polynomial quadrature metadata. Degree-zero polynomials use one node.
- Define clipped MinMax derivatives as zero at and outside boundaries; reject
  degenerate PCA whitening explicitly.

### Models, numerical behavior, and performance

- Add LinearSVC, LinearSVR, and supported binary/regression kernel SVM backends.
- Avoid cancellation in RBF outputs for inputs with large common offsets.
- Support Poisson MLP exponential output links and preserve float32/mixed-input
  MLP behavior and float32 affine results.
- Optimize selected-output MLP reverse passes and large-batch row parallelism;
  recreate the persistent executor safely after process forks.
- Accept NumPy integer targets and improve sparse-input diagnostics.

### Documentation, licensing, and release checks

- Switch to BSD-3-Clause and include its SPDX metadata and license file.
- Add Sphinx/PyData documentation, worked examples, API reference, and Pages
  deployment. Document standalone estimators and original/transformed features.
- Add reproducible numerical-gradient and PyTorch benchmarks. Correct the README:
  unsupported models have no numerical fallback inside skgrad.
- Include documentation assets in source packages and fix README links for PyPI.
- Gate publishing on the full Python/OS test matrix, older supported dependencies,
  strict documentation, and installed-wheel tests/examples. Test extras include
  pandas so feature-name checks run rather than skip.

### Compatibility

- `GradientProperties` now contains `constant_jacobian` and
  `exact_quadrature_steps`. Update one-argument positional construction and
  single-item tuple unpacking from 0.1.1; prefer named-field access.
- Historical Git tags v0.1.2–v0.1.4 retained package metadata version 0.1.1.
  This release reconciles the changes under a matching 0.1.5 package/tag.

## 0.1.1

- Add constant-Jacobian metadata for downstream optimization while preserving
  one unified model-support predicate.

## 0.1.0

- Stabilize the public value-and-input-Jacobian API established in dev0.
- Validate the API against Integrated Gradients and analytic chain-rule
  composition in downstream consumers.
- Preserve raw score/logit semantics for every classification model.

## 0.1.0.dev0

- Establish the public value-and-input-Jacobian API.
- Add analytic affine gradients for selected sklearn regression and
  classification models.
- Add analytically composed sklearn MLP gradients for identity, logistic,
  tanh, and ReLU hidden activations.
- Define classification outputs as decision scores or logits, never
  probabilities.
