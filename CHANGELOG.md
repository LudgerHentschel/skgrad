# Changelog

## Unreleased

- Speed up selected-output MLP input gradients with a two-dimensional reverse
  pass while retaining the complete-Jacobian APIs.
- Parallelize selected-output MLP gradients across large sample batches with a
  bounded, hardware-aware worker heuristic.
- Preserve scikit-learn MLP float32 and mixed-input dtype behavior in values
  and input gradients.
- Support the exponential output link and input gradients of Poisson-loss
  `MLPRegressor` models.
- Recreate the persistent MLP gradient executor after process forks and add
  macOS and Windows CI coverage.

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
