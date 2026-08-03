# Changelog

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
