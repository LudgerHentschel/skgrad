# The Integrated Gradients stack

| Package | Responsibility |
|---|---|
| [CBaseline](https://github.com/LudgerHentschel/cbaseline) | Construct empirical reference baseline distributions |
| skgrad | Evaluate analytic input derivatives of supported sklearn functions |
| [TreeIG](https://github.com/LudgerHentschel/treeig) | Account for prediction jumps along supported tree paths |
| [UnifiedIG](https://github.com/LudgerHentschel/unifiedig) | Choose backends and compute feature attributions |

An input gradient describes local sensitivity. Integrated Gradients also needs
a baseline and integration along a path. skgrad supplies the derivative needed
inside that integration; its metadata lets consumers recognize constant
Jacobians or known polynomial quadrature orders.

The [polynomial integration example](examples.md#integrating-a-polynomial-gradient)
shows this composition explicitly and checks that feature contributions sum to
the prediction difference. It is an educational example for a single baseline.
Use UnifiedIG for the complete attribution interface and baseline distributions.

Keep output scales aligned across packages. Differentiating decision scores,
logits, and probabilities describes different contrasts. In particular, do not
combine a score gradient with a probability-valued baseline prediction.
