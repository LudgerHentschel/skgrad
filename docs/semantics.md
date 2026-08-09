# Output and shape semantics

`skgrad` differentiates model outputs with respect to input features. It does
not compute gradients with respect to fitted parameters.

All public functions add a leading sample dimension to a single input and use:

```text
values:   (samples, outputs)
jacobian: (samples, outputs, features)
```

Regressors return their fitted prediction output. Binary classifiers expose
one score for the positive class. Multiclass classifiers expose one score per
class in `model.classes_` order. Logistic regression returns its decision
function; MLP classifiers return logits before the logistic or softmax output
activation. Binary support-vector classifiers return their decision score.
Probabilities are deliberately not part of V1.

MLP Jacobians are composed analytically through fitted weight matrices and
activation derivatives. ReLU's derivative is defined as zero at the origin.
