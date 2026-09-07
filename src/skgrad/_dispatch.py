"""Conservative prediction-method checks shared by estimator backends."""

import inspect


def inherits_prediction(model, supported_types):
    """Allow inherited methods, rejecting overrides anywhere in the MRO.

    Static lookup also detects descriptors and instance-level monkey patches
    without executing user code. Classifier scores use decision_function;
    MLP logits have no public scorer, so protect predict and predict_proba.
    """
    for base in supported_types:
        if isinstance(model, base):
            methods = ("decision_function",) if hasattr(base, "decision_function") else ("predict",)
            if hasattr(base, "predict_proba") and not hasattr(base, "decision_function"):
                methods += ("predict_proba",)
            return all(
                inspect.getattr_static(model, name) is inspect.getattr_static(base, name)
                for name in methods
            )
    return False
