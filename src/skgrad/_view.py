"""Explicit feature-space selection for fitted sklearn pipelines."""

from typing import Optional, Sequence

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .api import GradientResult
from ._inputs import normalize_data
from ._pipeline import _identity, _transform, _transformer_supports


class PipelineView:
    """A fitted predictor viewed after a named preprocessing step.

    Create with :func:`skgrad.pipeline_view`. ``model`` accepts transformed
    inputs. Convenience gradient methods accept original pipeline inputs and
    differentiate in the selected space. Fitted steps are shared, not refitted
    or copied; do not refit the source pipeline while using this view.
    """

    def __init__(
        self, source_model: Pipeline, model: object,
        prefix: Sequence[object], after: Optional[str],
    ) -> None:
        self.source_model = source_model
        self.model = model
        self.after = after
        self._prefix = tuple(prefix)

    def transform(self, X: object) -> np.ndarray:
        """Map original pipeline inputs to the selected feature space."""
        columns = getattr(X, "columns", None)
        expected = getattr(self.source_model, "feature_names_in_", None)
        if columns is not None and expected is not None:
            if list(columns) != list(expected):
                raise ValueError("input columns must match the fitted pipeline order")
        data = normalize_data(X)
        if data.shape[1] != self.source_model.n_features_in_:
            raise ValueError(
                f"expected {self.source_model.n_features_in_} original input features; "
                "pass original inputs, not already transformed data"
            )
        # Preserve DataFrame labels through sklearn transforms where available.
        value = X if columns is not None else data
        for step in self._prefix:
            value = _transform(step, value)
        return normalize_data(value).copy()

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        """Names of coordinates in the selected feature space."""
        expected = getattr(self.source_model, "feature_names_in_", None)
        if input_features is None:
            input_features = expected
        elif expected is not None and list(input_features) != list(expected):
            raise ValueError("input feature names must match the fitted pipeline order")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.source_model.n_features_in_)]
        names = np.asarray(input_features, dtype=object)
        if names.ndim != 1 or len(names) != self.source_model.n_features_in_:
            raise ValueError("input feature names must align with original features")
        for step in self._prefix:
            names = step.get_feature_names_out(names)
        return np.asarray(names, dtype=object)

    def model_output(self, X: object) -> np.ndarray:
        """Evaluate model outputs from original pipeline inputs."""
        from .api import model_output
        return model_output(self.model, self.transform(X))

    def input_gradient(self, X: object, target: Optional[int] = None) -> np.ndarray:
        """Differentiate in selected coordinates, accepting original inputs."""
        from .api import input_gradient
        return input_gradient(self.model, self.transform(X), target=target)

    def input_jacobian(self, X: object) -> np.ndarray:
        """Return all output derivatives in selected coordinates."""
        from .api import input_jacobian
        return input_jacobian(self.model, self.transform(X))

    def value_and_jacobian(self, X: object) -> GradientResult:
        """Return outputs and selected-coordinate Jacobians."""
        from .api import value_and_jacobian
        return value_and_jacobian(self.model, self.transform(X))


def pipeline_view(model: object, *, after: Optional[str] = None) -> PipelineView:
    """Select the inputs or an intermediate feature space of a fitted Pipeline.

    ``after=None`` keeps original pipeline inputs. Otherwise give a preprocessing
    step name, or a nested path such as ``'preprocess__scale'``. The view transforms
    original observations with the fitted prefix and exposes the remaining
    predictor as ``view.model``. Prefix steps must be supported continuous
    transformations. Suffix backend eligibility is checked by the caller.
    """
    if after is not None and (not isinstance(after, str) or not after):
        raise TypeError("after must be None or a non-empty pipeline step name")
    if type(model) is not Pipeline or not model.steps:
        raise TypeError("pipeline_view requires a non-empty sklearn Pipeline")
    leaves = []
    boundaries = {}

    def visit(pipeline, path=""):
        if not pipeline.steps:
            raise ValueError("empty nested pipelines are not supported")
        for name, step in pipeline.steps:
            key = f"{path}__{name}" if path else name
            if key in boundaries:
                raise ValueError("pipeline step names must be unique")
            if type(step) is Pipeline:
                visit(step, key)
            elif not _identity(step):
                leaves.append(step)
            boundaries[key] = len(leaves)

    visit(model)
    tail = model
    while type(tail) is Pipeline:
        tail = tail.steps[-1][1]
    if _identity(tail) or not callable(getattr(tail, "predict", None)):
        raise TypeError("pipeline must end in a fitted predictor")
    check_is_fitted(tail)
    if not hasattr(model, "n_features_in_"):
        raise ValueError("pipeline must expose its fitted original input dimension")
    if after is None:
        return PipelineView(model, model, (), None)
    if after not in boundaries:
        choices = [name for name, end in boundaries.items() if end < len(leaves)]
        raise ValueError(f"unknown preprocessing step {after!r}; choose from {choices}")
    end = boundaries[after]
    if end >= len(leaves):
        raise ValueError("after must name a preprocessing step, not the final predictor")
    prefix, suffix = leaves[:end], leaves[end:]
    for step in prefix:
        if not _transformer_supports(step):
            raise TypeError(f"unsupported preprocessing step: {type(step).__name__}")
        check_is_fitted(step)
    predictor = suffix[0] if len(suffix) == 1 else Pipeline(
        [(f"step{i}", step) for i, step in enumerate(suffix)]
    )
    return PipelineView(model, predictor, prefix, after)
