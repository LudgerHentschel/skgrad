"""Analytic input gradients for fitted scikit-learn models."""

from importlib.metadata import PackageNotFoundError, version

from ._view import PipelineView, pipeline_view

from .api import (
    GradientProperties,
    GradientResult,
    gradient_properties,
    input_gradient,
    input_jacobian,
    model_output,
    supports,
    value_and_jacobian,
)

__all__ = [
    "PipelineView",
    "pipeline_view",
    "GradientProperties",
    "GradientResult",
    "gradient_properties",
    "input_gradient",
    "input_jacobian",
    "model_output",
    "supports",
    "value_and_jacobian",
]

try:
    __version__ = version("skgrad")
except PackageNotFoundError:
    __version__ = "0+unknown"
