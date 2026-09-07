# API reference

All functions accept a fitted supported estimator. See [semantics](semantics.md)
for output axes, target selection, errors, and classification scales.

```{eval-rst}
.. autofunction:: skgrad.supports

.. autofunction:: skgrad.gradient_properties

.. autofunction:: skgrad.model_output

.. autofunction:: skgrad.input_gradient

.. autofunction:: skgrad.input_jacobian

.. autofunction:: skgrad.value_and_jacobian

.. autoclass:: skgrad.GradientResult

.. autoclass:: skgrad.GradientProperties
```

## Pipeline feature-space views

```{eval-rst}
.. autofunction:: skgrad.pipeline_view

.. autoclass:: skgrad.PipelineView
   :members: transform, get_feature_names_out, model_output, input_gradient, input_jacobian, value_and_jacobian
```
