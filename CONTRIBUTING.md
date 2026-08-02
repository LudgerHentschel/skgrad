# Contributing

Changes should preserve skgrad's narrow scope: analytic input derivatives of
fitted scikit-learn prediction functions. New backends must define their output
scale explicitly and test analytic Jacobians against independent calculations.

```console
python -m pip install -e ".[test]"
pytest
python -m build
python -m twine check dist/*
```
