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

Build the guide with `python -m pip install -e ".[docs]"` and
`python -m sphinx -W --keep-going -b html docs docs/_build/html`.
Run `python scripts/check_examples.py` when changing examples.
