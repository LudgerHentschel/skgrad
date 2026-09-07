# Performance


For affine estimators, the input Jacobian is simply the fitted coefficient
matrix and is effectively free to reuse. MLP gradients require a forward pass
and reverse pass. The benchmarks below answer three separate questions:
whether the analytic gradients agree with numerical differentiation, how much
work generic numerical differentiation requires, and how the MLP implementation
compares with a highly optimized automatic-differentiation system.

## Numerical agreement

`skgrad` agrees closely with generic two-sided central differences. Each
numerical derivative below used
`(f(x + h e_j) - f(x - h e_j)) / (2h)` with `h = 1e-5`, applied to a batch of
100 rows and 100 features. Relative error is the maximum absolute error divided
by the largest absolute numerical-gradient entry:

| Model | Differentiated output | Maximum absolute error | Relative error |
|---|---|---:|---:|
| LogisticRegression | Decision score | 1.260e-10 | 7.511e-11 |
| MLPRegressor `(64, 64)`, tanh | Prediction | 1.216e-10 | 7.488e-11 |
| MLPClassifier `(32,)`, tanh | First class logit | 4.354e-10 | 1.090e-10 |

## Speed versus numerical differentiation

Central differences require two model evaluations for every input feature.
The following benchmark holds the evaluation batch at 100 rows while varying
the number of features. It compares a direct analytic gradient
(`LogisticRegression`) with a gradient composed by a forward and reverse pass
(`MLPRegressor`). Timings exclude fitting and are warm-run medians of seven
repetitions after two warmups:

| Model | Features | skgrad | Central differences | Speedup |
|---|---:|---:|---:|---:|
| LogisticRegression | 10 | 0.009 ms | 0.117 ms | 13.8× |
| LogisticRegression | 100 | 0.011 ms | 1.769 ms | 167.8× |
| LogisticRegression | 1,000 | 0.078 ms | 74.022 ms | 954.6× |
| MLPRegressor `(64, 64)`, tanh | 10 | 0.085 ms | 1.167 ms | 13.7× |
| MLPRegressor `(64, 64)`, tanh | 100 | 0.213 ms | 23.023 ms | 108.1× |
| MLPRegressor `(64, 64)`, tanh | 1,000 | 0.480 ms | 634.666 ms | 1,322.6× |

The numerical method perturbs one feature at a time but evaluates all 100 rows
in one model call, so it retains the estimator's batch efficiency. Its linear
growth in model evaluations with feature count is inherent to generic central
differences. The complete benchmark, including deterministic model generation
and environment reporting, is in
[`benchmarks/numerical_gradients.py`](https://github.com/LudgerHentschel/skgrad/blob/main/benchmarks/numerical_gradients.py). The speedup generally grows with feature count; its magnitude depends on the model and runtime environment.

## Speed versus PyTorch autodiff

PyTorch CPU autodiff provides a more demanding speed comparison for MLPs
because, like `skgrad`, it obtains all feature derivatives in one reverse pass.

The following controlled benchmark used the same 20-input, two-hidden-layer
`(64, 64)` tanh network, weights, biases, float64 inputs, and scalar output in
scikit-learn/skgrad and PyTorch. Predictions and gradients agreed to floating-
point precision. Timings are warm-run medians for gradient calculation only;
fitting, model conversion, and weight copying were excluded. Each median uses
25 repetitions after five warmups. PyTorch used four intra-operation CPU
threads. `skgrad` used its automatic row-parallel policy:

| Sample rows | skgrad workers | skgrad | PyTorch autodiff | Relative result |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.012 ms | 0.039 ms | skgrad 3.31× faster |
| 10 | 1 | 0.024 ms | 0.052 ms | skgrad 2.19× faster |
| 100 | 1 | 0.137 ms | 0.175 ms | skgrad 1.28× faster |
| 1,000 | 1 | 0.759 ms | 0.680 ms | PyTorch 1.12× faster |
| 5,000 | 2 | 3.398 ms | 2.679 ms | PyTorch 1.27× faster |
| 10,000 | 2 | 6.498 ms | 5.425 ms | PyTorch 1.20× faster |
| 100,000 | 2 | 54.462 ms | 51.697 ms | PyTorch 1.05× faster |

In this like-for-like comparison, `skgrad` has effectively the same speed as
PyTorch. The complete benchmark and deterministic model construction are in
[`benchmarks/pytorch_autodiff.py`](https://github.com/LudgerHentschel/skgrad/blob/main/benchmarks/pytorch_autodiff.py).

Both sets of benchmarks ran on an Apple-silicon macOS laptop with Python 3.13,
NumPy 2.4.6, and scikit-learn 1.9.0. The autodiff benchmark additionally used
PyTorch 2.12.0. Results will vary with network
shape, activation, dtype, CPU, BLAS implementation, and thread configuration;
the table is a transparent reference point rather than a universal performance
guarantee. It does not compare GPU execution.

For selected MLP outputs, batches below 5,000 rows use the low-overhead serial
path. Larger batches are split across a persistent, hardware-aware pool of up
to four workers. The pool is created lazily, so its first use includes a
one-time startup cost. Complete-Jacobian calls retain the general vectorized
path.


## Reproducing and interpreting the tables

Install skgrad before running the benchmark scripts from a checkout. The PyTorch
comparison additionally requires `pip install torch`. Preserve the raw output,
commit, CPU model, BLAS library and thread settings with any new measurements.
The tables above retain their original environment and are not universal bounds.

The numerical benchmark perturbs `skgrad.model_output`; it measures numerical
differentiation of the backend function. Independent sklearn checks in the tests
and worked examples serve a separate correctness purpose. CPU float64 PyTorch
results apply only to the tested network and thread configuration.

Kernel Jacobians can allocate arrays proportional to samples × support vectors ×
features. Polynomial expansions can also be large. Split evaluation rows into
batches when needed; selected MLP gradients avoid constructing a full Jacobian.
