# Model coverage


| Family | Models | Differentiated output |
|---|---|---|
| Linear regression | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet` | Prediction |
| Linear classification | `LogisticRegression`, `RidgeClassifier`, `LinearSVC` | Decision score |
| Linear support-vector regression | `LinearSVR` | Prediction |
| Kernel support-vector regression | `SVR`, `NuSVR` | Prediction |
| Binary kernel classification | `SVC`, `NuSVC` | Decision score |
| Neural-network regression | `MLPRegressor` with squared-error or Poisson loss | Prediction, including the Poisson exponential output link |
| Neural-network classification | `MLPClassifier` | Binary or multiclass logits before logistic/softmax |
| Continuous pipelines | Supported scalers, polynomial expansion, PCA, and fitted feature selectors, then any supported estimator | Final estimator output, differentiated with respect to pipeline input features |

MLP hidden activations may be identity, logistic, tanh, or ReLU. At ReLU's
nondifferentiable origin, `skgrad` uses a zero derivative, matching
scikit-learn's backpropagation convention. Scalar and multi-output regression,
binary classification, and multiclass classification are supported.

Kernel SVMs support scikit-learn's `linear`, `poly`, `rbf`, and `sigmoid`
kernels. Multiclass kernel classifiers, callable kernels, and precomputed
kernels are not currently supported.

Tree models are intentionally excluded. Their predictions are piecewise
constant, so ordinary gradients are zero almost everywhere and undefined at
split boundaries. Use [TreeIG](https://github.com/LudgerHentschel/treeig),
which computes exact Integrated Gradients from the prediction jumps at tree
split crossings.

`skgrad` expects finite dense numeric inputs. Sequential and nested sklearn
pipelines may combine `StandardScaler`, `RobustScaler`, `MaxAbsScaler`,
`MinMaxScaler`, `PolynomialFeatures`, `PCA`, and supported fitted feature
selectors before a supported estimator. Gradients refer to the inputs of the
supplied pipeline, including every supported preprocessing chain rule.
Unknown transformers reject the entire analytic route; preprocessing is never
silently removed. See [pipeline conventions](https://ludgerhentschel.github.io/skgrad/pipelines.html)
for clipping, whitening, feature selection, and attribution coordinates.


## Important gaps

Coverage is focused rather than exhaustive. `ColumnTransformer`, categorical
encoders, arbitrary `FunctionTransformer` functions, and unsupported custom
transformer subclasses are not composed automatically. `KernelRidge`, Gaussian
processes, generalized linear regressors such as `PoissonRegressor`, and many
meta-estimators are also outside the current backend list. Having a `coef_`
attribute alone does not confer support.

Transformer recognition uses an explicit class list so an overridden `transform`
method cannot silently inherit an incorrect derivative. Estimator subclasses are accepted only when they inherit the differentiated
prediction method unchanged, including through intermediate base classes.
Regressors protect `predict`; score classifiers protect `decision_function`.
MLP classifiers protect `predict` and `predict_proba` because their exposed
logits derive from that prediction path. Overrides of these methods are refused.
This guard does not certify arbitrary changes to fitting or private helpers.

Nonlinear kernel SVMs depend on sklearn's private fitted `_gamma` attribute to
resolve `gamma="scale"` and `gamma="auto"`. If it is unavailable, evaluation raises
a clear compatibility error. CI tests scikit-learn pre-releases without imposing
an upper dependency bound.

See [pipeline gradients](pipelines.md) for the exact supported transformations
and [numerical fallback policy](numerical.md) for unsupported models.
