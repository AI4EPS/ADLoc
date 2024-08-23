# Author: Johannes Schönberger
#
# License: BSD 3 clause

import warnings
from numbers import Integral, Real

import numpy as np
import sklearn
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    MultiOutputMixin,
    RegressorMixin,
    _fit_context,
    clone,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._base import LinearRegression
from sklearn.utils import check_consistent_length, check_random_state
from sklearn.utils._param_validation import (
    HasMethods,
    Interval,
    Options,
    RealNotInt,
    StrOptions,
)
from sklearn.utils.metadata_routing import (
    _raise_for_unsupported_routing,
    _RoutingNotSupportedMixin,
)
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    has_fit_parameter,
)

_EPSILON = np.spacing(1)


def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.

    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.

    n_samples : int
        Total number of samples in the data.

    min_samples : int
        Minimum number of samples chosen randomly from original data.

    probability : float
        Probability (confidence) that one outlier-free sample is generated.

    Returns
    -------
    trials : int
        Number of trials.

    """
    inlier_ratio = n_inliers / float(n_samples)
    nom = max(_EPSILON, 1 - probability)
    denom = max(_EPSILON, 1 - inlier_ratio**min_samples)
    if nom == 1:
        return 0
    if denom == 1:
        return float("inf")
    return abs(float(np.ceil(np.log(nom) / np.log(denom))))


class RANSACRegressor(
    _RoutingNotSupportedMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    MultiOutputMixin,
    BaseEstimator,
):
    """RANSAC (RANdom SAmple Consensus) algorithm.

    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set.

    Read more in the :ref:`User Guide <ransac_regression>`.

    Parameters
    ----------
    estimator : object, default=None
        Base estimator object which implements the following methods:

         * `fit(X, y)`: Fit model to given training data and target values.
         * `score(X, y)`: Returns the mean accuracy on the given test data,
           which is used for the stop criterion defined by `stop_score`.
           Additionally, the score is used to decide which of two equally
           large consensus sets is chosen as the better one.
         * `predict(X)`: Returns predicted values using the linear model,
           which is used to compute residual error using loss function.

        If `estimator` is None, then
        :class:`~sklearn.linear_model.LinearRegression` is used for
        target values of dtype float.

        Note that the current implementation only supports regression
        estimators.

    min_samples : int (>= 1) or float ([0, 1]), default=None
        Minimum number of samples chosen randomly from original data. Treated
        as an absolute number of samples for `min_samples >= 1`, treated as a
        relative number `ceil(min_samples * X.shape[0])` for
        `min_samples < 1`. This is typically chosen as the minimal number of
        samples necessary to estimate the given `estimator`. By default a
        :class:`~sklearn.linear_model.LinearRegression` estimator is assumed and
        `min_samples` is chosen as ``X.shape[1] + 1``. This parameter is highly
        dependent upon the model, so if a `estimator` other than
        :class:`~sklearn.linear_model.LinearRegression` is used, the user must
        provide a value.

    residual_threshold : float, default=None
        Maximum residual for a data sample to be classified as an inlier.
        By default the threshold is chosen as the MAD (median absolute
        deviation) of the target values `y`. Points whose residuals are
        strictly equal to the threshold are considered as inliers.

    is_data_valid : callable, default=None
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(X, y)`. If its return value is
        False the current randomly chosen sub-sample is skipped.

    is_model_valid : callable, default=None
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, X, y)`. If its return value is
        False the current randomly chosen sub-sample is skipped.
        Rejecting samples with this function is computationally costlier than
        with `is_data_valid`. `is_model_valid` should therefore only be used if
        the estimated model is needed for making the rejection decision.

    max_trials : int, default=100
        Maximum number of iterations for random sample selection.

    max_skips : int, default=np.inf
        Maximum number of iterations that can be skipped due to finding zero
        inliers or invalid data defined by ``is_data_valid`` or invalid models
        defined by ``is_model_valid``.

        .. versionadded:: 0.19

    stop_n_inliers : int, default=np.inf
        Stop iteration if at least this number of inliers are found.

    stop_score : float, default=np.inf
        Stop iteration if score is greater equal than this threshold.

    stop_probability : float in range [0, 1], default=0.99
        RANSAC iteration stops if at least one outlier-free set of the training
        data is sampled in RANSAC. This requires to generate at least N
        samples (iterations)::

            N >= log(1 - probability) / log(1 - e**m)

        where the probability (confidence) is typically set to high value such
        as 0.99 (the default) and e is the current fraction of inliers w.r.t.
        the total number of samples.

    loss : str, callable, default='absolute_error'
        String inputs, 'absolute_error' and 'squared_error' are supported which
        find the absolute error and squared error per sample respectively.

        If ``loss`` is a callable, then it should be a function that takes
        two arrays as inputs, the true and predicted value and returns a 1-D
        array with the i-th value of the array corresponding to the loss
        on ``X[i]``.

        If the loss on a sample is greater than the ``residual_threshold``,
        then this sample is classified as an outlier.

        .. versionadded:: 0.18

    random_state : int, RandomState instance, default=None
        The generator used to initialize the centers.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    estimator_ : object
        Best fitted model (copy of the `estimator` object).

    n_trials_ : int
        Number of random selection trials until one of the stop criteria is
        met. It is always ``<= max_trials``.

    inlier_mask_ : bool array of shape [n_samples]
        Boolean mask of inliers classified as ``True``.

    n_skips_no_inliers_ : int
        Number of iterations skipped due to finding zero inliers.

        .. versionadded:: 0.19

    n_skips_invalid_data_ : int
        Number of iterations skipped due to invalid data defined by
        ``is_data_valid``.

        .. versionadded:: 0.19

    n_skips_invalid_model_ : int
        Number of iterations skipped due to an invalid model defined by
        ``is_model_valid``.

        .. versionadded:: 0.19

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    HuberRegressor : Linear regression model that is robust to outliers.
    TheilSenRegressor : Theil-Sen Estimator robust multivariate regression model.
    SGDRegressor : Fitted by minimizing a regularized empirical loss with SGD.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/RANSAC
    .. [2] https://www.sri.com/wp-content/uploads/2021/12/ransac-publication.pdf
    .. [3] http://www.bmva.org/bmvc/2009/Papers/Paper355/Paper355.pdf

    Examples
    --------
    >>> from sklearn.linear_model import RANSACRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(
    ...     n_samples=200, n_features=2, noise=4.0, random_state=0)
    >>> reg = RANSACRegressor(random_state=0).fit(X, y)
    >>> reg.score(X, y)
    0.9885...
    >>> reg.predict(X[:1,])
    array([-31.9417...])
    """  # noqa: E501

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "score", "predict"]), None],
        "min_samples": [
            Interval(Integral, 1, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="both"),
            None,
        ],
        # "residual_threshold": [Interval(Real, 0, None, closed="left"), None],
        "is_data_valid": [callable, None],
        "is_model_valid": [callable, None],
        "max_trials": [
            Interval(Integral, 0, None, closed="left"),
            Options(Real, {np.inf}),
        ],
        "max_skips": [
            Interval(Integral, 0, None, closed="left"),
            Options(Real, {np.inf}),
        ],
        "stop_n_inliers": [
            Interval(Integral, 0, None, closed="left"),
            Options(Real, {np.inf}),
        ],
        "stop_score": [Interval(Real, None, None, closed="both")],
        "stop_probability": [Interval(Real, 0, 1, closed="both")],
        "loss": [StrOptions({"absolute_error", "squared_error"}), callable],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        estimator=None,
        *,
        min_samples=None,
        residual_threshold=None,
        is_data_valid=None,
        is_model_valid=None,
        max_trials=100,
        max_skips=np.inf,
        stop_n_inliers=np.inf,
        stop_score=np.inf,
        stop_probability=0.99,
        loss="absolute_error",
        random_state=None,
    ):
        self.estimator = estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.is_data_valid = is_data_valid
        self.is_model_valid = is_model_valid
        self.max_trials = max_trials
        self.max_skips = max_skips
        self.stop_n_inliers = stop_n_inliers
        self.stop_score = stop_score
        self.stop_probability = stop_probability
        self.random_state = random_state
        self.loss = loss

    @_fit_context(
        # RansacRegressor.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit estimator using RANSAC algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample
            raises error if sample_weight is passed and estimator
            fit method does not support it.

            .. versionadded:: 0.18

        Returns
        -------
        self : object
            Fitted `RANSACRegressor` estimator.

        Raises
        ------
        ValueError
            If no valid consensus set could be found. This occurs if
            `is_data_valid` and `is_model_valid` return False for all
            `max_trials` randomly chosen sub-samples.
        """
        _raise_for_unsupported_routing(self, "fit", sample_weight=sample_weight)
        # Need to validate separately here. We can't pass multi_output=True
        # because that would allow y to be csr. Delay expensive finiteness
        # check to the estimator's own input validation.
        check_X_params = dict(accept_sparse="csr", force_all_finite=False)
        check_y_params = dict(ensure_2d=False)
        X, y = self._validate_data(X, y, validate_separately=(check_X_params, check_y_params))
        check_consistent_length(X, y)

        if self.estimator is not None:
            estimator = clone(self.estimator)
        else:
            estimator = LinearRegression()

        if self.min_samples is None:
            if not isinstance(estimator, LinearRegression):
                raise ValueError(
                    "`min_samples` needs to be explicitly set when estimator " "is not a LinearRegression."
                )
            min_samples = X.shape[1] + 1
        elif 0 < self.min_samples < 1:
            min_samples = np.ceil(self.min_samples * X.shape[0])
        elif self.min_samples >= 1:
            min_samples = self.min_samples
        if min_samples > X.shape[0]:
            raise ValueError(
                "`min_samples` may not be larger than number " "of samples: n_samples = %d." % (X.shape[0])
            )

        if self.residual_threshold is None:
            # MAD (median absolute deviation)
            residual_threshold = np.median(np.abs(y - np.median(y)))
        else:
            residual_threshold = self.residual_threshold

        if self.loss == "absolute_error":
            if y.ndim == 1:
                loss_function = lambda y_true, y_pred: np.abs(y_true - y_pred)
            else:
                # loss_function = lambda y_true, y_pred: np.sum(np.abs(y_true - y_pred), axis=1)
                loss_function = lambda y_true, y_pred: np.abs(y_true - y_pred)  ## for time + amplitude
        elif self.loss == "squared_error":
            if y.ndim == 1:
                loss_function = lambda y_true, y_pred: (y_true - y_pred) ** 2
            else:
                # loss_function = lambda y_true, y_pred: np.sum((y_true - y_pred) ** 2, axis=1)
                loss_function = lambda y_true, y_pred: (y_true - y_pred) ** 2  ## for time + amplitude

        elif callable(self.loss):
            loss_function = self.loss

        random_state = check_random_state(self.random_state)

        try:  # Not all estimator accept a random_state
            estimator.set_params(random_state=random_state)
        except ValueError:
            pass

        estimator_fit_has_sample_weight = has_fit_parameter(estimator, "sample_weight")
        estimator_name = type(estimator).__name__
        if sample_weight is not None and not estimator_fit_has_sample_weight:
            raise ValueError(
                "%s does not support sample_weight. Samples"
                " weights are only used for the calibration"
                " itself." % estimator_name
            )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        n_inliers_best = 1
        score_best = -np.inf
        inlier_mask_best = None
        X_inlier_best = None
        y_inlier_best = None
        inlier_best_idxs_subset = None
        self.n_skips_no_inliers_ = 0
        self.n_skips_invalid_data_ = 0
        self.n_skips_invalid_model_ = 0

        # number of data samples
        n_samples = X.shape[0]
        sample_idxs = np.arange(n_samples)
        prob = X[:, 2] / np.sum(X[:, 2])  # X: idx_sta, type, score, amp

        self.n_trials_ = 0
        max_trials = self.max_trials
        while self.n_trials_ < max_trials:
            self.n_trials_ += 1

            if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ + self.n_skips_invalid_model_) > self.max_skips:
                break

            # choose random sample set
            # subset_idxs = sample_without_replacement(n_samples, min_samples, random_state=random_state)
            subset_idxs = np.random.choice(sample_idxs, min_samples, replace=False, p=prob)
            X_subset = X[subset_idxs]
            y_subset = y[subset_idxs]

            # check if random sample set is valid
            if self.is_data_valid is not None and not self.is_data_valid(X_subset, y_subset):
                self.n_skips_invalid_data_ += 1
                continue

            # fit model for current random sample set
            if sample_weight is None:
                estimator.fit(X_subset, y_subset, **kwargs)
            else:
                estimator.fit(X_subset, y_subset, sample_weight=sample_weight[subset_idxs])

            # check if estimated model is valid
            if self.is_model_valid is not None and not self.is_model_valid(estimator, X_subset, y_subset):
                self.n_skips_invalid_model_ += 1
                continue

            # residuals of all data for current random sample model
            y_pred = estimator.predict(X, **kwargs)
            residuals_subset = loss_function(y, y_pred)

            # classify data into inliers and outliers
            inlier_mask_subset = residuals_subset <= residual_threshold
            inlier_mask_subset = (inlier_mask_subset).all(axis=-1)  ## for time + amplitude
            n_inliers_subset = np.sum(inlier_mask_subset)

            # less inliers -> skip current random sample
            if n_inliers_subset < n_inliers_best:
                self.n_skips_no_inliers_ += 1
                continue

            # extract inlier data set
            inlier_idxs_subset = sample_idxs[inlier_mask_subset]
            X_inlier_subset = X[inlier_idxs_subset]
            y_inlier_subset = y[inlier_idxs_subset]

            # score of inlier data set
            score_subset = estimator.score(X_inlier_subset, y_inlier_subset, **kwargs)

            # same number of inliers but worse score -> skip current random
            # sample
            if n_inliers_subset == n_inliers_best and score_subset < score_best:
                continue

            # save current random sample as best sample
            n_inliers_best = n_inliers_subset
            score_best = score_subset
            inlier_mask_best = inlier_mask_subset
            X_inlier_best = X_inlier_subset
            y_inlier_best = y_inlier_subset
            inlier_best_idxs_subset = inlier_idxs_subset

            ## ADLoc
            # estimator_best = clone(estimator)

            max_trials = min(
                max_trials,
                _dynamic_max_trials(n_inliers_best, n_samples, min_samples, self.stop_probability),
            )

            # break if sufficient number of inliers or score is reached
            if n_inliers_best >= self.stop_n_inliers or score_best >= self.stop_score:
                break

        # if none of the iterations met the required criteria
        if inlier_mask_best is None:
            if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ + self.n_skips_invalid_model_) > self.max_skips:
                raise ValueError(
                    "RANSAC skipped more iterations than `max_skips` without"
                    " finding a valid consensus set. Iterations were skipped"
                    " because each randomly chosen sub-sample failed the"
                    " passing criteria. See estimator attributes for"
                    " diagnostics (n_skips*)."
                )
            else:
                raise ValueError(
                    "RANSAC could not find a valid consensus set. All"
                    " `max_trials` iterations were skipped because each"
                    " randomly chosen sub-sample failed the passing criteria."
                    " See estimator attributes for diagnostics (n_skips*)."
                )
        else:
            if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ + self.n_skips_invalid_model_) > self.max_skips:
                warnings.warn(
                    (
                        "RANSAC found a valid consensus set but exited"
                        " early due to skipping more iterations than"
                        " `max_skips`. See estimator attributes for"
                        " diagnostics (n_skips*)."
                    ),
                    ConvergenceWarning,
                )

        # estimate final model using all inliers
        if sample_weight is None:
            estimator.fit(X_inlier_best, y_inlier_best, **kwargs)
        else:
            estimator.fit(
                X_inlier_best,
                y_inlier_best,
                sample_weight=sample_weight[inlier_best_idxs_subset],
            )

        # self.estimator_ = estimator
        # self.inlier_mask_ = inlier_mask_best

        # ADLoc
        y_pred = estimator.predict(X, **kwargs)
        residuals = loss_function(y, y_pred)
        inlier_mask = residuals <= residual_threshold
        inlier_mask = inlier_mask.all(axis=-1)  ## for time + amplitude
        # self.estimator_ = estimator_best
        self.estimator_ = estimator
        self.inlier_mask_ = inlier_mask

        return self

    def predict(self, X, **kwargs):
        """Predict using the estimated model.

        This is a wrapper for `estimator_.predict(X)`.

        Parameters
        ----------
        X : {array-like or sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y : array, shape = [n_samples] or [n_samples, n_targets]
            Returns predicted values.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X,
            force_all_finite=False,
            accept_sparse=True,
            reset=False,
        )
        return self.estimator_.predict(X, **kwargs)

    def score(self, X, y, **kwargs):
        """Return the score of the prediction.

        This is a wrapper for `estimator_.score(X, y)`.

        Parameters
        ----------
        X : (array-like or sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        z : float
            Score of the prediction.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X,
            force_all_finite=False,
            accept_sparse=True,
            reset=False,
        )
        return self.estimator_.score(X, y, **kwargs)

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": ("zero sample_weight is not equivalent to removing samples"),
            }
        }
