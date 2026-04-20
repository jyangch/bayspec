"""Significance calculators for count detections against a noisy background.

Adapted from ``gv_significance`` (Giacomo Vianello, 2018, BSD 3-Clause,
https://github.com/giacomov/gv_significance) and the corresponding
Vianello 2018 ApJS (https://doi.org/10.3847/1538-4365/aab780) and Li &
Ma (1983) formulations. ``pgsig`` treats the background as Gaussian;
``ppsig`` dispatches to Li & Ma, Vianello eq. 7, or Vianello eq. 9
based on the systematic-uncertainty inputs.
"""

import numpy as np
from math import log
import scipy.optimize
from numpy import sqrt, squeeze


def xlogy(x, y):
    """Return ``x * log(y)``, evaluating to 0 when ``x`` is 0.

    Avoids the ``0 * -inf = nan`` trap that naive multiplication hits.

    Args:
        x: Scalar multiplier.
        y: Scalar argument to the logarithm; must be positive when
            ``x`` is non-zero.

    Returns:
        ``0.0`` if ``x == 0``; otherwise ``x * log(y)``.
    """

    if x == 0.0:

        return 0.0

    else:

        return x * log(y)


def xlogyv(x, y):
    """Vectorized :func:`xlogy` that returns 0 where ``x`` is 0.

    Args:
        x: Scalar or array multiplier.
        y: Array argument to the logarithm; must be positive wherever
            ``x`` is non-zero.

    Returns:
        Squeezed array of ``x * log(y)`` with 0 in place of ``0 * log(0)``.
    """

    x = np.array(x, ndmin=1)
    y = np.array(y, ndmin=1)

    results = np.zeros_like(y)

    idx = (x != 0)

    results[idx] = x[idx] * np.log(y[idx])

    return np.squeeze(results)


def size_one_or_n(value, other_array, name):
    """Broadcast a scalar or length-``n`` array to match ``other_array``.

    Args:
        value: Scalar or 1D array.
        other_array: Reference array whose length defines ``n``.
        name: Label used in the assertion message.

    Returns:
        A 1D float array of the same length as ``other_array``.

    Raises:
        AssertionError: If ``value`` is neither length 1 nor length ``n``.
    """

    value_ = np.array(value, dtype=float, ndmin=1)

    if value_.shape[0] == 1:

        value_ = np.zeros(other_array.shape[0], dtype=float) + value

    else:

        assert value_.shape[0] == other_array.shape[0], \
            f"The size of {name} must be either 1 or the same size of n"

    return value_


def pgsig(n, b, sigma):
    """Compute the Gaussian-background detection significance.

    Treats the background estimate ``b`` as a Gaussian with standard
    deviation ``sigma``, maximizes the likelihood under the null
    hypothesis, and returns the signed square-root of the likelihood
    ratio.

    Args:
        n: Observed counts; scalar or array.
        b: Background expectation; same shape as ``n``.
        sigma: Background uncertainty; broadcasts against ``n``.

    Returns:
        Signed significance (``z`` score); negative when ``n < b``.
    """

    n_ = np.array(n, dtype=float, ndmin=1)
    b_ = np.array(b, dtype=float, ndmin=1)

    sigma_ = size_one_or_n(sigma, n_, "sigma")

    sign = np.where(n_ >= b_, 1, -1)

    B0_mle = 0.5 * (b_ - sigma_ ** 2 + sqrt(b_ ** 2 - 2 * b_ * sigma_ ** 2 + 4 * n_ * sigma_ ** 2 + sigma_ ** 4))

    # Clip tiny negative values produced by finite-precision arithmetic to zero.
    assert np.all(B0_mle > -0.01), "This is a bug. B0_mle cannot be negative."

    B0_mle = np.clip(B0_mle, 0, None)

    return squeeze(sqrt(2) * sqrt(xlogyv(n_, n_ / B0_mle) + (b_ - B0_mle)**2 / (2 * sigma_**2) + B0_mle - n_) * sign)


def _li_and_ma(n_, b_, alpha):
    """Return the Li & Ma (1983) significance for array inputs."""

    # Nudge by 1e-25 to sidestep the 0 * log(0) singularity; negligible downstream.
    n_ += 1E-25  # type: np.ndarray
    b_ += 1E-25  # type: np.ndarray

    n_plus_b = n_ + b_
    ap1 = alpha + 1

    res = n_ * np.log(ap1 / alpha * (n_ / n_plus_b))

    res += b_ * np.log(ap1 * (b_ / n_plus_b))

    return np.sqrt(2 * res)


def _likelihood_with_sys(o, b, a, s, k, B, M):
    """Log-likelihood for the Vianello 2018 eq. 9 model with a systematic term.

    Returns a large negative number in unphysical regions so that
    :func:`scipy.optimize.minimize` steers away.
    """

    if M + a * B <= 0 or k + 1 <= 0 or B <= 0:
        return -1000

    Ba = B * a
    Bak = B * a * k

    res = -Bak - Ba - B - M + xlogyv(b, B) - k ** 2 / (2 * s ** 2) + xlogyv(o, Bak + Ba + M)

    return res


def _get_TS_by_numerical_optimization(n_, b_, alpha, sigma):
    """Return the likelihood-ratio test statistic for Vianello 2018 eq. 9.

    Uses the paper's closed-form ``B_mle`` to reduce the null-hypothesis
    optimization to a single variable ``kk``, then computes
    ``TS = 2 * (logL_H0 - logL_H1)``.
    """

    # minimize() minimizes, so the log-likelihood is negated.
    wrapper = lambda kk: -1 * _likelihood_with_sys(n_, b_, alpha, sigma, kk,
                                                   B=(b_ + n_) / (alpha * kk + alpha + 1),
                                                   M=0)

    res = scipy.optimize.minimize(wrapper,
                                  [0.0],
                                  tol=1e-3)

    h0_mlike_value = res['fun']

    h1_mlike_value = -(xlogy(b_, b_) - b_ + xlogy(n_, n_) - n_)

    TS = 2 * (h0_mlike_value - h1_mlike_value)

    return TS


_get_TS_by_numerical_optimization_v = np.vectorize(_get_TS_by_numerical_optimization)


def ppsig(n, b, alpha, sigma=0, k=0):
    """Compute the Poisson-background detection significance.

    Dispatches per-element among three formulations:

    - Both ``sigma == 0`` and ``k == 0``: classic Li & Ma (1983).
    - ``k > 0``: Vianello 2018 eq. 7, treating ``k`` as an upper bound
      on the fractional systematic uncertainty (``sigma`` is ignored).
    - ``sigma > 0``: Vianello 2018 eq. 9, assuming a Gaussian systematic
      with standard deviation ``sigma`` (``k`` is ignored).

    Args:
        n: Observed counts; scalar or array.
        b: Expected background counts under the null; same shape as ``n``.
        alpha: Ratio of source to background observation efficiencies;
            scalar or same shape as ``n``.
        sigma: Gaussian systematic standard deviation; scalar or matching ``n``.
        k: Upper bound on fractional systematic uncertainty; scalar or
            matching ``n``.

    Returns:
        Signed significance (``z`` score) for each element; negative when
        ``n < alpha * b``.
    """

    n_ = np.array(n, dtype=float, ndmin=1)
    b_ = np.array(b, dtype=float, ndmin=1)

    k_ = size_one_or_n(k, n_, "k")

    sigma_ = size_one_or_n(sigma, n_, "sigma")

    alpha_ = size_one_or_n(alpha, n_, "alpha")

    sign = np.where(n_ >= alpha_ * b_, 1, -1)

    res = np.zeros(n_.shape[0], dtype=float)

    idx_lima = (sigma_ == 0) & (k_ == 0)

    res[idx_lima] = _li_and_ma(n_[idx_lima], b_[idx_lima], alpha_[idx_lima])

    # Vianello 2018 eq. 7 reduces to Li & Ma with alpha -> alpha * (k + 1).
    idx_eq7 = (k_ > 0)
    res[idx_eq7] = _li_and_ma(n_[idx_eq7], b_[idx_eq7], alpha_[idx_eq7] * (k_[idx_eq7] + 1))

    idx_eq9 = (sigma_ > 0)

    if np.any(idx_eq9):

        TS = _get_TS_by_numerical_optimization_v(n_[idx_eq9], b_[idx_eq9], alpha_[idx_eq9], sigma_[idx_eq9])

        res[idx_eq9] = np.sqrt(TS)

    return np.squeeze(sign * res)
