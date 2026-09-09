"""Significance calculators for count detections against a noisy background.

Adapted from ``gv_significance`` (Giacomo Vianello, 2018, BSD 3-Clause,
https://github.com/giacomov/gv_significance) and the corresponding
Vianello 2018 ApJS (https://doi.org/10.3847/1538-4365/aab780) and Li &
Ma (1983) formulations. ``pgsig`` treats the background as Gaussian;
``ppsig`` dispatches to Li & Ma, Vianello eq. 7, or Vianello eq. 9
based on the systematic-uncertainty inputs.
"""

from math import log

import numpy as np
from numpy import sqrt, squeeze
import scipy.optimize


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

    idx = x != 0

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
        assert value_.shape[0] == other_array.shape[0], (
            f'The size of {name} must be either 1 or the same size of n'
        )

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

    sigma_ = size_one_or_n(sigma, n_, 'sigma')

    sign = np.where(n_ >= b_, 1, -1)

    B0_mle = 0.5 * (
        b_ - sigma_**2 + sqrt(b_**2 - 2 * b_ * sigma_**2 + 4 * n_ * sigma_**2 + sigma_**4)
    )

    # Clip tiny negative values produced by finite-precision arithmetic to zero.
    assert np.all(B0_mle > -0.01), 'This is a bug. B0_mle cannot be negative.'

    B0_mle = np.clip(B0_mle, 0, None)

    return squeeze(
        sqrt(2)
        * sqrt(xlogyv(n_, n_ / B0_mle) + (b_ - B0_mle) ** 2 / (2 * sigma_**2) + B0_mle - n_)
        * sign
    )


def _li_and_ma(n_, b_, alpha):
    """Return the Li & Ma (1983) significance for array inputs."""

    # Nudge by 1e-25 to sidestep the 0 * log(0) singularity; negligible downstream.
    n_ += 1e-25  # type: np.ndarray
    b_ += 1e-25  # type: np.ndarray

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

    res = -Bak - Ba - B - M + xlogyv(b, B) - k**2 / (2 * s**2) + xlogyv(o, Bak + Ba + M)

    return res


def _get_TS_by_numerical_optimization(n_, b_, alpha, sigma):
    """Return the likelihood-ratio test statistic for Vianello 2018 eq. 9.

    Uses the paper's closed-form ``B_mle`` to reduce the null-hypothesis
    optimization to a single variable ``kk``, then computes
    ``TS = 2 * (logL_H0 - logL_H1)``.
    """

    # minimize() minimizes, so the log-likelihood is negated.
    def wrapper(kk):
        return -1 * _likelihood_with_sys(
            n_, b_, alpha, sigma, kk, B=(b_ + n_) / (alpha * kk + alpha + 1), M=0
        )

    res = scipy.optimize.minimize(wrapper, [0.0], tol=1e-3)

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

    k_ = size_one_or_n(k, n_, 'k')

    sigma_ = size_one_or_n(sigma, n_, 'sigma')

    alpha_ = size_one_or_n(alpha, n_, 'alpha')

    sign = np.where(n_ >= alpha_ * b_, 1, -1)

    res = np.zeros(n_.shape[0], dtype=float)

    idx_lima = (sigma_ == 0) & (k_ == 0)

    res[idx_lima] = _li_and_ma(n_[idx_lima], b_[idx_lima], alpha_[idx_lima])

    # Vianello 2018 eq. 7 reduces to Li & Ma with alpha -> alpha * (k + 1).
    idx_eq7 = k_ > 0
    res[idx_eq7] = _li_and_ma(n_[idx_eq7], b_[idx_eq7], alpha_[idx_eq7] * (k_[idx_eq7] + 1))

    idx_eq9 = sigma_ > 0

    if np.any(idx_eq9):
        TS = _get_TS_by_numerical_optimization_v(
            n_[idx_eq9], b_[idx_eq9], alpha_[idx_eq9], sigma_[idx_eq9]
        )

        res[idx_eq9] = np.sqrt(TS)

    return np.squeeze(sign * res)


def _solve_inverse_scalar(forward, sig, zero_point, negative_upper=None):
    """Solve one signed-significance inverse on a selected monotonic branch."""

    if sig == 0:
        return zero_point, (zero_point, None)

    if sig < 0:
        upper = zero_point if negative_upper is None else negative_upper
        upper_is_zero = negative_upper is None

        if upper <= 0:
            raise ValueError(f'sig={sig} is not reachable for non-negative counts')

        lower_value = forward(0.0)
        upper_value = 0.0 if upper_is_zero else forward(upper)

        if sig < lower_value or sig > upper_value:
            raise ValueError(f'sig={sig} is not reachable on the low-count branch')

        if sig == lower_value:
            return 0.0, (0.0, upper)
        if sig == upper_value:
            return upper, (0.0, upper)

        root = scipy.optimize.brentq(
            lambda n: -sig if upper_is_zero and n == upper else forward(n) - sig,
            0.0,
            upper,
        )
        return root, (0.0, upper)

    lower = zero_point
    upper = max(lower + 1.0, 2.0 * lower)

    for _ in range(100):
        upper_value = forward(upper)
        if upper_value >= sig:
            break
        upper = max(upper + 1.0, 2.0 * upper)
    else:
        raise ValueError(f'could not bracket sig={sig} on the high-count branch')

    root = scipy.optimize.brentq(
        lambda n: -sig if n == lower else forward(n) - sig,
        lower,
        upper,
    )
    return root, (lower, None)


def _integer_inverse(forward, sig, root, branch):
    """Return the first integer count reaching ``sig`` on ``branch``."""

    lower, upper = branch
    lower_integer = max(0, int(np.ceil(lower)))
    upper_integer = None if upper is None else int(np.floor(upper))
    candidate = max(lower_integer, int(np.floor(root)))

    def significance(count):
        if upper is None and count == lower:
            return 0.0
        return forward(count)

    while significance(candidate) < sig:
        candidate += 1
        if upper_integer is not None and candidate > upper_integer:
            raise ValueError(f'sig={sig} is not reachable at an integer count on this branch')

    while candidate > lower_integer and significance(candidate - 1) >= sig:
        candidate -= 1

    return candidate


def _inverse_elementwise(function, sig, parameters, integer):
    """Broadcast inverse inputs and evaluate ``function`` element by element."""

    if not isinstance(integer, (bool, np.bool_)):
        raise ValueError('integer must be a boolean')

    arrays = np.broadcast_arrays(
        np.asarray(sig, dtype=float),
        *(np.asarray(value, dtype=float) for value in parameters),
    )

    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError('inverse-significance inputs must be finite')

    result = np.empty(arrays[0].shape, dtype=int if integer else float)
    flat_arrays = [array.ravel() for array in arrays]

    for index, values in enumerate(zip(*flat_arrays, strict=True)):
        result.ravel()[index] = function(*map(float, values), integer=integer)

    return result.item() if result.ndim == 0 else result


def _pgsig_inv_scalar(sig, b, sigma, integer):
    if b < 0:
        raise ValueError('b must be non-negative')
    if sigma <= 0:
        raise ValueError('sigma must be positive')

    def forward(n):
        return float(pgsig(n, b, sigma))

    root, branch = _solve_inverse_scalar(forward, sig, zero_point=b)

    return _integer_inverse(forward, sig, root, branch) if integer else root


def pgsig_inv(sig, b, sigma, integer=False):
    """Invert :func:`pgsig` to obtain the observed source-region counts.

    Args:
        sig: Target signed significance; scalar or array.
        b: Gaussian background estimate; non-negative and broadcastable with
            the other inputs.
        sigma: Positive background uncertainty; broadcastable with the other
            inputs.
        integer: If true, return the smallest non-negative integer count whose
            significance reaches ``sig``. By default, return the continuous
            real-valued root.

    Returns:
        The inferred observed count or an array with the broadcast input shape.

    Raises:
        ValueError: If inputs are invalid or the target is unreachable.
    """

    return _inverse_elementwise(_pgsig_inv_scalar, sig, (b, sigma), integer)


def _ppsig_inv_scalar(sig, b, alpha, sigma, k, integer):
    if b < 0:
        raise ValueError('b must be non-negative')
    if alpha <= 0:
        raise ValueError('alpha must be positive')
    if sigma < 0:
        raise ValueError('sigma must be non-negative')
    if k < 0:
        raise ValueError('k must be non-negative')

    def forward(n):
        return float(ppsig(n, b, alpha, sigma=sigma, k=k))

    if sigma == 0 and k > 0:
        zero_point = alpha * (1 + k) * b
        negative_upper = np.nextafter(alpha * b, 0.0)
    else:
        zero_point = alpha * b
        negative_upper = None

    root, branch = _solve_inverse_scalar(
        forward,
        sig,
        zero_point=zero_point,
        negative_upper=negative_upper,
    )

    return _integer_inverse(forward, sig, root, branch) if integer else root


def ppsig_inv(sig, b, alpha, sigma=0, k=0, integer=False):
    """Invert :func:`ppsig` to obtain the observed source-region counts.

    The inverse follows the same Li & Ma / Vianello dispatch as :func:`ppsig`.
    For Vianello eq. 7 (``k > 0`` and ``sigma == 0``), positive targets use
    the high-count branch beginning at ``alpha * (1 + k) * b`` and negative
    targets use the low-count branch below ``alpha * b``. Targets in the gap
    introduced by the existing signed eq. 7 definition are rejected.

    Args:
        sig: Target signed significance; scalar or array.
        b: Expected Poisson background counts; non-negative and broadcastable.
        alpha: Positive source-to-background efficiency ratio; broadcastable.
        sigma: Non-negative Gaussian systematic standard deviation.
        k: Non-negative upper bound on fractional systematic uncertainty.
        integer: If true, return the smallest non-negative integer count on the
            selected branch whose significance reaches ``sig``. By default,
            return the continuous real-valued root.

    Returns:
        The inferred observed count or an array with the broadcast input shape.

    Raises:
        ValueError: If inputs are invalid or the target is unreachable on the
            selected branch.
    """

    return _inverse_elementwise(
        _ppsig_inv_scalar,
        sig,
        (b, alpha, sigma, k),
        integer,
    )
