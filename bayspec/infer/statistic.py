"""Likelihood-statistic kernels for spectral fitting.

Implements the Gaussian (``Gstat``), Poisson (``Pstat``), Poisson-source
plus Poisson-background (``PPstat``/``cstat``), and Poisson-source plus
Gaussian-background (``PGstat``) statistics. ``Statistic`` exposes the
Numba-accelerated implementation; ``StatisticNB`` remains as a compatibility
alias.

Every statistic takes the same keyword arguments (``S``, ``B``, ``m``,
``ts``, ``tb``, ``sigma_S``, ``sigma_B``) and returns a
:class:`StatisticResult`. Likelihood results expose channel-wise relative
log-likelihoods and signed pseudo-residuals.
"""

from dataclasses import dataclass
from functools import wraps

import numba as nb
import numpy as np


@dataclass
class StatisticResult:
    """Pointwise likelihood output and its derived fit diagnostics."""

    pointwise_loglike: np.ndarray
    pointwise_sign: np.ndarray

    def __post_init__(self):
        self.pointwise_loglike = np.asarray(self.pointwise_loglike, dtype=np.float64)
        self.pointwise_sign = np.asarray(self.pointwise_sign, dtype=np.float64)

        if (
            self.pointwise_loglike.ndim != 1
            or self.pointwise_sign.shape != self.pointwise_loglike.shape
        ):
            raise ValueError(
                'pointwise_loglike and pointwise_sign must have the same one-dimensional shape'
            )

        if np.isnan(self.pointwise_loglike).any() or np.isnan(self.pointwise_sign).any():
            raise ValueError('statistic result must not contain NaN')

        if (self.pointwise_loglike > 0.0).any():
            raise ValueError('positive relative log-likelihood cannot define a pseudo-residual')

    @property
    def pointwise_stat(self):
        """Per-observation deviance contributions."""

        return -2.0 * self.pointwise_loglike

    @property
    def loglike(self):
        """Total relative log-likelihood."""

        return np.sum(self.pointwise_loglike)

    @property
    def stat(self):
        """Total fit statistic."""

        return np.sum(self.pointwise_stat)

    @property
    def residual(self):
        """Signed square-root statistic contribution per channel."""

        return self.pointwise_sign * np.sqrt(self.pointwise_stat)


def guard_statistic(func):
    """Return an infinite statistic result when model predictions are invalid."""

    @wraps(func)
    def guarded(**kwargs):
        model = np.asarray(kwargs['m'])
        if not np.isfinite(model).all():
            return StatisticResult(
                np.full_like(model, -np.inf, dtype=np.float64),
                np.ones_like(model, dtype=np.float64),
            )
        return func(**kwargs)

    return guarded


@nb.vectorize([nb.float64(nb.float64, nb.float64)], cache=True)
def poisson_relative_loglike(count, mean):
    """Return the Poisson log-likelihood relative to its saturated value."""

    if (
        np.isnan(count)
        or np.isnan(mean)
        or count == np.inf
        or count == -np.inf
        or mean == np.inf
        or mean == -np.inf
        or count < 0.0
        or mean < 0.0
    ):
        return -np.inf
    if count == 0.0:
        return -mean
    if mean == 0.0:
        return -np.inf

    delta = (mean - count) / count
    if np.abs(delta) < 1e-4:
        return (
            -count
            * delta
            * delta
            * (0.5 - delta / 3.0 + delta * delta / 4.0 - delta * delta * delta / 5.0)
        )
    if delta != np.inf and delta != -np.inf and delta > -1.0:
        return -count * (delta - np.log1p(delta))

    return count * (np.log(mean) - np.log(count)) - mean + count


@nb.njit(cache=True, fastmath=True)
def _gstat_core(S, B, m, ts, tb, sigma_S, sigma_B):
    """Numba kernel for pointwise ``Gstat`` likelihood and residual sign."""

    n = S.shape[0]
    pointwise_loglike = np.empty(n, dtype=np.float64)
    pointwise_sign = np.empty(n, dtype=np.float64)

    ratio = 0.0
    if tb != 0.0:
        ratio = ts / tb

    for i in range(n):
        bi = B[i]
        sigma_bi = sigma_B[i]
        if tb != 0.0:
            bi = bi * ratio
            sigma_bi = sigma_bi * ratio

        sigma = np.sqrt(sigma_S[i] * sigma_S[i] + sigma_bi * sigma_bi)
        di = S[i] - bi
        mi = m[i] * ts
        delta = di - mi

        if sigma != 0.0:
            z = delta / sigma
            logli = -0.5 * z * z
        else:
            logli = 0.0 if delta == 0.0 else -np.inf

        sign = 0.0
        if delta > 0.0:
            sign = 1.0
        elif delta < 0.0:
            sign = -1.0

        pointwise_loglike[i] = logli
        pointwise_sign[i] = sign

    return pointwise_loglike, pointwise_sign


@nb.njit(cache=True, fastmath=True)
def _pstat_core(S, m, ts):
    """Numba kernel for pointwise pure-Poisson likelihood and residual sign."""

    n = S.shape[0]
    pointwise_loglike = np.empty(n, dtype=np.float64)
    pointwise_sign = np.empty(n, dtype=np.float64)

    for i in range(n):
        si = S[i]
        mu = m[i] * ts

        logli = poisson_relative_loglike(si, mu)
        delta = si - mu
        sign = 0.0
        if delta > 0.0:
            sign = 1.0
        elif delta < 0.0:
            sign = -1.0

        pointwise_loglike[i] = logli
        pointwise_sign[i] = sign

    return pointwise_loglike, pointwise_sign


@nb.njit(cache=True, fastmath=True)
def _ppstat_core(S, B, m, ts, tb):
    """Numba kernel for ``PPstat``/``cstat``: Poisson source + Poisson background.

    Profiles out the true background rate ``b`` per bin via the standard
    ``cstat`` closed-form, then sums the profiled Poisson log-likelihoods.
    """

    n = S.shape[0]
    pointwise_loglike = np.empty(n, dtype=np.float64)
    pointwise_sign = np.empty(n, dtype=np.float64)

    aa = ts + tb

    for i in range(n):
        si = S[i]
        bi = B[i]
        mi = m[i]

        bb = aa * mi - si - bi
        cc = -bi * mi
        dd = np.sqrt(bb * bb - 4.0 * aa * cc)

        if bb >= 0.0:
            b = 0.0 if (bb + dd) == 0.0 else -2.0 * cc / (bb + dd)
        else:
            b = -(bb - dd) / (2.0 * aa)

        mu_s = ts * (b + mi)
        mu_b = tb * b

        logli = poisson_relative_loglike(si, mu_s) + poisson_relative_loglike(bi, mu_b)
        delta = si / ts - bi / tb - mi
        sign = 0.0
        if delta > 0.0:
            sign = 1.0
        elif delta < 0.0:
            sign = -1.0

        pointwise_loglike[i] = logli
        pointwise_sign[i] = sign

    return pointwise_loglike, pointwise_sign


@nb.njit(cache=True, fastmath=True)
def _pgstat_core(S, B, m, ts, tb, sigma_B):
    """Numba kernel for ``PGstat``: Poisson source + Gaussian background.

    Profiles the background rate via the quadratic closed-form used in
    XSPEC's ``pgstat`` and combines the profiled Poisson source and
    Gaussian background log-likelihoods.
    """

    n = S.shape[0]
    pointwise_loglike = np.empty(n, dtype=np.float64)
    pointwise_sign = np.empty(n, dtype=np.float64)

    aa = tb * tb

    for i in range(n):
        si = S[i]
        bi = B[i]
        mi = m[i]
        sigma = sigma_B[i]

        bb = ts * sigma * sigma - tb * bi + tb * tb * mi
        cc = ts * sigma * sigma * mi - si * sigma * sigma - tb * bi * mi
        dd = np.sqrt(bb * bb - 4.0 * aa * cc)

        sgn = 1.0
        if bb < 0.0:
            sgn = -1.0

        qq = -0.5 * (bb + sgn * dd)

        b1 = qq / aa
        b2 = cc / qq if qq != 0.0 else 0.0
        b = b1 if b1 > 0.0 else b2

        mu_s = ts * (b + mi)

        pois_logli = poisson_relative_loglike(si, mu_s)

        gauss_logli = 0.0
        if sigma != 0.0:
            z = (bi - tb * b) / sigma
            gauss_logli = -0.5 * z * z

        logli = pois_logli + gauss_logli
        delta = si / ts - bi / tb - mi
        sign = 0.0
        if delta > 0.0:
            sign = 1.0
        elif delta < 0.0:
            sign = -1.0

        pointwise_loglike[i] = logli
        pointwise_sign[i] = sign

    return pointwise_loglike, pointwise_sign


class Statistic:
    """Numba-accelerated statistic dispatch table.

    Every method takes the standard keyword bundle (``S``, ``B``, ``m``,
    ``ts``, ``tb``, ``sigma_S``, ``sigma_B``) and returns a
    :class:`StatisticResult`.
    """

    @staticmethod
    @guard_statistic
    def Gstat(**kwargs):
        """Return the pointwise Gaussian statistic result."""

        pointwise_loglike, pointwise_sign = _gstat_core(
            kwargs['S'],
            kwargs['B'],
            kwargs['m'],
            kwargs['ts'],
            kwargs['tb'],
            kwargs['sigma_S'],
            kwargs['sigma_B'],
        )

        return StatisticResult(pointwise_loglike, pointwise_sign)

    @staticmethod
    @guard_statistic
    def Pstat(**kwargs):
        """Return the pointwise pure-Poisson statistic result."""

        pointwise_loglike, pointwise_sign = _pstat_core(kwargs['S'], kwargs['m'], kwargs['ts'])

        return StatisticResult(pointwise_loglike, pointwise_sign)

    @staticmethod
    @guard_statistic
    def PPstat(**kwargs):
        """Return the pointwise profiled Poisson-Poisson result."""

        pointwise_loglike, pointwise_sign = _ppstat_core(
            kwargs['S'], kwargs['B'], kwargs['m'], kwargs['ts'], kwargs['tb']
        )
        return StatisticResult(pointwise_loglike, pointwise_sign)

    @staticmethod
    @guard_statistic
    def PGstat(**kwargs):
        """Return the pointwise profiled Poisson-Gaussian result."""

        pointwise_loglike, pointwise_sign = _pgstat_core(
            kwargs['S'],
            kwargs['B'],
            kwargs['m'],
            kwargs['ts'],
            kwargs['tb'],
            kwargs['sigma_B'],
        )

        return StatisticResult(pointwise_loglike, pointwise_sign)


StatisticNB = Statistic
