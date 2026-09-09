"""Likelihood-statistic kernels for spectral fitting.

Implements the Gaussian (``Gstat``), Poisson (``Pstat``), Poisson-source
plus Poisson-background (``PPstat``/``cstat``), Poisson-source plus
Gaussian-background (``PGstat``) statistics. ``StatisticNB`` exposes the
numba-accelerated fast path and ``Statistic`` is a pure-numpy fallback with
the same interface.

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

        if (self.pointwise_loglike > 1e-12).any():
            raise ValueError('positive relative log-likelihood cannot define a pseudo-residual')

        self.pointwise_loglike = np.where(self.pointwise_loglike > 0.0, 0.0, self.pointwise_loglike)

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

        klogmu = 0.0
        if si != 0.0:
            klogmu = si * np.log(mu)

        klogk = 0.0
        if si != 0.0:
            klogk = si * np.log(si)

        logli = klogmu - mu - klogk + si
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

        s_klogmu = 0.0
        if si != 0.0:
            s_klogmu = si * np.log(mu_s)

        s_klogk = 0.0
        if si != 0.0:
            s_klogk = si * np.log(si)

        b_klogmu = 0.0
        if bi != 0.0:
            b_klogmu = bi * np.log(mu_b)

        b_klogk = 0.0
        if bi != 0.0:
            b_klogk = bi * np.log(bi)

        logli = (s_klogmu - mu_s - s_klogk + si) + (b_klogmu - mu_b - b_klogk + bi)
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

        s_klogmu = 0.0
        if si != 0.0:
            s_klogmu = si * np.log(mu_s)

        s_klogk = 0.0
        if si != 0.0:
            s_klogk = si * np.log(si)

        pois_logli = s_klogmu - mu_s - s_klogk + si

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


class StatisticNB:
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


class Statistic:
    """Pure-numpy fallback mirror of :class:`StatisticNB`.

    Same :class:`StatisticResult` contract as the numba-accelerated class,
    intended for debugging and for environments where numba cannot be used.
    """

    @staticmethod
    def xlogy(x, y):
        """Return ``x * log(y)`` element-wise, treating ``0 * log(y)`` as 0 for any ``y``."""

        res = np.zeros_like(x, dtype=np.float64)

        zero = x == 0
        res[~zero] = x[~zero] * np.log(y[~zero])

        return res

    @staticmethod
    def xdivy(x, y):
        """Return ``x / y`` element-wise, treating ``0 / 0`` as 0."""

        res = np.zeros_like(x, dtype=np.float64)

        zero = (x == 0) & (y == 0)
        res[~zero] = x[~zero] / y[~zero]

        return res

    @staticmethod
    def poisson_logpmf(k, mu):
        """Poisson log-PMF with Stirling's approximation for ``log(k!)``.

        Drops the ``log(2πk) / 2`` term, so the result is accurate up to
        an additive constant that cancels in likelihood ratios.
        """

        return Statistic.xlogy(k, mu) - mu - Statistic.xlogy(k, k) + k

    @staticmethod
    def gaussian_logpdf(x, loc, scale):
        """Gaussian log-PDF dropping the ``log(2π σ²)`` normalization constant."""

        return -0.5 * (Statistic.xdivy(x - loc, scale) ** 2)

    @staticmethod
    @guard_statistic
    def Gstat(**kwargs):
        """Return the pointwise Gaussian source-background result."""

        S = kwargs['S']
        B = kwargs['B']
        m = kwargs['m']

        ts = kwargs['ts']
        tb = kwargs['tb']

        sigma_S = kwargs['sigma_S']
        sigma_B = kwargs['sigma_B']

        if tb != 0:
            B = B / tb * ts
            sigma_B = sigma_B / tb * ts

        sigma = np.sqrt(sigma_S**2 + sigma_B**2)

        pointwise_sign = np.sign(S - B - m * ts)
        pointwise_loglike = Statistic.gaussian_logpdf(S - B, m * ts, sigma)

        return StatisticResult(pointwise_loglike, pointwise_sign)

    @staticmethod
    @guard_statistic
    def Pstat(**kwargs):
        """Return the pointwise pure-Poisson result."""

        S = kwargs['S']
        m = kwargs['m']
        ts = kwargs['ts']

        pointwise_sign = np.sign(S - m * ts)
        pointwise_loglike = Statistic.poisson_logpmf(S, m * ts)

        return StatisticResult(pointwise_loglike, pointwise_sign)

    @staticmethod
    @guard_statistic
    def PPstat(**kwargs):
        """Return the pointwise profiled Poisson-Poisson result."""

        S = kwargs['S']
        B = kwargs['B']
        m = kwargs['m']

        ts = kwargs['ts']
        tb = kwargs['tb']

        aa = ts + tb
        bb = (ts + tb) * m - S - B
        cc = -B * m
        dd = np.sqrt(bb * bb - 4 * aa * cc)

        po = bb >= 0
        b = np.empty_like(B, dtype=np.float64)

        denom = bb + dd
        zero_denom = po & (denom == 0)
        safe_po = po & ~zero_denom
        b[zero_denom] = 0.0
        b[safe_po] = -2 * cc[safe_po] / denom[safe_po]
        b[~po] = -(bb[~po] - dd[~po]) / (2 * aa)

        pointwise_sign = np.sign(S / ts - B / tb - m)
        pointwise_loglike = Statistic.poisson_logpmf(S, ts * (b + m)) + Statistic.poisson_logpmf(
            B, tb * b
        )

        return StatisticResult(pointwise_loglike, pointwise_sign)

    @staticmethod
    @guard_statistic
    def PGstat(**kwargs):
        """Return the pointwise profiled Poisson-Gaussian result."""

        S = kwargs['S']
        B = kwargs['B']
        m = kwargs['m']

        ts = kwargs['ts']
        tb = kwargs['tb']

        sigma = kwargs['sigma_B']

        aa = tb**2
        bb = ts * sigma**2 - tb * B + tb**2 * m
        cc = ts * sigma**2 * m - S * sigma**2 - tb * B * m
        dd = np.sqrt(bb**2 - 4 * aa * cc)

        sign = np.where(bb >= 0, 1, -1)
        qq = -0.5 * (bb + sign * dd)

        b1 = qq / aa
        b2 = np.zeros_like(B, dtype=np.float64)
        nonzero_qq = qq != 0
        b2[nonzero_qq] = cc[nonzero_qq] / qq[nonzero_qq]
        b = np.where(b1 > 0, b1, b2)

        pointwise_sign = np.sign(S / ts - B / tb - m)
        pointwise_loglike = Statistic.poisson_logpmf(S, ts * (b + m)) + Statistic.gaussian_logpdf(
            B, tb * b, sigma
        )

        return StatisticResult(pointwise_loglike, pointwise_sign)
