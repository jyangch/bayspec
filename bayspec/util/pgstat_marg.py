"""Standalone Poisson likelihood with a marginalized Gaussian background.

The fitting statistic ``pgstat_marg`` (``Statistic.PGstatMarg``) uses this model
through ``pgstat_marg_saturation`` and ``pgstat_marg_relative``. The profile
statistic ``pgstat`` is unchanged.

Model (independent channels, source and background exposures ts and tb)::

    S | theta, b ~ Poisson(ts * (m(theta) + b))
    B | b        ~ Normal(tb * b, sigma_B**2)
    pi(b)        proportional to 1 on b >= 0

The output is log p(S | theta, B), integrating the *normalized* background
posterior p(b | B). All Poisson and truncated-normal normalization factors are
retained. The flat background prior is improper: this conditional likelihood
does not define an absolute joint evidence for (S, B). It also does not make
old profile-posterior draws valid draws from the marginal model.

For sigma_B > 0, in source-count units s=ts*m, beta=ts*B/tb, e=ts*sigma_B/tb::

    p(S | s, B) = integral_0^inf Poisson(S | s+x)
                  * phi((x-beta)/e) / (e*Phi(beta/e)) dx

Numerical integration is centered and scaled at the integrand's constrained
mode. Log-concavity gives explicit bounds on the omitted tails. Quadrature and
tail error estimates are checked, with failures raised rather than silently
returning an unconverged likelihood. ``pgstat_marg_loglike`` is this
accuracy-oriented reference implementation.

``pgstat_marg_loglike_fast`` evaluates the same model with the same mode,
scaling and tail bracketing, but integrates each side of the mode with fixed
Gauss-Legendre nodes inside a numba kernel. It makes no per-call error estimate;
its accuracy is checked against the reference in the tests.

``pgstat_marg_relative`` subtracts a per-channel saturated offset, the maximum
of log p(S | s, B) over source counts, computed once per data unit by
``pgstat_marg_saturation``. Minus twice the relative value is then a deviance.

Example::

    from bayspec.util.pgstat_marg import pgstat_marg_loglike
    loglike_i = pgstat_marg_loglike(
        S=[12, 0], B=[8, -0.2], m=[0.4, 0.1],
        ts=10.0, tb=20.0, sigma_B=[2.0, 0.5],
    )
    loglike = loglike_i.sum()
"""

from functools import lru_cache
import math

import numba as nb
import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx, gammaln, log_ndtr

__all__ = [
    'pgstat_marg_loglike',
    'pgstat_marg_loglike_fast',
    'pgstat_marg_relative',
    'pgstat_marg_saturation',
]

_LOG_2PI = math.log(2.0 * math.pi)


def _log1pmx(x):
    """log(1+x)-x without subtracting nearly equal numbers."""
    if abs(x) < 1e-3:
        return x * x * (-0.5 + x * (1 / 3 + x * (-1 / 4 + x * (1 / 5 + x * (-1 / 6 + x / 7)))))
    return math.log1p(x) - x


def _poisson_logpmf(count, mean):
    if count == 0:
        return -mean
    if mean == 0:
        return -math.inf

    # log Poisson(n | n); direct gammaln subtraction loses precision at large n.
    if count < 16:
        saturated = count * math.log(count) - count - gammaln(count + 1)
    else:
        inverse = 1.0 / count
        square = inverse * inverse
        correction = inverse * (
            1 / 12
            + square * (-1 / 360 + square * (1 / 1260 + square * (-1 / 1680 + square / 1188)))
        )
        saturated = -0.5 * (_LOG_2PI + math.log(count)) - correction
    if 0.5 * count <= mean <= 2.0 * count:
        relative = count * _log1pmx((mean - count) / count)
    else:
        relative = count * (math.log(mean) - math.log(count)) - mean + count
    return saturated + relative


def _log_erfcx_positive(x):
    if x < 1e4:
        return math.log(erfcx(x))
    inverse_square = (1.0 / x) ** 2
    return (
        -math.log(x)
        - 0.5 * math.log(math.pi)
        + math.log1p(inverse_square * (-0.5 + inverse_square * (0.75 - 1.875 * inverse_square)))
    )


def _channel_loglike(count, source, background, error, rtol):
    if error == 0:
        mean = source + background
        if not math.isfinite(mean):
            raise FloatingPointError('known-background mean exceeds float64 working range')
        result = _poisson_logpmf(count, mean)
        if mean > 0 and not math.isfinite(result):
            raise FloatingPointError('Poisson log probability exceeds float64 working range')
        return result

    variance = error * error
    coefficient = source + variance - background
    constant = variance * (count - source) + source * background
    if (
        not all(math.isfinite(v) for v in (variance, coefficient, constant))
        or variance < np.finfo(float).tiny
    ):
        raise FloatingPointError('background marginalization exceeds float64 working range')

    # Solve in background counts, avoiding subtraction of source from total mean.
    # constant < 0 with coefficient < 0 cannot occur: the former needs background < variance,
    # the latter background > source + variance. Hence sqrt(constant) below is always defined.
    if constant <= 0 and coefficient >= 0:
        mode = 0.0
    else:
        discriminant = math.hypot(coefficient, 2.0 * math.sqrt(constant))
        if coefficient >= 0:
            mode = constant / (0.5 * discriminant + 0.5 * coefficient)
        else:
            mode = 0.5 * discriminant - 0.5 * coefficient

    mean = source + mode
    slope = (count / mean if count else 0.0) - 1.0 + background / variance
    # The derivative is exactly zero at an interior mode. Avoid numerical
    # cancellation in its algebraic expression by using that identity.
    if mode > 0:
        slope = 0.0
    curvature_root = math.hypot(math.sqrt(count) / mean if count else 0.0, 1.0 / error)
    scale = 1.0 / (curvature_root + abs(slope))
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(mean):
        raise FloatingPointError(
            'background marginalization has an unrepresentable integration scale'
        )

    def log_shape(t):
        delta = scale * t
        poisson_change = 0.0
        if count:
            ratio = delta / mean
            if ratio <= -1:
                return -math.inf
            poisson_change = count * _log1pmx(ratio)
        return poisson_change + slope * delta - 0.5 * (delta / error) ** 2

    def derivative(t):
        delta = scale * t
        poisson_change = 0.0
        if count:
            ratio = delta / mean
            if ratio <= -1:
                return math.inf
            poisson_change = -(count / mean) * ratio / (1.0 + ratio) * scale
        return poisson_change + slope * scale - (scale / error) ** 2 * t

    log_drop = max(40.0, -math.log(rtol) + 8.0)

    def integrate_side(direction, boundary=math.inf):
        if boundary == 0:
            return 0.0, 0.0
        extent = min(1.0, boundary)
        for _ in range(128):
            if extent == boundary or log_shape(direction * extent) <= -log_drop:
                break
            extent = min(2.0 * extent, boundary)
        else:
            raise FloatingPointError('could not bracket the background marginalization tail')

        result = quad(
            lambda t: math.exp(log_shape(direction * t)),
            0.0,
            extent,
            epsabs=0.0,
            epsrel=rtol / 4.0,
            limit=200,
            full_output=1,
        )
        if len(result) != 3:
            raise FloatingPointError(f'background quadrature failed: {result[3]}')
        integral, quadrature_error, _ = result
        tail = 0.0
        if extent != boundary:
            # Concavity bounds the tail by the tangent exponential at the end.
            tail = math.exp(log_shape(direction * extent)) / abs(derivative(direction * extent))
        return integral, quadrature_error + tail

    right, right_error = integrate_side(1)
    left, left_error = integrate_side(-1, mode / scale)
    integral = left + right
    if not integral > 0 or left_error + right_error > rtol * integral:
        raise FloatingPointError(
            'background integral did not meet the requested relative tolerance'
        )

    a = background / error
    if a < 0:
        # Cancel the potentially huge common a^2/2 analytically, before rounding.
        u = mode / error
        normal_height = (
            a * u
            - 0.5 * u * u
            - math.log(error)
            - 0.5 * _LOG_2PI
            - math.log(0.5)
            - _log_erfcx_positive(-a / math.sqrt(2.0))
        )
    else:
        # At the interior mode z=(b-beta)/e=(n-s-beta)*e/(s+b+e^2).
        # This is stable even when b and beta are nearly indistinguishable.
        z = (math.fsum((count, -source, -background)) / (mean + variance)) * error if mode else -a
        normal_height = -0.5 * z * z - math.log(error) - 0.5 * _LOG_2PI - log_ndtr(a)
    logp = _poisson_logpmf(count, mean) + normal_height + math.log(scale) + math.log(integral)
    if not math.isfinite(logp) or logp > rtol:
        raise FloatingPointError('background marginalization produced an invalid log probability')
    return logp


def pgstat_marg_loglike(S, B, m, ts, tb, sigma_B, *, rtol=1e-10):
    """Return normalized pointwise log p(S | m, B), integrating nonnegative background.

    Args:
        S: Nonnegative integer source-region counts (not background-subtracted).
        B: Gaussian background measurement in background-exposure counts. May be negative.
        m: Nonnegative predicted source count rate after response folding.
        ts: Positive source exposure, including the same scaling convention as PG-stat.
        tb: Positive effective background exposure, including area/extraction scaling.
        sigma_B: Nonnegative Gaussian standard error on B. Zero denotes known background;
            then B must be nonnegative.
        rtol: Requested relative quadrature tolerance, 1e-12 <= rtol < 1.
            Includes estimated quadrature and omitted-tail error, not all floating-point
            error. Values outside float64 working range raise FloatingPointError;
            in particular, a positive source-scaled variance must be a normal float.

    All six data arguments follow NumPy broadcasting. A scalar input combination
    returns a float; otherwise returns an array of the broadcast shape, without
    mutating inputs. Sum the output for the total conditional log likelihood.

    A uniform prior on the nonnegative *true background rate* is fixed by this
    function's definition; no per-channel prior tuning is performed. Channels
    must be independent under that model. The output is not a profile PG-stat,
    a relative/saturated likelihood, or a signed residual. It is not registered
    with Statistic, Pair, Posterior, or the public package exports.

    Raises:
        ValueError: Invalid counts, exposures, errors, tolerances or array shapes.
        FloatingPointError: Unrepresentable intermediates or integration failure.
    """
    if not np.isfinite(rtol) or not 1e-12 <= rtol < 1:
        raise ValueError('rtol must be finite and satisfy 1e-12 <= rtol < 1')
    counts, source_counts, background_counts, background_errors = _scaled_inputs(
        S, B, m, ts, tb, sigma_B
    )

    output = np.empty(counts.shape, dtype=float)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for index in np.ndindex(output.shape):
            try:
                output[index] = _channel_loglike(
                    float(counts[index]),
                    float(source_counts[index]),
                    float(background_counts[index]),
                    float(background_errors[index]),
                    float(rtol),
                )
            except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
                raise FloatingPointError(f'channel {index}: {exc}') from exc
    return float(output) if output.ndim == 0 else output


def _scaled_inputs(S, B, m, ts, tb, sigma_B):
    """Validate and broadcast inputs, then convert them to source-count units."""

    arrays = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in (S, B, m, ts, tb, sigma_B)])
    counts, backgrounds, rates, source_exposure, background_exposure, errors = arrays
    if any(not np.all(np.isfinite(x)) for x in arrays):
        raise ValueError('counts, rates, exposures and background measurements must be finite')
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError('S must contain nonnegative integer counts')
    if np.any(rates < 0) or np.any(errors < 0):
        raise ValueError('m and sigma_B must be nonnegative')
    if np.any(source_exposure <= 0) or np.any(background_exposure <= 0):
        raise ValueError('ts and tb must be positive')
    if np.any((errors == 0) & (backgrounds < 0)):
        raise ValueError('negative B with sigma_B=0 has no admissible nonnegative background')

    with np.errstate(over='raise', invalid='raise', divide='raise'):
        source_counts = source_exposure * rates
        ratio = source_exposure / background_exposure
        background_counts = ratio * backgrounds
        background_errors = ratio * errors
        if np.any(ratio == 0):
            raise FloatingPointError('exposure ratio underflowed')
        if np.any((rates > 0) & (source_counts == 0)):
            raise FloatingPointError('source mean underflowed during exposure scaling')
        if np.any((backgrounds != 0) & (background_counts == 0)):
            raise FloatingPointError('background measurement underflowed during exposure scaling')
        if np.any((errors > 0) & (background_errors == 0)):
            raise FloatingPointError('background uncertainty underflowed during exposure scaling')
    return counts, source_counts, background_counts, background_errors


def pgstat_marg_loglike_fast(S, B, m, ts, tb, sigma_B, *, npts=32):
    """Fixed-node, numba-compiled counterpart of ``pgstat_marg_loglike``.

    Same model, arguments, validation and output. Each side of the mode uses ``npts``
    Gauss-Legendre nodes instead of adaptive quad, with no per-call error estimate.
    At the default npts=32 it agrees with the reference to about 1e-13 in log likelihood
    over the tested range. Raises FloatingPointError where the kernel cannot evaluate.
    """
    if isinstance(npts, bool) or not isinstance(npts, (int, np.integer)) or npts < 8:
        raise ValueError('npts must be an integer >= 8')
    counts, source_counts, background_counts, background_errors = _scaled_inputs(
        S, B, m, ts, tb, sigma_B
    )

    nodes, weights = _gauss_legendre(int(npts))

    output = _pgstat_marg_core(
        counts.ravel(),
        source_counts.ravel(),
        background_counts.ravel(),
        background_errors.ravel(),
        nodes,
        weights,
    ).reshape(counts.shape)
    failed = np.isnan(output)
    if np.any(failed):
        index = tuple(int(i) for i in np.argwhere(failed)[0])
        raise FloatingPointError(f'channel {index}: background marginalization failed')
    return float(output) if output.ndim == 0 else output


def pgstat_marg_saturation(S, B, ts, tb, sigma_B, *, npts=32):
    """Saturated offset of the ``pgstat_marg`` statistic for one data unit.

    For each channel, maximizes log p(S | s, B) over source counts s >= 0. Returns
    (log_saturated, source_saturated, ts, tb, background_counts, background_errors);
    the last four record the exposures and source-unit background used, so the offset
    stays a data-only constant. Computed once per data unit, not per likelihood call.
    """
    counts = np.ascontiguousarray(S, dtype=np.float64)
    ratio = np.asarray(ts, dtype=np.float64) / np.asarray(tb, dtype=np.float64)
    background_counts = np.ascontiguousarray(
        np.broadcast_to(ratio * np.asarray(B, dtype=np.float64), counts.shape)
    )
    background_errors = np.ascontiguousarray(
        np.broadcast_to(ratio * np.asarray(sigma_B, dtype=np.float64), counts.shape)
    )
    nodes, weights = _gauss_legendre(npts)

    log_saturated, source_saturated = _pgstat_marg_saturated_core(
        counts, background_counts, background_errors, nodes, weights
    )
    failed = np.isnan(log_saturated)
    if np.any(failed):
        raise FloatingPointError(
            f'channel {int(np.argmax(failed))}: saturated background marginalization failed'
        )
    return (
        log_saturated,
        source_saturated,
        np.array(ts, dtype=np.float64),
        np.array(tb, dtype=np.float64),
        background_counts,
        background_errors,
    )


def pgstat_marg_relative(S, m, ts, tb, saturation, *, npts=32):
    """Relative log likelihood and residual sign of ``pgstat_marg`` for one data unit.

    The relative value is log p(S | m, B) minus the fixed saturated offset, so it is
    <= 0 and -2 times its sum is a deviance. The sign is that of s_sat - ts * m.
    Channels with negative model rates get -inf. The exposures must be those used
    for the offset: a free or changed sf/bf would turn the offset into a function
    of the parameters, which is not supported yet.
    """
    log_saturated, source_saturated, ts_ref, tb_ref, background_counts, background_errors = (
        saturation
    )
    if not (np.array_equal(ts, ts_ref) and np.array_equal(tb, tb_ref)):
        raise ValueError(
            'pgstat_marg needs the exposures used for its saturated offset; '
            'free or changed source/background factors are not supported yet'
        )
    rates = np.asarray(m, dtype=np.float64)
    invalid = ~(rates >= 0)
    source_counts = np.ascontiguousarray(
        np.broadcast_to(np.where(invalid, 0.0, np.asarray(ts) * rates), log_saturated.shape)
    )
    nodes, weights = _gauss_legendre(npts)

    loglike = _pgstat_marg_core(
        np.ascontiguousarray(S, dtype=np.float64),
        source_counts,
        background_counts,
        background_errors,
        nodes,
        weights,
    )
    failed = np.isnan(loglike) & ~invalid
    if np.any(failed):
        raise FloatingPointError(
            f'channel {int(np.argmax(failed))}: background marginalization failed'
        )

    relative = loglike - log_saturated
    if np.any(relative > 1e-8):
        raise FloatingPointError('pgstat_marg likelihood exceeds its saturated offset')
    relative = np.where(invalid, -np.inf, np.minimum(relative, 0.0))
    sign = np.where(invalid, 1.0, np.sign(source_saturated - source_counts))
    return relative, sign


@lru_cache(maxsize=8)
def _gauss_legendre(npts):
    nodes, weights = np.polynomial.legendre.leggauss(npts)
    return nodes, weights


# ---- numba kernel; NaN marks a channel the kernel cannot evaluate ----
_TINY = float(np.finfo(float).tiny)
_LOG_DROP = 40.0


@nb.njit(cache=True)
def _log1pmx_nb(x):
    if abs(x) < 1e-3:
        return x * x * (-0.5 + x * (1 / 3 + x * (-1 / 4 + x * (1 / 5 + x * (-1 / 6 + x / 7)))))
    return math.log1p(x) - x


@nb.njit(cache=True)
def _poisson_logpmf_nb(count, mean):
    if count == 0:
        return -mean
    if mean == 0:
        return -np.inf
    if count < 16:
        saturated = count * math.log(count) - count - math.lgamma(count + 1)
    else:
        inverse = 1.0 / count
        square = inverse * inverse
        correction = inverse * (
            1 / 12
            + square * (-1 / 360 + square * (1 / 1260 + square * (-1 / 1680 + square / 1188)))
        )
        saturated = -0.5 * (_LOG_2PI + math.log(count)) - correction
    if 0.5 * count <= mean <= 2.0 * count:
        relative = count * _log1pmx_nb((mean - count) / count)
    else:
        relative = count * (math.log(mean) - math.log(count)) - mean + count
    return saturated + relative


@nb.njit(cache=True)
def _log_norm_nb(a):
    """log Phi(a) for a >= 0; log Phi(a) + a^2/2 for a < 0, as cancelled in the kernel."""
    if a >= 0:
        return math.log1p(-0.5 * math.erfc(a / math.sqrt(2.0)))
    x = -a / math.sqrt(2.0)
    if x < 25.0:
        return math.log(0.5) + math.log(math.erfc(x)) + x * x
    # asymptotic log erfcx(x); the omitted term is below 1e-16 for x >= 25
    y = 1.0 / (2.0 * x * x)
    series = 1.0 + y * (-1.0 + y * (3.0 + y * (-15.0 + y * (105.0 + y * (-945.0 + y * 10395.0)))))
    return math.log(0.5) - math.log(x) - 0.5 * math.log(math.pi) + math.log(series)


@nb.njit(cache=True)
def _log_shape_nb(t, count, mean, slope, scale, error):
    delta = scale * t
    poisson_change = 0.0
    if count:
        ratio = delta / mean
        if ratio <= -1:
            return -np.inf
        poisson_change = count * _log1pmx_nb(ratio)
    return poisson_change + slope * delta - 0.5 * (delta / error) ** 2


@nb.njit(cache=True)
def _side_integral_nb(direction, boundary, count, mean, slope, scale, error, nodes, weights):
    if boundary == 0:
        return 0.0
    extent = min(1.0, boundary)
    bracketed = False
    for _ in range(128):
        if (
            extent == boundary
            or _log_shape_nb(direction * extent, count, mean, slope, scale, error) <= -_LOG_DROP
        ):
            bracketed = True
            break
        extent = min(2.0 * extent, boundary)
    if not bracketed:
        return np.nan

    half = 0.5 * extent
    total = 0.0
    for j in range(nodes.shape[0]):
        t = half * (nodes[j] + 1.0)
        total += weights[j] * math.exp(
            _log_shape_nb(direction * t, count, mean, slope, scale, error)
        )
    return half * total


@nb.njit(cache=True)
def _channel_loglike_nb(count, source, background, error, nodes, weights):
    if error == 0:
        mean = source + background
        if not math.isfinite(mean):
            return np.nan
        result = _poisson_logpmf_nb(count, mean)
        if mean > 0 and not math.isfinite(result):
            return np.nan
        return result

    variance = error * error
    coefficient = source + variance - background
    constant = variance * (count - source) + source * background
    if not (math.isfinite(variance) and math.isfinite(coefficient) and math.isfinite(constant)):
        return np.nan
    if variance < _TINY:
        return np.nan

    # same constrained mode as _channel_loglike; constant < 0 implies coefficient >= 0
    if constant <= 0 and coefficient >= 0:
        mode = 0.0
    else:
        discriminant = math.hypot(coefficient, 2.0 * math.sqrt(constant))
        if coefficient >= 0:
            mode = constant / (0.5 * discriminant + 0.5 * coefficient)
        else:
            mode = 0.5 * discriminant - 0.5 * coefficient

    mean = source + mode
    slope = (count / mean if count else 0.0) - 1.0 + background / variance
    if mode > 0:
        slope = 0.0
    curvature_root = math.hypot(math.sqrt(count) / mean if count else 0.0, 1.0 / error)
    scale = 1.0 / (curvature_root + abs(slope))
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(mean):
        return np.nan

    right = _side_integral_nb(1.0, np.inf, count, mean, slope, scale, error, nodes, weights)
    left = _side_integral_nb(-1.0, mode / scale, count, mean, slope, scale, error, nodes, weights)
    integral = left + right
    if not integral > 0:
        return np.nan

    a = background / error
    log_norm = _log_norm_nb(a)
    if a < 0:
        u = mode / error
        normal_height = a * u - 0.5 * u * u - math.log(error) - 0.5 * _LOG_2PI - log_norm
    else:
        # at the interior mode z = (S - s - beta) * e / (s + mode + e^2), as in _channel_loglike
        z = ((count - source) - background) / (mean + variance) * error if mode > 0 else -a
        normal_height = -0.5 * z * z - math.log(error) - 0.5 * _LOG_2PI - log_norm
    logp = _poisson_logpmf_nb(count, mean) + normal_height + math.log(scale) + math.log(integral)
    if not math.isfinite(logp) or logp > 1e-10:
        return np.nan
    return logp


@nb.njit(cache=True)
def _pgstat_marg_core(counts, source_counts, background_counts, background_errors, nodes, weights):
    """Pointwise log p(S | m, B) in source-count units."""

    n = counts.shape[0]
    pointwise_loglike = np.empty(n, dtype=np.float64)
    for i in range(n):
        pointwise_loglike[i] = _channel_loglike_nb(
            counts[i],
            source_counts[i],
            background_counts[i],
            background_errors[i],
            nodes,
            weights,
        )
    return pointwise_loglike


@nb.njit(cache=True)
def _channel_saturated_nb(count, background, error, nodes, weights):
    """Golden-section maximum of log p(S | s, B) over source counts 0 <= s <= S.

    log p is concave in s, and its stationary point cannot exceed S because
    E[S / (s + x)] = 1 with x >= 0 needs s <= S.
    """
    lower = 0.0
    upper = count
    best_source = 0.0
    best = _channel_loglike_nb(count, 0.0, background, error, nodes, weights)
    if upper > 0:
        inverse_phi = 0.6180339887498949
        x1 = upper - inverse_phi * (upper - lower)
        x2 = lower + inverse_phi * (upper - lower)
        f1 = _channel_loglike_nb(count, x1, background, error, nodes, weights)
        f2 = _channel_loglike_nb(count, x2, background, error, nodes, weights)
        for _ in range(200):
            if upper - lower <= 1e-12 * max(1.0, upper):
                break
            if f1 < f2:
                lower, x1, f1 = x1, x2, f2
                x2 = lower + inverse_phi * (upper - lower)
                f2 = _channel_loglike_nb(count, x2, background, error, nodes, weights)
            else:
                upper, x2, f2 = x2, x1, f1
                x1 = upper - inverse_phi * (upper - lower)
                f1 = _channel_loglike_nb(count, x1, background, error, nodes, weights)
        for source, value in ((x1, f1), (x2, f2)):
            if value > best:
                best, best_source = value, source
        if math.isnan(f1) or math.isnan(f2):
            best = np.nan
    return best, best_source


@nb.njit(cache=True)
def _pgstat_marg_saturated_core(counts, background_counts, background_errors, nodes, weights):
    """Per-channel saturated log likelihood and the source counts that reach it."""

    n = counts.shape[0]
    log_saturated = np.empty(n, dtype=np.float64)
    source_saturated = np.empty(n, dtype=np.float64)
    for i in range(n):
        log_saturated[i], source_saturated[i] = _channel_saturated_nb(
            counts[i], background_counts[i], background_errors[i], nodes, weights
        )
    return log_saturated, source_saturated
