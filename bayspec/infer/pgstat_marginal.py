"""Standalone Poisson likelihood with a marginalized Gaussian background.

This module is deliberately not registered as a BaySpec statistic. It changes
neither XSPEC-compatible profile PG-stat nor any fitting/WAIC/evidence pipeline.

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
returning an unconverged likelihood. This is an accuracy-oriented reference
implementation, not a speed-optimized replacement for profile PG-stat.

Example::

    from bayspec.infer.pgstat_marginal import pgstat_marginal_loglike
    loglike_i = pgstat_marginal_loglike(
        S=[12, 0], B=[8, -0.2], m=[0.4, 0.1],
        ts=10.0, tb=20.0, sigma_B=[2.0, 0.5],
    )
    loglike = loglike_i.sum()
"""

import math

import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx, gammaln, log_ndtr

__all__ = ['pgstat_marginal_loglike']

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


def pgstat_marginal_loglike(S, B, m, ts, tb, sigma_B, *, rtol=1e-10):
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

    output = np.empty(counts.shape, dtype=float)
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
