"""Prior distributions used as parameter priors in Bayesian inference.

Wraps ``scipy.stats`` frozen distributions behind a uniform interface so
that the inference layer can query pdf, cdf, quantiles, and moments
without caring about the underlying distribution family.

Typical usage:
    from bayspec.util.prior import unif, norm
    p = unif(0.0, 1.0)
    p.pdf(0.5)
"""

from scipy import stats


class Prior(object):
    """Base wrapper around a ``scipy.stats`` frozen distribution.

    Subclasses pick a specific distribution family and expose it under a
    short expression tag. The ``stat`` attribute holds the frozen
    distribution; all statistical queries are delegated to it.

    Attributes:
        args: Positional parameters used to build the distribution.
        expr: Short tag identifying the distribution family.
        stat: Underlying frozen ``scipy.stats`` distribution.
    """

    def __init__(self, stat, *args):
        """Freeze ``stat`` with ``args`` and record both for introspection.

        Args:
            stat: A ``scipy.stats`` distribution factory.
            *args: Positional parameters forwarded to ``stat``.
        """

        self.args = args
        self.expr = 'stat'
        self.stat = stat(*args)


    def pdf(self, x):
        """Return the probability density evaluated at ``x``."""

        return self.stat.pdf(x)


    def logpdf(self, x):
        """Return the log-probability density evaluated at ``x``."""

        return self.stat.logpdf(x)


    def cdf(self, x):
        """Return the cumulative distribution evaluated at ``x``."""

        return self.stat.cdf(x)


    def logcdf(self, x):
        """Return the log of the cumulative distribution evaluated at ``x``."""

        return self.stat.logcdf(x)


    def ppf(self, q):
        """Return the quantile corresponding to lower-tail probability ``q``."""

        return self.stat.ppf(q)


    def interval(self, q):
        """Return the central interval that contains probability mass ``q``.

        Args:
            q: Requested probability mass, between 0 and 1.

        Returns:
            Tuple ``(low, high)`` bounding the central interval.
        """

        return self.stat.interval(q)


    @property
    def median(self):
        """Median of the distribution."""

        return self.stat.median()


    @property
    def mean(self):
        """Mean of the distribution."""

        return self.stat.mean()


    @property
    def var(self):
        """Variance of the distribution."""

        return self.stat.var()


    @property
    def std(self):
        """Standard deviation of the distribution."""

        return self.stat.std()


    @property
    def info(self):
        """Short description of the distribution family and its arguments."""

        return f'{self.expr}{self.args}'



class unif(Prior):
    """Uniform prior on the closed interval ``[min, max]``."""

    def __init__(self, min, max):
        """Build a uniform prior.

        Args:
            min: Lower bound of the support.
            max: Upper bound of the support.
        """

        self.args = (min, max)
        self.expr = 'unif'
        self.stat = stats.uniform(min, max - min)



class logunif(Prior):
    """Log-uniform (reciprocal) prior on ``[min, max]``."""

    def __init__(self, min, max, loc=0, scale=1):
        """Build a log-uniform prior.

        Args:
            min: Lower bound of the support.
            max: Upper bound of the support.
            loc: Location shift forwarded to ``scipy.stats.loguniform``.
            scale: Scale factor forwarded to ``scipy.stats.loguniform``.
        """

        self.args = (min, max, loc, scale)
        self.expr = 'logunif'
        self.stat = stats.loguniform(*self.args)



class norm(Prior):
    """Gaussian prior."""

    def __init__(self, loc, scale):
        """Build a Gaussian prior.

        Args:
            loc: Mean of the distribution.
            scale: Standard deviation of the distribution.
        """

        self.args = (loc, scale)
        self.expr = 'norm'
        self.stat = stats.norm(*self.args)



class lognorm(Prior):
    """Log-normal prior."""

    def __init__(self, s, loc, scale):
        """Build a log-normal prior.

        Args:
            s: Shape parameter (sigma of the underlying normal).
            loc: Location shift.
            scale: Scale factor.
        """

        self.args = (s, loc, scale)
        self.expr = 'lognorm'
        self.stat = stats.lognorm(*self.args)



class truncnorm(Prior):
    """Truncated Gaussian prior."""

    def __init__(self, a, b, loc, scale):
        """Build a truncated Gaussian prior.

        Args:
            a: Lower truncation bound, in standardized units.
            b: Upper truncation bound, in standardized units.
            loc: Mean of the underlying Gaussian.
            scale: Standard deviation of the underlying Gaussian.
        """

        self.args = (a, b, loc, scale)
        self.expr = 'truncnorm'
        self.stat = stats.truncnorm(*self.args)



class cauchy(Prior):
    """Cauchy (Lorentzian) prior."""

    def __init__(self, loc, scale):
        """Build a Cauchy prior.

        Args:
            loc: Location (median) of the distribution.
            scale: Half-width at half maximum.
        """

        self.args = (loc, scale)
        self.expr = 'cauchy'
        self.stat = stats.cauchy(*self.args)



class cosine(Prior):
    """Cosine-shaped prior on a bounded interval."""

    def __init__(self, loc, scale):
        """Build a cosine prior.

        Args:
            loc: Location shift.
            scale: Scale factor.
        """

        self.args = (loc, scale)
        self.expr = 'cosine'
        self.stat = stats.cosine(*self.args)



class beta(Prior):
    """Beta prior."""

    def __init__(self, a, b, loc, scale):
        """Build a Beta prior.

        Args:
            a: First shape parameter.
            b: Second shape parameter.
            loc: Location shift.
            scale: Scale factor.
        """

        self.args = (a, b, loc, scale)
        self.expr = 'beta'
        self.stat = stats.beta(*self.args)



class gamma(Prior):
    """Gamma prior."""

    def __init__(self, a, loc, scale):
        """Build a Gamma prior.

        Args:
            a: Shape parameter.
            loc: Location shift.
            scale: Scale factor.
        """

        self.args = (a, loc, scale)
        self.expr = 'gamma'
        self.stat = stats.gamma(*self.args)



class expon(Prior):
    """Exponential prior."""

    def __init__(self, loc, scale):
        """Build an exponential prior.

        Args:
            loc: Location shift.
            scale: Inverse-rate (mean) of the distribution.
        """

        self.args = (loc, scale)
        self.expr = 'expon'
        self.stat = stats.expon(*self.args)



class plaw(Prior):
    """Power-law prior."""

    def __init__(self, a, loc, scale):
        """Build a power-law prior.

        Args:
            a: Power-law index.
            loc: Location shift.
            scale: Scale factor.
        """

        self.args = (a, loc, scale)
        self.expr = 'plaw'
        self.stat = stats.powerlaw(*self.args)


all_priors = {name: cls for name, cls in globals().items()
              if isinstance(cls, type)
              and issubclass(cls, Prior)
              and name != 'Prior'}


def list_priors():
    """Return the names of every prior subclass registered in this module."""

    return list(all_priors.keys())


__all__ = list(all_priors.keys()) + ['list_priors', 'all_priors']
