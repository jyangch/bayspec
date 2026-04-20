"""Container for posterior samples of a single parameter."""

import numpy as np



class Post(object):
    """Posterior sample and its summary statistics for one parameter.

    Holds a 1D array of posterior draws together with an optional matching
    array of log-probabilities, and exposes common point estimates and
    credible intervals.

    Attributes:
        sample: 1D array of posterior draws.
        logprob: Matching log-probability for each draw, or ``None``.
    """

    def __init__(self, sample, logprob=None):
        """Store ``sample`` and optionally its log-probabilities.

        Args:
            sample: 1D array of posterior draws.
            logprob: Matching log-probability for each draw. Required for
                ``best`` to be defined.
        """

        self._sample = sample
        self._logprob = logprob
        
        
    @property
    def sample(self):
        
        return self._sample
    
    
    @sample.setter
    def sample(self, new_sample):
        
        if not (np.ndim(new_sample) == 1):
            raise ValueError('sample must be 1D arrays')
        
        else:
            self._sample = new_sample
            
            
    @property
    def logprob(self):
        
        return self._logprob
    
    
    @logprob.setter
    def logprob(self, new_logprob):
        
        if not (np.ndim(new_logprob) == 1):
            raise ValueError('logprob must be 1D arrays')
        
        else:
            self._logprob = new_logprob
            
            
    @property
    def nsample(self):
        """Number of posterior draws."""

        return self.sample.shape[0]


    @property
    def mean(self):
        """Sample mean of the posterior draws."""

        return np.mean(self.sample)


    @property
    def median(self):
        """Sample median of the posterior draws."""

        return np.median(self.sample)


    @property
    def best(self):
        """Draw with the highest log-probability, or ``None`` if unavailable."""

        if self.logprob is None:
            return None

        else:
            return self.sample[np.argmax(self.logprob)]
        
        
    @property
    def best_ci(self):
        
        return getattr(self, '_best_ci', None)
    
    
    @best_ci.setter
    def best_ci(self, new_best_ci):
        
        self._best_ci = new_best_ci
    
    
    @property
    def truth(self):
        
        return getattr(self, '_truth', None)
    
    
    @truth.setter
    def truth(self, new_truth):
        
        self._truth = new_truth
    
    
    def quantile(self, q):
        """Return the ``q``-quantile of the sample.

        Args:
            q: Probability (or array of probabilities) in ``[0, 1]``.

        Returns:
            The corresponding sample quantile.
        """

        return np.quantile(self.sample, q)


    def interval(self, q):
        """Return the central credible interval containing probability ``q``.

        Args:
            q: Requested central probability mass, between 0 and 1.

        Returns:
            ``[low, high]`` bounding the central interval.
        """

        return np.quantile(self.sample, [0.5 - q / 2, 0.5 + q / 2]).tolist()


    @property
    def Isigma(self):
        """One-sigma (68.27%) central credible interval."""

        return self.interval(68.27 / 100)


    @property
    def IIsigma(self):
        """Two-sigma (95.45%) central credible interval."""

        return self.interval(95.45 / 100)


    @property
    def IIIsigma(self):
        """Three-sigma (99.73%) central credible interval."""

        return self.interval(99.73 / 100)


    def error(self, par, q=0.6827):
        """Return the asymmetric errors of ``par`` against the ``q``-interval.

        Args:
            par: Reference point estimate (e.g. best fit or median).
            q: Central credible level. Defaults to one sigma.

        Returns:
            ``[lower_error, upper_error]`` — signed differences between
            ``par`` and the interval endpoints.
        """

        ci = self.interval(q)

        return np.diff([ci[0], par, ci[1]]).tolist()


    @property
    def info(self):
        """Dictionary of mean, median, best, and one-sigma interval."""

        return dict([('mean', self.mean),
                     ('median', self.median),
                     ('best', self.best),
                     ('Isigma', self.Isigma)])
