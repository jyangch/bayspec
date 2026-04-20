"""Post-fit analyzers for posterior samples and bootstrap ensembles.

:class:`SampleAnalyzer` absorbs an :class:`~bayspec.infer.infer.Infer`
instance, reads a 2D sample matrix (``param_sample`` plus a trailing
log-probability column), attaches the draws to every free parameter's
:class:`~bayspec.util.post.Post`, and exposes point estimates, credible
intervals, and model-selection scores (AIC/AICc/BIC, optionally ``lnZ``).

:class:`Posterior` and :class:`Bootstrap` are thin subclasses that pick
which attribute of the underlying ``Infer`` carries the sample matrix.
"""

import os
import numpy as np
from collections import OrderedDict

from .infer import Infer, BayesInfer, MaxLikeFit
from ..util.info import Info
from ..util.post import Post
from ..util.tools import json_dump



class SampleAnalyzer(Infer):
    """Wrap an :class:`Infer` with posterior-/bootstrap-driven summary views.

    The sample matrix is expected to have shape ``(nsample, nfree + 1)``,
    where the last column holds the log-probability associated with each
    draw. Subclasses set :attr:`sample_attribute` to the instance
    attribute (``posterior_sample`` or ``bootstrap_sample``) that stores
    the matrix.

    Attributes:
        sample_attribute: Name of the source ``Infer`` attribute; set by
            each subclass.
        analyzer_type: Display label shown in :meth:`__str__`.
        save_prefix: File-name prefix used by :meth:`save`.
    """

    sample_attribute = None
    analyzer_type = 'Sample Analysis Results'
    save_prefix = 'sample'

    def __init__(self, infer):
        """Absorb ``infer`` and populate the free-parameter posteriors.

        Args:
            infer: An :class:`Infer` instance whose sample matrix is
                stored under :attr:`sample_attribute`.

        Raises:
            TypeError: If ``infer`` is not an :class:`Infer`.
            AttributeError/ValueError: If the sample matrix is missing
                or has the wrong shape.
        """

        self.infer = infer
        
        
    @property
    def infer(self):
        
        return self._infer
    
    
    @infer.setter
    def infer(self, new_infer):
        
        if not isinstance(new_infer, Infer):
            raise TypeError('expected an instance of Infer')
        
        self._infer = new_infer
        self.__dict__.update(new_infer.__dict__)
        
        self._check_sample()
        self._allot_post()
    
    
    def _check_sample(self):
        """Load and validate the sample matrix from :attr:`sample_attribute`."""

        if self.sample_attribute is None:
            raise AttributeError('sample_attribute is not defined')
        
        self.sample = getattr(self, self.sample_attribute, None)
        if self.sample is None:
            raise AttributeError(f'{self.sample_attribute} is not available')
        
        self.sample = np.asarray(self.sample, dtype=float)
        if self.sample.ndim != 2:
            raise ValueError(f'{self.sample_attribute} is expected to be a 2D array')
        
        if self.sample.shape[1] != self.free_nparams + 1:
            raise ValueError(f'{self.sample_attribute} is expected to have {self.free_nparams + 1} columns')
        
        self.param_sample = self.sample[:, :self.free_nparams].copy()
        self.prob_sample = self.sample[:, -1].copy()


    def _allot_post(self):
        """Attach a :class:`Post` to every free parameter and seed the best-fit CI."""

        for i in range(self.free_nparams):
            self.free_par[i+1].post = Post(self.param_sample[:, i], self.prob_sample)

        self._allot_best_ci(q=0.6827)
        self.at_par(self.par_best)


    def _allot_best_ci(self, q=0.6827):
        """Pick the highest-probability draw that lies within every ``q``-interval.

        Args:
            q: Central credible level (0–1) that the chosen draw must
                satisfy on every dimension simultaneously.
        """

        argsort = np.argsort(self.prob_sample)[::-1]
        sort_param_sample = self.param_sample[argsort]
        
        for sample in sort_param_sample:
            if np.array([True if (ci[0] <= sample[i] <= ci[1]) else False \
                for i, ci in enumerate(self.par_interval(q))]).all():
                
                for par, value in zip(self.free_par.values(), sample):
                    par.post.best_ci = value
                
                break


    @property
    def sample_statistic(self):
        """Mean, median, and 1/2/3-sigma intervals of :attr:`param_sample`."""

        mean = np.mean(self.param_sample, axis=0)
        median = np.median(self.param_sample, axis=0)
        
        q = 68.27 / 100
        Isigma = np.quantile(self.param_sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        
        q = 95.45 / 100
        IIsigma = np.quantile(self.param_sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        
        q = 99.73 / 100
        IIIsigma = np.quantile(self.param_sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        
        return dict([('mean', mean), 
                     ('median', median), 
                     ('Isigma', Isigma), 
                     ('IIsigma', IIsigma), 
                     ('IIIsigma', IIIsigma)])


    @property
    def par_mean(self):
        """Per-parameter posterior means drawn from each :class:`Post`."""

        return [par.post.mean for par in self.free_par.values()]


    @property
    def par_median(self):
        """Per-parameter posterior medians."""

        return [par.post.median for par in self.free_par.values()]


    @property
    def par_best(self):
        """Per-parameter highest-log-probability draws."""

        return [par.post.best for par in self.free_par.values()]


    @property
    def par_best_ci(self):
        """Per-parameter draw selected via :meth:`_allot_best_ci`."""

        return [par.post.best_ci for par in self.free_par.values()]


    @property
    def par_truth(self):
        """Per-parameter truth value stored in each :class:`Post`, or ``None``."""

        return [par.post.truth for par in self.free_par.values()]


    def par_quantile(self, q):
        """Per-parameter ``q``-quantile of the posterior.

        Args:
            q: Probability or array of probabilities in ``[0, 1]``.
        """

        return [par.post.quantile(q) for par in self.free_par.values()]


    def par_interval(self, q):
        """Per-parameter central ``q``-credible interval."""

        return [par.post.interval(q) for par in self.free_par.values()]


    @property
    def par_Isigma(self):
        """Per-parameter one-sigma credible interval."""

        return [par.post.Isigma for par in self.free_par.values()]


    @property
    def par_IIsigma(self):
        """Per-parameter two-sigma credible interval."""

        return [par.post.IIsigma for par in self.free_par.values()]


    @property
    def par_IIIsigma(self):
        """Per-parameter three-sigma credible interval."""

        return [par.post.IIIsigma for par in self.free_par.values()]


    def par_error(self, par, q=0.6827):
        """Per-parameter asymmetric errors of ``par`` against the ``q``-interval.

        Args:
            par: Sequence of per-parameter point estimates.
            q: Central credible level.

        Returns:
            List of ``[lower_error, upper_error]`` pairs aligned with
            ``par``.
        """

        ci = self.par_interval(q)

        return [np.diff([c[0], p, c[1]]).tolist() for p, c in zip(par, ci)]
    
    
    @property
    def max_loglike(self):
        """Log-likelihood evaluated at the best-fit parameter vector."""

        self.at_par(self.par_best)

        return self.loglike


    @property
    def aic(self):
        """Akaike information criterion ``AIC = -2 ln L + 2 k``."""

        return -2 * self.max_loglike + 2 * self.free_nparams


    @property
    def aicc(self):
        """Finite-sample corrected AIC."""

        return self.aic + 2 * self.free_nparams * \
            (self.free_nparams + 1) / \
                (self.npoint - self.free_nparams - 1)


    @property
    def bic(self):
        """Bayesian information criterion ``BIC = -2 ln L + k ln n``."""

        return -2 * self.max_loglike + self.free_nparams * np.log(self.npoint)


    @property
    def lnZ(self):
        """Log-evidence supplied by the nested sampler, or ``None``."""

        return getattr(self, 'logevidence', None)


    @property
    def free_par_info(self):
        """Tabular :class:`Info` of free parameters with posterior summaries."""

        self._you_free()
        
        free_params = self.free_params.copy()
        
        free_params = Info.list_dict_to_dict(free_params)
        
        del free_params['Posterior']
        del free_params['Mates']
        del free_params['Frozen']
        del free_params['Prior']
        del free_params['Value']
        
        if None not in self.par_truth:
            free_params['Truth'] = [par for par in self.par_truth]
        
        free_params['Mean'] = [par for par in self.par_mean]
        free_params['Median'] = [par for par in self.par_median]
        free_params['Best'] = [par for par in self.par_best]
        free_params['1sigma Best'] = [par for par in self.par_best_ci]
        free_params['1sigma CI'] = ['[%.3f, %.3f]' % tuple(ci) for ci in self.par_Isigma]
        
        return Info.from_dict(free_params)
    
    
    @property
    def stat_info(self):
        """Tabular :class:`Info` of the fit statistic evaluated at the best fit."""

        self.at_par(self.par_best)

        all_stat = self.all_stat.copy()

        return Info.from_dict(all_stat)


    @property
    def all_IC(self):
        """Ordered dictionary of AIC/AICc/BIC/lnZ values."""

        all_IC = OrderedDict()
        all_IC['AIC'] = self.aic
        all_IC['AICc'] = self.aicc
        all_IC['BIC'] = self.bic
        all_IC['lnZ'] = self.lnZ

        return all_IC


    @property
    def IC_info(self):
        """Tabular :class:`Info` view of :attr:`all_IC`."""

        all_IC = self.all_IC.copy()

        return Info.from_dict(all_IC)


    def save(self, savepath):
        """Dump free-parameter, statistic, and IC tables under ``savepath``.

        Args:
            savepath: Directory path. Created if missing.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        json_dump(self.free_par_info.data_list_dict, savepath + f'/{self.save_prefix}_free_par.json')
        json_dump(self.stat_info.data_list_dict, savepath + f'/{self.save_prefix}_stat.json')
        json_dump(self.IC_info.data_list_dict, savepath + f'/{self.save_prefix}_IC.json')


    def __str__(self):
        
        return (
            f'*** {self.analyzer_type} ***\n'
            f'*** Parameters ***\n'
            f'{self.free_par_info.text_table}\n'
            f'*** Statistics ***\n'
            f'{self.stat_info.text_table}\n'
            f'*** Information Criterias ***\n'
            f'{self.IC_info.text_table}'
            )


    def __repr__(self):
        
        return self.__str__()
    
    
    def _repr_html_(self):
        
        return (
            f'{self.free_par_info.html_style}'
            f'<details open>'
            f'<summary style="margin-bottom: 10px;"><b>{self.analyzer_type}</b></summary>'
            f'<details open style="margin-top: 10px;">'
            f'<summary style="margin-bottom: 10px;"><b>Parameters</b></summary>'
            f'{self.free_par_info.html_table}'
            f'</details>'
            f'<details open style="margin-top: 10px;">'
            f'<summary style="margin-bottom: 10px;"><b>Statistics</b></summary>'
            f'{self.stat_info.html_table}'
            f'</details>'
            f'<details open style="margin-top: 10px;">'
            f'<summary style="margin-bottom: 10px;"><b>Information Criterias</b></summary>'
            f'{self.IC_info.html_table}'
            f'</details>'
            )



class Posterior(SampleAnalyzer):
    """Analyzer specialised for Bayesian posterior samples."""

    sample_attribute = 'posterior_sample'
    analyzer_type = 'Posterior Results'
    save_prefix = 'post'

    def __init__(self, infer):
        """Absorb a :class:`BayesInfer` and attach its ``posterior_sample``.

        Raises:
            TypeError: If ``infer`` is not a :class:`BayesInfer`.
        """

        if not isinstance(infer, BayesInfer):
            raise TypeError('expected an instance of BayesInfer')

        super().__init__(infer)



class Bootstrap(SampleAnalyzer):
    """Analyzer specialised for maximum-likelihood bootstrap ensembles.

    The first row of ``bootstrap_sample`` is treated as the best-fit
    truth; its value is copied onto each parameter's :class:`Post` so
    downstream consumers can access it as ``par.post.truth``.
    """

    sample_attribute = 'bootstrap_sample'
    analyzer_type = 'Bootstrap Results'
    save_prefix = 'boot'

    def __init__(self, infer):
        """Absorb a :class:`MaxLikeFit` and attach its ``bootstrap_sample``.

        Raises:
            TypeError: If ``infer`` is not a :class:`MaxLikeFit`.
        """

        if not isinstance(infer, MaxLikeFit):
            raise TypeError('expected an instance of MaxLikeFit')

        super().__init__(infer)


    def _allot_post(self):
        """Attach a :class:`Post` to every free parameter, plus best-CI and truth."""

        for i in range(self.free_nparams):
            self.free_par[i+1].post = Post(self.param_sample[:, i], self.prob_sample)

        self._allot_best_ci(q=0.6827)
        self._allot_truth()
        self.at_par(self.par_truth)


    def _allot_truth(self):
        """Store the first bootstrap row (the best fit) as each parameter's truth."""

        for par, value in zip(self.free_par.values(), self.param_sample[0].tolist()):
            par.post.truth = value


    @property
    def max_loglike(self):
        """Log-likelihood evaluated at the best-fit truth vector."""

        self.at_par(self.par_truth)

        return self.loglike