"""Post-fit analyzers for posterior samples and bootstrap ensembles.

:class:`SampleAnalyzer` absorbs an :class:`~bayspec.infer.infer.Infer`
instance, reads a 2D parameter-sample matrix, evaluates each draw, attaches
the draws to every free parameter's :class:`~bayspec.util.post.Post`, and
exposes point estimates, credible intervals, and model-selection scores.
Posterior analyzers also eagerly calculate WAIC and PSIS-LOO.

:class:`Posterior` and :class:`Bootstrap` are thin subclasses that pick
which attribute of the underlying ``Infer`` carries the sample matrix.
"""

from collections import OrderedDict
import os
import warnings

import arviz as az
import numpy as np
from scipy.special import logsumexp

from ..util.info import Info
from ..util.post import Post
from ..util.tools import json_dump, memoized
from .infer import BayesInfer, Infer, MaxLikeFit


class SampleAnalyzer(Infer):
    """Wrap an :class:`Infer` with posterior-/bootstrap-driven summary views.

    The sample matrix is expected to have shape ``(nsample, nfree)`` and
    contain parameter draws only. Subclasses set :attr:`sample_attribute`
    to the instance attribute (``posterior_sample`` or ``bootstrap_sample``)
    that stores the matrix. Pointwise log-likelihood, total log-likelihood,
    log-prior, and log-probability samples are evaluated eagerly.

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
        """Absorb ``new_infer``, then load its sample matrix and wire up posteriors.

        Raises:
            TypeError: If ``new_infer`` is not an :class:`Infer`.
            AttributeError/ValueError: If the sample matrix is missing or
                has the wrong shape.
        """

        if not isinstance(new_infer, Infer):
            raise TypeError('expected an instance of Infer')

        self._infer = new_infer
        self.__dict__.update(new_infer.__dict__)

        self._check_sample()
        self._allot_post()

    def _check_sample(self):
        """Load parameter draws and eagerly evaluate their probability terms."""

        if self.sample_attribute is None:
            raise AttributeError('sample_attribute is not defined')

        self.sample = getattr(self, self.sample_attribute, None)
        if self.sample is None:
            raise AttributeError(f'{self.sample_attribute} is not available')

        self.sample = np.asarray(self.sample, dtype=float)
        if self.sample.ndim != 2:
            raise ValueError(f'{self.sample_attribute} is expected to be a 2D array')

        if self.sample.shape[1] != self.free_nparams:
            raise ValueError(
                f'{self.sample_attribute} is expected to have {self.free_nparams} columns'
            )

        self.param_sample = self.sample.copy()

        par_now = [par.val for par in self.free_par.values()]
        try:
            self.pointwise_loglike_sample = self._calc_pointwise_loglike_sample()
            self.logprior_sample = self._calc_logprior_sample()
        finally:
            self.at_par(par_now)

        self.loglike_sample = np.sum(self.pointwise_loglike_sample, axis=1)
        self.logprob_sample = self.loglike_sample + self.logprior_sample

    def _calc_pointwise_loglike_sample(self):
        """Evaluate fitted-channel log-likelihoods for every parameter draw."""

        return np.vstack([self.calc_pointwise_loglike(theta) for theta in self.param_sample])

    def _calc_logprior_sample(self):
        """Evaluate the joint log-prior for every parameter draw."""

        return np.asarray([self.calc_logprior(theta) for theta in self.param_sample], dtype=float)

    @property
    def _ranking_sample(self):
        """Per-draw score used for ``Post.best`` and best-CI selection."""

        raise NotImplementedError

    def _allot_post(self):
        """Attach a :class:`Post` to every free parameter and seed the best-fit CI."""

        for i in range(self.free_nparams):
            self.free_par[i + 1].post = Post(self.param_sample[:, i], self._ranking_sample)

        self._allot_best_ci(q=0.6827)
        self.at_par(self.par_best)

    def _allot_best_ci(self, q=0.6827):
        """Pick the highest-probability draw that lies within every ``q``-interval.

        Args:
            q: Central credible level (0-1) that the chosen draw must
                satisfy on every dimension simultaneously.
        """

        argsort = np.argsort(self._ranking_sample)[::-1]
        sort_param_sample = self.param_sample[argsort]

        for sample in sort_param_sample:
            if np.array(
                [(ci[0] <= sample[i] <= ci[1]) for i, ci in enumerate(self.par_interval(q))]
            ).all():
                for par, value in zip(self.free_par.values(), sample, strict=False):
                    par.post.best_ci = value

                break

    @staticmethod
    def _reshape_draws(values, sampler_type, nwalkers=None):
        """Arrange flattened samples as ``(chain, draw, ...)`` for ArviZ."""

        values = np.asarray(values)
        if sampler_type != 'mcmc':
            return values[None, ...]

        if nwalkers is None or values.shape[0] % nwalkers != 0:
            raise ValueError('flattened emcee samples are incompatible with mcmc_chain')

        ndraw = values.shape[0] // nwalkers
        shape = (ndraw, nwalkers, *values.shape[1:])

        return values.reshape(shape).swapaxes(0, 1)

    @property
    def par_statistic(self):
        """Mean, median, and 1/2/3-sigma intervals of :attr:`param_sample`."""

        mean = np.mean(self.param_sample, axis=0)
        median = np.median(self.param_sample, axis=0)

        q = 68.27 / 100
        Isigma = np.quantile(self.param_sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)

        q = 95.45 / 100
        IIsigma = np.quantile(self.param_sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)

        q = 99.73 / 100
        IIIsigma = np.quantile(self.param_sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)

        q = 90 / 100
        ninety_percent = np.quantile(self.param_sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)

        return dict(
            [
                ('mean', mean),
                ('median', median),
                ('Isigma', Isigma),
                ('IIsigma', IIsigma),
                ('IIIsigma', IIIsigma),
                ('90%', ninety_percent),
            ]
        )

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
        """Per-parameter draws with the highest analyzer ranking score."""

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

    @property
    def par_ninety_percent(self):
        """Per-parameter 90% credible interval."""

        return [par.post.ninety_percent for par in self.free_par.values()]

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

        return [np.diff([c[0], p, c[1]]).tolist() for p, c in zip(par, ci, strict=False)]

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

        return self.aic + 2 * self.free_nparams * (self.free_nparams + 1) / (
            self.npoint - self.free_nparams - 1
        )

    @property
    def bic(self):
        """Bayesian information criterion ``BIC = -2 ln L + k ln n``."""

        return -2 * self.max_loglike + self.free_nparams * np.log(self.npoint)

    def to_arviz(self):
        """Return posterior draws and pointwise likelihood as ArviZ data."""

        sampler_type = getattr(self, 'sampler_type', 'independent')
        nwalkers = self.mcmc_chain.shape[1] if sampler_type == 'mcmc' else None
        reshaped_param_sample = self._reshape_draws(self.param_sample, sampler_type, nwalkers)
        reshaped_loglikel_sample = self._reshape_draws(
            self.pointwise_loglike_sample, sampler_type, nwalkers
        )

        return az.from_dict(
            posterior={'theta': reshaped_param_sample},
            log_likelihood={'obs': reshaped_loglikel_sample},
            coords={
                'parameter': list(self.clean_free_indexed_plabels),
                'observation': np.arange(reshaped_loglikel_sample.shape[-1]),
            },
            dims={'theta': ['parameter'], 'obs': ['observation']},
        )

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
        free_params['1sigma CI'] = ['[{:.3f}, {:.3f}]'.format(*tuple(ci)) for ci in self.par_Isigma]
        free_params['90% CI'] = [
            '[{:.3f}, {:.3f}]'.format(*tuple(ci)) for ci in self.par_ninety_percent
        ]

        return Info.from_dict(free_params)

    @property
    def stat_info(self):
        """Tabular :class:`Info` of the fit statistic evaluated at the best fit."""

        self.at_par(self.par_best)

        all_stat = self.all_stat.copy()

        return Info.from_dict(all_stat)

    @property
    def all_IC(self):
        """Ordered dictionary of AIC/AICc/BIC values."""

        all_IC = OrderedDict()
        all_IC['AIC'] = self.aic
        all_IC['AICc'] = self.aicc
        all_IC['BIC'] = self.bic

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

        json_dump(
            self.free_par_info.data_list_dict, savepath + f'/{self.save_prefix}_free_par.json'
        )
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

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='For one or more samples the posterior variance.*',
                category=UserWarning,
            )
            warnings.filterwarnings(
                'ignore',
                message='Estimated shape parameter of Pareto distribution.*',
                category=UserWarning,
            )
            warnings.filterwarnings(
                'ignore',
                message=r'^overflow encountered in .*',
                category=RuntimeWarning,
            )

            self.waic()
            self.loo()

    @property
    def _ranking_sample(self):
        """Log-posterior values used to rank posterior draws."""

        return self.logprob_sample

    def _warn_power_likelihood(self):
        """Warn when predictive criteria describe a weighted power likelihood."""

        if self.has_nonunit_weights:
            warnings.warn(
                'WAIC/LOO with non-unit data weights describes a power likelihood; '
                'the usual independent-observation interpretation does not apply literally.',
                UserWarning,
                stacklevel=3,
            )

    @staticmethod
    def _format_ic(value, error):
        """Format an information criterion with its standard error."""

        if value is None:
            return None

        if error is None:
            return f'{value:.3f}'

        return f'{value:.3f} ± {error:.3f}'

    @memoized()
    def waic(self, scale='log', pointwise=True):
        """Compute WAIC from fitted-channel log-likelihoods using ArviZ.

        The default full result is calculated during initialization. Results
        are cached separately for each normalized argument combination.

        Args:
            scale: ArviZ output scale: ``'log'``, ``'negative_log'``, or
                ``'deviance'``. Log scale is larger-is-better.
            pointwise: Include the per-channel ``waic_i`` values.
        """

        self._warn_power_likelihood()

        return az.waic(self.to_arviz(), var_name='obs', scale=scale, pointwise=pointwise)

    @memoized()
    def loo(self, scale='log', pointwise=True, reff=None):
        """Compute PSIS-LOO and Pareto-k diagnostics using ArviZ.

        The default full result is calculated during initialization. Results
        are cached separately for each normalized argument combination. Channels
        with numerically constant likelihood use raw importance sampling because
        their zero-width weight distribution has no Pareto tail to fit.

        Args:
            scale: ArviZ output scale: ``'log'``, ``'negative_log'``, or
                ``'deviance'``. Log scale is larger-is-better.
            pointwise: Include per-channel ``loo_i`` and ``pareto_k`` values.
            reff: Relative MCMC efficiency. Equal-weight MultiNest samples
                default to one; emcee samples let ArviZ estimate it.
        """

        self._warn_power_likelihood()

        if reff is None and getattr(self, 'sampler_type', None) == 'nested':
            reff = 1.0

        idata = self.to_arviz()
        loglike = np.asarray(self.pointwise_loglike_sample, dtype=float)
        degenerate = np.ptp(loglike, axis=0) <= np.sqrt(np.finfo(float).eps)

        if not degenerate.any():
            return az.loo(
                idata,
                var_name='obs',
                scale=scale,
                pointwise=pointwise,
                reff=reff,
            )

        scale = scale.lower()
        scale_value = {'log': 1, 'negative_log': -1, 'deviance': -2}.get(scale)
        if scale_value is None:
            raise TypeError('Valid scale values are "deviance", "log", "negative_log"')

        warnings.warn(
            f'PSIS tail fitting was skipped for {degenerate.sum()} channels with nearly '
            'constant log-likelihood; raw importance weights were used and their Pareto-k '
            'values are undefined.',
            UserWarning,
            stacklevel=2,
        )

        n_samples, n_data_points = loglike.shape
        good = ~degenerate
        loo_i_values = np.empty(n_data_points, dtype=float)
        pareto_k_values = np.full(n_data_points, np.nan, dtype=float)
        p_loo = 0.0
        warn_mg = False

        if good.any():
            good_result = az.loo(
                idata.isel(observation=np.flatnonzero(good)),
                var_name='obs',
                scale=scale,
                pointwise=True,
                reff=reff,
            )
            loo_i_values[good] = np.asarray(good_result.loo_i)
            pareto_k_values[good] = np.asarray(good_result.pareto_k)
            p_loo += good_result.p_loo
            warn_mg = bool(good_result.warning)

        degenerate_loglike = loglike[:, degenerate].T
        log_n_samples = np.log(n_samples)
        raw_loo_i = -(logsumexp(-degenerate_loglike, axis=1) - log_n_samples)
        lppd_i = logsumexp(degenerate_loglike, axis=1) - log_n_samples
        loo_i_values[degenerate] = scale_value * raw_loo_i
        p_loo += np.sum(lppd_i - raw_loo_i)

        elpd_loo = np.sum(loo_i_values)
        loo_se = np.sqrt(n_data_points * np.var(loo_i_values))
        good_k = min(1 - 1 / np.log10(n_samples), 0.7)

        if not pointwise:
            return az.ELPDData(
                data=[
                    elpd_loo,
                    loo_se,
                    p_loo,
                    n_samples,
                    n_data_points,
                    warn_mg,
                    scale,
                    good_k,
                ],
                index=[
                    'elpd_loo',
                    'se',
                    'p_loo',
                    'n_samples',
                    'n_data_points',
                    'warning',
                    'scale',
                    'good_k',
                ],
            )

        pointwise_template = idata.log_likelihood['obs'].isel(chain=0, draw=0, drop=True)
        loo_i = pointwise_template.copy(data=loo_i_values).rename('loo_i')
        pareto_k = pointwise_template.copy(data=pareto_k_values).rename('pareto_shape')

        return az.ELPDData(
            data=[
                elpd_loo,
                loo_se,
                p_loo,
                n_samples,
                n_data_points,
                warn_mg,
                loo_i,
                pareto_k,
                scale,
                good_k,
            ],
            index=[
                'elpd_loo',
                'se',
                'p_loo',
                'n_samples',
                'n_data_points',
                'warning',
                'loo_i',
                'pareto_k',
                'scale',
                'good_k',
            ],
        )

    @property
    def lnZ(self):
        """Log-evidence supplied by the nested sampler, or ``None``."""

        return getattr(self, 'logevidence', None)

    @property
    def lnZ_err(self):
        """Nested-sampler uncertainty on :attr:`lnZ`, or ``None``."""

        return getattr(self, 'logevidence_err', None)

    @property
    def all_IC(self):
        """AIC-family scores, predictive information criteria, and evidence."""

        waic = self.waic()
        loo = self.loo()

        all_IC = super().all_IC
        all_IC['WAIC'] = self._format_ic(-2.0 * waic.elpd_waic, 2.0 * waic.se)
        all_IC['LOOIC'] = self._format_ic(-2.0 * loo.elpd_loo, 2.0 * loo.se)
        all_IC['lnZ'] = self._format_ic(self.lnZ, self.lnZ_err)

        return all_IC


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

    @property
    def _ranking_sample(self):
        """Log-likelihood values used to rank bootstrap draws."""

        return self.loglike_sample

    def _allot_post(self):
        """Attach a :class:`Post` to every free parameter, plus best-CI and truth."""

        super()._allot_post()
        self._allot_truth()
        self.at_par(self.par_truth)

    def _allot_truth(self):
        """Store the first bootstrap row (the best fit) as each parameter's truth."""

        for par, value in zip(self.free_par.values(), self.param_sample[0].tolist(), strict=False):
            par.post.truth = value

    @property
    def max_loglike(self):
        """Log-likelihood evaluated at the best-fit truth vector."""

        self.at_par(self.par_truth)

        return self.loglike
