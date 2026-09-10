from importlib import import_module
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from bayspec.infer.analyzer import Posterior, SampleAnalyzer
from bayspec.infer.infer import BayesInfer
from bayspec.util.tools import SuperDict


class Value:
    def __init__(self, val):
        self.val = val
        self.post = SimpleNamespace(best=val)


def make_posterior(param_sample, pointwise, sampler_type='nested', nwalkers=None):
    post = object.__new__(Posterior)
    post.param_sample = np.asarray(param_sample, dtype=float)
    post.pointwise_loglike_sample = np.asarray(pointwise, dtype=float)
    post._free_nparams = post.param_sample.shape[1]
    post._free_plabels = [f'theta_{i + 1}' for i in range(post.free_nparams)]
    post._free_par = SuperDict((str(i + 1), Value(7.0)) for i in range(post.free_nparams))
    post.Pair = []
    post.sampler_type = sampler_type
    if nwalkers is not None:
        ndraw = post.param_sample.shape[0] // nwalkers
        post.mcmc_chain = post.param_sample.reshape(ndraw, nwalkers, -1)
    return post


def test_nested_samples_become_one_chain():
    pointwise = np.array([[-1.0, -2.0], [-1.5, -1.0], [-2.0, -1.5]])
    post = make_posterior([[0.0], [1.0], [2.0]], pointwise)

    idata = post.to_arviz()

    assert idata.posterior['theta'].shape == (1, 3, 1)
    assert idata.log_likelihood['obs'].shape == (1, 3, 2)
    np.testing.assert_allclose(idata.log_likelihood['obs'].values[0], pointwise)


def test_emcee_walkers_become_chains_and_steps_become_draws():
    params = np.arange(12.0).reshape(6, 2)
    pointwise = -np.arange(18.0).reshape(6, 3)
    post = make_posterior(params, pointwise, sampler_type='mcmc', nwalkers=2)

    idata = post.to_arviz()

    assert idata.posterior['theta'].shape == (2, 3, 2)
    assert idata.log_likelihood['obs'].shape == (2, 3, 3)
    np.testing.assert_allclose(
        idata.log_likelihood['obs'].values,
        pointwise.reshape(3, 2, 3).transpose(1, 0, 2),
    )


def test_waic_returns_arviz_elpd_on_log_scale():
    pointwise = np.array(
        [
            [-1.0, -2.0],
            [-2.0, -1.0],
            [-1.5, -1.5],
            [-1.2, -1.8],
        ]
    )
    post = make_posterior(np.arange(4.0)[:, None], pointwise)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = post.waic()

    assert result.scale == 'log'
    assert result.elpd_waic == pytest.approx(-3.1438605895230722)
    assert result.p_waic == pytest.approx(0.28375)
    assert result.waic_i.shape == (2,)


def test_loo_returns_psis_pointwise_values_and_pareto_k():
    arviz = pytest.importorskip('arviz')
    rng = np.random.default_rng(20260908)
    params = rng.normal(size=(200, 1))
    pointwise = -0.5 * (rng.normal(size=(200, 4)) ** 2)
    post = make_posterior(params, pointwise)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        actual = post.loo(reff=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected = arviz.loo(post.to_arviz(), var_name='obs', pointwise=True, scale='log', reff=1.0)

    assert actual.elpd_loo == pytest.approx(expected.elpd_loo)
    assert actual.p_loo == pytest.approx(expected.p_loo)
    np.testing.assert_allclose(actual.loo_i, expected.loo_i)
    np.testing.assert_allclose(actual.pareto_k, expected.pareto_k)
    assert actual.good_k == pytest.approx(expected.good_k)


def test_loo_uses_raw_weights_for_degenerate_psis_tail():
    delta = np.r_[
        np.zeros(80),
        np.full(5, 5e-18),
        np.geomspace(5e-17, 1e-12, 15),
    ]
    degenerate = -1e-3 - delta
    regular = -0.5 * np.linspace(-2.0, 2.0, 100) ** 2
    post = make_posterior(
        np.arange(100.0)[:, None],
        np.column_stack([degenerate, regular]),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = post.loo(reff=1.0)

    assert np.isfinite(result.elpd_loo)
    assert np.asarray(result.loo_i)[0] == pytest.approx(-0.001, abs=1e-12)
    assert np.isnan(np.asarray(result.pareto_k)[0])
    assert any('nearly constant' in str(item.message) for item in caught)
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)


def test_loo_preserves_runtime_warnings_outside_initialization(monkeypatch):
    post = make_posterior([[0.0], [1.0]], [[-1.0, -1.5], [-2.0, -1.0]])
    analyzer_module = import_module('bayspec.infer.analyzer')

    def loo_with_unrelated_warning(*args, **kwargs):
        warnings.warn('overflow encountered in multiply', RuntimeWarning, stacklevel=2)
        return object()

    monkeypatch.setattr(analyzer_module.az, 'loo', loo_with_unrelated_warning)

    with pytest.warns(RuntimeWarning, match='overflow encountered in multiply'):
        post.loo(reff=1.0)


def test_predictive_criteria_warn_for_power_likelihood():
    post = make_posterior([[0.0], [1.0]], [[-1.0, -1.5], [-2.0, -1.0]])
    post.Pair = [type('WeightedPair', (), {'has_nonunit_weights': True})()]

    with pytest.warns(UserWarning, match='non-unit data weights'):
        post.waic()


def test_posterior_initialization_eagerly_computes_default_criteria(monkeypatch):
    calls = []
    infer = object.__new__(BayesInfer)

    monkeypatch.setattr(SampleAnalyzer, '__init__', lambda self, value: None)
    monkeypatch.setattr(Posterior, 'waic', lambda self: calls.append('waic'))
    monkeypatch.setattr(Posterior, 'loo', lambda self: calls.append('loo'))

    Posterior(infer)

    assert calls == ['waic', 'loo']


def test_posterior_initialization_suppresses_arviz_predictive_warnings(monkeypatch):
    infer = object.__new__(BayesInfer)

    monkeypatch.setattr(SampleAnalyzer, '__init__', lambda self, value: None)
    monkeypatch.setattr(
        Posterior,
        'waic',
        lambda self: warnings.warn(
            'For one or more samples the posterior variance exceeds 0.4.',
            UserWarning,
            stacklevel=2,
        ),
    )
    monkeypatch.setattr(
        Posterior,
        'loo',
        lambda self: warnings.warn(
            'Estimated shape parameter of Pareto distribution is greater than 0.69.',
            UserWarning,
            stacklevel=2,
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        Posterior(infer)

    assert caught == []


def test_posterior_initialization_suppresses_all_overflow_warnings(monkeypatch):
    infer = object.__new__(BayesInfer)

    monkeypatch.setattr(SampleAnalyzer, '__init__', lambda self, value: None)
    monkeypatch.setattr(
        Posterior,
        'waic',
        lambda self: warnings.warn(
            'overflow encountered in exp', RuntimeWarning, stacklevel=2
        ),
    )

    def loo_with_overflows(self):
        warnings.warn('overflow encountered in reduce', RuntimeWarning, stacklevel=2)
        warnings.warn('overflow encountered in multiply', RuntimeWarning, stacklevel=2)

    monkeypatch.setattr(Posterior, 'loo', loo_with_overflows)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        Posterior(infer)

    assert caught == []


def test_posterior_initialization_preserves_non_overflow_runtime_warnings(monkeypatch):
    infer = object.__new__(BayesInfer)

    monkeypatch.setattr(SampleAnalyzer, '__init__', lambda self, value: None)
    monkeypatch.setattr(
        Posterior,
        'waic',
        lambda self: warnings.warn(
            'invalid value encountered in sqrt', RuntimeWarning, stacklevel=2
        ),
    )
    monkeypatch.setattr(Posterior, 'loo', lambda self: None)

    with pytest.warns(RuntimeWarning, match='invalid value encountered in sqrt'):
        Posterior(infer)


def test_posterior_initialization_preserves_unrelated_user_warnings(monkeypatch):
    infer = object.__new__(BayesInfer)

    monkeypatch.setattr(SampleAnalyzer, '__init__', lambda self, value: None)
    monkeypatch.setattr(
        Posterior,
        'waic',
        lambda self: warnings.warn('unrelated diagnostic', UserWarning, stacklevel=2),
    )
    monkeypatch.setattr(Posterior, 'loo', lambda self: None)

    with pytest.warns(UserWarning, match='unrelated diagnostic'):
        Posterior(infer)


def test_predictive_criteria_are_cached_by_normalized_arguments(monkeypatch):
    rng = np.random.default_rng(20260908)
    post = make_posterior(
        rng.normal(size=(200, 1)),
        -0.5 * rng.normal(size=(200, 4)) ** 2,
    )
    calls = {'waic': 0, 'loo': 0}
    analyzer_module = import_module('bayspec.infer.analyzer')
    arviz_waic = __import__('arviz').waic
    arviz_loo = __import__('arviz').loo

    def count_waic(*args, **kwargs):
        calls['waic'] += 1
        return arviz_waic(*args, **kwargs)

    def count_loo(*args, **kwargs):
        calls['loo'] += 1
        return arviz_loo(*args, **kwargs)

    monkeypatch.setattr(analyzer_module.az, 'waic', count_waic)
    monkeypatch.setattr(analyzer_module.az, 'loo', count_loo)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        waic = post.waic()
        loo = post.loo()
        deviance_waic = post.waic(scale='deviance', pointwise=True)
        deviance_loo = post.loo(scale='deviance', pointwise=True, reff=1.0)

        assert post.waic('log', True) is waic
        assert post.loo(scale='log', pointwise=True) is loo
        assert post.waic('deviance', True) is deviance_waic
        assert post.loo('deviance', True, 1.0) is deviance_loo

    assert calls == {'waic': 2, 'loo': 2}


def test_posterior_ic_info_includes_waic_and_looic():
    rng = np.random.default_rng(20260908)
    post = make_posterior(
        rng.normal(size=(200, 1)),
        -0.5 * rng.normal(size=(200, 4)) ** 2,
    )
    post.logevidence = -12.345
    post.logevidence_err = 0.678

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        all_ic = post.all_IC
        ic_info = post.IC_info

    assert list(all_ic) == ['AIC', 'AICc', 'BIC', 'WAIC', 'LOOIC', 'lnZ']
    assert list(ic_info.data_dict) == ['AIC', 'AICc', 'BIC', 'WAIC', 'LOOIC', 'lnZ']
    assert all_ic['WAIC'] == (
        f'{-2.0 * post.waic().elpd_waic:.3f} ± {2.0 * post.waic().se:.3f}'
    )
    assert all_ic['LOOIC'] == (
        f'{-2.0 * post.loo().elpd_loo:.3f} ± {2.0 * post.loo().se:.3f}'
    )
    assert all_ic['lnZ'] == '-12.345 ± 0.678'
    assert ic_info.data_dict['WAIC'][0] == all_ic['WAIC']
    assert ic_info.data_dict['LOOIC'][0] == all_ic['LOOIC']
    assert ic_info.data_dict['lnZ'][0] == all_ic['lnZ']


def test_posterior_ic_info_omits_uncertainty_when_evidence_is_unavailable():
    rng = np.random.default_rng(20260908)
    post = make_posterior(
        rng.normal(size=(200, 1)),
        -0.5 * rng.normal(size=(200, 4)) ** 2,
        sampler_type='mcmc',
        nwalkers=2,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert post.all_IC['lnZ'] is None
