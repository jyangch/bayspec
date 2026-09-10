from importlib import import_module
import json
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from scipy.special import logsumexp

from bayspec.infer.analyzer import Bootstrap, Posterior, SampleAnalyzer
from bayspec.infer.infer import BayesInfer
from bayspec.util.info import Info
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
        np.geomspace(5e-17, 1e-5, 15),
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
    assert np.asarray(result.loo_i)[0] == pytest.approx(-0.0010001184667300933)
    assert np.isnan(np.asarray(result.pareto_k)[0])
    assert result.warning
    assert any('PSIS-LOO failed' in str(item.message) for item in caught)
    assert not any('nearly constant' in str(item.message) for item in caught)
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)


@pytest.mark.parametrize('scale, factor', [('log', 1), ('negative_log', -1), ('deviance', -2)])
@pytest.mark.parametrize('pointwise', [True, False])
def test_loo_marks_failed_nonconstant_tail_as_unreliable(scale, factor, pointwise):
    loglike = -np.r_[np.zeros(80), np.linspace(0.01, 0.2, 19), 708.0]
    post = make_posterior(np.arange(100.0)[:, None], loglike[:, None])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = post.loo(scale=scale, pointwise=pointwise)

    assert result.elpd_loo == pytest.approx(factor * (-708.0 + np.log(100)))
    assert result.warning
    assert any('PSIS-LOO failed' in str(item.message) for item in caught)
    assert not any('nearly constant' in str(item.message) for item in caught)
    assert not any('Estimated shape parameter' in str(item.message) for item in caught)
    if pointwise:
        assert np.isnan(result.pareto_k[0])


def test_loo_nearly_constant_channels_do_not_signal_psis_failure():
    loglike = np.column_stack([np.full(100, -2.0), -3 + np.linspace(0, 1e-9, 100)])
    post = make_posterior(np.arange(100.0)[:, None], loglike)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = post.loo()

    np.testing.assert_allclose(result.loo_i, -logsumexp(-loglike, axis=0) + np.log(100))
    assert np.isnan(result.pareto_k).all()
    assert not result.warning
    assert any('nearly constant' in str(item.message) for item in caught)
    assert not any('PSIS-LOO failed' in str(item.message) for item in caught)


def test_loo_failure_warning_survives_initialization_and_cache(monkeypatch):
    loglike = np.column_stack(
        [
            -np.r_[np.zeros(80), np.linspace(0.01, 0.2, 19), 708.0],
            -0.5 * np.linspace(-1.0, 1.0, 100) ** 2,
        ]
    )
    prepared = make_posterior(np.arange(100.0)[:, None], loglike)
    monkeypatch.setattr(
        SampleAnalyzer, '__init__', lambda self, infer: self.__dict__.update(prepared.__dict__)
    )

    with pytest.warns(UserWarning, match='PSIS-LOO failed'):
        post = Posterior(object.__new__(BayesInfer))

    assert post.loo().warning
    assert post.loo() is post.loo()


def test_loo_preserves_infinite_pareto_k_as_unreliable():
    influential = np.r_[-10.0, np.zeros(99)]
    post = make_posterior(
        np.arange(100.0)[:, None],
        influential[:, None],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = post.loo(reff=1.0)

    assert np.isfinite(result.elpd_loo)
    assert np.isposinf(np.asarray(result.pareto_k)[0])
    assert result.warning
    assert any('Estimated shape parameter' in str(item.message) for item in caught)


def test_loo_preserves_runtime_warnings_outside_initialization(monkeypatch):
    rng = np.random.default_rng(20260910)
    post = make_posterior(
        rng.normal(size=(200, 1)),
        -0.5 * rng.normal(size=(200, 4)) ** 2,
    )
    analyzer_module = import_module('bayspec.infer.analyzer')
    arviz_loo = analyzer_module.az.loo

    def loo_with_unrelated_warning(*args, **kwargs):
        warnings.warn('overflow encountered in multiply', RuntimeWarning, stacklevel=2)
        return arviz_loo(*args, **kwargs)

    monkeypatch.setattr(analyzer_module.az, 'loo', loo_with_unrelated_warning)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='Estimated shape parameter of Pareto distribution.*',
            category=UserWarning,
        )
        with pytest.warns(RuntimeWarning, match='overflow encountered in multiply'):
            post.loo(reff=1.0)


def test_loo_suppresses_known_arviz_runtime_warnings(monkeypatch):
    rng = np.random.default_rng(20260910)
    post = make_posterior(
        rng.normal(size=(200, 1)),
        -0.5 * rng.normal(size=(200, 4)) ** 2,
    )
    analyzer_module = import_module('bayspec.infer.analyzer')
    arviz_loo = analyzer_module.az.loo

    def loo_with_arviz_warning(*args, **kwargs):
        warnings.warn_explicit(
            'divide by zero encountered in divide',
            RuntimeWarning,
            filename='arviz/stats/stats.py',
            lineno=1052,
            module='arviz.stats.stats',
        )
        return arviz_loo(*args, **kwargs)

    monkeypatch.setattr(analyzer_module.az, 'loo', loo_with_arviz_warning)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        post.loo(reff=1.0)

    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)


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
        lambda self: warnings.warn('overflow encountered in exp', RuntimeWarning, stacklevel=2),
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


@pytest.mark.parametrize('scale, factor', [('log', 1), ('deviance', -2)])
def test_replacing_infer_invalidates_predictive_criteria_cache(monkeypatch, scale, factor):
    rng = np.random.default_rng(833)
    params = rng.normal(size=(200, 1))
    loglike = -0.5 * rng.normal(size=(200, 4)) ** 2
    post = make_posterior(params, loglike)
    replacement = object.__new__(BayesInfer)
    replacement.__dict__.update(make_posterior(params, loglike - 2).__dict__)
    replacement.posterior_sample = params
    replacement.expected_pointwise = loglike - 2
    monkeypatch.setattr(
        SampleAnalyzer,
        '_calc_pointwise_loglike_sample',
        lambda self: self.expected_pointwise.copy(),
    )
    monkeypatch.setattr(SampleAnalyzer, '_calc_logprior_sample', lambda self: np.zeros(200))
    monkeypatch.setattr(SampleAnalyzer, '_allot_post', lambda self: None)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        old_waic = post.waic(scale=scale)
        old_loo = post.loo(scale=scale)
        post.infer = replacement
        new_waic = post.waic(scale=scale)
        new_loo = post.loo(scale=scale)

    np.testing.assert_array_equal(post.pointwise_loglike_sample, loglike - 2)
    assert new_waic.elpd_waic == pytest.approx(old_waic.elpd_waic - factor * 8)
    assert new_loo.elpd_loo == pytest.approx(old_loo.elpd_loo - factor * 8)
    assert new_waic.p_waic == pytest.approx(old_waic.p_waic)
    assert new_loo.p_loo == pytest.approx(old_loo.p_loo)
    assert new_waic is not old_waic
    assert new_loo is not old_loo
    assert post.waic(scale=scale) is new_waic
    assert post.loo(scale=scale) is new_loo


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
    assert all_ic['WAIC'] == (f'{-2.0 * post.waic().elpd_waic:.3f} ± {2.0 * post.waic().se:.3f}')
    assert all_ic['LOOIC'] == (f'{-2.0 * post.loo().elpd_loo:.3f} ± {2.0 * post.loo().se:.3f}')
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


def make_ic_posterior(pointwise=None):
    rng = np.random.default_rng(948)
    if pointwise is None:
        pointwise = -0.5 * rng.normal(size=(200, 6)) ** 2
    post = make_posterior(rng.normal(size=(len(pointwise), 1)), pointwise)
    bins = np.arange(7.0) + 1
    units = {
        name: SimpleNamespace(
            npoint=3,
            stat='pgstat',
            weight=1.0,
            rsp_chbin=np.column_stack([bins[start : start + 3], bins[start + 1 : start + 4]]),
        )
        for name, start in [('detector_a', 0), ('detector_b', 3)]
    }
    post.Pair = [
        SimpleNamespace(
            data=SimpleNamespace(data=units),
            model=SimpleNamespace(expr='pl'),
            npoint=6,
            loglike=float(np.max(pointwise.sum(axis=1))),
            has_nonunit_weights=False,
        )
    ]
    post.logevidence = -12.3456789
    post.logevidence_err = 0.123456789
    return post


def test_ic_bundle_contains_unrounded_numeric_criteria_and_channel_order():
    post = make_ic_posterior()
    par_before = [par.val for par in post.free_par.values()]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bundle = post.ic

    assert [par.val for par in post.free_par.values()] == par_before
    assert bundle['schema_version'] == 1
    assert bundle['n_params'] == 1
    assert bundle['n_data_points'] == 6
    assert bundle['n_samples'] == 200
    assert bundle['models'] == ['pl']
    assert [unit['name'] for unit in bundle['data']] == ['detector_a', 'detector_b']
    assert [unit['slice'] for unit in bundle['data']] == [[0, 3], [3, 6]]
    assert bundle['data'][1]['channel_bins'] == [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]]
    criteria = bundle['criteria']
    assert list(criteria) == ['AIC', 'AICc', 'BIC', 'WAIC', 'LOOIC', 'lnZ']
    assert criteria['AIC']['value'] == post.aic
    assert criteria['AICc']['value'] == post.aicc
    assert criteria['BIC']['value'] == post.bic
    for name, result, elpd, pointwise, penalty in [
        ('WAIC', post.waic(), 'elpd_waic', 'waic_i', 'p_waic'),
        ('LOOIC', post.loo(), 'elpd_loo', 'loo_i', 'p_loo'),
    ]:
        assert criteria[name]['value'] == -2 * result[elpd]
        assert criteria[name]['se'] == 2 * result.se
        assert criteria[name][penalty] == result[penalty]
        assert criteria[name]['warning'] == bool(result.warning)
        assert criteria[name]['scale'] == 'deviance'
        assert not criteria[name]['higher_is_better']
        np.testing.assert_array_equal(criteria[name]['pointwise'], -2 * result[pointwise])
        assert sum(criteria[name]['pointwise']) == pytest.approx(criteria[name]['value'])
    assert criteria['LOOIC']['good_k'] == post.loo().good_k
    assert criteria['lnZ']['value'] == -12.3456789
    assert criteria['lnZ']['error'] == 0.123456789
    assert criteria['lnZ']['higher_is_better']


def test_save_ic_roundtrip_preserves_failures_infinite_k_and_missing_evidence(tmp_path):
    regular = -0.5 * np.linspace(-1.0, 1.0, 100) ** 2
    post = make_ic_posterior(
        np.column_stack(
            [
                np.full(100, -2.0),
                np.r_[-10.0, np.zeros(99)],
                -np.r_[np.zeros(80), np.linspace(0.01, 0.2, 19), 708.0],
                regular,
                regular,
                regular,
            ]
        )
    )
    post.logevidence = None
    post.logevidence_err = None
    path = tmp_path / 'model' / 'ic.json'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        post.save_ic(path)

    def reject_nonstandard_constant(value):
        raise AssertionError(f'Non-standard JSON constant: {value}')

    loaded = json.loads(path.read_text(), parse_constant=reject_nonstandard_constant)
    assert loaded == post.ic
    loo = loaded['criteria']['LOOIC']
    assert loo['warning']
    assert loo['pareto_k'][:3] == [None, 'Infinity', None]
    restored_k = np.asarray(loo['pareto_k'], dtype=float)
    assert np.isnan(restored_k[0]) and np.isposinf(restored_k[1]) and np.isnan(restored_k[2])
    assert loaded['criteria']['lnZ']['value'] is None
    assert loaded['criteria']['lnZ']['error'] is None


def test_ic_files_support_paired_waic_comparison_without_posterior(tmp_path):
    left = make_ic_posterior()
    shift = np.linspace(0.1, 0.6, 6)
    right = make_ic_posterior(left.pointwise_loglike_sample - shift)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        left.save_ic(tmp_path / 'left.json')
        right.save_ic(tmp_path / 'right.json')
    first = json.loads((tmp_path / 'left.json').read_text())
    second = json.loads((tmp_path / 'right.json').read_text())
    difference = (
        np.array(second['criteria']['WAIC']['pointwise']) - first['criteria']['WAIC']['pointwise']
    )
    np.testing.assert_allclose(difference, 2 * shift)
    assert difference.sum() == pytest.approx(
        second['criteria']['WAIC']['value'] - first['criteria']['WAIC']['value']
    )
    assert np.sqrt(6 * difference.var()) == pytest.approx(np.sqrt(6 * (2 * shift).var()))


def test_save_writes_machine_readable_bundle_alongside_existing_tables(monkeypatch, tmp_path):
    post = make_ic_posterior()
    table = Info.from_dict({'test': 1.0})
    monkeypatch.setattr(SampleAnalyzer, 'free_par_info', property(lambda self: table))
    monkeypatch.setattr(SampleAnalyzer, 'stat_info', property(lambda self: table))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        post.save(tmp_path)
    assert (tmp_path / 'post_free_par.json').is_file()
    assert (tmp_path / 'post_stat.json').is_file()
    assert isinstance(json.loads((tmp_path / 'post_IC.json').read_text()), list)
    assert json.loads((tmp_path / 'post_ic_summary.json').read_text()) == post.ic


def test_bootstrap_ic_bundle_contains_only_available_criteria(tmp_path):
    bootstrap = object.__new__(Bootstrap)
    bootstrap.__dict__.update(make_ic_posterior().__dict__)
    bootstrap.free_par[1].post.truth = 7.0
    bootstrap.save_ic(tmp_path / 'bootstrap.json')
    bundle = json.loads((tmp_path / 'bootstrap.json').read_text())
    assert bundle['analyzer'] == 'Bootstrap'
    assert list(bundle['criteria']) == ['AIC', 'AICc', 'BIC']
