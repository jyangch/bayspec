import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bayspec.infer import analyzer as analyzer_module
from bayspec.infer.infer import BayesInfer


def test_get_multinest_evidence_from_plain_returns_value_and_error(tmp_path):
    stats_file = tmp_path / 'stats.dat'
    stats_file.write_text('Nested Sampling Global Log-Evidence : -0.295912E+03 +/- 0.206352E+00\n')

    assert BayesInfer._get_multinest_evidence_from_plain(stats_file) == pytest.approx(
        (-295.912, 0.206352)
    )


def test_get_multinest_evidence_from_stats_keeps_value_and_error_from_same_estimator():
    stats = {
        'nested sampling global log-evidence': -295.91,
        'nested sampling global log-evidence error': 0.21,
        'nested importance sampling global log-evidence': -295.85,
        'nested importance sampling global log-evidence error': 0.02,
    }

    assert BayesInfer._get_multinest_evidence_from_stats(stats, ins=True) == pytest.approx(
        (-295.85, 0.02)
    )
    assert BayesInfer._get_multinest_evidence_from_stats(stats, ins=False) == pytest.approx(
        (-295.91, 0.21)
    )


def test_multinest_resume_refreshes_parameter_sample_from_standard_output(tmp_path, monkeypatch):
    prefix = tmp_path / '1-'
    stale_sample = np.array([[90.0, 91.0], [92.0, 93.0]])
    equal_weighted = np.array([[1.0, 2.0, -10.0], [3.0, 4.0, -11.0]])
    np.savetxt(f'{prefix}posterior_sample.txt', stale_sample)
    (tmp_path / '1-max_iter_warning.txt').write_text('previous run stopped at cap\n')

    stats = {
        'nested sampling global log-evidence': -12.0,
        'nested sampling global log-evidence error': 0.2,
        'nested importance sampling global log-evidence': -11.8,
        'nested importance sampling global log-evidence error': 0.1,
    }

    class Analyzer:
        def __init__(self, **kwargs):
            pass

        def get_stats(self):
            return stats

        def get_equal_weighted_posterior(self):
            return equal_weighted

    monkeypatch.setitem(
        sys.modules,
        'pymultinest',
        SimpleNamespace(Analyzer=Analyzer, run=lambda **kwargs: None),
    )
    monkeypatch.setattr(analyzer_module, 'Posterior', lambda infer: infer)

    infer = object.__new__(BayesInfer)
    infer._free_nparams = 2
    infer._you_free = lambda: None

    with pytest.warns(UserWarning, match='reusing saved outputs'):
        result = infer.multinest(resume=True, savepath=str(tmp_path))

    np.testing.assert_allclose(result.posterior_sample, equal_weighted[:, :2])
    np.testing.assert_allclose(
        np.loadtxt(f'{prefix}posterior_sample.txt', ndmin=2),
        equal_weighted[:, :2],
    )
