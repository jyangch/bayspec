from importlib import import_module

import numpy as np
import pytest

from bayspec.infer import Statistic, StatisticNB, StatisticResult

statistic_module = import_module('bayspec.infer.statistic')


REAL_FIT_COUNT = 923.0
REAL_FIT_MEAN = 923.0000009160082
REAL_FIT_RELATIVE_LOGLIKE = -4.5453464348197324e-16


def test_statistic_nb_remains_a_compatibility_alias():
    assert StatisticNB is Statistic


def test_statistic_result_rejects_any_positive_relative_loglike():
    with pytest.raises(ValueError, match='positive relative log-likelihood'):
        StatisticResult(np.array([1e-15]), np.array([1.0]))


def _statistic_inputs():
    return {
        'S': np.array([REAL_FIT_COUNT]),
        'B': np.array([0.0]),
        'm': np.array([REAL_FIT_MEAN]),
        'ts': 1.0,
        'tb': 1.0,
        'sigma_S': np.array([0.0]),
        'sigma_B': np.array([0.0]),
    }


@pytest.mark.parametrize('statistic_name', ['Pstat', 'PPstat', 'PGstat'])
def test_poisson_statistics_preserve_tiny_negative_relative_loglike(statistic_name):
    result = getattr(Statistic, statistic_name)(**_statistic_inputs())

    np.testing.assert_allclose(
        result.pointwise_loglike,
        [REAL_FIT_RELATIVE_LOGLIKE],
        rtol=5e-15,
        atol=0.0,
    )


def _poisson_relative_loglike():
    helper = getattr(statistic_module, 'poisson_relative_loglike', None)
    assert callable(helper), 'poisson_relative_loglike must be a public helper'
    return helper


@pytest.mark.parametrize(
    ('count', 'mean', 'expected'),
    [
        (0.0, 0.0, 0.0),
        (0.0, 2.5, -2.5),
        (2.0, 0.0, -np.inf),
        (-1.0, 1.0, -np.inf),
        (1.0, -1.0, -np.inf),
        (np.nan, 1.0, -np.inf),
        (1.0, np.nan, -np.inf),
        (np.inf, 1.0, -np.inf),
        (1.0, np.inf, -np.inf),
    ],
)
def test_poisson_relative_loglike_handles_domain_boundaries(count, mean, expected):
    helper = _poisson_relative_loglike()

    assert helper(count, mean) == expected


def test_poisson_relative_loglike_supports_numpy_arrays():
    helper = _poisson_relative_loglike()

    actual = helper(
        np.array([REAL_FIT_COUNT, 0.0, 2.0]),
        np.array([REAL_FIT_MEAN, 2.5, 0.0]),
    )

    np.testing.assert_allclose(
        actual,
        [REAL_FIT_RELATIVE_LOGLIKE, -2.5, -np.inf],
        rtol=5e-15,
        atol=0.0,
    )
