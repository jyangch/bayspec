import numpy as np
import pytest

from bayspec.infer.analyzer import Bootstrap, Posterior
from bayspec.infer.infer import MaxLikeFit
from bayspec.util.tools import SuperDict


class Prior:
    @staticmethod
    def pdf(value):
        return np.exp(-2.0 * value)


class Parameter:
    def __init__(self, value):
        self.val = value
        self.prior = Prior()
        self.post = None


class Pair:
    npoint = 2

    def __init__(self, parameter):
        self.parameter = parameter

    @property
    def loglike(self):
        return -((self.parameter.val - 1.0) ** 2) - 0.5


def make_analyzer(analyzer_cls):
    analyzer = object.__new__(analyzer_cls)
    setattr(analyzer, analyzer.sample_attribute, np.array([[0.0], [1.0], [2.0]]))
    analyzer._free_nparams = 1
    analyzer._free_par = SuperDict([('1', Parameter(7.0))])
    analyzer.Pair = [Pair(analyzer.free_par[1])]
    analyzer._logprior_func = None
    analyzer._loglike_func = lambda _, theta: np.array([-((theta[0] - 1.0) ** 2), -0.5])
    return analyzer


def test_check_sample_eagerly_calculates_likelihood_prior_and_probability():
    post = make_analyzer(Posterior)

    post._check_sample()

    np.testing.assert_allclose(post.param_sample, [[0.0], [1.0], [2.0]])
    np.testing.assert_allclose(
        post.pointwise_loglike_sample,
        [[-1.0, -0.5], [0.0, -0.5], [-1.0, -0.5]],
    )
    np.testing.assert_allclose(post.loglike_sample, [-1.5, -0.5, -1.5])
    np.testing.assert_allclose(post.logprior_sample, [0.0, -2.0, -4.0])
    np.testing.assert_allclose(post.logprob_sample, [-1.5, -2.5, -5.5])
    assert post.free_par[1].val == 7.0


def test_check_sample_rejects_a_trailing_score_column():
    post = make_analyzer(Posterior)
    post.posterior_sample = np.array([[0.0, -1.0], [1.0, -2.0]])

    with pytest.raises(ValueError, match='expected to have 1 columns'):
        post._check_sample()


def test_posterior_ranks_draws_by_log_probability():
    post = make_analyzer(Posterior)

    post._check_sample()
    post._allot_post()

    assert post.free_par[1].post.best == 0.0


def test_bootstrap_ranks_draws_by_log_likelihood():
    boot = make_analyzer(Bootstrap)

    boot._check_sample()
    boot._allot_post()

    assert boot.free_par[1].post.best == 1.0


@pytest.mark.parametrize(
    'attribute, expected',
    [('max_loglike', -0.5), ('aic', 3.0), ('bic', 1.0 + np.log(2.0))],
)
def test_posterior_likelihood_criteria_use_highest_likelihood_sample(attribute, expected):
    post = make_analyzer(Posterior)
    post._check_sample()
    post._allot_post()

    assert post.par_best == [0.0]
    assert getattr(post, attribute) == pytest.approx(expected)
    assert post.par_best == [0.0]


def test_posterior_max_loglike_does_not_change_current_parameters():
    post = make_analyzer(Posterior)
    post._check_sample()
    post._allot_post()
    post.at_par([7.0])

    _ = post.max_loglike

    assert post.free_par[1].val == 7.0


def test_bootstrap_max_loglike_remains_at_truth_not_highest_sample():
    boot = make_analyzer(Bootstrap)
    boot._check_sample()
    boot._allot_post()

    assert boot.par_truth == [0.0]
    assert boot.par_best == [1.0]
    assert boot.max_loglike == -1.5


def test_bootstrap_sample_contains_only_parameter_draws():
    fit = object.__new__(MaxLikeFit)
    fit._free_pranges = [(-10.0, 10.0)]
    fit.at_par = lambda theta: None
    fit.calc_loglike = lambda theta: pytest.fail('bootstrap generation must not score draws')

    fit._make_bootstrap_sample([0.0], covar=[[1.0]], nsample=4, random_seed=3)

    assert fit.bootstrap_sample.shape == (4, 1)
    assert fit.bootstrap_sample[0, 0] == 0.0
