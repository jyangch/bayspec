from contextlib import suppress
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bayspec.infer import BayesInfer


@pytest.mark.parametrize('failing_stage', ['prior', 'loglikelihood'])
def test_multinest_rejects_results_after_callback_error(monkeypatch, tmp_path, failing_stage):
    analyzer_called = False
    callback_results = []

    def run(**kwargs):
        cube = [0.5]
        with suppress(BaseException):
            kwargs['Prior'](cube, 1, 1)
            callback_results.append(kwargs['LogLikelihood'](cube, 1, 1, 0.0))

    class Analyzer:
        def __init__(self, **kwargs):
            nonlocal analyzer_called
            analyzer_called = True
            raise AssertionError('callback failure must be checked before analyzing outputs')

    monkeypatch.setitem(sys.modules, 'pymultinest', SimpleNamespace(run=run, Analyzer=Analyzer))

    infer = BayesInfer.__new__(BayesInfer)
    infer._free_nparams = 1
    infer._you_free = lambda: None
    infer.multinest_prior_transform = lambda cube: np.asarray(cube)
    infer.multinest_calc_loglike = lambda theta: 0.0

    def fail(_):
        raise ValueError('boom')

    if failing_stage == 'prior':
        infer.multinest_prior_transform = fail
    else:
        infer.multinest_calc_loglike = fail

    with pytest.raises(RuntimeError, match=rf'MultiNest {failing_stage} callback failed: boom'):
        infer.multinest(nlive=1, resume=False, max_iter=1, savepath=str(tmp_path))

    assert not analyzer_called
    assert callback_results == [-2e100]
