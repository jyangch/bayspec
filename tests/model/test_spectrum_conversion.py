import warnings

import numpy as np

from bayspec.infer.pair import Pair
from bayspec.model.model import Model


class ConversionModel(Model):
    @property
    def conv_re_ctsspec(self):
        return [np.array([2.0, 0.0])]

    @property
    def re_phtspec_at_rsp(self):
        return [np.array([4.0, 0.0])]

    @property
    def re_flxspec_at_rsp(self):
        return [np.array([6.0, 0.0])]

    @property
    def re_ergspec_at_rsp(self):
        return [np.array([8.0, 0.0])]


class ConversionPair(Pair):
    @property
    def conv_re_ctsspec(self):
        return [np.array([2.0, 0.0])]

    @property
    def re_phtspec_at_rsp(self):
        return [np.array([4.0, 0.0])]

    @property
    def re_flxspec_at_rsp(self):
        return [np.array([6.0, 0.0])]

    @property
    def re_ergspec_at_rsp(self):
        return [np.array([8.0, 0.0])]


def test_rebinned_spectrum_conversion_marks_zero_model_counts_undefined():
    model = ConversionModel()

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        ratios = [model.re_cts_to_pht[0], model.re_cts_to_flx[0], model.re_cts_to_erg[0]]

    for ratio, finite_value in zip(ratios, [2.0, 3.0, 4.0], strict=True):
        assert ratio[0] == finite_value
        assert np.isnan(ratio[1])


def test_pair_rebinned_spectrum_conversion_matches_model_zero_count_behavior():
    pair = object.__new__(ConversionPair)

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        ratios = [pair.re_cts_to_pht[0], pair.re_cts_to_flx[0], pair.re_cts_to_erg[0]]

    for ratio, finite_value in zip(ratios, [2.0, 3.0, 4.0], strict=True):
        assert ratio[0] == finite_value
        assert np.isnan(ratio[1])
