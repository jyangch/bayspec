import numpy as np
import pytest

from bayspec.data import Background, BalrogResponse, DataUnit, Response, Source
from bayspec.util.param import Par


class _FakeBalrogDRM:
    ebounds = np.arange(8.0)
    monte_carlo_energies = np.array([0.0, 7.0])

    def set_location(self, ra, dec):
        self.matrix = np.array([0.0, 0.0, 1.0, 4.0, 3.0, 1.0, 0.0])[:, None]


def _response_with_three_channel_fwhm(nchan=7):
    chbin = np.column_stack([np.arange(nchan), np.arange(1, nchan + 1)]).astype(float)
    phbin = np.array([[0.0, float(nchan)]])
    drm = np.zeros((1, nchan))
    peak = nchan // 2
    drm[0, [peak - 1, peak, peak + 1, peak + 2]] = [1.0, 4.0, 3.0, 1.0]
    return Response(chbin, phbin, drm)


def _source(counts, quality=None):
    counts = np.asarray(counts, dtype=float)
    return Source(
        counts=counts,
        errors=np.sqrt(counts),
        exposure=1.0,
        quality=quality,
    )


def _background(counts):
    counts = np.asarray(counts, dtype=float)
    return Background(
        counts=counts,
        errors=np.sqrt(counts),
        exposure=1.0,
    )


def test_data_unit_dispatches_optimal_grouping():
    unit = DataUnit(
        src=_source(np.ones(7)),
        rsp=_response_with_three_channel_fwhm(),
        grpg={'method': 'optimal'},
    )

    np.testing.assert_array_equal(unit.grouping, [1, -1, -1, 1, -1, -1, 1])


@pytest.mark.parametrize('ra_frozen, dec_frozen', [(False, False), (False, True), (True, False)])
def test_optimal_grouping_rejects_free_balrog_location(ra_frozen, dec_frozen):
    with pytest.raises(ValueError, match='requires a fixed response'):
        DataUnit(
            src=_source(np.ones(7)),
            rsp=BalrogResponse(
                _FakeBalrogDRM(),
                ra=Par(0, frozen=ra_frozen),
                dec=Par(0, frozen=dec_frozen),
            ),
            grpg={'method': 'optimal'},
        )


def test_optimal_grouping_allows_frozen_balrog_location():
    unit = DataUnit(
        src=_source(np.ones(7)),
        rsp=BalrogResponse(
            _FakeBalrogDRM(),
            ra=Par(0, frozen=True),
            dec=Par(0, frozen=True),
        ),
        grpg={'method': 'optimal'},
    )

    np.testing.assert_array_equal(unit.grouping, [1, -1, -1, 1, -1, -1, 1])


def test_optimal_grouping_does_not_cross_quality_gap():
    unit = DataUnit(
        src=_source(np.ones(7), quality=[0, 0, 1, 0, 0, 0, 0]),
        rsp=_response_with_three_channel_fwhm(),
        grpg={'method': 'optimal'},
    )

    np.testing.assert_array_equal(unit.grouping, [1, -1, 0, 1, -1, -1, 1])


def test_optimal_grouping_does_not_cross_noticing_gap():
    unit = DataUnit(
        src=_source(np.ones(7)),
        rsp=_response_with_three_channel_fwhm(),
        notc=[[0, 2], [3, 7]],
        grpg={'method': 'optimal'},
    )

    np.testing.assert_array_equal(unit.grouping, [1, -1, 0, 1, -1, -1, 1])


def test_optimal_grouping_merges_under_threshold_tail_backward():
    unit = DataUnit(
        src=_source(np.ones(7)),
        rsp=_response_with_three_channel_fwhm(),
        grpg={'method': 'optimal', 'min_evt': 4},
    )

    np.testing.assert_array_equal(unit.grouping, [1, -1, -1, -1, -1, -1, -1])


def test_optimal_grouping_extends_each_bin_until_minimum_events_are_met():
    unit = DataUnit(
        src=_source(np.ones(8)),
        rsp=_response_with_three_channel_fwhm(8),
        grpg={'method': 'optimal', 'min_evt': 4},
    )

    np.testing.assert_array_equal(unit.grouping, [1, -1, -1, -1, 1, -1, -1, -1])
    np.testing.assert_array_equal(unit.src_counts, [4.0, 4.0])


def test_grouping_rejects_unknown_method():
    with pytest.raises(ValueError, match='unsupported grouping method'):
        DataUnit(
            src=_source(np.ones(7)),
            rsp=_response_with_three_channel_fwhm(),
            grpg={'method': 'unknown'},
        )


def test_grouping_rejects_old_opt_method_name():
    with pytest.raises(ValueError, match='unsupported grouping method'):
        DataUnit(
            src=_source(np.ones(7)),
            rsp=_response_with_three_channel_fwhm(),
            grpg={'method': 'opt'},
        )


def test_optimal_grouping_uses_bayspec_net_count_threshold():
    unit = DataUnit(
        src=_source(np.ones(7)),
        bkg=_background(np.full(7, 0.5)),
        rsp=_response_with_three_channel_fwhm(),
        grpg={'method': 'optimal', 'min_nevt': 2},
    )

    np.testing.assert_array_equal(unit.grouping, [1, -1, -1, -1, -1, -1, -1])


def test_optimal_grouping_uses_bayspec_significance_threshold():
    unit = DataUnit(
        src=_source(np.ones(7)),
        rsp=_response_with_three_channel_fwhm(),
        stat='cstat',
        grpg={'method': 'optimal', 'min_sigma': 2.2},
    )

    np.testing.assert_array_equal(unit.grouping, [1, -1, -1, -1, -1, -1, -1])


def test_optimal_grouping_keeps_bin_when_threshold_is_unreachable():
    unit = DataUnit(
        src=_source(np.ones(7)),
        rsp=_response_with_three_channel_fwhm(),
        grpg={'method': 'optimal', 'min_evt': 8},
    )

    np.testing.assert_array_equal(unit.grouping, [1, -1, -1, -1, -1, -1, -1])


def test_optimal_grouping_rejects_maximum_smaller_than_optimal_width():
    with pytest.raises(ValueError, match='max_bin cannot be smaller'):
        DataUnit(
            src=_source(np.ones(7)),
            rsp=_response_with_three_channel_fwhm(),
            grpg={'method': 'optimal', 'max_bin': 2},
        )


def test_optimal_grouping_stops_at_maximum_before_threshold_is_met():
    unit = DataUnit(
        src=_source(np.ones(7)),
        rsp=_response_with_three_channel_fwhm(),
        grpg={'method': 'optimal', 'min_evt': 4, 'max_bin': 3},
    )

    np.testing.assert_array_equal(unit.grouping, [1, -1, -1, 1, -1, -1, 1])


def test_optimal_grouping_does_not_merge_tail_past_maximum():
    unit = DataUnit(
        src=_source(np.ones(7)),
        rsp=_response_with_three_channel_fwhm(),
        grpg={'method': 'optimal', 'min_evt': 4, 'max_bin': 4},
    )

    np.testing.assert_array_equal(unit.grouping, [1, -1, -1, -1, 1, -1, -1])


def test_pure_optimal_grouping_is_independent_of_background():
    source = _source(np.ones(7))
    response = _response_with_three_channel_fwhm()
    without_background = DataUnit(src=source, rsp=response, grpg={'method': 'optimal'})
    with_background = DataUnit(
        src=source,
        bkg=_background(np.full(7, 100.0)),
        rsp=response,
        grpg={'method': 'optimal'},
    )

    np.testing.assert_array_equal(without_background.grouping, with_background.grouping)


def test_optimal_maximum_without_thresholds_does_not_use_background():
    source = _source(np.ones(7))
    response = _response_with_three_channel_fwhm()
    without_background = DataUnit(
        src=source,
        rsp=response,
        grpg={'method': 'optimal', 'max_bin': 4},
    )
    with_background = DataUnit(
        src=source,
        bkg=_background(np.full(7, 100.0)),
        rsp=response,
        grpg={'method': 'optimal', 'max_bin': 4},
    )

    np.testing.assert_array_equal(without_background.grouping, with_background.grouping)


def test_explicit_threshold_method_preserves_existing_grouping_behavior():
    source = _source(np.ones(7))
    response = _response_with_three_channel_fwhm()

    implicit = DataUnit(src=source, rsp=response, grpg={'min_evt': 2})
    explicit = DataUnit(
        src=source,
        rsp=response,
        grpg={'method': 'threshold', 'min_evt': 2},
    )

    np.testing.assert_array_equal(implicit.grouping, explicit.grouping)


def test_rebinning_changes_plot_grid_without_changing_grouping_grid():
    unit = DataUnit(
        src=_source(np.ones(8)),
        rsp=_response_with_three_channel_fwhm(8),
        grpg={'min_evt': 2},
        rebn={'min_evt': 4},
    )

    np.testing.assert_array_equal(unit.grouping, [1, -1, 1, -1, 1, -1, 1, -1])
    np.testing.assert_array_equal(unit.rebining, [1, -1, -1, -1, 1, -1, -1, -1])
    np.testing.assert_array_equal(unit.src_counts, [2.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(unit.src_re_counts, [4.0, 4.0])
