import numpy as np
import pytest


def test_build_grouping_closes_threshold_bins_at_minimum_events():
    from bayspec.util.group import build_grouping

    np.testing.assert_array_equal(
        build_grouping(
            src_cts=np.ones(4),
            bkg_cts=np.zeros(4),
            bkg_err=np.zeros(4),
            src_expo=1.0,
            bkg_expo=1.0,
            src_scal=1.0,
            bkg_scal=1.0,
            min_evt=2,
        ),
        [1, -1, 1, -1],
    )


def test_threshold_grouping_only_accepts_valid_flag_equal_to_one():
    from bayspec.util.group import build_grouping

    np.testing.assert_array_equal(
        build_grouping(
            src_cts=np.ones(3),
            bkg_cts=np.zeros(3),
            bkg_err=np.zeros(3),
            src_expo=1.0,
            bkg_expo=1.0,
            src_scal=1.0,
            bkg_scal=1.0,
            min_evt=2,
            valid=[1, 2, 1],
        ),
        [1, 0, -1],
    )


def test_build_grouping_extends_optimal_bins_to_threshold():
    from bayspec.util.group import build_grouping

    np.testing.assert_array_equal(
        build_grouping(
            np.ones(8),
            np.zeros(8),
            np.zeros(8),
            1.0,
            1.0,
            1.0,
            1.0,
            method='optimal',
            rsp_fwhm=np.full(8, 3.0),
            min_evt=4,
        ),
        [1, -1, -1, -1, 1, -1, -1, -1],
    )


def test_optimal_bin_widths_use_cpp_half_away_from_zero_rounding():
    from bayspec.util.group import calculate_optimal_bin_widths

    rsp_fwhm = np.full(16, 3.0)
    src_cts = np.ones(16, dtype=int)

    np.testing.assert_array_equal(
        calculate_optimal_bin_widths(rsp_fwhm=rsp_fwhm, src_cts=src_cts),
        np.full(16, 3),
    )


def test_optimal_grouping_uses_most_restrictive_width_inside_candidate_bin():
    from bayspec.util.group import build_optimal_grouping

    rsp_fwhm = np.array([4.0, 1.0, 4.0, 4.0, 4.0])
    src_cts = np.zeros(5, dtype=int)

    np.testing.assert_array_equal(
        build_optimal_grouping(rsp_fwhm=rsp_fwhm, src_cts=src_cts),
        [1, -1, 1, -1, -1],
    )


def test_optimal_bin_widths_round_rate_counts_like_ftgrouppha():
    from bayspec.util.group import calculate_optimal_bin_widths

    rsp_fwhm = np.full(8, 3.0)
    rate_cts = np.full(8, 1.5)

    np.testing.assert_array_equal(
        calculate_optimal_bin_widths(rsp_fwhm, rate_cts),
        [2, 2, 2, 2, 2, 2, 2, 3],
    )


def test_optimal_bin_widths_reject_nonpositive_fwhm():
    from bayspec.util.group import calculate_optimal_bin_widths

    with pytest.raises(ValueError, match='FWHM must be positive'):
        calculate_optimal_bin_widths([2.0, -1.0], [1, 1])


def test_optimal_bin_widths_reject_mismatched_arrays():
    from bayspec.util.group import calculate_optimal_bin_widths

    with pytest.raises(ValueError, match='same length'):
        calculate_optimal_bin_widths([2.0, 2.0], [1])


def test_optimal_bin_widths_reject_nonfinite_fwhm():
    from bayspec.util.group import calculate_optimal_bin_widths

    with pytest.raises(ValueError, match='finite'):
        calculate_optimal_bin_widths([2.0, np.inf], [1, 1])


def test_optimal_bin_widths_reject_nonfinite_counts():
    from bayspec.util.group import calculate_optimal_bin_widths

    with pytest.raises(ValueError, match='counts must be finite'):
        calculate_optimal_bin_widths([2.0, 2.0], [1, np.nan])
