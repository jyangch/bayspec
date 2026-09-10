"""Numerical utilities for response-aware spectral grouping."""

import numpy as np

from .significance import pgsig, ppsig


def _round_half_away_from_zero(value):
    """Round a value to the nearest integer, rounding away from zero."""

    if value >= 0:
        return int(np.floor(value + 0.5))

    return int(np.ceil(value - 0.5))


def _energy_bin_index(energy, low, high):
    """Reproduce HEASP's response-energy bin lookup."""

    if len(low) == 1:
        return 0

    increasing = low[1] > low[0]
    if increasing:
        if energy < low[0]:
            return 0
        if energy > high[-1]:
            return len(low) - 1
    else:
        if energy > low[0]:
            return 0
        if energy < high[-1]:
            return len(low) - 1

    ilo = 0
    ihi = len(low) - 1
    while ihi - ilo > 1:
        mid = (ilo + ihi) // 2
        if (increasing and energy > low[mid]) or (not increasing and energy < low[mid]):
            ilo = mid
        else:
            ihi = mid

    if low[ilo] < energy <= high[ilo]:
        return ilo
    return ihi


def estimate_channel_fwhm(chbin, phbin, drm):
    """Estimate the HEASP-compatible response FWHM for each detector channel."""

    chbin = np.asarray(chbin)
    phbin = np.asarray(phbin)
    drm = np.asarray(drm)

    rsp_fwhm = np.empty(len(phbin), dtype=float)
    for i, row in enumerate(drm):
        peak = int(np.argmax(row))
        half_max = row[peak] / 2.0

        high = peak + 1 if peak < len(row) - 1 else peak
        while high < len(row) - 1 and row[high] > half_max:
            high += 1

        low = peak - 1 if peak > 0 else peak
        while low > 0 and row[low] > half_max:
            low -= 1

        good_high = row[-1] <= half_max
        good_low = row[0] <= half_max

        width = 0.0
        if good_high:
            width += high - peak
        if good_low:
            width += peak - low
        if good_high != good_low:
            width *= 2
        if not good_high and not good_low:
            width = -1.0
        rsp_fwhm[i] = width

    ch_mean = np.mean(chbin, axis=1)
    idx = [_energy_bin_index(energy, phbin[:, 0], phbin[:, 1]) for energy in ch_mean]

    return rsp_fwhm[idx]


def calculate_optimal_bin_widths(rsp_fwhm, src_cts):
    """Calculate HEASP-compatible Kaastra-Bleeker bin widths."""

    rsp_fwhm = np.asarray(rsp_fwhm, dtype=float)
    src_cts = np.asarray(src_cts)

    if not np.all(np.isfinite(rsp_fwhm)):
        raise ValueError('FWHM must be finite for optimal grouping')
    if np.any(rsp_fwhm <= 0):
        raise ValueError('FWHM must be positive for optimal grouping')
    if len(rsp_fwhm) != len(src_cts):
        raise ValueError('FWHM and counts must have the same length')
    if not np.all(np.isfinite(src_cts)):
        raise ValueError('counts must be finite for optimal grouping')

    n_res = 1.0 + np.sum(1.0 / rsp_fwhm)
    log_n_res = np.log(n_res)
    src_cts = np.array([_round_half_away_from_zero(value) for value in src_cts], dtype=int)

    src_cts_per_res = np.empty(len(rsp_fwhm), dtype=float)
    for i, width in enumerate(rsp_fwhm):
        start = max(0, _round_half_away_from_zero(i - width / 2.0))
        stop = min(len(rsp_fwhm) - 1, _round_half_away_from_zero(i + width / 2.0))
        src_cts_per_res[i] = 1.314 * np.sum(src_cts[start : stop + 1])

    opt_widths = rsp_fwhm.copy()
    for i, src_cts_res in enumerate(src_cts_per_res):
        arg = src_cts_res * (1.0 + 0.2 * log_n_res)
        if arg <= 0:
            continue
        x = np.log(arg)
        if x > 2.119:
            opt_widths[i] *= (0.08 * x + 7.0 + 1.8 / x) / (x + 5.9)

    return np.maximum(opt_widths.astype(int), 1)


def build_optimal_grouping(rsp_fwhm, src_cts, valid=None):
    """Return OGIP grouping flags for HEASP-compatible optimal bins."""

    opt_widths = calculate_optimal_bin_widths(rsp_fwhm, src_cts)

    if valid is None:
        valid = np.ones(len(opt_widths), dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
        if len(valid) != len(opt_widths):
            raise ValueError('valid and FWHM must have the same length')

    grouping = np.zeros(len(opt_widths), dtype=int)

    start = 0
    last = len(opt_widths) - 1
    while start <= last:
        if not valid[start]:
            start += 1
            continue

        seg_stop = start
        while seg_stop < last and valid[seg_stop + 1]:
            seg_stop += 1

        grouping[start] = 1
        stop = min(seg_stop, start + opt_widths[start] - 1)
        for i in range(start + 1, stop + 1):
            stop = min(stop, i + opt_widths[i] - 1)
        grouping[start + 1 : stop + 1] = -1
        start = stop + 1

    return grouping


def _grouping_significance(src, bkg, bkg_err, alpha, stat):

    if stat in ['pstat', 'cstat', 'ppstat']:
        if (bkg < 0 or src < 0) and bkg != src:
            return 0
        return ppsig(src, bkg, alpha)

    if stat in ['gstat', 'chi2', 'pgstat']:
        if src <= 0 or bkg_err == 0:
            return 0
        return pgsig(src, bkg * alpha, bkg_err * alpha)

    raise AttributeError(f'unsupported stat: {stat}')


def build_threshold_grouping(
    src_cts,
    bkg_cts,
    bkg_err,
    src_expo,
    bkg_expo,
    src_scal,
    bkg_scal,
    min_sigma=None,
    min_evt=None,
    min_nevt=None,
    max_bin=None,
    stat=None,
    valid=None,
):
    """Group channels using BaySpec count, significance, and width thresholds."""

    if valid is None:
        valid = np.ones(len(src_cts), dtype=bool)

    if min_sigma is None:
        min_sigma = -np.inf
    if min_evt is None:
        min_evt = 0
    if min_nevt is None:
        min_nevt = 0
    if max_bin is None:
        max_bin = np.inf
    if stat is None:
        stat = 'pgstat'

    alpha = src_expo * src_scal / (bkg_expo * bkg_scal)

    grouping = []
    starts = []
    active = False
    src_sum = 0
    bkg_sum = 0
    bkg_err_sum = 0
    width = 0

    for i in range(len(src_cts)):
        if valid[i] != 1:
            grouping.append(0)
            if active and len(starts) >= 2:
                grouping[starts[-1]] = -1
            active = False
            src_sum = 0
            bkg_sum = 0
            bkg_err_sum = 0
            width = 0
            continue

        if not active:
            grouping.append(1)
            starts.append(i)
            width = 1
        else:
            grouping.append(-1)
            width += 1

        src_sum += src_cts[i]
        bkg_sum += bkg_cts[i]
        bkg_err_sum = np.sqrt(bkg_err_sum**2 + bkg_err[i] ** 2)

        sigma = _grouping_significance(
            src_sum,
            bkg_sum,
            bkg_err_sum,
            alpha,
            stat,
        )

        evts = src_sum
        net_evts = src_sum - bkg_sum * alpha

        if (sigma >= min_sigma and evts >= min_evt and net_evts >= min_nevt) or width == max_bin:
            active = False
            src_sum = 0
            bkg_sum = 0
            bkg_err_sum = 0
            width = 0
        else:
            active = True

        if active and i == len(src_cts) - 1 and len(starts) >= 2:
            grouping[starts[-1]] = -1

    return np.asarray(grouping)


def build_optimal_threshold_grouping(
    rsp_fwhm,
    src_cts,
    bkg_cts,
    bkg_err,
    src_expo,
    bkg_expo,
    src_scal,
    bkg_scal,
    min_sigma=None,
    min_evt=None,
    min_nevt=None,
    max_bin=None,
    stat=None,
    valid=None,
):
    """Apply BaySpec thresholds on top of HEASP-compatible optimal widths."""

    if min_sigma is None and min_evt is None and min_nevt is None and max_bin is None:
        return build_optimal_grouping(rsp_fwhm, src_cts, valid=valid)

    src_cts = np.asarray(src_cts)
    bkg_cts = np.asarray(bkg_cts)
    bkg_err = np.asarray(bkg_err)

    opt_widths = calculate_optimal_bin_widths(rsp_fwhm, src_cts)
    valid = np.ones(len(src_cts), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)

    if max_bin is None:
        max_bin = np.inf
    if stat is None:
        stat = 'pgstat'

    alpha = src_expo * src_scal / (bkg_expo * bkg_scal)

    def thresholds_met(start, stop):
        src_sum = np.sum(src_cts[start : stop + 1])
        bkg_sum = np.sum(bkg_cts[start : stop + 1])

        if min_evt is not None and src_sum < min_evt:
            return False
        if min_nevt is not None and src_sum - bkg_sum * alpha < min_nevt:
            return False
        if min_sigma is not None:
            bkg_err_sum = np.sqrt(np.sum(bkg_err[start : stop + 1] ** 2))
            sigma = _grouping_significance(
                src_sum,
                bkg_sum,
                bkg_err_sum,
                alpha,
                stat,
            )
            if sigma < min_sigma:
                return False
        return True

    bins = []
    start = 0
    last = len(src_cts) - 1
    while start <= last:
        if not valid[start]:
            start += 1
            continue

        seg_stop = start
        while seg_stop < last and valid[seg_stop + 1]:
            seg_stop += 1

        seg_bins = []
        while start <= seg_stop:
            cand_stop = min(seg_stop, start + opt_widths[start] - 1)
            stop = cand_stop
            for i in range(start + 1, cand_stop + 1):
                stop = min(stop, i + opt_widths[i] - 1)

            if stop - start + 1 > max_bin:
                raise ValueError('max_bin cannot be smaller than the optimal bin width')

            while (
                not thresholds_met(start, stop) and stop < seg_stop and stop - start + 1 < max_bin
            ):
                stop += 1

            seg_bins.append((start, stop))
            start = stop + 1

        tail_start, tail_stop = seg_bins[-1]
        while not thresholds_met(tail_start, tail_stop) and len(seg_bins) >= 2:
            prev_start = seg_bins[-2][0]
            if tail_stop - prev_start + 1 > max_bin:
                break
            seg_bins.pop()
            seg_bins[-1] = (prev_start, tail_stop)
            tail_start = prev_start

        bins.extend(seg_bins)

    grouping = np.zeros(len(src_cts), dtype=int)
    for start, stop in bins:
        grouping[start] = 1
        grouping[start + 1 : stop + 1] = -1

    return grouping


def build_grouping(
    src_cts,
    bkg_cts,
    bkg_err,
    src_expo,
    bkg_expo,
    src_scal,
    bkg_scal,
    method='threshold',
    rsp_fwhm=None,
    min_sigma=None,
    min_evt=None,
    min_nevt=None,
    max_bin=None,
    stat=None,
    valid=None,
):
    """Build OGIP grouping flags using threshold or optimal grouping."""

    if method == 'threshold':
        return build_threshold_grouping(
            src_cts,
            bkg_cts,
            bkg_err,
            src_expo,
            bkg_expo,
            src_scal,
            bkg_scal,
            min_sigma=min_sigma,
            min_evt=min_evt,
            min_nevt=min_nevt,
            max_bin=max_bin,
            stat=stat,
            valid=valid,
        )

    if method == 'optimal':
        if rsp_fwhm is None:
            raise ValueError('rsp_fwhm is required for optimal grouping')
        return build_optimal_threshold_grouping(
            rsp_fwhm,
            src_cts,
            bkg_cts,
            bkg_err,
            src_expo,
            bkg_expo,
            src_scal,
            bkg_scal,
            min_sigma=min_sigma,
            min_evt=min_evt,
            min_nevt=min_nevt,
            max_bin=max_bin,
            stat=stat,
            valid=valid,
        )

    raise ValueError(f'unsupported grouping method: {method}')
