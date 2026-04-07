import numba as nb
import numpy as np

from ..util.significance import ppsig, pgsig


@nb.njit(cache=True, fastmath=True)
def _gstat_core(S, B, m, ts, tb, sigma_S, sigma_B):
    
    n = S.shape[0]
    residual = np.empty(n, dtype=np.float64)
    stat = 0.0

    ratio = 0.0
    if tb != 0.0:
        ratio = ts / tb

    for i in range(n):
        bi = B[i]
        sigma_bi = sigma_B[i]
        if tb != 0.0:
            bi = bi * ratio
            sigma_bi = sigma_bi * ratio

        sigma = np.sqrt(sigma_S[i] * sigma_S[i] + sigma_bi * sigma_bi)
        di = S[i] - bi
        mi = m[i] * ts
        z = (di - mi) / sigma
        logli = -0.5 * z * z

        stati = -2.0 * logli
        stat += stati

        sign = 0.0
        delta = di - mi
        if delta > 0.0:
            sign = 1.0
        elif delta < 0.0:
            sign = -1.0

        residual[i] = sign * np.sqrt(stati)

    return stat, residual


@nb.njit(cache=True, fastmath=True)
def _pstat_core(S, m, ts):
    
    n = S.shape[0]
    residual = np.empty(n, dtype=np.float64)
    stat = 0.0

    for i in range(n):
        si = S[i]
        mu = m[i] * ts

        klogmu = 0.0
        if si != 0.0:
            klogmu = si * np.log(mu)

        klogk = 0.0
        if si != 0.0:
            klogk = si * np.log(si)

        logli = klogmu - mu - klogk + si
        stati = -2.0 * logli
        stat += stati

        delta = si - mu
        sign = 0.0
        if delta > 0.0:
            sign = 1.0
        elif delta < 0.0:
            sign = -1.0

        residual[i] = sign * np.sqrt(stati)

    return stat, residual


@nb.njit(cache=True, fastmath=True)
def _ppstat_core(S, B, m, ts, tb):
    
    n = S.shape[0]
    residual = np.empty(n, dtype=np.float64)
    stat = 0.0

    aa = ts + tb

    for i in range(n):
        si = S[i]
        bi = B[i]
        mi = m[i]

        bb = aa * mi - si - bi
        cc = -bi * mi
        dd = np.sqrt(bb * bb - 4.0 * aa * cc)

        if bb >= 0.0:
            b = -2.0 * cc / (bb + dd)
        else:
            b = -(bb - dd) / (2.0 * aa)

        mu_s = ts * (b + mi)
        mu_b = tb * b

        s_klogmu = 0.0
        if si != 0.0:
            s_klogmu = si * np.log(mu_s)

        s_klogk = 0.0
        if si != 0.0:
            s_klogk = si * np.log(si)

        b_klogmu = 0.0
        if bi != 0.0:
            b_klogmu = bi * np.log(mu_b)

        b_klogk = 0.0
        if bi != 0.0:
            b_klogk = bi * np.log(bi)

        logli = (s_klogmu - mu_s - s_klogk + si) + (b_klogmu - mu_b - b_klogk + bi)
        stati = -2.0 * logli
        stat += stati

        delta = si / ts - bi / tb - mi
        sign = 0.0
        if delta > 0.0:
            sign = 1.0
        elif delta < 0.0:
            sign = -1.0

        residual[i] = sign * np.sqrt(stati)

    return stat, residual


@nb.njit(cache=True, fastmath=True)
def _pgstat_core(S, B, m, ts, tb, sigma_B):
    
    n = S.shape[0]
    residual = np.empty(n, dtype=np.float64)
    stat = 0.0

    aa = tb * tb

    for i in range(n):
        si = S[i]
        bi = B[i]
        mi = m[i]
        sigma = sigma_B[i]

        bb = ts * sigma * sigma - tb * bi + tb * tb * mi
        cc = ts * sigma * sigma * mi - si * sigma * sigma - tb * bi * mi
        dd = np.sqrt(bb * bb - 4.0 * aa * cc)

        sgn = 1.0
        if bb < 0.0:
            sgn = -1.0

        qq = -0.5 * (bb + sgn * dd)

        b1 = qq / aa
        b2 = cc / qq
        b = b1 if b1 > 0.0 else b2

        mu_s = ts * (b + mi)

        s_klogmu = 0.0
        if si != 0.0:
            s_klogmu = si * np.log(mu_s)

        s_klogk = 0.0
        if si != 0.0:
            s_klogk = si * np.log(si)
            
        pois_logli = s_klogmu - mu_s - s_klogk + si

        z = (bi - tb * b) / sigma
        gauss_logli = -0.5 * z * z

        logli = pois_logli + gauss_logli
        stati = -2.0 * logli
        stat += stati

        delta = si / ts - bi / tb - mi
        sign = 0.0
        if delta > 0.0:
            sign = 1.0
        elif delta < 0.0:
            sign = -1.0

        residual[i] = sign * np.sqrt(stati)

    return stat, residual



class StatisticNB(object):

    @staticmethod
    def Gstat(**kwargs):
        return _gstat_core(
            kwargs['S'],
            kwargs['B'],
            kwargs['m'],
            kwargs['ts'],
            kwargs['tb'],
            kwargs['sigma_S'],
            kwargs['sigma_B'])


    @staticmethod
    def Pstat(**kwargs):
        return _pstat_core(
            kwargs['S'],
            kwargs['m'],
            kwargs['ts'])


    @staticmethod
    def PPstat(**kwargs):
        return _ppstat_core(
            kwargs['S'],
            kwargs['B'],
            kwargs['m'],
            kwargs['ts'],
            kwargs['tb'])


    @staticmethod
    def PGstat(**kwargs):
        return _pgstat_core(
            kwargs['S'],
            kwargs['B'],
            kwargs['m'],
            kwargs['ts'],
            kwargs['tb'],
            kwargs['sigma_B'])


    @staticmethod
    def PPstat_UL(**kwargs):
        
        B = kwargs['B']
        m = kwargs['m']

        ts = kwargs['ts']
        tb = kwargs['tb']
        alpha = ts / tb

        bkg_cts = np.sum(B)
        mo_cts = np.sum(m * ts)

        ul_sigma = 3.0
        sigma = ppsig(mo_cts + bkg_cts * alpha, bkg_cts, alpha)

        stat = (sigma - ul_sigma) ** 2
        residual = np.array([sigma - ul_sigma], dtype=np.float64)

        return stat, residual


    @staticmethod
    def PGstat_UL(**kwargs):
        
        B = kwargs['B']
        m = kwargs['m']

        ts = kwargs['ts']
        tb = kwargs['tb']
        alpha = ts / tb

        sigma_B = kwargs['sigma_B']

        bkg_cts = np.sum(B)
        mo_cts = np.sum(m * ts)
        bkg_err = np.sqrt(np.sum(sigma_B * sigma_B))

        ul_sigma = 3.0
        sigma = pgsig(mo_cts + bkg_cts * alpha, bkg_cts * alpha, bkg_err * alpha)

        stat = (sigma - ul_sigma) ** 2
        residual = np.array([sigma - ul_sigma], dtype=np.float64)

        return stat, residual



class Statistic(object):
    
    @staticmethod
    def xlogy(x, y):

        res = np.zeros_like(x, dtype=np.float64)
        
        zero = (x == 0)
        res[~zero] = x[~zero] * np.log(y[~zero])
        
        return res
    
    
    @staticmethod
    def xdivy(x, y):
        
        res = np.zeros_like(x, dtype=np.float64)
        
        zero = (x == 0) & (y == 0)
        res[~zero] = x[~zero] / y[~zero]
        
        return res
    
    
    @staticmethod
    def poisson_logpmf(k, mu):
        """
        NOTE
        logpmf (for poisson) = klogmu - mu - log(k!)
        Stirling's approximation: ln(K!) ~ klogk - k + log(2πk) / 2
        Here omit the term log(2πk) / 2
        """
        
        return Statistic.xlogy(k, mu) - mu - Statistic.xlogy(k, k) + k
    
    
    @staticmethod
    def gaussian_logpdf(x, loc, scale):
        """
        NOTE
        logpdf (for gaussian) = -0.5 * ((x-loc)^2/scale^2 + log(2π*scale^2))
        Here omit the term log(2π*scale^2)
        """

        return -0.5 * (Statistic.xdivy(x - loc, scale) ** 2)

    
    @staticmethod
    def Gstat(**kwargs):
        
        S = kwargs['S']
        B = kwargs['B']
        m = kwargs['m']
        
        ts = kwargs['ts']
        tb = kwargs['tb']
        
        sigma_S = kwargs['sigma_S']
        sigma_B = kwargs['sigma_B']
        
        if tb != 0:
            B = B / tb * ts
            sigma_B = sigma_B / tb * ts
        
        sigma = np.sqrt(sigma_S ** 2 + sigma_B ** 2)
        
        sign = np.sign(S - B - m * ts)
        loglike = Statistic.gaussian_logpdf(S - B, m * ts, sigma)
        
        stat = (-2 * loglike).sum()
        residual = sign * np.sqrt(-2 * loglike)
        
        return stat, residual
    
    
    @staticmethod
    def Pstat(**kwargs):
        
        S = kwargs['S']
        m = kwargs['m']
        ts = kwargs['ts']
        
        sign = np.sign(S - m * ts)
        loglike = Statistic.poisson_logpmf(S, m * ts)
        
        stat = (-2 * loglike).sum()
        residual = sign * np.sqrt(-2 * loglike)
        
        return stat, residual


    @staticmethod
    def PPstat(**kwargs):
        
        S = kwargs['S']
        B = kwargs['B']
        m = kwargs['m']
        
        ts = kwargs['ts']
        tb = kwargs['tb']
        
        aa = ts + tb
        bb = (ts + tb) * m - S - B
        cc = -B * m
        dd = np.sqrt(bb * bb - 4 * aa * cc)
        
        po = bb >= 0
        b = np.empty_like(B, dtype=np.float64)
        
        b[po] = -2 * cc[po] / (bb[po] + dd[po])
        b[~po] = -(bb[~po] - dd[~po]) / (2 * aa)
        
        sign = np.sign(S / ts - B / tb - m)
        loglike = Statistic.poisson_logpmf(S, ts * (b + m)) + Statistic.poisson_logpmf(B, tb * b)
        
        stat = (-2 * loglike).sum()
        residual = sign * np.sqrt(-2 * loglike)
        
        return stat, residual


    def PGstat(**kwargs):
        
        S = kwargs['S']
        B = kwargs['B']
        m = kwargs['m']
        
        ts = kwargs['ts']
        tb = kwargs['tb']
        
        sigma = kwargs['sigma_B']
        
        aa = tb ** 2
        bb = ts * sigma ** 2 - tb * B + tb ** 2 * m
        cc = ts * sigma ** 2 * m - S * sigma ** 2 - tb * B * m
        dd = np.sqrt(bb ** 2 - 4 * aa * cc)
        
        sign = np.where(bb >= 0, 1, -1)
        qq = -0.5 * (bb + sign * dd)
        
        b1 = qq / aa
        b2 = cc / qq
        b = np.where(b1 > 0, b1, b2)
        
        sign = np.sign(S / ts - B / tb - m)
        loglike = Statistic.poisson_logpmf(S, ts * (b + m)) + Statistic.gaussian_logpdf(B, tb * b, sigma)
        
        stat = (-2 * loglike).sum()
        residual = sign * np.sqrt(-2 * loglike)
        
        return stat, residual


    @staticmethod
    def PPstat_UL(**kwargs):
        
        B = kwargs['B']
        m = kwargs['m']
        
        ts = kwargs['ts']
        tb = kwargs['tb']
        alpha = ts / tb
        
        bkg_cts = np.sum(B)
        mo_cts = np.sum(m * ts)
        
        ul_sigma = 3.0
        sigma = ppsig(mo_cts + bkg_cts * alpha, bkg_cts, alpha)
        
        stat = (sigma - ul_sigma) ** 2
        residual = np.array([sigma - ul_sigma], dtype=np.float64)

        return stat, residual
    
    
    @staticmethod
    def PGstat_UL(**kwargs):
        
        B = kwargs['B']
        m = kwargs['m']
        
        ts = kwargs['ts']
        tb = kwargs['tb']
        alpha = ts / tb
        
        sigma_B = kwargs['sigma_B']
        
        bkg_cts = np.sum(B)
        mo_cts = np.sum(m * ts)
        bkg_err = np.sqrt(np.sum(sigma_B * sigma_B))
        
        ul_sigma = 3.0
        sigma = pgsig(mo_cts + bkg_cts * alpha, bkg_cts * alpha, bkg_err * alpha)

        stat = (sigma - ul_sigma) ** 2
        residual = np.array([sigma - ul_sigma], dtype=np.float64)

        return stat, residual