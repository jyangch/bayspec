import numpy as np

from ..util.significance import ppsig, pgsig



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
        
        return -2 * Statistic.gaussian_logpdf(S - B, m * ts, sigma).sum()
    
    
    @staticmethod
    def Pstat(**kwargs):
        
        S = kwargs['S']
        m = kwargs['m']
        ts = kwargs['ts']
        
        return -2 * Statistic.poisson_logpmf(S, m * ts).sum()


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
        
        return -2 * (Statistic.poisson_logpmf(S, ts * (b + m)).sum() \
            + Statistic.poisson_logpmf(B, tb * b).sum())


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
        
        return -2 * (Statistic.poisson_logpmf(S, ts * (b + m)).sum() \
            + Statistic.gaussian_logpdf(B, tb * b, sigma).sum())

    
    @staticmethod
    def PPstat_Xspec(**kwargs):
        
        S = kwargs['S']
        B = kwargs['B']
        m = kwargs['m']
        
        ts = kwargs['ts']
        tb = kwargs['tb']
        
        FLOOR = 1.0e-5
        stat = 0

        for i in range(len(S)):
            si = S[i]
            bi = B[i]
            mi = m[i]
            tt = ts + tb
            mi = max(mi, FLOOR / ts)

            if si == 0.0:
                stat += ts * mi - bi * np.log(tb / tt)
            else:
                if bi == 0.0:
                    if mi <= si / tt:
                        stat += -tb * mi - si * np.log(ts / tt)
                    else:
                        stat += ts * mi + si * (np.log(si) - np.log(ts * mi) - 1)
                else:
                    # now the main case where both data and background !=0
                    # Solve quadratic equation for f. Use the positive root to ensure
                    # that f > 0.
                    a = tt
                    b = tt * mi - si - bi
                    c = -bi * mi
                    d = np.sqrt(b * b - 4.0 * a * c)
                    # Avoid round-off error problems if b^2 >> 4ac (see eg Num.Recipes)
                    if b >= 0:
                        fi = -2 * c / (b + d)
                    else:
                        fi = -(b - d) / (2 * a)
                    # note that at this point f must be > 0 so the log
                    # functions below will be valid.
                    stat += ts * mi + tt * fi - si * np.log(ts * mi + ts * fi) - bi * np.log(tb * fi) \
                            - si * (1 - np.log(si)) - bi * (1 - np.log(bi))

        return 2.0 * stat


    @staticmethod
    def PGstat_Xspec(**kwargs):
        
        S = kwargs['S']
        B = kwargs['B']
        m = kwargs['m']
        
        ts = kwargs['ts']
        tb = kwargs['tb']
        
        sigma = kwargs['sigma_B']

        FLOOR = 1.0e-5
        stat = 0

        for i in range(len(S)):
            si = S[i]
            bi = B[i]
            mi = m[i]
            sigmai = sigma[i]
            tr = ts / tb

            # special case for sigmai = 0
            if sigmai == 0.0:
                mbi = max(mi + bi / tb, FLOOR / ts)
                stat += ts * mbi
                if si > 0.0:
                    stat += si * (np.log(si) - np.log(ts * mbi) - 1)
            else:
                if si == 0.0:
                    stat += ts * mi + bi * tr - 0.5 * (sigmai * tr) ** 2
                else:
                    # Solve quadratic equation for fi, using Numerical Recipes technique
                    # to avoid round-off error that can easily cause problems here
                    # when b^2 >> ac.
                    a = tb ** 2
                    b = ts * sigmai ** 2 - tb * bi + tb ** 2 * mi
                    c = ts * sigmai ** 2 * mi - si * sigmai ** 2 - tb * bi * mi
                    if b >= 0.0:
                        sign = 1.0
                    else:
                        sign = -1.0
                    q = -0.5 * (b + sign * np.sqrt(b ** 2 - 4.0 * a * c))
                    fi = q / a
                    if fi < 0.0:
                        fi = c / q
                    # note that at this point fi must be > 0 so the log
                    # functions below will be valid.
                    stat += ts * (mi + fi) - si * np.log(ts * mi + ts * fi) + \
                            0.5 * (bi - tb * fi) * (bi - tb * fi) / sigmai ** 2 - si * (1 - np.log(si))

        return 2.0 * stat


    @staticmethod
    def PPstat_UL(**kwargs):
        
        B = kwargs['B']
        m = kwargs['m']
        
        ts = kwargs['ts']
        tb = kwargs['tb']
        alpha = ts / tb
        
        bkg_cts = np.sum(B)
        mo_cts = np.sum(m * ts)
        
        ul_sigma = 3
        sigma = ppsig(mo_cts + bkg_cts * alpha, bkg_cts, alpha)

        return (sigma - ul_sigma) ** 2 / 0.01
    
    
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
        
        ul_sigma = 3
        sigma = pgsig(mo_cts + bkg_cts * alpha, bkg_cts * alpha, bkg_err * alpha)

        return (sigma - ul_sigma) ** 2 / 0.01
