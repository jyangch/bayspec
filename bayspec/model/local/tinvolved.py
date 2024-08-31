import toml
import numpy as np
import subprocess as sp
import astropy.units as u
from ..model import Tinvolved
from ...util.prior import unif
from ...util.param import Par, Cfg
from collections import OrderedDict
from os.path import dirname, abspath
from astropy.cosmology import Planck18
docs_path = dirname(dirname(dirname(abspath(__file__)))) + '/docs'



class hlecpl(Tinvolved):
    # 10.1088/0004-637X/690/1/L10
    
    def __init__(self):
        super().__init__()
        
        self.expr = 'hlecpl'
        self.comment = 'curvature effect model for cpl function'

        self.params = OrderedDict()
        self.params['$\\alpha$'] = Par(-1, unif(-2, 2))
        self.params['log$E_{p,c}$'] = Par(2, unif(0, 4))
        self.params['log$A_{c}$'] = Par(0, unif(-6, 6))
        self.params['$t_{0}$'] = Par(0, unif(-20, 20))
        self.params['$t_{c}$'] = Par(10, unif(0, 50))


    def func(self, E, T, O=None):
        alpha = self.params['$\\alpha$'].value
        logEpc = self.params['log$E_{p,c}$'].value
        logAc = self.params['log$A_{c}$'].value
        t0 = self.params['$t_{0}$'].value
        tc = self.params['$t_{c}$'].value

        Epc = 10 ** logEpc
        Ac = 10 ** logAc

        if tc <= t0 or tc > np.min(T):
            return np.ones_like(E) * np.nan

        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        Ept = Epc * ((T - t0) / (tc - t0)) ** (- 1)
        At = Ac * ((T - t0) / (tc - t0)) ** (alpha - 1)

        phtspec = At * (E / 100) ** alpha * np.exp(-(2 + alpha) * E / Ept)
        return phtspec



class hleband(Tinvolved):
    # 10.1088/0004-637X/690/1/L10
    
    def __init__(self):
        super().__init__()
        
        self.expr = 'hleband'
        self.comment = 'curvature effect model for band function'

        self.params = OrderedDict()
        self.params['$\\alpha$'] = Par(-1, unif(-2, 2))
        self.params['$\\beta$'] = Par(-4, unif(-6, -2))
        self.params['log$E_{p,c}$'] = Par(2, unif(0, 4))
        self.params['log$A_{c}$'] = Par(0, unif(-6, 6))
        self.params['$t_{0}$'] = Par(0, unif(-20, 20))
        self.params['$t_{c}$'] = Par(10, unif(0, 50))


    def func(self, E, T, O=None):
        alpha = self.params['$\\alpha$'].value
        beta = self.params['$\\beta$'].value
        logEpc = self.params['log$E_{p,c}$'].value
        logAc = self.params['log$A_{c}$'].value
        t0 = self.params['$t_{0}$'].value
        tc = self.params['$t_{c}$'].value

        Epc = 10 ** logEpc
        Ac = 10 ** logAc

        if tc <= t0 or tc > np.min(T):
            return np.ones_like(E) * np.nan

        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        Ept = Epc * ((T - t0) / (tc - t0)) ** (- 1)
        At = Ac * ((T - t0) / (tc - t0)) ** (alpha - 1)
        Ebt = (alpha - beta) / (alpha + 2) * Ept
        phtspec = np.zeros_like(E, dtype=float)

        i1 = E < Ebt; i2 = E >= Ebt
        phtspec[i1] = At[i1] * (E[i1] / 100) ** alpha * np.exp(-(2 + alpha) * E[i1] / Ept[i1])
        phtspec[i2] = At[i2] * (Ebt[i2] / 100) ** (alpha - beta) * (E[i2] / 100) ** beta * np.exp(beta - alpha)
        return phtspec
