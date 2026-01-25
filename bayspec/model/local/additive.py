import os
import toml
import numpy as np
import subprocess as sp
import astropy.units as u
from scipy.integrate import quad
from collections import OrderedDict
from os.path import dirname, abspath
from astropy.cosmology import Planck18
docs_path = dirname(dirname(dirname(abspath(__file__)))) + '/docs'

from ..model import Additive
from ...util.prior import unif
from ...util.param import Par, Cfg
from ...util.tools import cached_property



class pl(Additive):

    def __init__(self):
        
        self.expr = 'pl'
        self.comment = 'power-law model'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['pivot_energy'] = Cfg(1.0)
        
        self.params = OrderedDict()
        self.params[r'$\alpha$'] = Par(0, unif(-10, 10))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))


    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        epiv = self.config['pivot_energy'].value
        
        alpha = self.params[r'$\alpha$'].value
        
        logA = self.params[r'log$A$'].value
        Amp = 10 ** logA
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]
        
        zi = 1 + redshift
        E = E * zi

        phtspec = Amp * (E / epiv) ** alpha
        
        return phtspec[0] if scalar else phtspec



class cpl(Additive):

    def __init__(self):
        
        self.expr = 'cpl'
        self.comment = 'power-law model with high-energy cutoff'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['pivot_energy'] = Cfg(1.0)
        self.config['vfv_peak'] = Cfg(True)
        
        
    @cached_property(lambda self: self.config['vfv_peak'].value)
    def params(self):
        
        params = OrderedDict()
        
        if self.config['vfv_peak'].value:
            params[r'$\alpha$'] = Par(-1, unif(-2, 2))
            params[r'log$E_p$'] = Par(2, unif(0, 4))
            params[r'log$A$'] = Par(0, unif(-10, 10))

        elif not self.config['vfv_peak'].value:
            params[r'$\alpha$'] = Par(-1, unif(-8, 4))
            params[r'log$E_c$'] = Par(2, unif(0, 4))
            params[r'log$A$'] = Par(0, unif(-10, 10))
            
        else:
            raise ValueError('Invalid value for vfv_peak config.')

        return params


    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        epiv = self.config['pivot_energy'].value
        peak = self.config['vfv_peak'].value
        
        alpha = self.params[r'$\alpha$'].value
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]
        
        if peak:
            logEp = self.params[r'log$E_p$'].value
            Ep = 10 ** logEp
            
            if not alpha > -2:
                return np.nan if scalar else np.ones_like(E) * np.nan

            Ec = Ep / (2 + alpha)
            
        else:
            logEc = self.params[r'log$E_c$'].value
            Ec = 10 ** logEc

        logA = self.params[r'log$A$'].value
        Amp = 10 ** logA

        zi = 1 + redshift
        E = E * zi

        phtspec = Amp * (E / epiv) ** alpha * np.exp(-E / Ec)
        
        return phtspec[0] if scalar else phtspec



class sbpl(Additive):
    # 10.1086/505911
    
    def __init__(self):
        
        self.expr = 'sbpl'
        self.comment = 'smoothly broken power-law model'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['pivot_energy'] = Cfg(1.0)
        self.config['vfv_peak'] = Cfg(True)
        self.config['smoothness'] = Cfg(0.3)
        
        
    @cached_property(lambda self: self.config['vfv_peak'].value)
    def params(self):
        
        params = OrderedDict()
        
        if self.config['vfv_peak'].value:
            params[r'$\alpha$'] = Par(-1, unif(-2, 2))
            params[r'$\beta$'] = Par(-3, unif(-6, -2))
            params[r'log$E_p$'] = Par(2, unif(0, 4))
            params[r'log$A$'] = Par(0, unif(-10, 10))
        
        elif not self.config['vfv_peak'].value:
            params[r'$\alpha_1$'] = Par(-1, unif(-6, 4))
            params[r'$\alpha_2$'] = Par(-3, unif(-6, 4))
            params[r'log$E_b$'] = Par(2, unif(0, 4))
            params[r'log$A$'] = Par(0, unif(-10, 10))
            
        else:
            raise ValueError('Invalid value for vfv_peak config.')

        return params


    @staticmethod
    def _log_cosh(q):

        return np.logaddexp(q, -q) - np.log(2.0)


    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        epiv = self.config['pivot_energy'].value
        peak = self.config['vfv_peak'].value
        delta = self.config['smoothness'].value
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]
        
        if peak:
            alpha1 = self.params[r'$\alpha$'].value
            alpha2 = self.params[r'$\beta$'].value
            
            logEp = self.params[r'log$E_p$'].value
            Ep = 10 ** logEp
            
            if not (alpha1 > -2 and alpha2 < -2):
                return np.nan if scalar else np.ones_like(E) * np.nan
            
            Eb = Ep / (10 ** (delta * np.arctanh((alpha1 + alpha2 + 4) / (alpha1 - alpha2))))
        
        else:
            alpha1 = self.params[r'$\alpha_1$'].value
            alpha2 = self.params[r'$\alpha_2$'].value
            
            logEb = self.params[r'log$E_b$'].value
            Eb = 10 ** logEb

        b = (alpha1 + alpha2) / 2
        m = (alpha2 - alpha1) / 2
            
        logA = self.params[r'log$A$'].value
        Amp = 10 ** logA
        
        zi = 1 + redshift
        E = E * zi
            
        q = np.log10(E / Eb) / delta
        qpiv = np.log10(epiv / Eb) / delta
        
        a = m * delta * self._log_cosh(q)
        apiv = m * delta * self._log_cosh(qpiv)
        
        phtspec = Amp * (E / epiv) ** b * 10 ** (a - apiv)
        
        return phtspec[0] if scalar else phtspec
    
    
    def slope_func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        peak = self.config['vfv_peak'].value
        delta = self.config['smoothness'].value
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]
        
        if peak:
            alpha1 = self.params[r'$\alpha$'].value
            alpha2 = self.params[r'$\beta$'].value
            
            logEp = self.params[r'log$E_p$'].value
            Ep = 10 ** logEp
            
            if not (alpha1 > -2 and alpha2 < -2):
                return np.nan if scalar else np.ones_like(E) * np.nan
            
            Eb = Ep / (10 ** (delta * np.arctanh((alpha1 + alpha2 + 4) / (alpha1 - alpha2))))
            
        else:
            alpha1 = self.params[r'$\alpha_1$'].value
            alpha2 = self.params[r'$\alpha_2$'].value
            
            logEb = self.params[r'log$E_b$'].value
            Eb = 10 ** logEb

        b = (alpha1 + alpha2) / 2
        m = (alpha2 - alpha1) / 2

        zi = 1 + redshift
        E = E * zi
        
        q = np.log10(E / Eb) / delta
        
        slope = b + m * np.tanh(q)
        
        return slope[0] if scalar else slope



class csbpl(Additive):
    
    def __init__(self):
        
        self.expr = 'csbpl'
        self.comment = 'smoothly broken power-law model with high-energy cutoff'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['pivot_energy'] = Cfg(1.0)
        self.config['vfv_peak'] = Cfg(True)
        self.config['smoothness'] = Cfg(0.3)
        
        
    @cached_property(lambda self: self.config['vfv_peak'].value)
    def params(self):
        
        params = OrderedDict()
        
        if self.config['vfv_peak'].value:
            params[r'$\alpha_1$'] = Par(1, unif(-2, 2))
            params[r'$\alpha_2$'] = Par(-1, unif(-2, 2))
            params[r'log$E_b$'] = Par(1, unif(-1, 3))
            params[r'log$E_p$'] = Par(2, unif(0, 4))
            params[r'log$A$'] = Par(0, unif(-10, 10))
        
        elif not self.config['vfv_peak'].value:
            params[r'$\alpha_1$'] = Par(1, unif(-6, 4))
            params[r'$\alpha_2$'] = Par(-1, unif(-6, 4))
            params[r'log$E_b$'] = Par(1, unif(-1, 3))
            params[r'log$E_c$'] = Par(2, unif(0, 4))
            params[r'log$A$'] = Par(0, unif(-10, 10))
            
        else:
            raise ValueError('Invalid value for vfv_peak config.')

        return params


    @staticmethod
    def _log_cosh(q):

        return np.logaddexp(q, -q) - np.log(2.0)


    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        epiv = self.config['pivot_energy'].value
        peak = self.config['vfv_peak'].value
        delta = self.config['smoothness'].value
        
        alpha1 = self.params[r'$\alpha_1$'].value
        alpha2 = self.params[r'$\alpha_2$'].value
        
        logEb = self.params[r'log$E_b$'].value
        Eb = 10 ** logEb
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]
        
        if peak:
            logEp = self.params[r'log$E_p$'].value
            Ep = 10 ** logEp
            
            if not alpha2 > -2:
                return np.nan if scalar else np.ones_like(E) * np.nan
            
            Ec = Ep / (2 + alpha2)
            
        else:
            logEc = self.params[r'log$E_c$'].value
            Ec = 10 ** logEc

        logA = self.params[r'log$A$'].value
        Amp = 10 ** logA

        zi = 1 + redshift
        E = E * zi

        b = (alpha1 + alpha2) / 2
        m = (alpha2 - alpha1) / 2
        
        q = np.log10(E / Eb) / delta
        qpiv = np.log10(epiv / Eb) / delta
        
        a = m * delta * self._log_cosh(q)
        apiv = m * delta * self._log_cosh(qpiv)
        
        phtspec = Amp * (E / epiv) ** b * 10 ** (a - apiv) * np.exp(-E / Ec)
        
        return phtspec[0] if scalar else phtspec



class dsbpl(Additive):

    def __init__(self):

        self.expr = 'dsbpl'
        self.comment = 'double smoothly broken power-law model'

        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['pivot_energy'] = Cfg(1.0)
        self.config['smoothness1'] = Cfg(0.3)
        self.config['smoothness2'] = Cfg(0.3)

        self.params = OrderedDict()
        self.params[r'$\alpha_1$'] = Par(1, unif(-6, 4))
        self.params[r'$\alpha_2$'] = Par(-1, unif(-6, 4))
        self.params[r'$\alpha_3$'] = Par(-3, unif(-6, 4))
        self.params[r'log$E_{b1}$'] = Par(1, unif(-1, 3))
        self.params[r'log$E_{b2}$'] = Par(2, unif(0, 4))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))


    @staticmethod
    def _log_cosh(q):

        return np.logaddexp(q, -q) - np.log(2.0)


    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        epiv = self.config['pivot_energy'].value
        delta1 = self.config['smoothness1'].value
        delta2 = self.config['smoothness2'].value
        
        alpha1 = self.params[r'$\alpha_1$'].value
        alpha2 = self.params[r'$\alpha_2$'].value
        alpha3 = self.params[r'$\alpha_3$'].value
        logEb1 = self.params[r'log$E_{b1}$'].value
        logEb2 = self.params[r'log$E_{b2}$'].value
        logA = self.params[r'log$A$'].value

        Eb1 = 10.0 ** logEb1
        Eb2 = 10.0 ** logEb2
        Amp = 10.0 ** logA
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]

        zi = 1.0 + redshift
        E = E * zi
        
        b = 0.5 * (alpha1 + alpha3)
        m1 = 0.5 * (alpha2 - alpha1)
        m2 = 0.5 * (alpha3 - alpha2)

        q1 = np.log10(E / Eb1) / delta1
        q2 = np.log10(E / Eb2) / delta2
        
        qpiv1 = np.log10(epiv / Eb1) / delta1
        qpiv2 = np.log10(epiv / Eb2) / delta2
        
        a1 = m1 * delta1 * self._log_cosh(q1)
        a2 = m2 * delta2 * self._log_cosh(q2)
        
        apiv1 = m1 * delta1 * self._log_cosh(qpiv1)
        apiv2 = m2 * delta2 * self._log_cosh(qpiv2)
        
        phtspec = Amp * (E / epiv) ** b * 10.0 ** ((a1 + a2) - (apiv1 + apiv2))
        
        return phtspec[0] if scalar else phtspec
    
    
    def slope_func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        delta1 = self.config['smoothness1'].value
        delta2 = self.config['smoothness2'].value
        
        alpha1 = self.params[r'$\alpha_1$'].value
        alpha2 = self.params[r'$\alpha_2$'].value
        alpha3 = self.params[r'$\alpha_3$'].value
        logEb1 = self.params[r'log$E_{b1}$'].value
        logEb2 = self.params[r'log$E_{b2}$'].value

        Eb1 = 10 ** logEb1
        Eb2 = 10 ** logEb2
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]

        zi = 1 + redshift
        E = E * zi

        b = 0.5 * (alpha1 + alpha3)
        m1 = 0.5 * (alpha2 - alpha1)
        m2 = 0.5 * (alpha3 - alpha2)

        q1 = np.log10(E / Eb1) / delta1
        q2 = np.log10(E / Eb2) / delta2
        
        slope = b + m1 * np.tanh(q1) + m2 * np.tanh(q2)
        
        return slope[0] if scalar else slope



class tsbpl(Additive):
    
    def __init__(self):
        
        self.expr = 'tsbpl'
        self.comment = 'triple smoothly broken power-law model'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['pivot_energy'] = Cfg(1.0)
        self.config['smoothness1'] = Cfg(0.3)
        self.config['smoothness2'] = Cfg(0.3)
        self.config['smoothness3'] = Cfg(0.3)

        self.params = OrderedDict()
        self.params[r'$\alpha_1$'] = Par(2, unif(-6, 4))
        self.params[r'$\alpha_2$'] = Par(1, unif(-6, 4))
        self.params[r'$\alpha_3$'] = Par(-1, unif(-6, 4))
        self.params[r'$\alpha_4$'] = Par(-3, unif(-6, 4))
        self.params[r'log$E_{b1}$'] = Par(1, unif(-1, 3))
        self.params[r'log$E_{b2}$'] = Par(2, unif(0, 4))
        self.params[r'log$E_{b3}$'] = Par(3, unif(1, 5))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))


    @staticmethod
    def _log_cosh(q):

        return np.logaddexp(q, -q) - np.log(2.0)


    def func(self, E, T=None, O=None):

        redshift = self.config['redshift'].value
        epiv = self.config['pivot_energy'].value
        delta1 = self.config['smoothness1'].value
        delta2 = self.config['smoothness2'].value
        delta3 = self.config['smoothness3'].value

        alpha1 = self.params[r'$\alpha_1$'].value
        alpha2 = self.params[r'$\alpha_2$'].value
        alpha3 = self.params[r'$\alpha_3$'].value
        alpha4 = self.params[r'$\alpha_4$'].value
        logEb1 = self.params[r'log$E_{b1}$'].value
        logEb2 = self.params[r'log$E_{b2}$'].value
        logEb3 = self.params[r'log$E_{b3}$'].value
        logA = self.params[r'log$A$'].value
        
        Eb1 = 10.0 ** logEb1
        Eb2 = 10.0 ** logEb2
        Eb3 = 10.0 ** logEb3
        Amp = 10.0 ** logA
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]
        
        zi = 1.0 + redshift
        E = E * zi

        b = 0.5 * (alpha1 + alpha4) 
        m1 = 0.5 * (alpha2 - alpha1)
        m2 = 0.5 * (alpha3 - alpha2)
        m3 = 0.5 * (alpha4 - alpha3)
        
        q1 = np.log10(E / Eb1) / delta1
        q2 = np.log10(E / Eb2) / delta2
        q3 = np.log10(E / Eb3) / delta3
        
        qpiv1 = np.log10(epiv / Eb1) / delta1
        qpiv2 = np.log10(epiv / Eb2) / delta2
        qpiv3 = np.log10(epiv / Eb3) / delta3

        a1 = m1 * delta1 * self._log_cosh(q1)
        a2 = m2 * delta2 * self._log_cosh(q2)
        a3 = m3 * delta3 * self._log_cosh(q3)

        apiv1 = m1 * delta1 * self._log_cosh(qpiv1)
        apiv2 = m2 * delta2 * self._log_cosh(qpiv2)
        apiv3 = m3 * delta3 * self._log_cosh(qpiv3)

        shape_term = (a1 + a2 + a3) - (apiv1 + apiv2 + apiv3)
        phtspec = Amp * (E / epiv) ** b * 10.0 ** shape_term
        
        return phtspec[0] if scalar else phtspec


    def slope_func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        delta1 = self.config['smoothness1'].value
        delta2 = self.config['smoothness2'].value
        delta3 = self.config['smoothness3'].value
        
        alpha1 = self.params[r'$\alpha_1$'].value
        alpha2 = self.params[r'$\alpha_2$'].value
        alpha3 = self.params[r'$\alpha_3$'].value
        alpha4 = self.params[r'$\alpha_4$'].value
        logEb1 = self.params[r'log$E_{b1}$'].value
        logEb2 = self.params[r'log$E_{b2}$'].value
        logEb3 = self.params[r'log$E_{b3}$'].value

        Eb1 = 10.0 ** logEb1
        Eb2 = 10.0 ** logEb2
        Eb3 = 10.0 ** logEb3
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]

        zi = 1 + redshift
        E = E * zi

        b = 0.5 * (alpha1 + alpha4) 
        m1 = 0.5 * (alpha2 - alpha1)
        m2 = 0.5 * (alpha3 - alpha2)
        m3 = 0.5 * (alpha4 - alpha3)
        
        q1 = np.log10(E / Eb1) / delta1
        q2 = np.log10(E / Eb2) / delta2
        q3 = np.log10(E / Eb3) / delta3
        
        slope = b + m1 * np.tanh(q1) + m2 * np.tanh(q2) + m3 * np.tanh(q3)
        
        return slope[0] if scalar else slope



class sb2pl(Additive):
    # 10.1051/0004-6361/201732245
    
    def __init__(self):
        
        self.expr = 'sb2pl'
        self.comment = '2-segment smoothly broken power-law model (always convex)'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['pivot_energy'] = Cfg(1.0)
        self.config['vfv_peak'] = Cfg(True)
        self.config['smoothness'] = Cfg(2.0)
        
        
    @cached_property(lambda self: self.config['vfv_peak'].value)
    def params(self):
        
        params = OrderedDict()
        
        if self.config['vfv_peak'].value:
            params[r'$\alpha$'] = Par(-1, unif(-2, 2))
            params[r'$\beta$'] = Par(-3, unif(-6, -2))
            params[r'log$E_p$'] = Par(2, unif(0, 4))
            params[r'log$A$'] = Par(0, unif(-10, 10))
        
        elif not self.config['vfv_peak'].value:
            params[r'$\alpha_1$'] = Par(-1, unif(-6, 4))
            params[r'$\alpha_2$'] = Par(-3, unif(-6, 4))
            params[r'log$E_b$'] = Par(2, unif(0, 4))
            params[r'log$A$'] = Par(0, unif(-10, 10))
            
        else:
            raise ValueError('Invalid value for vfv_peak config.')

        return params


    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        epiv = self.config['pivot_energy'].value
        peak = self.config['vfv_peak'].value
        omega = self.config['smoothness'].value
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]
        
        if peak:
            alpha1 = self.params[r'$\alpha$'].value
            alpha2 = self.params[r'$\beta$'].value
            
            logEp = self.params[r'log$E_p$'].value
            Ep = 10 ** logEp
            
            if not (alpha1 > -2 and alpha2 < -2):
                return np.nan if scalar else np.ones_like(E) * np.nan
            
            Eb = Ep * (-(alpha1 + 2) / (alpha2 + 2)) ** (1 / ((alpha2 - alpha1) * omega))

        else:
            alpha1 = self.params[r'$\alpha_1$'].value
            alpha2 = self.params[r'$\alpha_2$'].value
            
            logEb = self.params[r'log$E_b$'].value
            Eb = 10 ** logEb
            
            if not alpha1 > alpha2:
                return np.nan if scalar else np.ones_like(E) * np.nan
            
        logA = self.params[r'log$A$'].value
        Amp = 10 ** logA
        
        zi = 1 + redshift
        E = E * zi
            
        f = ((E / Eb) ** (-alpha1 * omega) + (E / Eb) ** (-alpha2 * omega)) ** (-1 / omega)
        fpiv = ((epiv / Eb) ** (-alpha1 * omega) + (epiv / Eb) ** (-alpha2 * omega)) ** (-1 / omega)
            
        phtspec = Amp * (f / fpiv)

        return phtspec[0] if scalar else phtspec



class csb2pl(Additive):

    def __init__(self):
        
        self.expr = 'csb2pl'
        self.comment = '2-segment smoothly broken power-law model (always convex) with high-energy cutoff'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['pivot_energy'] = Cfg(1.0)
        self.config['vfv_peak'] = Cfg(True)
        self.config['smoothness'] = Cfg(2.0)
        
        
    @cached_property(lambda self: self.config['vfv_peak'].value)
    def params(self):
        
        params = OrderedDict()
        
        if self.config['vfv_peak'].value:
            params[r'$\alpha_1$'] = Par(1, unif(-2, 2))
            params[r'$\alpha_2$'] = Par(-1, unif(-2, 2))
            params[r'log$E_b$'] = Par(1, unif(-1, 3))
            params[r'log$E_p$'] = Par(2, unif(0, 4))
            params[r'log$A$'] = Par(0, unif(-10, 10))
        
        elif not self.config['vfv_peak'].value:
            params[r'$\alpha_1$'] = Par(1, unif(-6, 4))
            params[r'$\alpha_2$'] = Par(-1, unif(-6, 4))
            params[r'log$E_b$'] = Par(1, unif(-1, 3))
            params[r'log$E_c$'] = Par(2, unif(0, 4))
            params[r'log$A$'] = Par(0, unif(-10, 10))
            
        else:
            raise ValueError('Invalid value for vfv_peak config.')

        return params


    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        epiv = self.config['pivot_energy'].value
        peak = self.config['vfv_peak'].value
        omega = self.config['smoothness'].value
        
        alpha1 = self.params[r'$\alpha_1$'].value
        alpha2 = self.params[r'$\alpha_2$'].value
        
        logEb = self.params[r'log$E_b$'].value
        Eb = 10 ** logEb
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]
        
        if not alpha1 > alpha2:
            return np.nan if scalar else np.ones_like(E) * np.nan
        
        if peak:
            logEp = self.params[r'log$E_p$'].value
            Ep = 10 ** logEp
            
            if not alpha2 > -2:
                return np.nan if scalar else np.ones_like(E) * np.nan
            
            Ec = Ep / (2 + alpha2)
            
        else:
            logEc = self.params[r'log$E_c$'].value
            Ec = 10 ** logEc

        logA = self.params[r'log$A$'].value
        Amp = 10 ** logA

        zi = 1 + redshift
        E = E * zi
        
        f = ((E / Eb) ** (-alpha1 * omega) + (E / Eb) ** (-alpha2 * omega)) \
            ** (-1 / omega) * np.exp(-E / Ec)
        fpiv = ((epiv / Eb) ** (-alpha1 * omega) + (epiv / Eb) ** (-alpha2 * omega)) \
            ** (-1 / omega)
        phtspec = Amp * (f / fpiv)

        return phtspec[0] if scalar else phtspec



class sb3pl(Additive):
    
    def __init__(self):
        
        self.expr = 'sb3pl'
        self.comment = '3-segment smoothly broken power-law model (always convex)'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['pivot_energy'] = Cfg(1.0)
        self.config['smoothness1'] = Cfg(2.0)
        self.config['smoothness2'] = Cfg(2.0)
        
        self.params = OrderedDict()
        self.params[r'$\alpha_1$'] = Par(1, unif(-6, 4))
        self.params[r'$\alpha_2$'] = Par(-1, unif(-6, 4))
        self.params[r'$\alpha_3$'] = Par(-3, unif(-6, 4))
        self.params[r'log$E_{b1}$'] = Par(1, unif(-1, 3))
        self.params[r'log$E_{b2}$'] = Par(2, unif(0, 4))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))


    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        epiv = self.config['pivot_energy'].value
        omega1 = self.config['smoothness1'].value
        omega2 = self.config['smoothness2'].value
        
        alpha1 = self.params[r'$\alpha_1$'].value
        alpha2 = self.params[r'$\alpha_2$'].value
        alpha3 = self.params[r'$\alpha_3$'].value
        logEb1 = self.params[r'log$E_{b1}$'].value
        logEb2 = self.params[r'log$E_{b2}$'].value
        logA = self.params[r'log$A$'].value

        Eb1 = 10 ** logEb1
        Eb2 = 10 ** logEb2
        Amp = 10 ** logA
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]
        
        if not alpha1 > alpha2 > alpha3:
            return np.nan if scalar else np.ones_like(E) * np.nan

        zi = 1 + redshift
        E = E * zi
        
        f = self._sb3pl(E, [alpha1, alpha2, alpha3, Eb1, Eb2, omega1, omega2])
        fpiv = self._sb3pl(epiv, [alpha1, alpha2, alpha3, Eb1, Eb2, omega1, omega2])
        
        phtspec = Amp * (f / fpiv)

        return phtspec[0] if scalar else phtspec
    
    
    def _pl(self, E, P):
        
        alpha = P[0]
        Eb = P[1]
        
        return (E / Eb) ** alpha


    def _sb2pl(self, E, P):
        
        alpha1 = P[0]
        alpha2 = P[1]
        Eb = P[2]
        omega = P[3]
        
        F1 = self._pl(E, [alpha1, Eb])
        F2 = self._pl(E, [alpha2, Eb])
        F12 = (F1 ** (- omega) + F2 ** (- omega)) ** (- 1 / omega)
        
        return F12


    def _sb3pl(self, E, P):
        
        alpha1 = P[0]
        alpha2 = P[1]
        alpha3 = P[2]
        Eb1 = P[3]
        Eb2 = P[4]
        omega1 = P[5]
        omega2 = P[6]
        
        F12 = self._sb2pl(E, [alpha1, alpha2, Eb1, omega1])
        F3 = self._pl(E, [alpha3, Eb2]) * self._sb2pl(Eb2, [alpha1, alpha2, Eb1, omega1])
        F123 = (F12 ** (- omega2) + F3 ** (- omega2)) ** (- 1 / omega2)
        
        return F123



class sb4pl(Additive):

    def __init__(self):
        
        self.expr = 'sb4pl'
        self.comment = '4-segment smoothly broken power-law model (always convex)'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['pivot_energy'] = Cfg(1.0)
        self.config['smoothness1'] = Cfg(2.0)
        self.config['smoothness2'] = Cfg(2.0)
        self.config['smoothness3'] = Cfg(2.0)
        
        self.params = OrderedDict()
        self.params[r'$\alpha_1$'] = Par(2, unif(-6, 4))
        self.params[r'$\alpha_2$'] = Par(1, unif(-6, 4))
        self.params[r'$\alpha_3$'] = Par(-1, unif(-6, 4))
        self.params[r'$\alpha_4$'] = Par(-3, unif(-6, 4))
        self.params[r'log$E_{b1}$'] = Par(1, unif(-1, 3))
        self.params[r'log$E_{b2}$'] = Par(2, unif(0, 4))
        self.params[r'log$E_{b3}$'] = Par(3, unif(1, 5))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))
        
        
    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        epiv = self.config['pivot_energy'].value
        omega1 = self.config['smoothness1'].value
        omega2 = self.config['smoothness2'].value
        omega3 = self.config['smoothness3'].value
        
        alpha1 = self.params[r'$\alpha_1$'].value
        alpha2 = self.params[r'$\alpha_2$'].value
        alpha3 = self.params[r'$\alpha_3$'].value
        alpha4 = self.params[r'$\alpha_4$'].value
        logEb1 = self.params[r'log$E_{b1}$'].value
        logEb2 = self.params[r'log$E_{b2}$'].value
        logEb3 = self.params[r'log$E_{b3}$'].value
        logA = self.params[r'log$A$'].value
        
        Eb1 = 10 ** logEb1
        Eb2 = 10 ** logEb2
        Eb3 = 10 ** logEb3
        Amp = 10 ** logA
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]

        zi = 1 + redshift
        E = E * zi
        
        if not alpha1 > alpha2 > alpha3 > alpha4:
            return np.nan if scalar else np.ones_like(E) * np.nan
        
        f = self._sb4pl(E, [alpha1, alpha2, alpha3, alpha4, Eb1, Eb2, Eb3, omega1, omega2, omega3])
        fpiv = self._sb4pl(epiv, [alpha1, alpha2, alpha3, alpha4, Eb1, Eb2, Eb3, omega1, omega2, omega3])
        phtspec = Amp * (f / fpiv)

        return phtspec[0] if scalar else phtspec


    def _pl(self, E, P):
        
        alpha = P[0]
        Eb = P[1]
        
        return (E / Eb) ** alpha


    def _sb2pl(self, E, P):
        
        alpha1 = P[0]
        alpha2 = P[1]
        Eb = P[2]
        omega = P[3]
        
        F1 = self._pl(E, [alpha1, Eb])
        F2 = self._pl(E, [alpha2, Eb])
        F12 = (F1 ** (- omega) + F2 ** (- omega)) ** (- 1 / omega)
        
        return F12


    def _sb3pl(self, E, P):
        
        alpha1 = P[0]
        alpha2 = P[1]
        alpha3 = P[2]
        Eb1 = P[3]
        Eb2 = P[4]
        omega1 = P[5]
        omega2 = P[6]
        
        F12 = self._sb2pl(E, [alpha1, alpha2, Eb1, omega1])
        F3 = self._pl(E, [alpha3, Eb2]) * self._sb2pl(Eb2, [alpha1, alpha2, Eb1, omega1])
        F123 = (F12 ** (- omega2) + F3 ** (- omega2)) ** (- 1 / omega2)
        
        return F123


    def _sb4pl(self, E, P):
        
        alpha1 = P[0]
        alpha2 = P[1]
        alpha3 = P[2]
        alpha4 = P[3]
        Eb1 = P[4]
        Eb2 = P[5]
        Eb3 = P[6]
        omega1 = P[7]
        omega2 = P[8]
        omega3 = P[9]
        
        F123 = self._sb3pl(E, [alpha1, alpha2, alpha3, Eb1, Eb2, omega1, omega2])
        F4 = self._pl(E, [alpha4, Eb3]) * self._sb3pl(Eb3, [alpha1, alpha2, alpha3, Eb1, Eb2, omega1, omega2])
        F1234 = (F123 ** (- omega3) + F4 ** (- omega3)) ** (- 1 / omega3)
        
        return F1234



class band(Additive):
    # 10.1086/172995

    def __init__(self):
        
        self.expr = 'band'
        self.comment = 'band function'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['pivot_energy'] = Cfg(100.0)

        self.params = OrderedDict()
        self.params[r'$\alpha$'] = Par(-1, unif(-2, 2))
        self.params[r'$\beta$'] = Par(-3, unif(-6, -2))
        self.params[r'log$E_p$'] = Par(2, unif(0, 4))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))


    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        epiv = self.config['pivot_energy'].value
        
        alpha = self.params[r'$\alpha$'].value
        beta = self.params[r'$\beta$'].value
        logEp = self.params[r'log$E_p$'].value
        logA = self.params[r'log$A$'].value

        Ep = 10 ** logEp
        Amp = 10 ** logA
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]

        zi = 1 + redshift
        E = E * zi

        Ec = Ep / (2 + alpha)
        Eb = (alpha - beta) * Ec
        phtspec = np.zeros_like(E, dtype=float)

        i1 = E <= Eb; i2 = E > Eb
        phtspec[i1] = Amp * (E[i1] / epiv) ** alpha * np.exp(-E[i1] / Ec)
        phtspec[i2] = Amp * (Eb / epiv) ** (alpha - beta) * np.exp(beta - alpha) \
            * (E[i2] / epiv) ** beta
        
        return phtspec[0] if scalar else phtspec



class cband(Additive):
    # 10.1088/0004-637X/751/2/90
    
    def __init__(self):
        
        self.expr = 'cband'
        self.comment = 'band function with high-energy cutoff'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['pivot_energy'] = Cfg(1.0)

        self.params = OrderedDict()
        self.params[r'$\alpha_1$'] = Par(1, unif(-2, 2))
        self.params[r'$\alpha_2$'] = Par(-1, unif(-2, 2))
        self.params[r'log$E_b$'] = Par(1, unif(0, 3))
        self.params[r'log$E_p$'] = Par(3, unif(1, 4))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))


    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        epiv = self.config['pivot_energy'].value
        
        alpha1 = self.params[r'$\alpha_1$'].value
        alpha2 = self.params[r'$\alpha_2$'].value
        logEb = self.params[r'log$E_b$'].value
        logEp = self.params[r'log$E_p$'].value
        logA = self.params[r'log$A$'].value

        Eb = 10 ** logEb
        Ep = 10 ** logEp
        Amp = 10 ** logA
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]

        zi = 1 + redshift
        E = E * zi

        Ec2 = Ep / (2 + alpha2)
        Ec1 = 1 / (1 / Ec2 + (alpha1 - alpha2) / Eb)
        phtspec = np.zeros_like(E, dtype=float)

        i1 = E <= Eb; i2 = E > Eb
        phtspec[i1] = Amp * (E[i1] / epiv) ** alpha1 * np.exp(-E[i1] / Ec1)
        phtspec[i2] = Amp * (Eb / epiv) ** (alpha1 - alpha2) * np.exp(alpha2 - alpha1) \
            * (E[i2] / epiv) ** alpha2 * np.exp(-E[i2] / Ec2)
        
        return phtspec[0] if scalar else phtspec



class dband(Additive):
    # 10.1088/0004-637X/751/2/90
    
    def __init__(self):
        
        self.expr = 'dband'
        self.comment = 'double band functions'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['pivot_energy'] = Cfg(1.0)

        self.params = OrderedDict()
        self.params[r'$\alpha_1$'] = Par(1, unif(-2, 2))
        self.params[r'$\alpha_2$'] = Par(-1, unif(-2, 2))
        self.params[r'$\beta$'] = Par(-3, unif(-6, -2))
        self.params[r'log$E_b$'] = Par(1, unif(0, 3))
        self.params[r'log$E_p$'] = Par(3, unif(1, 4))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))


    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        epiv = self.config['pivot_energy'].value
        
        alpha1 = self.params[r'$\alpha_1$'].value
        alpha2 = self.params[r'$\alpha_2$'].value
        beta = self.params[r'$\beta$'].value
        logEb = self.params[r'log$E_b$'].value
        logEp = self.params[r'log$E_p$'].value
        logA = self.params[r'log$A$'].value

        Eb = 10 ** logEb
        Ep = 10 ** logEp
        Amp = 10 ** logA
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]

        zi = 1 + redshift
        E = E * zi
        
        Eb1 = Eb
        Ec2 = Ep / (2 + alpha2)
        Ec1 = 1 / (1 / Ec2 + (alpha1 - alpha2) / Eb1)
        Eb2 = Ec2 * (alpha2 - beta)
        phtspec = np.zeros_like(E, dtype=float)

        i1 = E <= Eb1; i2 = (E > Eb1) & (E <= Eb2); i3 = E > Eb2
        phtspec[i1] = Amp * (E[i1] / epiv) ** alpha1 * np.exp(-E[i1] / Ec1)
        phtspec[i2] = Amp * (Eb1 / epiv) ** (alpha1 - alpha2) * np.exp(alpha2 - alpha1) \
            * (E[i2] / epiv) ** alpha2 * np.exp(-E[i2] / Ec2)
        phtspec[i3] = Amp * (Eb1 / epiv) ** (alpha1 - alpha2) * np.exp(alpha2 - alpha1) \
            * (Eb2 / epiv) ** (alpha2 - beta) * np.exp(beta - alpha2) * (E[i3] / epiv) ** beta

        return phtspec[0] if scalar else phtspec



class bb(Additive):

    def __init__(self):
        
        self.expr = 'bb'
        self.comment = 'black-body model'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)

        self.params = OrderedDict()
        self.params[r'log$kT$'] = Par(1, unif(-2, 4))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))


    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
    
        logKT = self.params[r'log$kT$'].value
        logA = self.params[r'log$A$'].value

        kT = 10 ** logKT
        Amp = 10 ** logA
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]

        zi = 1 + redshift
        E = E * zi

        phtspec = Amp * 8.0525 * E ** 2 / (kT ** 4 * (np.exp(E / kT) - 1))
        
        return phtspec[0] if scalar else phtspec



class mbb(Additive):
    # 10.3847/1538-4357/aadc07

    def __init__(self):
        
        self.expr = 'mbb'
        self.comment = 'multi-color black-body model'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)

        self.params = OrderedDict()
        self.params[r'log$kT_{min}$'] = Par(0, unif(-1, 3))
        self.params[r'log$kT_{max}$'] = Par(2, unif(0, 4))
        self.params[r'$m$'] = Par(0, unif(-2, 2))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))


    @staticmethod
    def _integrand(x, m):
        if x > 700: 
            return 0.0
        try:
            return (x**(2.0 - m)) / (np.exp(x) - 1.0)
        except OverflowError:
            return 0.0


    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        
        logkTmin = self.params[r'log$kT_{min}$'].value
        logkTmax = self.params[r'log$kT_{max}$'].value
        m = self.params[r'$m$'].value
        logA = self.params[r'log$A$'].value
        
        kTmin = 10 ** logkTmin
        kTmax = 10 ** logkTmax
        Amp = 10 ** logA
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]
        
        zi = 1 + redshift
        E = E * zi
        
        if not kTmax > kTmin:
            return np.nan if scalar else np.ones_like(E) * np.nan

        ratio_T = kTmax / kTmin

        term1 = 8.0525 * (m + 1) * Amp
        term2 = (ratio_T**(m + 1)) - 1
        term3 = (kTmin) ** (-2)
        prefactor = (term1 / term2) * term3
        
        intspec = []
        for Ei in E:
            lower_limit = Ei / kTmax
            upper_limit = Ei / kTmin
            integral_val, _ = quad(self._integrand, lower_limit, upper_limit, args=(m,))
            
            scaling_factor = (Ei / kTmin) ** (m - 1)
            intspec.append(scaling_factor * integral_val)

        intspec = np.array(intspec)
        phtspec = prefactor * intspec
        
        return phtspec[0] if scalar else phtspec



class hlecpl(Additive):
    # 10.1088/0004-637X/690/1/L10
    
    def __init__(self):
        
        self.expr = 'hlecpl'
        self.comment = 'curvature effect model for cpl function'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['pivot_energy'] = Cfg(1.0)

        self.params = OrderedDict()
        self.params[r'$\alpha$'] = Par(-1, unif(-2, 2))
        self.params[r'log$E_{p,c}$'] = Par(2, unif(0, 4))
        self.params[r'log$A_c$'] = Par(0, unif(-6, 6))
        self.params[r'$t_0$'] = Par(0, unif(-20, 20))
        self.params[r'$t_c$'] = Par(10, unif(0, 50))


    def func(self, E, T, O=None):
        
        redshift = self.config['redshift'].value
        epiv = self.config['pivot_energy'].value
        
        alpha = self.params[r'$\alpha$'].value
        logEpc = self.params[r'log$E_{p,c}$'].value
        logAc = self.params[r'log$A_c$'].value
        t0 = self.params[r'$t_0$'].value
        tc = self.params[r'$t_c$'].value

        Epc = 10 ** logEpc
        Ac = 10 ** logAc
        
        E = np.asarray(E)
        E_scalar = E.ndim == 0
        if E_scalar: E = E[np.newaxis]
        
        T = np.asarray(T)
        T_scalar = T.ndim == 0
        if T_scalar: T = T[np.newaxis]
        
        if E_scalar == T_scalar:
            if E.shape != T.shape:
                raise ValueError('E and T must have the same shape')
            else:
                scalar = E_scalar
        else:
            raise ValueError('E and T must both be scalars or both be arrays')

        if tc <= t0 or tc > np.min(T):
            return np.nan if scalar else np.ones_like(E) * np.nan

        zi = 1 + redshift
        E = E * zi

        Ept = Epc * ((T - t0) / (tc - t0)) ** (- 1)
        At = Ac * ((T - t0) / (tc - t0)) ** (alpha - 1)
        phtspec = At * (E / epiv) ** alpha * np.exp(-(2 + alpha) * E / Ept)
        
        return phtspec[0] if scalar else phtspec
    
    

class hleband(Additive):
    
    def __init__(self):
        
        self.expr = 'hleband'
        self.comment = 'curvature effect model for band function'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['pivot_energy'] = Cfg(100.0)

        self.params = OrderedDict()
        self.params[r'$\alpha$'] = Par(-1, unif(-2, 2))
        self.params[r'$\beta$'] = Par(-4, unif(-6, -2))
        self.params[r'log$E_{p,c}$'] = Par(2, unif(0, 4))
        self.params[r'log$A_c$'] = Par(0, unif(-6, 6))
        self.params[r'$t_0$'] = Par(0, unif(-20, 20))
        self.params[r'$t_c$'] = Par(10, unif(0, 50))


    def func(self, E, T, O=None):
        
        redshift = self.config['redshift'].value
        epiv = self.config['pivot_energy'].value
        
        alpha = self.params[r'$\alpha$'].value
        beta = self.params[r'$\beta$'].value
        logEpc = self.params[r'log$E_{p,c}$'].value
        logAc = self.params[r'log$A_c$'].value
        t0 = self.params[r'$t_0$'].value
        tc = self.params[r'$t_c$'].value

        Epc = 10 ** logEpc
        Ac = 10 ** logAc
        
        E = np.asarray(E)
        E_scalar = E.ndim == 0
        if E_scalar: E = E[np.newaxis]
        
        T = np.asarray(T)
        T_scalar = T.ndim == 0
        if T_scalar: T = T[np.newaxis]
        
        if E_scalar == T_scalar:
            if E.shape != T.shape:
                raise ValueError('E and T must have the same shape')
            else:
                scalar = E_scalar
        else:
            raise ValueError('E and T must both be scalars or both be arrays')

        if tc <= t0 or tc > np.min(T):
            return np.nan if scalar else np.ones_like(E) * np.nan

        zi = 1 + redshift
        E = E * zi

        Ept = Epc * ((T - t0) / (tc - t0)) ** (- 1)
        At = Ac * ((T - t0) / (tc - t0)) ** (alpha - 1)
        Ebt = (alpha - beta) / (alpha + 2) * Ept
        phtspec = np.zeros_like(E, dtype=float)

        i1 = E < Ebt; i2 = E >= Ebt
        phtspec[i1] = At[i1] * (E[i1] / epiv) ** alpha * np.exp(-(2 + alpha) * E[i1] / Ept[i1])
        phtspec[i2] = At[i2] * (Ebt[i2] / epiv) ** (alpha - beta) * np.exp(beta - alpha) \
            * (E[i2] / epiv) ** beta
        
        return phtspec[0] if scalar else phtspec
    
    

class zxhsync(Additive):

    def __init__(self):
        
        self.expr = 'zxhsync'
        self.comment = "zxh's synchrotron model"
        
        self.mo_prefix = docs_path + '/ZXHSYNC'
        self.mo_dir = self.mo_prefix + '/spec_lc_ele_z_dL_gm_gmax_injpl_v2.o'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(1.0)
        self.config['max_time'] = Cfg(26.0)
        self.config['zero_offset'] = Cfg(0.0)
        self.config['spec_prec'] = Cfg(2)
        self.config['temp_prec'] = Cfg(25)

        self.params = OrderedDict()
        self.params[r'log$B_0$'] = Par(1, unif(0, 2))
        self.params[r'$\alpha_B$'] = Par(2, unif(0, 3))
        self.params[r'log$\gamma_{min}$'] = Par(5, unif(3, 7))
        self.params[r'log$\gamma_{max}$'] = Par(7, unif(5, 9))
        self.params[r'log$\Gamma$'] = Par(2, unif(1, 3))
        self.params[r'$p$'] = Par(2, unif(1.5, 3.5))
        self.params[r'$t_{inj}$'] = Par(10, unif(-0.5, 26.00))
        self.params[r'$q$'] = Par(5, unif(0, 10))
        self.params[r'log$R_0$'] = Par(15, unif(12, 17))
        self.params[r'log$Q_0$'] = Par(40, unif(30, 50))


    def func(self, E, T, O=None):
        
        logB0 = self.params[r'log$B_0$'].value
        alphaB = self.params[r'$\alpha_B$'].value
        loggamma_min = self.params[r'log$\gamma_{min}$'].value
        loggamma_max = self.params[r'log$\gamma_{max}$'].value
        logGamma = self.params[r'log$\Gamma$'].value
        p = self.params[r'$p$'].value
        tinj = self.params[r'$t_{inj}$'].value
        q = self.params[r'$q$'].value
        logR0 = self.params[r'log$R_0$'].value
        logQ0 = self.params[r'log$Q_0$'].value   # in unit of s^-1

        B0 = 10 ** logB0
        gamma_min = 10 ** loggamma_min
        gamma_max = 10 ** loggamma_max
        Gamma = 10 ** logGamma
        R0 = 10 ** logR0
        Q0 = 10 ** logQ0

        B0_str = str(B0)
        alphaB_str = str(alphaB)
        gamma_min_str = str(gamma_min)
        gamma_max_str = str(gamma_max)
        Gamma_str = str(Gamma)
        p_str = str(p)
        
        zero_dt_str = '%.4f' % self.config['zero_offset'].value
        tinj_str = str(tinj)
        q_str = str(q)
        max_time_str = '%.4f' % self.config['max_time'].value
        R0_str = str(R0)
        Q0_str = str(Q0)
        z_str = '%.4f' % self.config['redshift'].value
        dL_str = '%.4e' % self.luminosity_distance
        spec_prec_str = '%.2f' % self.config['spec_prec'].value
        temp_prec_str = '%.2f' % self.config['temp_prec'].value
        
        E = np.asarray(E)
        E_scalar = E.ndim == 0
        if E_scalar: E = E[np.newaxis]
        
        T = np.asarray(T)
        T_scalar = T.ndim == 0
        if T_scalar: T = T[np.newaxis]
        
        if E_scalar == T_scalar:
            if E.shape != T.shape:
                raise ValueError('E and T must have the same shape')
            else:
                scalar = E_scalar
        else:
            raise ValueError('E and T must both be scalars or both be arrays')

        phtspec = np.zeros_like(E, dtype=float)
        
        for Ti in set(T):
            idx = np.where(T == Ti)[0]
            t_obs_str = str(Ti)
            E_str = ' '.join([str(Ei) for Ei in E[idx]])
            n_str = str(len(E[idx]))
            it_str = '0'
            ielec_str = '0'

            cmd = self.mo_dir + ' ' + self.mo_prefix + ' ' + B0_str + ' ' + alphaB_str + ' ' + gamma_min_str \
                + ' ' + gamma_max_str + ' ' + Gamma_str + ' ' + p_str + ' ' + t_obs_str + ' ' + zero_dt_str \
                + ' ' + tinj_str + ' ' + q_str + ' ' + max_time_str + ' ' + R0_str + ' ' + Q0_str + ' ' + z_str \
                + ' ' + dL_str + ' ' + n_str + ' ' + it_str + ' ' + ielec_str + ' ' + spec_prec_str \
                + ' ' + temp_prec_str + ' ' + E_str

            process = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
            (out, err) = process.communicate()
            Fd_str = out.split()
            if out == '' or len(Fd_str) != len(E[idx]):
                print('+++++ Error Message +++++')
                print('out: ', out)
                print('err: ', err)
                print('cmd: ', cmd)
                print('+++++ +++++++++++++ +++++')
                raise RuntimeError('ZXHSYNC model execution failed')
            Fd = np.array([float(Fdi) for Fdi in Fd_str])   # in unit of mJy
            Fv = Fd / (E[idx] * 1.6022e-9) / (6.62607e-34 * 6.2415e15) / 1.e26   # in unit of photons/s/cm^2/keV
            phtspec[idx] = Fv

        return phtspec[0] if scalar else phtspec
    
    
    @property
    def luminosity_distance(self):
        
        return Planck18.luminosity_distance(self.config['redshift'].value).to(u.cm).value


    def elecspec(self, ielec_list=None):
        """
        Generate the electron spectrum for each step
        ielec_list: list of steps
        return: list of electron spectra
        """
        logB0 = self.params[r'log$B_0$'].value
        alphaB = self.params[r'$\alpha_B$'].value
        loggamma_min = self.params[r'log$\gamma_{min}$'].value
        loggamma_max = self.params[r'log$\gamma_{max}$'].value
        logGamma = self.params[r'log$\Gamma$'].value
        p = self.params[r'$p$'].value
        tinj = self.params[r'$t_{inj}$'].value
        q = self.params[r'$q$'].value
        logR0 = self.params[r'log$R_0$'].value
        logQ0 = self.params[r'log$Q_0$'].value   # in unit of s^-1

        B0 = 10 ** logB0
        gamma_min = 10 ** loggamma_min
        gamma_max = 10 ** loggamma_max
        Gamma = 10 ** logGamma
        R0 = 10 ** logR0
        Q0 = 10 ** logQ0

        B0_str = str(B0)
        alphaB_str = str(alphaB)
        gamma_min_str = str(gamma_min)
        gamma_max_str = str(gamma_max)
        Gamma_str = str(Gamma)
        p_str = str(p)
        
        zero_dt_str = '%.4f' % self.config['zero_offset'].value
        tinj_str = str(tinj)
        q_str = str(q)
        max_time_str = '%.4f' % self.config['max_time'].value
        R0_str = str(R0)
        Q0_str = str(Q0)
        z_str = '%.4f' % self.config['redshift'].value
        dL_str = '%.4e' % self.luminosity_distance
        spec_prec_str = '%.2f' % self.config['spec_prec'].value
        temp_prec_str = '%.2f' % self.config['temp_prec'].value

        t_obs_str = '1.0'
        n_str = '0'
        it_str = '0'
        
        if ielec_list is None:
            ielec_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        beta = np.sqrt(Gamma ** 2 - 1) / Gamma
        tobspre = self.config['zero_offset'].value
        tobs0 = R0 * (1 - beta) * (1 + 0.36) / beta / 3e10
        tobsmax = self.config['max_time'].value
        Rmax = 3e10 * beta * (tobsmax + tobs0 + tobspre) / (1 + 0.36) / (1 - beta)
        
        elec_spec = []

        for ielec in ielec_list:
            Rn = 10 ** (np.log10(R0) + (ielec - 1) * np.log10(Rmax / R0) / 399)
            tn = (Rn - R0) / beta / 3e10
            
            ielec_str = str(ielec)
            
            cmd = self.mo_dir + ' ' + self.mo_prefix + ' ' + B0_str + ' ' + alphaB_str + ' ' + gamma_min_str \
                + ' ' + gamma_max_str + ' ' + Gamma_str + ' ' + p_str + ' ' + t_obs_str + ' ' + zero_dt_str \
                + ' ' + tinj_str + ' ' + q_str + ' ' + max_time_str + ' ' + R0_str + ' ' + Q0_str + ' ' + z_str \
                + ' ' + dL_str + ' ' + n_str + ' ' + it_str + ' ' + ielec_str + ' ' + spec_prec_str \
                + ' ' + temp_prec_str
                
            process = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
            (out, err) = process.communicate()
            ge_str = out.split()[::2]
            Ne_str = out.split()[1::2]
            if err != '':
                print('+++++ Error Message +++++')
                print('out: ', out)
                print('err: ', err)
                print('cmd: ', cmd)
                print('+++++ +++++++++++++ +++++')
                raise RuntimeError('ZXHSYNC model execution failed')
            ge = np.array([float(gei) for gei in ge_str])
            Ne = np.array([float(Nei) for Nei in Ne_str])
            
            elec_spec.append({'nstep': ielec, 'Rn': Rn, 'tn': tn, 'ge': ge, 'Ne': Ne})
            
        return elec_spec


    def lightcurve(self, response_list, time_list, tarr):
        """
        Generate the model-predicted counts spectrum for each time
        response_list: list of response files
        time_list: list of times
        tarr: time array of the light curve
        return: list of model-predicted counts spectrum for each time
        """
        idx = np.argmin(np.abs(tarr[:, None] - time_list), axis=1)
        
        lc = []
        for i, j in enumerate(idx):
            ctsrate, _ = self.convolve_response(response_list[j], tarr[i])
            lc.append(ctsrate)
            
        return lc
    
    

class katu(Additive):

    def __init__(self):
        
        self.expr = 'katu'
        self.comment = 'katu model'
        
        self.pwd = os.getcwd()
        self.mo_prefix = docs_path + '/Katu'
        self.mo_dir = self.mo_prefix + '/GRB_MZ'
        self.cfg_dir = self.mo_prefix + '/prompt.toml'
        
        with open(self.cfg_dir, 'r') as f_obj:
            self.mo_cfg = toml.load(f_obj)
            
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(1.0)
        self.config['t_obs_start'] = Cfg(0)
        self.config['t_obs_end'] = Cfg(20)
            
        self.params = OrderedDict()
        self.params[r'log$B_0$'] = Par(1, unif(0, 2))
        self.params[r'$\alpha_B$'] = Par(1, unif(0, 3))
        self.params[r'log$\gamma_{min}$'] = Par(5, unif(3, 7))
        self.params[r'log$\Gamma$'] = Par(2, unif(1, 3))
        self.params[r'$p$'] = Par(3, unif(1.5, 3.5))
        self.params[r'$t_{inj}$'] = Par(10, unif(0, 20))
        self.params[r'log$R_0$'] = Par(15, unif(12, 17))
        self.params[r'log$Q_0$'] = Par(40, unif(30, 50))


    def func(self, E, T, O=None):
        
        redshift = self.config['redshift'].value
        t_obs_start = self.config['t_obs_start'].value
        t_obs_end = self.config['t_obs_end'].value
        
        self.set_cfg('Flux.z', redshift)
        self.set_cfg('General.t_obs_start', t_obs_start)
        self.set_cfg('General.t_obs_end', t_obs_end)
        
        logB0 = self.params[r'log$B_0$'].value
        alphaB = self.params[r'$\alpha_B$'].value
        loggamma_min = self.params[r'log$\gamma_{min}$'].value
        logGamma = self.params[r'log$\Gamma$'].value
        p = self.params[r'$p$'].value
        tinj = self.params[r'$t_{inj}$'].value
        logR0 = self.params[r'log$R_0$'].value
        logQ0 = self.params[r'log$Q_0$'].value

        self.set_cfg('Prompt.B_0', logB0)
        self.set_cfg('Prompt.alpha', alphaB)
        self.set_cfg('Prompt.prompt_gmin', loggamma_min)
        self.set_cfg('Prompt.Gamma_init', logGamma)
        self.set_cfg('Prompt.prompt_p', p)
        self.set_cfg('Prompt.t_inj', tinj)
        self.set_cfg('Prompt.R_0', logR0)
        self.set_cfg('Prompt.Q_0', logQ0)

        with open(self.cfg_dir, 'w') as f_obj:
            toml.dump(self.mo_cfg, f_obj)
            
        E = np.asarray(E)
        E_scalar = E.ndim == 0
        if E_scalar: E = E[np.newaxis]
        
        T = np.asarray(T)
        T_scalar = T.ndim == 0
        if T_scalar: T = T[np.newaxis]
        
        if E_scalar == T_scalar:
            if E.shape != T.shape:
                raise ValueError('E and T must have the same shape')
            else:
                scalar = E_scalar
        else:
            raise ValueError('E and T must both be scalars or both be arrays')

        E_list = [str(Ei) for Ei in E]
        T_list = [str(Ti) for Ti in T]
        
        E_split_list = [E_list[i:i + 10000] for i in range(0, len(E_list), 10000)]
        T_split_list = [T_list[i:i + 10000] for i in range(0, len(T_list), 10000)]
        
        os.chdir(self.mo_prefix)
        
        sed_split_list = []
        for E_split, T_split in zip(E_split_list, T_split_list):
            cmd = [self.mo_dir, 'prompt.toml', '--energy'] + E_split + ['--time'] + T_split
            process = sp.Popen(cmd, shell=False, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
            (out, err) = process.communicate()
            if out == '':
                print('+++++ Error Message +++++')
                print('out: ', out)
                print('err: ', err)
                print('cmd: ', cmd)
                print('+++++ +++++++++++++ +++++')
                raise RuntimeError('Katu model execution failed')
            sed_split = np.array([float(val) for val in out.split()[3::4]])
            sed_split_list.append(sed_split)

        os.chdir(self.pwd)
        
        sed = np.concatenate(sed_split_list)    # in unit of erg/cm^2/s
        phtspec = sed / (E * E * 1.60218e-9)    # in unit of photons/s/cm^2/keV
        
        return phtspec[0] if scalar else phtspec


    def set_cfg(self, key, value):

        key_list = key.split('.')
        n_key = len(key_list)

        cfg_ = self.mo_cfg
        for i, key_i in enumerate(key_list):
            assert key_i in cfg_, 'no key (%s) in cfg' % key_i
            if i < n_key - 1:
                cfg_ = cfg_[key_i]
            else:
                cfg_[key_i] = value