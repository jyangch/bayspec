import numpy as np
import subprocess as sp
import astropy.units as u
from ..model import Additive
from ...util.prior import unif
from ...util.param import Par, Cfg
from collections import OrderedDict
from os.path import dirname, abspath
from scipy.special import gamma, zeta
from astropy.cosmology import Planck18
docs_path = dirname(dirname(dirname(abspath(__file__)))) + '/docs'



class pl(Additive):

    def __init__(self):
        super().__init__()
        
        self.expr = 'pl'
        self.comment = 'power law model'
        
        self.params = OrderedDict()
        self.params[r'$\alpha$'] = Par(-1, unif(-8, 5))
        self.params[r'log$A$'] = Par(-1, unif(-10, 8))


    def func(self, E, T=None, O=None):
        alpha = self.params[r'$\alpha$'].value
        logA = self.params[r'log$A$'].value
        
        Amp = 10 ** logA
        
        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        phtspec = Amp * (E / 100) ** alpha
        return phtspec



class bb(Additive):

    def __init__(self):
        super().__init__()
        
        self.expr = 'bb'
        self.comment = 'black-body model'

        self.params = OrderedDict()
        self.params[r'log$kT$'] = Par(2, unif(0, 3))
        self.params[r'log$A$'] = Par(-1, unif(-6, 5))


    def func(self, E, T=None, O=None):
        logKT = self.params[r'log$kT$'].value
        logA = self.params[r'log$A$'].value

        kT = 10 ** logKT
        Amp = 10 ** logA

        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        phtspec = Amp * 8.0525 * E ** 2 / (kT ** 4 * (np.exp(E / kT) - 1))
        return phtspec



class cpl(Additive):

    def __init__(self):
        super().__init__()
        
        self.expr = 'cpl'
        self.comment = 'cutoff power law model'

        self.params = OrderedDict()
        self.params[r'$\alpha$'] = Par(-1, unif(-8, 4))
        self.params[r'log$E_{c}$'] = Par(2, unif(0, 4))
        self.params[r'log$A$'] = Par(-1, unif(-6, 5))


    def func(self, E, T=None, O=None):
        alpha = self.params[r'$\alpha$'].value
        logEc = self.params[r'log$E_{c}$'].value
        logA = self.params[r'log$A$'].value

        Ec = 10 ** logEc
        Amp = 10 ** logA

        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        phtspec = Amp * (E / 100) ** alpha * np.exp(-1.0 * E / Ec)
        return phtspec



class ppl(Additive):

    def __init__(self):
        super().__init__()
        
        self.expr = 'ppl'
        self.comment = 'cutoff power law model with peak energy'

        self.params = OrderedDict()
        self.params[r'$\alpha$'] = Par(-1, unif(-2, 2))
        self.params[r'log$E_{p}$'] = Par(2, unif(0, 4))
        self.params[r'log$A$'] = Par(-1, unif(-6, 5))


    def func(self, E, T=None, O=None):
        alpha = self.params[r'$\alpha$'].value
        logEp = self.params[r'log$E_{p}$'].value
        logA = self.params[r'log$A$'].value

        Ep = 10 ** logEp
        Amp = 10 ** logA

        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        Ec = Ep / (2 + alpha)
        phtspec = Amp * (E / 100) ** alpha * np.exp(-1.0 * E / Ec)
        return phtspec
    
    
    
class band(Additive):

    def __init__(self):
        super().__init__()
        
        self.expr = 'band'
        self.comment = 'grb band function'

        self.params = OrderedDict()
        self.params[r'$\alpha$'] = Par(-1, unif(-2, 2))
        self.params[r'$\beta$'] = Par(-4, unif(-6, -2))
        self.params[r'log$E_{p}$'] = Par(2, unif(0, 4))
        self.params[r'log$A$'] = Par(-1, unif(-6, 5))


    def func(self, E, T=None, O=None):
        alpha = self.params[r'$\alpha$'].value
        beta = self.params[r'$\beta$'].value
        logEp = self.params[r'log$E_{p}$'].value
        logA = self.params[r'log$A$'].value

        Ep = 10 ** logEp
        Amp = 10 ** logA
        
        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        Ec = Ep / (2 + alpha)
        Eb = (alpha - beta) * Ec
        phtspec = np.zeros_like(E, dtype=float)

        i1 = E <= Eb; i2 = E > Eb
        phtspec[i1] = Amp * (E[i1] / 100) ** alpha * np.exp(-E[i1] / Ec)
        phtspec[i2] = Amp * (Eb / 100) ** (alpha - beta) * (E[i2] / 100) ** beta * np.exp(beta - alpha)
        return phtspec
    
    
    
class cband(Additive):
    # 10.1088/0004-637X/751/2/90
    
    def __init__(self):
        super().__init__()
        
        self.expr = 'cband'
        self.comment = 'band function with cut-off'

        self.params = OrderedDict()
        self.params[r'$\alpha_1$'] = Par(1, unif(-2, 2))
        self.params[r'$\alpha_2$'] = Par(-1, unif(-2, 2))
        self.params[r'log$E_{b}$'] = Par(1, unif(0, 3))
        self.params[r'log$E_{p}$'] = Par(3, unif(1, 4))
        self.params[r'log$A$'] = Par(0, unif(-6, 6))


    def func(self, E, T=None, O=None):
        alpha1 = self.params[r'$\alpha_1$'].value
        alpha2 = self.params[r'$\alpha_2$'].value
        logEb = self.params[r'log$E_{b}$'].value
        logEp = self.params[r'log$E_{p}$'].value
        logA = self.params[r'log$A$'].value

        Eb = 10 ** logEb
        Ep = 10 ** logEp
        Amp = 10 ** logA

        if alpha1 <= alpha2:
            return np.ones_like(E) * np.nan

        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        E2 = Ep / (2 + alpha2)
        E1 = 1 / (1 / E2 + (alpha1 - alpha2) / Eb)
        phtspec = np.zeros_like(E, dtype=float)

        i1 = E <= Eb; i2 = E > Eb
        phtspec[i1] = Amp * E[i1] ** alpha1 * np.exp(- E[i1] / E1)
        phtspec[i2] = Amp * Eb ** (alpha1 - alpha2) * np.exp(alpha2 - alpha1) * E[i2] ** alpha2 * np.exp(- E[i2] / E2)
        return phtspec
    
    
    
class dband(Additive):
    # 10.1088/0004-637X/751/2/90
    
    def __init__(self):
        super().__init__()
        
        self.expr = 'dband'
        self.comment = 'double band functions'

        self.params = OrderedDict()
        self.params[r'$\alpha_{1}$'] = Par(1, unif(-2, 2))
        self.params[r'$\alpha_{2}$'] = Par(-1, unif(-2, 2))
        self.params[r'$\beta$'] = Par(-3, unif(-5, -2))
        self.params[r'log$E_{b}$'] = Par(1, unif(0, 3))
        self.params[r'log$E_{p}$'] = Par(3, unif(1, 4))
        self.params[r'log$A$'] = Par(0, unif(-6, 6))


    def func(self, E, T=None, O=None):
        alpha1 = self.params[r'$\alpha_{1}$'].value
        alpha2 = self.params[r'$\alpha_{2}$'].value
        beta = self.params[r'$\beta$'].value
        logEb = self.params[r'log$E_{b}$'].value
        logEp = self.params[r'log$E_{p}$'].value
        logA = self.params[r'log$A$'].value

        Eb = 10 ** logEb
        Ep = 10 ** logEp
        Amp = 10 ** logA

        if alpha1 <= alpha2:
            return np.ones_like(E) * np.nan

        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        E2 = Ep / (alpha2 - beta)
        E1 = 1 / (1 / E2 + (alpha1 - alpha2) / Eb)
        phtspec = np.zeros_like(E, dtype=float)

        i1 = E <= Eb; i2 = (E > Eb) & (E <= Ep); i3 = E > Ep
        phtspec[i1] = Amp * E[i1] ** alpha1 * np.exp(- E[i1] / E1)
        phtspec[i2] = Amp * Eb ** (alpha1 - alpha2) * np.exp(alpha2 - alpha1) * E[i2] ** alpha2 * np.exp(- E[i2] / E2)
        phtspec[i3] = Amp * Eb ** (alpha1 - alpha2) * np.exp(alpha2 - alpha1) * ((alpha2 - beta) * E2) ** (alpha2 - beta) * np.exp(beta - alpha2) * E[i3] ** beta
        return phtspec
    


class sbpl(Additive):
    # Kaneko_ApJS_2006#
    
    def __init__(self):
        super().__init__()
        
        self.expr = 'sbpl'
        self.comment = 'smoothly broken power law'

        self.params = OrderedDict()
        self.params[r'$\alpha$'] = Par(-1, unif(-4, 3))
        self.params[r'$\beta$'] = Par(-2, unif(-5, 2))
        self.params[r'log$E_{b}$'] = Par(2, unif(0, 4))
        self.params[r'log$A$'] = Par(-1, unif(-6, 5))
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0)
        self.config['smoothness'] = Cfg(0.3)


    def func(self, E, T=None, O=None):
        alpha = self.params[r'$\alpha$'].value
        beta = self.params[r'$\beta$'].value
        logEb = self.params[r'log$E_{b}$'].value
        logA = self.params[r'log$A$'].value

        Eb = 10 ** logEb
        Amp = 10 ** logA

        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        Delta = self.config['smoothness'].value
        m = (beta - alpha) / 2
        b = (alpha + beta) / 2
        q = np.log10(E / Eb) / Delta
        qpiv = np.log10(100 / Eb) / Delta
        a = m * Delta * np.log((np.e ** q + np.e ** (-q)) / 2)
        apiv = m * Delta * np.log((np.e ** qpiv + np.e ** (-qpiv)) / 2)
        phtspec = Amp * (E / 100) ** b * 10 ** (a - apiv)
        return phtspec
    


class ssbpl(Additive):
    #Ravasio_A&A_2018#
    
    def __init__(self):
        super().__init__()
        
        self.expr = 'ssbpl'
        self.comment = 'single smoothly broken power law'

        self.params = OrderedDict()
        self.params[r'$\alpha$'] = Par(-1, unif(-4, 3))
        self.params[r'$\beta$'] = Par(-3, unif(-5, -2))
        self.params[r'log$E_{p}$'] = Par(2, unif(0, 4))
        self.params[r'log$A$'] = Par(0, unif(-6, 6))
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0)
        self.config['smoothness'] = Cfg(2)


    def func(self, E, T=None, O=None):
        alpha = self.params[r'$\alpha$'].value
        beta = self.params[r'$\beta$'].value
        logEp = self.params[r'log$E_{p}$'].value
        logA = self.params[r'log$A$'].value

        Ep = 10 ** logEp
        Amp = 10 ** logA

        if alpha <= beta:
            return np.ones_like(E) * np.nan

        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        n = self.config['smoothness'].value
        Ej = Ep * (-(alpha + 2) / (beta + 2)) ** (1 / ((beta - alpha) * n))
        phtspec = Amp * Ej ** alpha * ((E / Ej) ** (-alpha * n) + (E / Ej) ** (-beta * n)) ** (-1 / n)
        return phtspec



class csbpl(Additive):

    def __init__(self):
        super().__init__()
        
        self.expr = 'csbpl'
        self.comment = 'smoothly broken power law with cutoff'

        self.params = OrderedDict()
        self.params[r'$\alpha_{1}$'] = Par(1, unif(-2, 2))
        self.params[r'$\alpha_{2}$'] = Par(-1, unif(-2, 2))
        self.params[r'log$E_{b}$'] = Par(1, unif(-1, 2))
        self.params[r'log$E_{p}$'] = Par(3, unif(1, 4))
        self.params[r'log$A$'] = Par(0, unif(-6, 6))
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0)
        self.config['smoothness'] = Cfg(2)


    def func(self, E, T=None, O=None):
        alpha1 = self.params[r'$\alpha_{1}$'].value
        alpha2 = self.params[r'$\alpha_{2}$'].value
        logEb = self.params[r'log$E_{b}$'].value
        logEp = self.params[r'log$E_{p}$'].value
        logA = self.params[r'log$A$'].value

        Eb = 10 ** logEb
        Ep = 10 ** logEp
        Amp = 10 ** logA

        if alpha1 <= alpha2:
            return np.ones_like(E) * np.nan

        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        n = self.config['smoothness'].value
        Ec = Ep / (2 + alpha2)
        phtspec = Amp * Eb ** alpha1 * ((E / Eb) ** (-alpha1 * n) + (E / Eb) ** (-alpha2 * n)) ** (-1 / n) * np.exp(-E / Ec)
        return phtspec
    
    
    
class dsbpl(Additive):
    # Ravasio_A&A_2018#
    
    def __init__(self):
        super().__init__()
        
        self.expr = 'dsbpl'
        self.comment = 'double smoothly broken power laws'

        self.params = OrderedDict()
        self.params[r'$\alpha_{1}$'] = Par(1, unif(-2, 2))
        self.params[r'$\alpha_{2}$'] = Par(-1, unif(-2, 2))
        self.params[r'$\beta$'] = Par(-4, unif(-6, -2))
        self.params[r'log$E_{b}$'] = Par(1, unif(0, 3))
        self.params[r'log$E_{p}$'] = Par(3, unif(1, 4))
        self.params[r'log$A$'] = Par(0, unif(-6, 6))
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0)
        self.config['smoothness1'] = Cfg(2)
        self.config['smoothness2'] = Cfg(2)


    def func(self, E, T=None, O=None):
        alpha1 = self.params[r'$\alpha_{1}$'].value
        alpha2 = self.params[r'$\alpha_{2}$'].value
        beta = self.params[r'$\beta$'].value
        logEb = self.params[r'log$E_{b}$'].value
        logEp = self.params[r'log$E_{p}$'].value
        logA = self.params[r'log$A$'].value

        Eb = 10 ** logEb
        Ep = 10 ** logEp
        Amp = 10 ** logA

        if alpha1 <= alpha2:
            return np.ones_like(E) * np.nan

        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        n1 = self.config['smoothness1'].value
        n2 = self.config['smoothness2'].value
        Ej = Ep * (-(alpha2 + 2) / (beta + 2)) ** (1 / ((beta - alpha2) * n2))
        sbpl1 = ((E / Eb) ** (-alpha1 * n1) + (E / Eb) ** (-alpha2 * n1)) ** (n2 / n1)
        sbpl2 = ((Ej / Eb) ** (-alpha1 * n1) + (Ej / Eb) ** (-alpha2 * n1)) ** (n2 / n1)
        phtspec = Amp * Eb ** alpha1 * (sbpl1 + (E / Ej) ** (-beta * n2) * sbpl2) ** (-1 / n2)
        return phtspec
    
    

class tsbpl(Additive):

    def __init__(self):
        super().__init__()
        
        self.expr = 'tsbpl'
        self.comment = 'triple smoothly broken power laws'
        
        self.params = OrderedDict()
        self.params[r'$\alpha_{a}$'] = Par(2, unif(0, 4))
        self.params[r'$\alpha_{m}$'] = Par(1, unif(-2, 2))
        self.params[r'$\alpha_{p}$'] = Par(-1, unif(-2, 2))
        self.params[r'$\beta$'] = Par(-4, unif(-6, -2))
        self.params[r'log$E_{a}$'] = Par(1, unif(0, 2))
        self.params[r'log$E_{m}$'] = Par(2, unif(0, 3))
        self.params[r'log$E_{p}$'] = Par(3, unif(1, 4))
        self.params[r'log$A$'] = Par(0, unif(-6, 6))
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0)
        self.config['smoothness1'] = Cfg(2)
        self.config['smoothness2'] = Cfg(2)
        self.config['smoothness3'] = Cfg(2)
        
        
    def func(self, E, T=None, O=None):
        alpha_a = self.params[r'$\alpha_{a}$'].value
        alpha_m = self.params[r'$\alpha_{m}$'].value
        alpha_p = self.params[r'$\alpha_{p}$'].value
        beta = self.params[r'$\beta$'].value
        logEa = self.params[r'log$E_{a}$'].value
        logEm = self.params[r'log$E_{m}$'].value
        logEp = self.params[r'log$E_{p}$'].value
        logA = self.params[r'log$A$'].value
        
        Ea = 10 ** logEa
        Em = 10 ** logEm
        Ep = 10 ** logEp
        Amp = 10 ** logA

        if alpha_m <= alpha_p or alpha_a <= alpha_m:
            return np.ones_like(E) * np.nan
    
        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        n1 = self.config['smoothness1'].value
        n2 = self.config['smoothness2'].value
        n3 = self.config['smoothness3'].value

        Ej = Ep * (-(alpha_p + 2) / (beta + 2)) ** (1 / ((beta - alpha_p) * n3))
        phtspec = Amp * Ea ** alpha_a * self._sb4pl(E, [alpha_a, alpha_m, alpha_p, beta, Ea, Em, Ej, n1, n2, n3])
        return phtspec


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



class mbb(Additive):

    def __init__(self):
        super().__init__()
        
        self.expr = 'mbb'
        self.comment = 'multi-color black-body model'

        self.params = OrderedDict()
        self.params[r'log$kT_{min}$'] = Par(1, unif(0, 2))
        self.params[r'log$kT_{max}$'] = Par(2, unif(1, 4))
        self.params[r'$m$'] = Par(0, unif(-5, 5))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))


    def func(self, E, T=None, O=None):
        logkTmin = self.params[r'log$kT_{min}$'].value
        logkTmax = self.params[r'log$kT_{max}$'].value
        m = self.params[r'$m$'].value
        logA = self.params[r'log$A$'].value
        
        kTmin = 10 ** logkTmin
        kTmax = 10 ** logkTmax
        Amp = 10 ** logA
        
        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        Emin = E / kTmin
        Emax = E / kTmax
        s1 = -2.015 - 1.665 * m - 0.125 * (m ** 2.0)
        s2 = 3.0025 + 0.9275 * m + 0.0875 * (m ** 2.0)

        phtspec = Amp * E ** (-2) * (1 + np.abs(m) * Emax ** (-1 - m) / (gamma(1 - m) * zeta(-1 * m, 1))) * \
            kTmin ** (-4) / ((np.exp(Emax / s1) - 1) ** s1 + 1) / (((m + 1) / m * Emin ** (-3)) ** s2 + \
                (Emin ** (-4 - m) / (gamma(1 - m) * zeta(-1 * m, 1))) ** s2) ** (1 / s2)
        return phtspec
