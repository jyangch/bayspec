import numpy as np
from collections import OrderedDict

from ..model import Multiplicative
from ...util.prior import unif
from ...util.param import Par, Cfg



class hecut(Multiplicative):

    def __init__(self):
        
        self.expr = 'hecut'
        self.comment = 'high-energy cutoff model'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['threshold_energy'] = Cfg(0.0)
        
        self.params = OrderedDict()
        self.params[r'log$E_c$'] = Par(2, unif(-1, 5))


    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        ethr = self.config['threshold_energy'].value
        
        logEc = self.params[r'log$E_c$'].value
        
        Ec = 10 ** logEc
        
        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]
        
        zi = 1 + redshift
        E = E * zi
        
        nouspec = np.zeros_like(E, dtype=float)

        i1 = E <= ethr; i2 = E > ethr
        nouspec[i1] = 1.0
        nouspec[i2] = np.exp((ethr - E[i2]) / Ec)

        return nouspec[0] if scalar else nouspec



class tinvabs(Multiplicative):
    
    def __init__(self):
        
        from astromodels import TbAbs
        
        self.expr = 'tinvabs'
        self.comment = 'time-involved absorption model'
        self.tbabs = TbAbs()
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)

        self.params = OrderedDict()
        self.params[r'$N_{H,0}$'] = Par(1, unif(1e-4, 50))
        self.params[r'$\tau$'] = Par(50, unif(0, 300))
        
        
    def func(self, E, T, O=None):
        
        redshift = self.config['redshift'].value
        
        NH0 = self.params[r'$N_{H,0}$'].value
        tau = self.params[r'$\tau$'].value
        
        self.tbabs.parameters['redshift'].value = redshift
        
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
        
        fracspec = np.zeros_like(E, dtype=float)
        
        for Ti in set(T):
            idx = np.where(T == Ti)[0]
            NH = NH0 * np.exp(- Ti / tau)
            
            if NH < 1e-4:
                return np.nan if scalar else np.ones_like(E) * np.nan
            
            self.tbabs.parameters['NH'].value = NH
            
            res = self.tbabs(np.array(E, dtype=float))
            fracspec[idx] = res

        return fracspec[0] if scalar else fracspec