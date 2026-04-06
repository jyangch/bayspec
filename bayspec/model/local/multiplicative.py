import numpy as np
import numba as nb
from collections import OrderedDict
from os.path import dirname, abspath
docs_path = dirname(dirname(dirname(abspath(__file__)))) + '/docs'

from ..model import Multiplicative
from ...util.prior import unif
from ...util.param import Par, Cfg
from ...util.tools import memoized



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



class wabs(Multiplicative):
    
    def __init__(self):
        
        self.expr = 'wabs'
        self.comment = 'Wisconsin ISM absorption model'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)

        self.params = OrderedDict()
        self.params[r'$N_H$'] = Par(1, unif(1e-4, 50))
        
        self.xsect_prefix = docs_path + '/xsect'
        self.xsect_dir = self.xsect_prefix + '/xsect_wabs_angr.npz'
        self.xsect_data = np.load(self.xsect_dir)
        self.xsect_energy = self.xsect_data['energy']
        self.xsect_sigma = self.xsect_data['sigma']
    
    
    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        nh = self.params[r'$N_H$'].value
        
        sigma = self._get_cached_sigma(E, redshift)
        
        fracspec = np.exp(-nh * sigma)
        
        return fracspec
    
    
    @memoized()
    def _get_cached_sigma(self, E, redshift):
        
        E = np.asarray(E, dtype=np.float64)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]
        
        zi = 1 + redshift
        E = E * zi
        
        sigma = np.interp(E, self.xsect_energy, self.xsect_sigma, right=0.0)

        return sigma[0] if scalar else sigma



class phabs(Multiplicative):
    
    def __init__(self):
        
        self.expr = 'phabs'
        self.comment = 'photoelectric absorption model'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)

        self.params = OrderedDict()
        self.params[r'$N_H$'] = Par(1, unif(1e-4, 50))
        
        self.xsect_prefix = docs_path + '/xsect'
        self.xsect_dir = self.xsect_prefix + '/xsect_phabs_aspl.npz'
        self.xsect_data = np.load(self.xsect_dir)
        self.xsect_energy = self.xsect_data['energy']
        self.xsect_sigma = self.xsect_data['sigma']
    
    
    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        nh = self.params[r'$N_H$'].value
        
        sigma = self._get_cached_sigma(E, redshift)
        
        fracspec = np.exp(-nh * sigma)
        
        return fracspec
    
    
    @memoized()
    def _get_cached_sigma(self, E, redshift):
        
        E = np.asarray(E, dtype=np.float64)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]
        
        zi = 1 + redshift
        E = E * zi
        
        sigma = np.interp(E, self.xsect_energy, self.xsect_sigma, right=0.0)

        return sigma[0] if scalar else sigma



class tbabs(Multiplicative):
    
    def __init__(self):
        
        self.expr = 'tbabs'
        self.comment = 'Tuebingen-Boulder ISM absorption model'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)

        self.params = OrderedDict()
        self.params[r'$N_H$'] = Par(1, unif(1e-4, 50))
        
        self.xsect_prefix = docs_path + '/xsect'
        self.xsect_dir = self.xsect_prefix + '/xsect_tbabs_wilm.npz'
        self.xsect_data = np.load(self.xsect_dir)
        self.xsect_energy = self.xsect_data['energy']
        self.xsect_sigma = self.xsect_data['sigma']
    
    
    def func(self, E, T=None, O=None):
        
        redshift = self.config['redshift'].value
        nh = self.params[r'$N_H$'].value
        
        sigma = self._get_cached_sigma(E, redshift)
        
        fracspec = np.exp(-nh * sigma)
        
        return fracspec
    
    
    @memoized()
    def _get_cached_sigma(self, E, redshift):
        
        E = np.asarray(E, dtype=np.float64)
        scalar = E.ndim == 0
        if scalar: E = E[np.newaxis]
        
        zi = 1 + redshift
        E = E * zi
        
        sigma = np.interp(E, self.xsect_energy, self.xsect_sigma, right=0.0)

        return sigma[0] if scalar else sigma



class tinvabs(Multiplicative):
    
    def __init__(self):
        
        self.expr = 'tinvabs'
        self.comment = 'time-involved absorption model'
        self.tbabs = tbabs()
        
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
            
            self.tbabs.params[r'$N_H$'].value = NH
            
            res = self.tbabs(np.array(E, dtype=float))
            fracspec[idx] = res

        return fracspec[0] if scalar else fracspec