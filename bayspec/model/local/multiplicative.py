import numpy as np
from collections import OrderedDict

from ..model import Multiplicative
from ...util.prior import unif
from ...util.param import Par, Cfg



class hecut(Multiplicative):

    def __init__(self):
        super().__init__()
        
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
        
        zi = 1 + redshift
        E = E * zi
        
        nouspec = np.zeros_like(E, dtype=float)

        i1 = E <= ethr; i2 = E > ethr
        nouspec[i1] = 1.0
        nouspec[i2] = np.exp((ethr - E[i2]) / Ec)

        return nouspec
