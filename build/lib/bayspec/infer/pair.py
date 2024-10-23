import numpy as np
from ..data.data import Data
from ..model.model import Model
from .statistic import Statistic



class Pair(object):
    
    _allowed_stats = {'gstat': Statistic.Gstat, 
                      'chi2': Statistic.Gstat, 
                      'pstat': Statistic.Pstat, 
                      'ppstat': Statistic.PPstat, 
                      'cstat': Statistic.PPstat, 
                      'pgstat': Statistic.PGstat, 
                      'Xppstat': Statistic.PPstat_Xspec, 
                      'Xcstat': Statistic.PPstat_Xspec, 
                      'Xpgstat': Statistic.PGstat_Xspec, 
                      'ULppstat': Statistic.PPstat_UL, 
                      'ULpgstat': Statistic.PGstat_UL}

    def __init__(self, data, model):
        
        self._data = data
        self._model = model
        
        self._pair()
        
        
    @property
    def data(self):
        
        return self._data
    
    
    @data.setter
    def data(self, new_data):
        
        self._data = new_data
        
        self._pair()


    @property
    def model(self):
        
        return self._model
    
    
    @model.setter
    def model(self, new_model):
        
        self._model = new_model
        
        self._pair()
        
        
    def _pair(self):
        
        if not isinstance(self.data, Data):
            raise ValueError('data argument should be Data type')
        
        if not isinstance(self.model, Model):
            raise ValueError('model argument should be Model type')
        
        self.data.fit_with = self.model


    @property
    def stat_func(self):
        
        return lambda S, B, m, ts, tb, sigma_S, sigma_B, stat: \
            np.inf if np.isnan(m).any() or np.isinf(m).any() else \
                self._allowed_stats[stat](**{'S': np.float128(S), 'B': np.float128(B), \
                    'm': np.float128(m), 'ts': np.float128(ts), 'tb': np.float128(tb), \
                    'sigma_S': np.float128(sigma_S), 'sigma_B': np.float128(sigma_B)})


    def _stat_calculate(self):
        
        return np.array(list(map(self.stat_func, 
                                 self.data.src_counts, 
                                 self.data.bkg_counts, 
                                 self.model.ctsrate, 
                                 self.data.corr_src_efficiency, 
                                 self.data.corr_bkg_efficiency, 
                                 self.data.src_errors, 
                                 self.data.bkg_errors, 
                                 self.data.stats))).astype(float)


    @property
    def stat_list(self):
        
        return self._stat_calculate()
    
    
    @property
    def weight_list(self):
        
        return self.data.weights


    @property
    def stat(self):
        
        return np.sum(self.stat_list * self.weight_list)
    
    
    @property
    def loglike_list(self):
        
        return -0.5 * self.stat_list
    
    
    @property
    def loglike(self):
        
        return -0.5 * self.stat
    
    
    @property
    def npoint_list(self):
        
        return self.data.npoints
    
    
    @property
    def npoint(self):
        
        return np.sum(self.npoint_list)
