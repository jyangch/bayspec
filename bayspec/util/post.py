import numpy as np



class Post(object):
    
    def __init__(self, sample, logprob=None):
        
        self._sample = sample
        self._logprob = logprob
        
        
    @property
    def sample(self):
        
        return self._sample
    
    
    @sample.setter
    def sample(self, new_sample):
        
        if not (np.ndim(new_sample) == 1):
            raise ValueError('sample must be 1D arrays')
        
        else:
            self._sample = new_sample
            
            
    @property
    def logprob(self):
        
        return self._logprob
    
    
    @logprob.setter
    def logprob(self, new_logprob):
        
        if not (np.ndim(new_logprob) == 1):
            raise ValueError('logprob must be 1D arrays')
        
        else:
            self._logprob = new_logprob
            
            
    @property
    def nsample(self):
        
        return self.sample.shape[0]
            
            
    @property
    def mean(self):
        
        return np.mean(self.sample)
    
    
    @property
    def median(self):
        
        return np.median(self.sample)
    
    
    @property
    def best(self):
        
        if self.logprob is None:
            return None
        
        else:
            return self.sample[np.argmax(self.logprob)]
        
        
    @property
    def best_ci(self):
        
        return self._best_ci
    
    
    @best_ci.setter
    def best_ci(self, new_best_ci):
        
        self._best_ci = new_best_ci
    
    
    def quantile(self, q):
        
        return np.quantile(self.sample, q)
    
    
    def interval(self, q):
        
        return np.quantile(self.sample, [0.5 - q / 2, 0.5 + q / 2]).tolist()
    
    
    @property
    def Isigma(self):
        
        return self.interval(68.27 / 100)
    
    
    @property
    def IIsigma(self):
        
        return self.interval(95.45 / 100)
    
    
    @property
    def IIIsigma(self):
        
        return self.interval(99.73 / 100)
    
    
    def error(self, par, q=0.6827):
        
        ci = self.interval(q)
        
        return np.diff([ci[0], par, ci[1]]).tolist()
    
    
    @property
    def info(self):
        
        return dict([('mean', self.mean), 
                     ('median', self.median), 
                     ('best', self.best), 
                     ('Isigma', self.Isigma)])
