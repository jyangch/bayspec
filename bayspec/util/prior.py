from scipy import stats


class Prior(object):
    
    def __init__(self, stat, *args):
        
        self.args = args
        self.expr = 'stat'
        self.stat = stat(*args)
        
        
    def pdf(self, x):
        
        return self.stat.pdf(x)
    
    
    def logpdf(self, x):
        
        return self.stat.logpdf(x)
    
    
    def cdf(self, x):
        
        return self.stat.cdf(x)
    
    
    def logcdf(self, x):
        
        return self.stat.logcdf(x)
    
    
    def ppf(self, q):
        
        return self.stat.ppf(q)
    
    
    def interval(self, q):
        
        return self.stat.interval(q)
    
    
    @property
    def median(self):
        
        return self.stat.median()
        
        
    @property
    def mean(self):
        
        return self.stat.mean()
    
    
    @property
    def var(self):
        
        return self.stat.var()
    
    
    @property
    def std(self):
        
        return self.stat.std()
    
    
    @property
    def info(self):
        
        return f'{self.expr}{self.args}'



class unif(Prior):
    
    def __init__(self, min, max):
        
        self.args = (min, max)
        self.expr = 'unif'
        self.stat = stats.uniform(min, max - min)



class logunif(Prior):
    
    def __init__(self, min, max, loc=0, scale=1):
        
        self.args = (min, max, loc, scale)
        self.expr = 'logunif'
        self.stat = stats.loguniform(*self.args)



class norm(Prior):
    
    def __init__(self, loc, scale):
        
        self.args = (loc, scale)
        self.expr = 'norm'
        self.stat = stats.norm(*self.args)



class lognorm(Prior):
    
    def __init__(self, s, loc, scale):
        
        self.args = (s, loc, scale)
        self.expr = 'lognorm'
        self.stat = stats.lognorm(*self.args)



class truncnorm(Prior):
    
    def __init__(self, a, b, loc, scale):
        
        self.args = (a, b, loc, scale)
        self.expr = 'truncnorm'
        self.stat = stats.truncnorm(*self.args)



class cauchy(Prior):
    
    def __init__(self, loc, scale):
        
        self.args = (loc, scale)
        self.expr = 'cauchy'
        self.stat = stats.cauchy(*self.args)



class cosine(Prior):
    
    def __init__(self, loc, scale):
        
        self.args = (loc, scale)
        self.expr = 'cosine'
        self.stat = stats.cosine(*self.args)



class beta(Prior):
    
    def __init__(self, a, b, loc, scale):
        
        self.args = (a, b, loc, scale)
        self.expr = 'beta'
        self.stat = stats.beta(*self.args)



class gamma(Prior):
    
    def __init__(self, a, loc, scale):
        
        self.args = (a, loc, scale)
        self.expr = 'gamma'
        self.stat = stats.gamma(*self.args)



class expon(Prior):
    
    def __init__(self, loc, scale):
        
        self.args = (loc, scale)
        self.expr = 'expon'
        self.stat = stats.expon(*self.args)



class plaw(Prior):
    
    def __init__(self, a, loc, scale):
        
        self.args = (a, loc, scale)
        self.expr = 'plaw'
        self.stat = stats.powerlaw(*self.args)
        
        
all_priors = {name: cls for name, cls in globals().items() 
              if isinstance(cls, type) 
              and issubclass(cls, Prior) 
              and name != 'Prior'}


def list_priors():
    return list(all_priors.keys())


__all__ = list(all_priors.keys()) + ['list_priors', 'all_priors']
