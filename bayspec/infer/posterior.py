import os
import numpy as np
from collections import OrderedDict

from .infer import Infer
from ..util.info import Info
from ..util.post import Post
from ..util.tools import json_dump



class Posterior(Infer):
    
    def __init__(self, infer):
        
        self.infer = infer
        
        
    @property
    def infer(self):
        
        return self._infer
    
    
    @infer.setter
    def infer(self, new_infer):
        
        if not isinstance(new_infer, Infer):
            raise TypeError('expected an instance of Infer')
        
        self._infer = new_infer
        self.__dict__.update(new_infer.__dict__)
        
        self._post()


    def _post(self):
        
        for i in range(self.free_nparams):
            sample = self.posterior_sample[:, i].copy()
            logprob = self.posterior_sample[:, -1].copy()
            
            self.free_par[i+1].post = Post(sample, logprob)
            
        self._par_best_ci(q=0.6827)
        self.at_par(self.par_best)


    def _par_best_ci(self, q=0.6827):
        
        argsort = np.argsort(self.posterior_sample[: ,-1])[::-1]
        sort_posterior_sample = self.posterior_sample[: ,0:-1].copy()[argsort]
        
        for sample in sort_posterior_sample:
            if np.array([True if (ci[0] <= sample[i] <= ci[1]) else False \
                for i, ci in enumerate(self.par_interval(q))]).all():
                
                for par, value in zip(self.free_par.values(), sample):
                    par.post.best_ci = value
                
                break


    @property
    def posterior_statistic(self):
        
        mean = np.mean(self.posterior_sample, axis=0)
        median = np.median(self.posterior_sample, axis=0)
        
        q = 68.27 / 100
        Isigma = np.quantile(self.posterior_sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        
        q = 95.45 / 100
        IIsigma = np.quantile(self.posterior_sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        
        q = 99.73 / 100
        IIIsigma = np.quantile(self.posterior_sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        
        return dict([('mean', mean), 
                     ('median', median), 
                     ('Isigma', Isigma), 
                     ('IIsigma', IIsigma), 
                     ('IIIsigma', IIIsigma)])


    @property
    def par_mean(self):
        
        return [par.post.mean for par in self.free_par.values()]
    
    
    @property
    def par_median(self):
        
        return [par.post.median for par in self.free_par.values()]
    
    
    @property
    def par_best(self):
        
        return [par.post.best for par in self.free_par.values()]
    
    
    @property
    def par_best_ci(self):
        
        return [par.post.best_ci for par in self.free_par.values()]


    def par_quantile(self, q):
        
        return [par.post.quantile(q) for par in self.free_par.values()]
    
    
    def par_interval(self, q):
        
        return [par.post.interval(q) for par in self.free_par.values()]
    
    
    @property
    def par_Isigma(self):
        
        return [par.post.Isigma for par in self.free_par.values()]
    
    
    @property
    def par_IIsigma(self):
        
        return [par.post.IIsigma for par in self.free_par.values()]
    
    
    @property
    def par_IIIsigma(self):
        
        return [par.post.IIIsigma for par in self.free_par.values()]
    
    
    def par_error(self, par, q=0.6827):
        
        ci = self.par_interval(q)
        
        return [np.diff([c[0], p, c[1]]).tolist() for p, c in zip(par, ci)]
    
    
    @property
    def max_loglike(self):
        
        self.at_par(self.par_best)
        
        return self.loglike
    
    
    @property
    def aic(self):
        
        return -2 * self.max_loglike + 2 * self.free_nparams
    
    
    @property
    def aicc(self):
        
        return self.aic + 2 * self.free_nparams * \
            (self.free_nparams + 1) / \
                (self.npoint - self.free_nparams - 1)
    
    
    @property
    def bic(self):
        
        return -2 * self.max_loglike + self.free_nparams * np.log(self.npoint)


    @property
    def lnZ(self):
        
        try:
            return self.logevidence
        except AttributeError:
            return None


    @property
    def free_par_info(self):
        
        self._you_free()
        
        free_par_info = Info.list_dict_to_dict(self.free_params)
        
        del free_par_info['Posterior']
        del free_par_info['Mates']
        del free_par_info['Frozen']
        del free_par_info['Prior']
        del free_par_info['Value']
        
        free_par_info['Mean'] = ['%.3f'%par for par in self.par_mean]
        free_par_info['Median'] = ['%.3f'%par for par in self.par_median]
        free_par_info['Best'] = ['%.3f'%par for par in self.par_best]
        free_par_info['1sigma Best'] = ['%.3f'%par for par in self.par_best_ci]
        free_par_info['1sigma CI'] = ['[%.3f, %.3f]'%tuple(ci) for ci in self.par_Isigma]
        
        return Info.from_dict(free_par_info)
    
    
    @property
    def stat_info(self):
        
        self.at_par(self.par_best)
        
        return Info.from_dict(self.all_stat)


    @property
    def IC_info(self):
        
        IC_info = OrderedDict()
        IC_info['AIC'] = ['%.2f'%self.aic]
        IC_info['AICc'] = ['%.2f'%self.aicc]
        IC_info['BIC'] = ['%.2f'%self.bic]
        IC_info['lnZ'] = [f'{self.lnZ}' if self.lnZ is None else '%.2f'%self.lnZ]
        
        return Info.from_dict(IC_info)
    
    
    def save(self, savepath):
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        json_dump(self.free_par_info.data_list_dict, savepath + '/post_free_par.json')
        json_dump(self.stat_info.data_list_dict, savepath + '/post_stat.json')
        json_dump(self.IC_info.data_list_dict, savepath + '/post_IC.json')


    def __str__(self):
        
        print(self.free_par_info.table)
        print(self.stat_info.table)
        print(self.IC_info.table)
        
        return ''
