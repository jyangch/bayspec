import os
import numpy as np
from collections import OrderedDict

from .infer import BayesInfer
from ..util.info import Info
from ..util.post import Post
from ..util.tools import json_dump



class Posterior(BayesInfer):
    
    def __init__(self, infer):
        
        self.infer = infer
        
        
    @property
    def infer(self):
        
        return self._infer
    
    
    @infer.setter
    def infer(self, new_infer):
        
        if not isinstance(new_infer, BayesInfer):
            raise TypeError('expected an instance of BayesInfer')
        
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
        
        free_params = self.free_params.copy()
        
        free_params = Info.list_dict_to_dict(free_params)
        
        del free_params['Posterior']
        del free_params['Mates']
        del free_params['Frozen']
        del free_params['Prior']
        del free_params['Value']
        
        free_params['Mean'] = ['%.3f' % par for par in self.par_mean]
        free_params['Median'] = ['%.3f' % par for par in self.par_median]
        free_params['Best'] = ['%.3f' % par for par in self.par_best]
        free_params['1sigma Best'] = ['%.3f' % par for par in self.par_best_ci]
        free_params['1sigma CI'] = ['[%.3f, %.3f]' % tuple(ci) for ci in self.par_Isigma]
        
        return Info.from_dict(free_params)
    
    
    @property
    def stat_info(self):
        
        self.at_par(self.par_best)
        
        all_stat = self.all_stat.copy()
        
        return Info.from_dict(all_stat)
    
    
    @property
    def all_IC(self):
        
        all_IC = OrderedDict()
        all_IC['AIC'] = self.aic
        all_IC['AICc'] = self.aicc
        all_IC['BIC'] = self.bic
        all_IC['lnZ'] = self.lnZ
        
        return all_IC


    @property
    def IC_info(self):
        
        all_IC = self.all_IC.copy()
        
        return Info.from_dict(all_IC)
    
    
    def save(self, savepath):
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        json_dump(self.free_par_info.data_list_dict, savepath + '/post_free_par.json')
        json_dump(self.stat_info.data_list_dict, savepath + '/post_stat.json')
        json_dump(self.IC_info.data_list_dict, savepath + '/post_IC.json')


    def __str__(self):
        
        return (
            f'*** Posterior Results ***\n'
            f'*** Parameters ***\n'
            f'{self.free_par_info.text_table}\n'
            f'*** Statistics ***\n'
            f'{self.stat_info.text_table}\n'
            f'*** Information Criterias ***\n'
            f'{self.IC_info.text_table}'
            )


    def __repr__(self):
        
        return self.__str__()
    
    
    def _repr_html_(self):
        
        return (
            f'{self.free_par_info.html_style}'
            f'<details open>'
            f'<summary style="margin-bottom: 10px;"><b>Posterior Results</b></summary>'
            f'<details open style="margin-top: 10px;">'
            f'<summary style="margin-bottom: 10px;"><b>Parameters</b></summary>'
            f'{self.free_par_info.html_table}'
            f'</details>'
            f'<details open style="margin-top: 10px;">'
            f'<summary style="margin-bottom: 10px;"><b>Statistics</b></summary>'
            f'{self.stat_info.html_table}'
            f'</details>'
            f'<details open style="margin-top: 10px;">'
            f'<summary style="margin-bottom: 10px;"><b>Information Criterias</b></summary>'
            f'{self.IC_info.html_table}'
            f'</details>'
            )