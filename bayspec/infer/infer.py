import os
import json
import ctypes
import numpy as np
from .pair import Pair
from ..util.info import Info
from ..data.data import Data
from ..model.model import Model
from scipy.optimize import minimize
from collections import OrderedDict
from ..util.tools import SuperDict, JsonEncoder



class Infer(object):
    
    def __init__(self, pairs=None):
        
        self.pairs = pairs
        
        
    @property
    def pairs(self):
        
        return self._pairs
    
    
    @pairs.setter
    def pairs(self, new_pairs):
        
        self._pairs = list()

        if new_pairs is None:
            pass
            
        elif isinstance(new_pairs, list):
            for pair in new_pairs:
                if isinstance(pair, (tuple, list)):
                    self._addpair(*pair)
                    
            self._extract()
            
        else:
            raise ValueError('unsupported pair type')

        
    def _addpair(self, *pair):
        
        p1, p2 = pair
        
        if isinstance(p1, Data):
            data = p1
            if isinstance(p2, Model):
                model = p2
            else:
                raise ValueError('p1 is Data type, then p2 should be Model type')
            
        elif isinstance(p1, Model):
            model = p1
            if isinstance(p2, Data):
                data = p2
            else:
                raise ValueError('p1 is Model type, then p2 should be Data type')
            
        else:
            raise ValueError('unsupported pair type')
        
        self._pairs.append((data, model))
        
        
    def append(self, *pair):
        
        self._addpair(*pair)
        self._extract()


    def _extract(self):
        
        if self.pairs is None:
            raise ValueError('pairs is None')
        
        self.Data = [pair[0] for pair in self.pairs]
        self.Model = [pair[1] for pair in self.pairs]
        self.Pair = [Pair(*pair) for pair in self.pairs]
        
        self.data_exprs = [key for data in self.Data for key in data.exprs]
        self.model_exprs = [model.expr for model in self.Model]
        
        
        self._you_free()


    @property
    def pdicts(self):
        
        return OrderedDict([(md.expr, md.pdicts) for md in (self.Model + self.Data)])


    @property
    def cdicts(self):
        
        return OrderedDict([(mo.expr, mo.pdicts) for mo in self.Model])


    @property
    def cfg(self):

        cid = 0
        cfg = SuperDict()
        
        for mo in self.Model:
            for config in mo.cdicts.values():
                for cg in config.values():
                    cid += 1
                    cfg[str(cid)] = cg
                
        return cfg


    @property
    def par(self):
        
        pid = 0
        par = SuperDict()
        
        for md in (self.Model + self.Data):
            for params in md.pdicts.values():
                for pr in params.values():
                    pid += 1
                    par[str(pid)] = pr
                
        return par

    
    @staticmethod
    def foo(id):
        
        return ctypes.cast(id, ctypes.py_object).value


    @property
    def idpid(self):
        
        pid = 0
        idpid = SuperDict()
        
        for md in (self.Model + self.Data):
            for params in md.pdicts.values():
                for pr in params.values():
                    pid += 1
                    if str(id(pr)) not in idpid:
                        idpid[str(id(pr))] = {str(pid)}
                    else:
                        idpid[str(id(pr))].add(str(pid))
                        
        return idpid
    
    
    @property
    def all_config(self):
        
        cid = 0
        all_config = list()
        
        for mo in self.Model:
            for expr, config in mo.cdicts.items():
                for cl, cg in config.items():
                    cid += 1
                    
                    all_config.append(\
                        {'cfg#': str(cid), 
                         'Expression': mo.expr, 
                         'Component': expr, 
                         'Parameter': cl, 
                         'Value': cg.val})
                        
        return all_config


    @property
    def all_params(self):
        
        pid = 0
        all_params = list()
        
        for md in (self.Model + self.Data):
            for expr, params in md.pdicts.items():
                for pl, pr in params.items():
                    pid += 1
                    
                    self_id = self.idpid[str(id(pr))]
                    mate_id = [self.idpid[str(id(mate))] for mate in pr.mates]
                    mates = self_id.union(*mate_id)
                    mates.remove(str(pid))
                    
                    all_params.append(\
                        {'par#': str(pid), 
                         'Expression': md.expr, 
                         'Component': expr, 
                         'Parameter': pl, 
                         'Value': pr.val, 
                         'Prior': f'{pr.prior_info}', 
                         'Frozen': pr.frozen, 
                         'Mates': mates, 
                         'Posterior': f'{pr.post_info}'})
                        
        return all_params
   

    def _you_free(self):
        
        unfree_par = set()
        self._free_par = SuperDict()
        self._free_params = list()
        
        for param in self.all_params:
            pid = param['par#']
            
            if param['Frozen']:
                unfree_par.update(param['Mates'])

            else:
                if pid not in unfree_par:
                    self._free_par[pid] = self.par[pid]
                    self._free_params.append(param)
                    unfree_par.update(param['Mates'])
                else:
                    unfree_par.update(param['Mates'])

        self._free_plabels = [param['Parameter'] for param in self._free_params]
        self._free_pvalues = [param['Value'] for param in self._free_params]
        self._free_pranges = [par.range for par in self._free_par.values()]
        self._free_nparams = len(self._free_plabels)


    def link(self, pids):
        
        for i, ip in enumerate(pids):
            for j, jp in enumerate(pids):
                if j > i:
                    if id(self.par[ip]) != id(self.par[jp]):
                        self.par[ip].link(self.par[jp])

        self._you_free()
        
        
    def unlink(self, pids):
        
        for i, ip in enumerate(pids):
            for j, jp in enumerate(pids):
                if j > i:
                    if id(self.par[ip]) != id(self.par[jp]):
                        self.par[ip].unlink(self.par[jp])

        self._you_free()
        
        
    @property
    def free_par(self):
        
        return self._free_par
    
    
    @property
    def free_params(self):
        
        return self._free_params
    
    
    @property
    def free_plabels(self):
        
        return self._free_plabels
    
    
    @property
    def free_pvalues(self):
        
        return self._free_pvalues
    
    
    @property
    def free_pranges(self):
        
        return self._free_pranges


    @property
    def free_nparams(self):
        
        return self._free_nparams
        
        
    @property
    def data_chbin_mean(self):
        
        return [value for data in self.Data for value in data.rsp_chbin_mean]
    
    
    @property
    def data_re_chbin_mean(self):
        
        return [value for data in self.Data for value in data.rsp_re_chbin_mean]
    
    
    @property
    def data_chbin_width(self):
        
        return [value for data in self.Data for value in data.rsp_chbin_width]
    
    
    @property
    def data_re_chbin_width(self):
        
        return [value for data in self.Data for value in data.rsp_re_chbin_width]
    
    
    @property
    def data_ctsrate(self):
        
        return [value for data in self.Data for value in data.net_ctsrate]
    
    
    @property
    def data_re_ctsrate(self):
        
        return [value for data in self.Data for value in data.net_re_ctsrate]
    
    
    @property
    def data_ctsrate_error(self):
        
        return [value for data in self.Data for value in data.net_ctsrate_error]
    
    
    @property
    def data_re_ctsrate_error(self):
        
        return [value for data in self.Data for value in data.net_re_ctsrate_error]
    
    
    @property
    def data_ctsspec(self):
        
        return [value for data in self.Data for value in data.net_ctsspec]
    
    
    @property
    def data_re_ctsspec(self):
        
        return [value for data in self.Data for value in data.net_re_ctsspec]
    
    
    @property
    def data_ctsspec_error(self):
        
        return [value for data in self.Data for value in data.net_ctsspec_error]
    
    
    @property
    def data_re_ctsspec_error(self):
        
        return [value for data in self.Data for value in data.net_re_ctsspec_error]
    
    
    @property
    def data_phtspec(self):
        
        return [value for data in self.Data for value in data.deconv_phtspec]
    
    
    @property
    def data_re_phtspec(self):
        
        return [value for data in self.Data for value in data.deconv_re_phtspec]
    
    
    @property
    def data_phtspec_error(self):
        
        return [value for data in self.Data for value in data.deconv_phtspec_error]
    
    
    @property
    def data_re_phtspec_error(self):
        
        return [value for data in self.Data for value in data.deconv_re_phtspec_error]
    
    
    @property
    def model_ctsrate(self):
        
        return [value for model in self.Model for value in model.conv_ctsrate]
    
    
    @property
    def model_re_ctsrate(self):
        
        return [value for model in self.Model for value in model.conv_re_ctsrate]
    
    
    @property
    def model_ctsspec(self):
        
        return [value for model in self.Model for value in model.conv_ctsspec]
    
    
    @property
    def model_re_ctsspec(self):
        
        return [value for model in self.Model for value in model.conv_re_ctsspec]
    
    
    @property
    def model_phtspec(self):
        
        return [value for model in self.Model for value in model.phtspec_at_rsp]
    
    
    @property
    def model_re_phtspec(self):
        
        return [value for model in self.Model for value in model.re_phtspec_at_rsp]
    
    
    @property
    def model_cts_to_pht(self):
        
        return [value for model in self.Model for value in model.cts_to_pht]
    
    
    @property
    def model_re_cts_to_pht(self):
        
        return [value for model in self.Model for value in model.re_cts_to_pht]


    @property
    def stat_list(self):
        
        return np.hstack([pair.stat_list for pair in self.Pair])
    
    
    @property
    def weight_list(self):
        
        return np.hstack([pair.weight_list for pair in self.Pair])
    
    
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
        
        return np.hstack([pair.npoint_list for pair in self.Pair])
    
    
    @property
    def npoint(self):
        
        return np.sum(self.npoint_list)
    
    
    @property
    def dof(self):
        
        return self.npoint - self.free_nparams
        

    @property
    def prior_list(self):
        
        return [par.prior.pdf(par.val) for par in self.free_par.values()]
    
    
    @property
    def logprior(self):
        
        return np.log(np.prod(self.prior_list))
        
        
    @property
    def all_stat(self):
        
        all_stat = OrderedDict([('Data', ['Total']), 
                                ('Model', ['Total']), 
                                ('Statistic', ['stat/dof']), 
                                ('Value', ['{:.2f}/{:d}'.format(self.stat, self.dof)]), 
                                ('Bins', ['{:d}'.format(self.npoint)])])

        for dt, mo in zip(self.Data, self.Model):
            mex = mo.expr
            for sex, stat in zip(dt.exprs, dt.stats):
                all_stat['Data'].insert(-1, sex)
                all_stat['Model'].insert(-1, mex)
                all_stat['Statistic'].insert(-1, stat)
                
        all_stat['Value'] = ['%.2f' % stat for stat in self.stat_list] + all_stat['Value']
        all_stat['Bins'] = ['%d' % point for point in self.npoint_list] + all_stat['Bins']

        return all_stat


    @property
    def cfg_info(self):
        
        return Info.from_list_dict(self.all_config)


    @property
    def par_info(self):
        
        self._you_free()
        
        all_params = self.all_params.copy()
        
        for par in all_params:
            if par['par#'] in self.free_par:
                par['par#'] = par['par#'] + '*'
            else: 
                if par['Frozen']:
                    par['Prior'] = 'frozen'
                else:
                    par['Prior'] = '=par#{%s}'%(','.join(par['Mates']))
        
        par_info = Info.list_dict_to_dict(all_params)
        
        del par_info['Posterior']
        del par_info['Mates']
        del par_info['Frozen']
        
        return Info.from_dict(par_info)
        
        
    @property
    def free_par_info(self):
        
        self._you_free()
        
        free_par_info = Info.list_dict_to_dict(self.free_params)
        
        del free_par_info['Posterior']
        del free_par_info['Mates']
        del free_par_info['Frozen']
        
        return Info.from_dict(free_par_info)


    @property
    def stat_info(self):
        
        return Info.from_dict(self.all_stat)


    def at_par(self, theta):
        
        theta = np.array(theta, dtype=float)
        
        for i, thi in enumerate(theta): 
            self.free_par[i+1].val = thi
            

    def _loglike(self, theta):
        
        self.at_par(theta)
        
        return np.sum([[pair.loglike for pair in self.Pair]])


    def _prior_transform(self, cube):
        
        theta = np.array(cube)
        
        for i, cui in enumerate(cube): 
            theta[i] = self.free_par[i+1].prior.ppf(cui)
            
        return theta
    
    
    def _logprior(self, theta):
        
        pprs = np.zeros_like(theta)

        for i, thi in enumerate(theta):
            pprs[i] = self.free_par[i+1].prior.pdf(thi)
            
        ppr = np.prod(pprs)
        
        if ppr == 0:
            return -np.inf
        else:
            return np.log(ppr)
        
        
    def _logprior_sample(self, theta_sample):
        
        pprs_sample = np.zeros_like(theta_sample)
        
        for i in range(theta_sample.shape[1]):
            pprs_sample[:, i] = self.free_par[i+1].prior.pdf(theta_sample[:, i])
            
        ppr_sample = np.prod(pprs_sample, axis=1)
        
        return np.where(ppr_sample == 0, -np.inf, np.log(ppr_sample))


    def _logprob(self, theta):

        return self._logprior(theta) + self._loglike(theta)


    def __str__(self):
        
        print(self.cfg_info.table)
        print(self.par_info.table)
        
        return ''

        
    def multinest_loglike(self, cube, ndim, nparams):
        
        theta = np.array([cube[i] for i in range(ndim)], dtype=float)
        
        for i, thi in enumerate(theta): 
            self.free_par[i+1].val = thi
        
        return np.sum([[pair.loglike for pair in self.Pair]])


    def multinest_prior_transform(self, cube, ndim, nparams):
        
        for i in range(ndim):
            cube[i] = self.free_par[i+1].prior.ppf(cube[i])
        
    
    def multinest(self, nlive=500, resume=True, savepath='./'):
        
        import pymultinest
        from .posterior import Posterior
        
        self._you_free()
        
        self.nlive = nlive
        self.resume = resume
        self.prefix = savepath + '/1-'
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        pymultinest.run(self.multinest_loglike, self.multinest_prior_transform, self.free_nparams, resume=resume, 
                        verbose=True, n_live_points=nlive, outputfiles_basename=self.prefix, 
                        sampling_efficiency=0.8, importance_nested_sampling=True, multimodal=True)

        self.Analyzer = pymultinest.Analyzer(outputfiles_basename=self.prefix, n_params=self.free_nparams)
        
        self.posterior_stats = self.Analyzer.get_stats()
        self.posterior_sample = self.Analyzer.get_equal_weighted_posterior()

        self.posterior_sample[:, -1] = self.posterior_sample[:, -1] + \
            self._logprior_sample(self.posterior_sample[:, 0:-1])

        self.logevidence = self.posterior_stats['nested importance sampling global log-evidence']
        
        json.dump(self.nlive, open(self.prefix + 'nlive.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.logevidence, open(self.prefix + 'log_evidence.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.posterior_stats, open(self.prefix + 'posterior_stats.json', 'w'), indent=4, cls=JsonEncoder)

        return Posterior(self)


    def emcee_logprob(self, theta):
        
        lp = self._logprior(theta)
        
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._loglike(theta)
    
    
    def emcee(self, nstep=1000, discard=100, resume=True, savepath='./'):

        import emcee
        from .posterior import Posterior
        
        self._you_free()
        
        self.nstep = nstep
        self.discard = discard
        self.resume = resume
        self.prefix = savepath + '/1-'
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        np.random.seed(42)
        ndim = self.free_nparams
        nwalkers = 32 if 2 * ndim < 32 else 2 * ndim
        pos = self.free_pvalues + 1e-4 * np.random.randn(nwalkers, ndim)

        if (not self.resume) or (not os.path.exists(self.prefix + '.npz')):
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.emcee_logprob)
            sampler.run_mcmc(pos, self.nstep, progress=True)
            
            self.params_samples = sampler.get_chain()
            np.savez(self.prefix, samples=self.params_samples)
            
            self.logprob_sample = sampler.get_log_prob()
            np.savetxt(self.prefix + 'logprob.dat', self.logprob_sample)
            
            try:
                self.autocorr_time = sampler.get_autocorr_time()
                json.dump(self.autocorr_time, open(self.prefix + 'autocorr_time.json', 'w'), indent=4, cls=JsonEncoder)
            except:
                pass

        self.params_samples = np.load(self.prefix + '.npz')['samples']
        self.logprob_sample = np.loadtxt(self.prefix + 'logprob.dat')
        
        flat_params_sample = self.params_samples[self.discard:, :, :].reshape(-1, ndim)
        flat_logprob_sample = self.logprob_sample[self.discard:, :].reshape(-1)
        
        self.posterior_sample = np.hstack((flat_params_sample, np.reshape(flat_logprob_sample, (-1, 1))))
        
        np.savetxt(self.prefix + 'post_equal_weights.dat', self.posterior_sample)
        json.dump(self.nstep, open(self.prefix + 'nstep.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.discard, open(self.prefix + 'discard.json', 'w'), indent=4, cls=JsonEncoder)
        
        return Posterior(self)
    
    
    def minimize(self, method='Nelder-Mead'):
        
        """
        method: 'Nelder-Mead', 'TNC', 'SLSQP', 'Powell', 'trust-constr', 'L-BFGS-B'
        """
        
        np.random.seed(42)
        nll = lambda *args: -2 * self._loglike(*args)
        pos = self.free_pvalues + 1e-4 * np.random.randn(self.free_nparams)
        soln = minimize(nll, pos, method=method, bounds=self.free_pranges)
        
        return soln.x
