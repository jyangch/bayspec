import os
import warnings
import numpy as np
from collections import OrderedDict

from ..util.info import Info
from ..util.prior import unif
from ..util.param import Par, Cfg
from ..util.tools import SuperDict
from ..util.tools import cached_property, json_dump



class Model(object):
    
    _allowed_types = ('add', 'mul', 'conv', 'tinv', 'math')

    def __init__(self):
        
        self.expr = 'model'
        self.type = 'add'
        self.comment = 'model base class'
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0)
        
        self.params = OrderedDict()
        self.params['p'] = Par(1, unif(0, 2))

        
    def func(self, E, T=None, O=None):
        
        pass
        
        
    @cached_property()
    def mdicts(self):
        
        return OrderedDict([(self.expr, self)])
    
    
    @cached_property()
    def fdicts(self):
        
        return OrderedDict([(ex, mo.func) for ex, mo in self.mdicts.items()])
    
    
    @cached_property()
    def cdicts(self):
        
        return OrderedDict([(ex, mo.config) for ex, mo in self.mdicts.items()])

    
    @cached_property()
    def pdicts(self):
        
        return OrderedDict([(ex, mo.params) for ex, mo in self.mdicts.items()])
    
    
    @cached_property()
    def cfg(self):

        cid = 0
        cfg = SuperDict()
        
        for config in self.cdicts.values():
            for cg in config.values():
                cid += 1
                cfg[str(cid)] = cg
                
        return cfg
    
    
    @cached_property()
    def par(self):
        
        pid = 0
        par = SuperDict()
        
        for params in self.pdicts.values():
            for pr in params.values():
                pid += 1
                par[str(pid)] = pr
                
        return par
    
    
    @property
    def pvalues(self):

        return tuple([pr.value for pr in self.par.values()])
                

    @property
    def all_config(self):
        
        cid = 0
        all_config = list()
        
        for expr, config in self.cdicts.items():
            for cl, cg in config.items():
                cid += 1
                
                all_config.append(
                    {'cfg#': str(cid), 
                     'Component': expr, 
                     'Parameter': cl, 
                     'Value': cg.val})

        return all_config
    
    
    @property
    def all_params(self):
        
        pid = 0
        all_params = list()
        
        for expr, params in self.pdicts.items():
            for pl, pr in params.items():
                pid += 1
                
                all_params.append(
                    {'par#': str(pid), 
                     'Component': expr, 
                     'Parameter': pl, 
                     'Value': pr.val, 
                     'Prior': f'{pr.prior_info}', 
                     'Frozen': pr.frozen, 
                     'Posterior': f'{pr.post_info}'})

        return all_params


    @property
    def cfg_info(self):
            
        return Info.from_list_dict(self.all_config)

  
    @property
    def par_info(self):
        
        par_info = Info.list_dict_to_dict(self.all_params)
        
        del par_info['Posterior']
        
        return Info.from_dict(par_info)
    
    
    def save(self, savepath):
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        json_dump(self.cfg_info.data_list_dict, savepath + '/model_cfg.json')
        json_dump(self.par_info.data_list_dict, savepath + '/model_par.json')


    @property
    def fit_to(self):
        
        try:
            return self._fit_to
        except AttributeError:
            raise AttributeError('no data fit to')


    @fit_to.setter
    def fit_to(self, new_data):
        
        from ..data.data import Data

        self._fit_to = new_data

        if not isinstance(self._fit_to, Data): 
            raise ValueError('fit_to argument should be Data type!')

        try:
            self._fit_to.fit_with
        except AttributeError:
            self._fit_to.fit_with = self
        else:
            if self._fit_to.fit_with != self:
                self._fit_to.fit_with = self


    def integ(self, ebin, tarr=None, ngrid=5):
        
        scale = np.linspace(0, 1, ngrid)
        egrid = ebin[:, [0]] + (ebin[:, [1]] - ebin[:, [0]]) * scale
        
        egrid = np.maximum(egrid, 1e-10)
        
        if self.type == 'add':
            fgrid = self.func(egrid.ravel()).reshape(-1, ngrid)
            return np.trapz(fgrid, egrid, axis=1)
            
        elif self.type == 'tinv':
            tgrid = np.repeat(tarr[:, None], ngrid, axis=1)
            fgrid = self.func(egrid.ravel(), tgrid.ravel()).reshape(-1, ngrid)
            return np.trapz(fgrid, egrid, axis=1)
            
        elif self.type == 'mul':
            ewidt = ebin[:, 1] - ebin[:, 0]
            fgrid = self.func(egrid.ravel()).reshape(-1, ngrid)
            return np.trapz(fgrid, egrid, axis=1) / ewidt
            
        else:
            msg = f'integ is invalid for {self.type} type model'
            raise TypeError(msg)


    def convolve_response(self, response, time=None):
        
        nbin = response.phbin.shape[0]
        tarr = np.repeat(time, nbin)
        
        phtflux = self.integ(response.phbin, tarr)
        ctsrate = np.dot(phtflux, response.drm)
        ctsspec = ctsrate / response.chbin_width
        
        return ctsrate, ctsspec
        
        
    def convolve_dataunit(self, dataunit):
        
        phtflux = self.integ(dataunit.ebin, dataunit.tarr)
        ctsrate = np.dot(phtflux, dataunit.corr_rsp_drm)
        ctsspec = ctsrate / dataunit.rsp_chbin_width
        
        return ctsrate, ctsspec
        
        
    def convolve_data(self, data):
        
        flat_phtflux = self.integ(data.ebin, data.tarr)
        phtflux = [flat_phtflux[i:j] for (i, j) in zip(data.bin_start, data.bin_stop)]
        ctsrate = [np.dot(pf, drm) for (pf, drm) in zip(phtflux, data.corr_rsp_drm)]
        ctsspec = [cr / chw for (cr, chw) in zip(ctsrate, data.rsp_chbin_width)]
        
        return ctsrate, ctsspec
        
        
    def _convolve(self):
        
        flat_phtflux = self.integ(self.fit_to.ebin, self.fit_to.tarr)
        phtflux = [flat_phtflux[i:j] for (i, j) in zip(self.fit_to.bin_start, self.fit_to.bin_stop)]
        ctsrate = [np.dot(pf, drm) for (pf, drm) in zip(phtflux, self.fit_to.corr_rsp_drm)]
        
        return ctsrate
    
    
    def _convolve_f64(self):
        
        flat_phtflux = self.integ(self.fit_to.ebin, self.fit_to.tarr)
        phtflux = [flat_phtflux[i:j] for (i, j) in zip(self.fit_to.bin_start, self.fit_to.bin_stop)]
        ctsrate = [np.dot(pf, drm).astype(np.float64) for (pf, drm) in zip(phtflux, self.fit_to.corr_rsp_drm)]
        
        return ctsrate
    
    
    def _re_convolve(self):
        
        flat_phtflux = self.integ(self.fit_to.ebin, self.fit_to.tarr)
        phtflux = [flat_phtflux[i:j] for (i, j) in zip(self.fit_to.bin_start, self.fit_to.bin_stop)]
        re_ctsrate = [np.dot(pf, drm) for (pf, drm) in zip(phtflux, self.fit_to.corr_rsp_re_drm)]
        
        return re_ctsrate


    @property
    def conv_ctsrate(self):
        
        return self._convolve()
    
    
    @property
    def conv_ctsrate_f64(self):
        
        return self._convolve_f64()


    @property
    def conv_re_ctsrate(self):
        
        return self._re_convolve()


    @property
    def conv_ctsspec(self):
        
        return [cr / chw for (cr, chw) in zip(self.conv_ctsrate, self.fit_to.rsp_chbin_width)]
    
    
    @property
    def conv_re_ctsspec(self):
        
        return [cr / chw for (cr, chw) in zip(self.conv_re_ctsrate, self.fit_to.rsp_re_chbin_width)]
    
    
    @property
    def phtspec_at_rsp(self):
        
        return [self.phtspec(E, T) for (E, T) in \
            zip(self.fit_to.rsp_chbin_mean, self.fit_to.rsp_chbin_tarr)]
        
        
    @property
    def re_phtspec_at_rsp(self):
        
        return [self.phtspec(E, T) for (E, T) in \
            zip(self.fit_to.rsp_re_chbin_mean, self.fit_to.rsp_re_chbin_tarr)]
        
        
    @property
    def flxspec_at_rsp(self):
        
        return [self.flxspec(E, T) for (E, T) in \
            zip(self.fit_to.rsp_chbin_mean, self.fit_to.rsp_chbin_tarr)]
        
        
    @property
    def re_flxspec_at_rsp(self):
        
        return [self.flxspec(E, T) for (E, T) in \
            zip(self.fit_to.rsp_re_chbin_mean, self.fit_to.rsp_re_chbin_tarr)]
        
        
    @property
    def ergspec_at_rsp(self):
        
        return [self.ergspec(E, T) for (E, T) in \
            zip(self.fit_to.rsp_chbin_mean, self.fit_to.rsp_chbin_tarr)]
        
        
    @property
    def re_ergspec_at_rsp(self):
        
        return [self.ergspec(E, T) for (E, T) in \
            zip(self.fit_to.rsp_re_chbin_mean, self.fit_to.rsp_re_chbin_tarr)]
        
        
    @property
    def cts_to_pht(self):
        
        return [pht / cts for (cts, pht) in zip(self.conv_ctsspec, self.phtspec_at_rsp)]
    
    
    @property
    def re_cts_to_pht(self):
        
        return [pht / cts for (cts, pht) in zip(self.conv_re_ctsspec, self.re_phtspec_at_rsp)]
    
    
    @property
    def cts_to_flx(self):
        
        return [flx / cts for (cts, flx) in zip(self.conv_ctsspec, self.flxspec_at_rsp)]
    
    
    @property
    def re_cts_to_flx(self):
        
        return [flx / cts for (cts, flx) in zip(self.conv_re_ctsspec, self.re_flxspec_at_rsp)]
    
    
    @property
    def cts_to_erg(self):
        
        return [erg / cts for (cts, erg) in zip(self.conv_ctsspec, self.ergspec_at_rsp)]
    
    
    @property
    def re_cts_to_erg(self):
        
        return [erg / cts for (cts, erg) in zip(self.conv_re_ctsspec, self.re_ergspec_at_rsp)]
    
    
    @property
    def rate_to_flux(self):
        
        ctsrate = [np.sum(cr) for cr in self.fit_to.net_ctsrate]
        ergflux = [np.sum([self.ergflux(emin, emax, 1000) for emin, emax in notc])
                   for notc in self.fit_to.notcs]
        
        return [flux / rate for (flux, rate) in zip(ergflux, ctsrate)]
    
    
    @property
    def conv_rate_to_flux(self):
        
        ctsrate = [np.sum(cr) for cr in self.conv_ctsrate]
        ergflux = [np.sum([self.ergflux(emin, emax, 1000) for emin, emax in notc])
                   for notc in self.fit_to.notcs]
        
        return [flux / rate for (flux, rate) in zip(ergflux, ctsrate)]
    
    
    def rate_to_fluxdensity(self, at=1, unit='Fv'):
        
        ctsrate = [np.sum(cr) for cr in self.fit_to.net_ctsrate]
        if unit == 'NE':    # photons cm-2 s-1 keV-1
            fluxdensity = self.phtspec(at)
        elif unit == 'Fv':  # erg cm-2 s-1 keV-1
            fluxdensity = self.flxspec(at)
        elif unit == 'Jy':  # Jansky
            fluxdensity = self.flxspec(at) * 1e6 / 2.416
        else:
            raise ValueError(f'unsupported value of unit: {unit}')
            
        return [fluxdensity / rate for rate in ctsrate]
    
    
    def conv_rate_to_fluxdensity(self, at=1, unit='fv'):
        
        ctsrate = [np.sum(cr) for cr in self.conv_ctsrate]
        if unit == 'NE':    # photons cm-2 s-1 keV-1
            fluxdensity = self.phtspec(at)
        elif unit == 'Fv':  # erg cm-2 s-1 keV-1
            fluxdensity = self.flxspec(at)
        elif unit == 'Jy':  # Jansky
            fluxdensity = self.flxspec(at) * 1e6 / 2.416
        else:
            raise ValueError(f'unsupported value of unit: {unit}')
            
        return [fluxdensity / rate for rate in ctsrate]


    def phtspec(self, E, T=None):
        # NE in units of photons cm-2 s-1 keV-1
        
        if self.type not in ['add', 'tinv']:
            msg = f'ne is invalid for {self.type} type model'
            raise TypeError(msg)
        else:
            return self.func(E, T)
        
        
    def nouspec(self, E):
        # dimensionless
        
        if self.type not in ['mul', 'math']:
            msg = f'fracspec is invalid for {self.type} type model'
            raise TypeError(msg)
        else:
            return self.func(E)


    def flxspec(self, E, T=None):
        # Fv in units of erg cm-2 s-1 keV-1
        
        if self.type not in ['add', 'tinv']:
            msg = f'ene is invalid for {self.type} type model'
            raise TypeError(msg)
        else:
            return 1.60218e-9 * E * self.phtspec(E, T)
        
    
    def ergspec(self, E, T=None):
        # vFv in units of erg cm-2 s-1
        
        if self.type not in ['add', 'tinv']:
            msg = f'eene is invalid for {self.type} type model'
            raise TypeError(msg)
        else:
            return 1.60218e-9 * E * E * self.phtspec(E, T)


    def phtflux(self, emin, emax, ngrid, epoch=None):
        # integ(NE, E) in units of photons cm-2 s-1
        
        if self.type not in ['add', 'tinv']:
            msg = f'phtflux is invalid for {self.type} type model'
            raise TypeError(msg)
        elif self.type == 'add':
            egrid = np.logspace(np.log10(emin), np.log10(emax), ngrid)
            return np.trapz(self.phtspec(egrid), egrid)
        elif self.type == 'tinv':
            assert epoch is not None
            egrid = np.logspace(np.log10(emin), np.log10(emax), ngrid)
            tgrid = np.ones_like(egrid) * epoch
            return np.trapz(self.phtspec(egrid, tgrid), egrid)
            
        
    def ergflux(self, emin, emax, ngrid, epoch=None):
        # integ(Fv, E) in units of erg cm-2 s-1
        
        if self.type not in ['add', 'tinv']:
            msg = f'ergflux is invalid for {self.type} type model'
            raise TypeError(msg)
        elif self.type == 'add':
            egrid = np.logspace(np.log10(emin), np.log10(emax), ngrid)
            return np.trapz(self.flxspec(egrid), egrid)
        elif self.type == 'tinv':
            assert epoch is not None
            egrid = np.logspace(np.log10(emin), np.log10(emax), ngrid)
            tgrid = np.ones_like(egrid) * epoch
            return np.trapz(self.flxspec(egrid, tgrid), egrid)
        
        
    def at_par(self, theta):
        
        for i, thi in enumerate(theta): 
            self.par[i+1].val = thi


    @property
    def par_mean(self):
        
        return [par.val if par.frozen else par.post.mean for par in self.par.values()]


    @property
    def par_median(self):
        
        return [par.val if par.frozen else par.post.median for par in self.par.values()]


    @property
    def par_best(self):
        
        return [par.val if par.frozen else par.post.best for par in self.par.values()]


    @property
    def par_best_ci(self):
        
        return [par.val if par.frozen else par.post.best_ci for par in self.par.values()]


    def best_phtspec(self, E, T=None):
        
        self.at_par(self.par_best)
        
        return self.phtspec(E, T)


    def best_ci_phtspec(self, E, T=None):
        
        self.at_par(self.par_best_ci)
        
        return self.phtspec(E, T)
    
    
    def median_phtspec(self, E, T=None):
        
        self.at_par(self.par_median)
        
        return self.phtspec(E, T)
    
    
    def mean_phtspec(self, E, T=None):
        
        self.at_par(self.par_mean)
        
        return self.phtspec(E, T)
    
    
    def best_nouspec(self, E):
        
        self.at_par(self.par_best)
        
        return self.nouspec(E)
    
    
    def best_ci_nouspec(self, E):
        
        self.at_par(self.par_best_ci)
        
        return self.nouspec(E)
    
    
    def median_nouspec(self, E):
        
        self.at_par(self.par_median)
        
        return self.nouspec(E)
    
    
    def mean_nouspec(self, E):
        
        self.at_par(self.par_mean)
        
        return self.nouspec(E)
    
    
    def best_flxspec(self, E, T=None):
        
        self.at_par(self.par_best)
        
        return self.flxspec(E, T)
    
    
    def best_ci_flxspec(self, E, T=None):
        
        self.at_par(self.par_best_ci)
        
        return self.flxspec(E, T)
    
    
    def median_flxspec(self, E, T=None):
        
        self.at_par(self.par_median)
        
        return self.flxspec(E, T)
    
    
    def mean_flxspec(self, E, T=None):
        
        self.at_par(self.par_mean)
        
        return self.flxspec(E, T)
    
    
    def best_ergspec(self, E, T=None):
        
        self.at_par(self.par_best)
        
        return self.ergspec(E, T)
    
    
    def best_ci_ergspec(self, E, T=None):
        
        self.at_par(self.par_best_ci)
        
        return self.ergspec(E, T)
    
    
    def median_ergspec(self, E, T=None):
        
        self.at_par(self.par_median)
        
        return self.ergspec(E, T)
    
    
    def mean_ergspec(self, E, T=None):
        
        self.at_par(self.par_mean)
        
        return self.ergspec(E, T)
    
    
    def best_phtflux(self, emin, emax, ngrid, epoch=None):
        
        self.at_par(self.par_best)
        
        return self.phtflux(emin, emax, ngrid, epoch)
    
    
    def best_ci_phtflux(self, emin, emax, ngrid, epoch=None):
        
        self.at_par(self.par_best_ci)
        
        return self.phtflux(emin, emax, ngrid, epoch)
    
    
    def median_phtflux(self, emin, emax, ngrid, epoch=None):
        
        self.at_par(self.par_median)
        
        return self.phtflux(emin, emax, ngrid, epoch)
    
    
    def mean_phtflux(self, emin, emax, ngrid, epoch=None):
        
        self.at_par(self.par_mean)
        
        return self.phtflux(emin, emax, ngrid, epoch)
    
    
    def best_ergflux(self, emin, emax, ngrid, epoch=None):
        
        self.at_par(self.par_best)
        
        return self.ergflux(emin, emax, ngrid, epoch)
    
    
    def best_ci_ergflux(self, emin, emax, ngrid, epoch=None):
        
        self.at_par(self.par_best_ci)
        
        return self.ergflux(emin, emax, ngrid, epoch)
    
    
    def median_ergflux(self, emin, emax, ngrid, epoch=None):
        
        self.at_par(self.par_median)
        
        return self.ergflux(emin, emax, ngrid, epoch)
    
    
    def mean_ergflux(self, emin, emax, ngrid, epoch=None):
        
        self.at_par(self.par_mean)
        
        return self.ergflux(emin, emax, ngrid, epoch)
    

    def best_ergflux_ratio(self, erange1, erange2, ngrid, epoch=None):
        
        self.at_par(self.par_best)

        emin1, emax1 = erange1

        emin2, emax2 = erange2
        
        return self.ergflux(emin1, emax1, ngrid, epoch) / self.ergflux(emin2, emax2, ngrid, epoch)
    
    
    def best_ci_ergflux_ratio(self, erange1, erange2, ngrid, epoch=None):
        
        self.at_par(self.par_best_ci)

        emin1, emax1 = erange1

        emin2, emax2 = erange2
        
        return self.ergflux(emin1, emax1, ngrid, epoch) / self.ergflux(emin2, emax2, ngrid, epoch)
    
    
    def median_ergflux_ratio(self, erange1, erange2, ngrid, epoch=None):
        
        self.at_par(self.par_median)

        emin1, emax1 = erange1

        emin2, emax2 = erange2
        
        return self.ergflux(emin1, emax1, ngrid, epoch) / self.ergflux(emin2, emax2, ngrid, epoch)
    
    
    def mean_ergflux_ratio(self, erange1, erange2, ngrid, epoch=None):
        
        self.at_par(self.par_mean)

        emin1, emax1 = erange1

        emin2, emax2 = erange2
        
        return self.ergflux(emin1, emax1, ngrid, epoch) / self.ergflux(emin2, emax2, ngrid, epoch)


    @property
    def posterior_nsample(self):
        
        nsample = max([1 if par.frozen else par.post.nsample \
            for par in self.par.values()])
        
        return nsample
        
    
    @property
    def posterior_sample(self):
        
        sample = np.vstack([np.full(self.posterior_nsample, par.val) \
            if par.frozen else par.post.sample.copy() \
                for par in self.par.values()]).T
            
        return sample
    
    
    def sample_statistic(self, sample):
        
        mean = np.mean(sample, axis=0)
        median = np.median(sample, axis=0)
        
        q = 68.27 / 100
        Isigma = np.quantile(sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        
        q = 95.45 / 100
        IIsigma = np.quantile(sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        
        q = 99.73 / 100
        IIIsigma = np.quantile(sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        
        return dict([('mean', mean), 
                     ('median', median), 
                     ('Isigma', Isigma), 
                     ('IIsigma', IIsigma), 
                     ('IIIsigma', IIIsigma)])
    
    
    @property
    def par_sample(self):
        
        return self.sample_statistic(self.posterior_sample)


    def phtspec_sample(self, E, T=None):
        
        sample = np.zeros([self.posterior_nsample, len(E)], dtype=float)
        
        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.phtspec(E, T)
            
        return self.sample_statistic(sample)
    
    
    def nouspec_sample(self, E):
        
        sample = np.zeros([self.posterior_nsample, len(E)], dtype=float)
        
        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.nouspec(E)
            
        return self.sample_statistic(sample)
    
    
    def flxspec_sample(self, E, T=None):
        
        sample = np.zeros([self.posterior_nsample, len(E)], dtype=float)
        
        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.flxspec(E, T)
            
        return self.sample_statistic(sample)
    
    
    def ergspec_sample(self, E, T=None):
        
        sample = np.zeros([self.posterior_nsample, len(E)], dtype=float)
        
        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.ergspec(E, T)
            
        return self.sample_statistic(sample)
    
    
    def phtflux_sample(self, emin, emax, ngrid, epoch=None):
        
        sample = np.zeros(self.posterior_nsample, dtype=float)
        
        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.phtflux(emin, emax, ngrid, epoch)
            
        return self.sample_statistic(sample)
    
    
    def ergflux_sample(self, emin, emax, ngrid, epoch=None):
        
        sample = np.zeros(self.posterior_nsample, dtype=float)
        
        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.ergflux(emin, emax, ngrid, epoch)
            
        return self.sample_statistic(sample)


    def ergflux_ratio_sample(self, erange1, erange2, ngrid, epoch=None):
        
        sample = np.zeros(self.posterior_nsample, dtype=float)

        emin1, emax1 = erange1
        
        emin2, emax2 = erange2
        
        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.ergflux(emin1, emax1, ngrid, epoch) / self.ergflux(emin2, emax2, ngrid, epoch)
            
        return self.sample_statistic(sample)


    def __add__(self, other):
        
        return CompositeModel(self, '+', other)


    def __radd__(self, other):
        
        return self.__add__(other)


    def __sub__(self, other):
        
        return CompositeModel(self, '-', other)


    def __rsub__(self, other):
        
        return CompositeModel(other, '-', self)


    def __mul__(self, other):
        
        return CompositeModel(self, '*', other)


    def __rmul__(self, other):
        
        return self.__mul__(other)


    def __truediv__(self, other):
        
        return CompositeModel(self, '/', other)


    def __rtruediv__(self, other):
        
        return CompositeModel(other, '/', self)
    
    
    def __call__(self, other=None):
        
        return CompositeModel(self, '()', other)


    def __str__(self):
        
        print(f'{self.expr} [{self.type}]')
        print(self.comment)
        
        print(self.cfg_info.table)
        print(self.par_info.table)
        
        return ''



class Additive(Model):
    
    def __init__(self):
        super().__init__()
        self.type = 'add'



class Tinvolved(Model):
    
    def __init__(self):
        super().__init__()
        self.type = 'tinv'



class Multiplicative(Model):
    
    def __init__(self):
        super().__init__()
        self.type = 'mul'



class Mathematic(Model):
    
    def __init__(self):
        super().__init__()
        self.type = 'math'

    

class FrozenConst(Mathematic):

    def __init__(self, value):
        super().__init__()
        
        self.expr = 'const'
        self.comment = f'constant model with value {value}'

        self.params = OrderedDict()
        self.params['$C$'] = Par(value, frozen=True)
        
    
    def func(self, E, T=None, O=None):
        
        C = self.params['$C$'].value
        return C



class CompositeModel(Model):

    def __init__(self, m1, op, m2):
        
        self.op = op
        
        if isinstance(m1, Model):
            self.m1 = m1
        elif isinstance(m1, (int, float)):
            self.m1 = FrozenConst(m1)
        else:
            raise ValueError(f'unsupported model type for {op}')
        
        if isinstance(m2, Model):
            self.m2 = m2
        elif isinstance(m2, (int, float)):
            self.m2 = FrozenConst(m2)
        else:
            raise ValueError(f'unsupported model type for {op}')
        
        for ex in set(self.m1.mdicts.keys()) & set(self.m2.mdicts.keys()):
            if id(self.m1.mdicts[ex]) == id(self.m2.mdicts[ex]):
                msg = f'note that the same object ({ex}) is used multiple times!'
                warnings.warn(msg)
            else:
                msg = f'note that the objects with same name ({ex}) are used!'
                warnings.warn(msg)
                
                family = set(self.m1.mdicts.keys()) | set(self.m2.mdicts.keys())
                self.m2.mdicts[ex].expr = self._generate_unique_name(ex, family)


    @cached_property()
    def expr(self):
        
        if self.op == '()':
            if self.m2.expr[0] == '(' and self.m2.expr[-1] == ')':
                return f'{self.m1.expr}{self.m2.expr}'
            else:
                return f'{self.m1.expr}({self.m2.expr})'
        else:
            return f'({self.m1.expr}{self.op}{self.m2.expr})'
        

    @cached_property()
    def type(self):
        
        assert self.m1.type in self._allowed_types, f'unsupported model.type: {self.m1.type}'
        assert self.m2.type in self._allowed_types, f'unsupported model.type: {self.m2.type}'
        
        if self.op == '()':
            type_op = f'{self.m1.type}({self.m2.type})'
        else:
            type_op = f'{self.m1.type}{self.op}{self.m2.type}'
            
        if not self.tdict[type_op]:
            msg = f'unsupported model.type {(self.m1.type, self.m2.type)} for {self.op}'
            raise ValueError(msg)
        else:
            return self.tdict[type_op]
            
            
    @cached_property()
    def comment(self):
        
        return '\n'.join([f'{expr}: {mo.comment}' for expr, mo in self.mdicts.items()])


    def func(self, E, T=None, O=None):
        
        if self.op == '+':
            return self.m1.func(E, T, O) + self.m2.func(E, T, O)
        elif self.op == '-':
            return self.m1.func(E, T, O) - self.m2.func(E, T, O)
        elif self.op == '*':
            return self.m1.func(E, T, O) * self.m2.func(E, T, O)
        elif self.op == '/':
            return self.m1.func(E, T, O) / self.m2.func(E, T, O)
        elif self.op == '()':
            return self.m1.func(E, T, self.m2)
        else:
            raise ValueError(f'Unknown operation: {self.op}')
        
        
    @cached_property()
    def mdicts(self):
        
        return OrderedDict({**self.m1.mdicts, **self.m2.mdicts})
    

    @staticmethod
    def _generate_unique_name(name, family, number=2):
        
        while True:
            new_name = f'{name}_{number}'
            if new_name in family:
                continue
            else:
                break
        return new_name


    def type_operation(self):
        
        assert self.m1.type in self._allowed_types, f'unsupported model.type: {self.m1.type}'
        assert self.m2.type in self._allowed_types, f'unsupported model.type: {self.m2.type}'
        
        types = (self.m1.type, self.m2.type)
        msg = f'unsupported model.type {types} for {self.op}'
        
        if self.op == '()':
            assert self.m1.type == 'conv', msg
            return self.m2.type
        elif self.op == '+' or self.op == '-':
            if set(types) < set(('add', 'tinv', 'math')):
                if 'tinv' in types:
                    return 'tinv'
                elif 'add' in types:
                    return 'add'
                else:
                    return 'math'
            elif set(types) <= set(('mul', 'math')):
                if 'mul' in types:
                    return 'mul'
                else:
                    return 'math'
            else:
                raise ValueError(msg)
        elif self.op == '*':
            assert 'mul' in types or 'math' in types, msg
            if 'conv' in types:
                raise ValueError(msg)
            elif 'tinv' in types:
                return 'tinv'
            elif 'add' in types:
                return 'add'
            elif 'mul' in types:
                return 'mul'
            else:
                return 'math'
        elif self.op == '/':
            assert self.m2.type in ('mul', 'math'), msg
            if 'conv' in types:
                raise ValueError(msg)
            elif 'tinv' in types:
                return 'tinv'
            elif 'add' in types:
                return 'add'
            elif 'mul' in types:
                return 'mul'
            else:
                return 'math'


    @cached_property()
    def tdict(self):
        
        return {'add(add)': False,
                'add(mul)': False,
                'add(conv)': False,
                'add(tinv)': False,
                'add(math)': False,
                'mul(add)': False,
                'mul(mul)': False,
                'mul(conv)': False,
                'mul(tinv)': False,
                'mul(math)': False,
                'conv(add)': 'add',
                'conv(mul)': 'mul',
                'conv(conv)': 'conv',
                'conv(tinv)': 'tinv',
                'conv(math)': 'math',
                'tinv(add)': False,
                'tinv(mul)': False,
                'tinv(conv)': False,
                'tinv(tinv)': False,
                'tinv(math)': False,
                'math(add)': False,
                'math(mul)': False,
                'math(conv)': False,
                'math(tinv)': False,
                'math(math)': False,
                'add+add': 'add',
                'add+mul': False,
                'add+conv': False,
                'add+tinv': 'tinv',
                'add+math': 'add',
                'mul+add': False,
                'mul+mul': 'mul',
                'mul+conv': False,
                'mul+tinv': False,
                'mul+math': 'mul',
                'conv+add': False,
                'conv+mul': False,
                'conv+conv': False,
                'conv+tinv': False,
                'conv+math': False,
                'tinv+add': 'tinv',
                'tinv+mul': False,
                'tinv+conv': False,
                'tinv+tinv': 'tinv',
                'tinv+math': 'tinv',
                'math+add': 'add',
                'math+mul': 'mul',
                'math+conv': False,
                'math+tinv': 'tinv',
                'math+math': 'math',
                'add-add': 'add',
                'add-mul': False,
                'add-conv': False,
                'add-tinv': 'tinv',
                'add-math': 'add',
                'mul-add': False,
                'mul-mul': 'mul',
                'mul-conv': False,
                'mul-tinv': False,
                'mul-math': 'mul',
                'conv-add': False,
                'conv-mul': False,
                'conv-conv': False,
                'conv-tinv': False,
                'conv-math': False,
                'tinv-add': 'tinv',
                'tinv-mul': False,
                'tinv-conv': False,
                'tinv-tinv': 'tinv',
                'tinv-math': 'tinv',
                'math-add': 'add',
                'math-mul': 'mul',
                'math-conv': False,
                'math-tinv': 'tinv',
                'math-math': 'math',
                'add*add': False,
                'add*mul': 'add',
                'add*conv': False,
                'add*tinv': False,
                'add*math': 'add',
                'mul*add': 'add',
                'mul*mul': 'mul',
                'mul*conv': False,
                'mul*tinv': 'tinv',
                'mul*math': 'mul',
                'conv*add': False,
                'conv*mul': False,
                'conv*conv': False,
                'conv*tinv': False,
                'conv*math': False,
                'tinv*add': False,
                'tinv*mul': 'tinv',
                'tinv*conv': False,
                'tinv*tinv': False,
                'tinv*math': 'tinv',
                'math*add': 'add',
                'math*mul': 'mul',
                'math*conv': False,
                'math*tinv': 'tinv',
                'math*math': 'math',
                'add/add': False,
                'add/mul': 'add',
                'add/conv': False,
                'add/tinv': False,
                'add/math': 'add',
                'mul/add': False,
                'mul/mul': 'mul',
                'mul/conv': False,
                'mul/tinv': False,
                'mul/math': 'mul',
                'conv/add': False,
                'conv/mul': False,
                'conv/conv': False,
                'conv/tinv': False,
                'conv/math': False,
                'tinv/add': False,
                'tinv/mul': 'tinv',
                'tinv/conv': False,
                'tinv/tinv': False,
                'tinv/math': 'tinv',
                'math/add': False,
                'math/mul': 'mul',
                'math/conv': False,
                'math/tinv': False,
                'math/math': 'math'}
