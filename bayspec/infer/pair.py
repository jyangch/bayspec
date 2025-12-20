import numpy as np

from ..data.data import Data
from ..model.model import Model
from .statistic import Statistic
from ..util.tools import cached_property, clear_cached_property



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
        
        self._PAIR = object()
        
        clear_cached_property(self)
        
        self.data.fit_with = self.model
        
        
    def _convolve(self):
        
        flat_phtflux = self.model.integ(self.data.ebin, self.data.tarr)
        phtflux = [flat_phtflux[i:j] for (i, j) in zip(self.data.bin_start, self.data.bin_stop)]
        ctsrate = [np.dot(pf, drm) for (pf, drm) in zip(phtflux, self.data.corr_rsp_drm)]
        
        return ctsrate
    
    
    def _re_convolve(self):
        
        flat_phtflux = self.model.integ(self.data.ebin, self.data.tarr)
        phtflux = [flat_phtflux[i:j] for (i, j) in zip(self.data.bin_start, self.data.bin_stop)]
        re_ctsrate = [np.dot(pf, drm) for (pf, drm) in zip(phtflux, self.data.corr_rsp_re_drm)]
        
        return re_ctsrate


    @property
    def conv_ctsrate(self):
        
        return self._convolve()
    
    
    @property
    def conv_re_ctsrate(self):
        
        return self._re_convolve()


    @property
    def conv_ctsspec(self):
        
        return [cr / chw for (cr, chw) in zip(self.conv_ctsrate, self.data.rsp_chbin_width)]
    
    
    @property
    def conv_re_ctsspec(self):
        
        return [cr / chw for (cr, chw) in zip(self.conv_re_ctsrate, self.data.rsp_re_chbin_width)]
        
        
    @property
    def phtspec_at_rsp(self):
        
        return [self.model.phtspec(E, T) for (E, T) in \
            zip(self.data.rsp_chbin_mean, self.data.rsp_chbin_tarr)]
        
        
    @property
    def re_phtspec_at_rsp(self):
        
        return [self.model.phtspec(E, T) for (E, T) in \
            zip(self.data.rsp_re_chbin_mean, self.data.rsp_re_chbin_tarr)]
        

    @property
    def flxspec_at_rsp(self):
        
        return [self.model.flxspec(E, T) for (E, T) in \
            zip(self.data.rsp_chbin_mean, self.data.rsp_chbin_tarr)]
        
        
    @property
    def re_flxspec_at_rsp(self):
        
        return [self.model.flxspec(E, T) for (E, T) in \
            zip(self.data.rsp_re_chbin_mean, self.data.rsp_re_chbin_tarr)]
        
        
    @property
    def ergspec_at_rsp(self):
        
        return [self.model.ergspec(E, T) for (E, T) in \
            zip(self.data.rsp_chbin_mean, self.data.rsp_chbin_tarr)]
        
        
    @property
    def re_ergspec_at_rsp(self):
        
        return [self.model.ergspec(E, T) for (E, T) in \
            zip(self.data.rsp_re_chbin_mean, self.data.rsp_re_chbin_tarr)]


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
        
        ctsrate = [np.sum(cr) for cr in self.data.net_ctsrate]
        ergflux = [np.sum([self.model.ergflux(emin, emax, 1000) for emin, emax in notc])
                   for notc in self.data.notcs]
        
        return [flux / rate for (flux, rate) in zip(ergflux, ctsrate)]
    
    
    @property
    def conv_rate_to_flux(self):
        
        ctsrate = [np.sum(cr) for cr in self.conv_ctsrate]
        ergflux = [np.sum([self.model.ergflux(emin, emax, 1000) for emin, emax in notc])
                   for notc in self.data.notcs]
        
        return [flux / rate for (flux, rate) in zip(ergflux, ctsrate)]
    
    
    def rate_to_fluxdensity(self, at=1, unit='fv'):
        
        ctsrate = [np.sum(cr) for cr in self.data.net_ctsrate]
        if unit == 'NE':    # photons cm-2 s-1 keV-1
            fluxdensity = self.model.phtspec(at)
        elif unit == 'Fv':  # erg cm-2 s-1 keV-1
            fluxdensity = self.model.flxspec(at)
        elif unit == 'Jy':  # Jansky
            fluxdensity = self.model.flxspec(at) * 1e6 / 2.416
        else:
            raise ValueError(f'unsupported value of unit: {unit}')
            
        return [fluxdensity / rate for rate in ctsrate]
    
    
    def conv_rate_to_fluxdensity(self, at=1, unit='fv'):
        
        ctsrate = [np.sum(cr) for cr in self.conv_ctsrate]
        if unit == 'NE':    # photons cm-2 s-1 keV-1
            fluxdensity = self.model.phtspec(at)
        elif unit == 'Fv':  # erg cm-2 s-1 keV-1
            fluxdensity = self.model.flxspec(at)
        elif unit == 'Jy':  # Jansky
            fluxdensity = self.model.flxspec(at) * 1e6 / 2.416
        else:
            raise ValueError(f'unsupported value of unit: {unit}')
            
        return [fluxdensity / rate for rate in ctsrate]


    @property
    def deconv_phtspec(self):
        
        return [factor * cts for (factor, cts) in zip(self.cts_to_pht, self.data.net_ctsspec)]
    
    
    @property
    def deconv_re_phtspec(self):
        
        return [factor * cts for (factor, cts) in zip(self.re_cts_to_pht, self.data.net_re_ctsspec)]
    
    
    @property
    def deconv_phtspec_error(self):
        
        return [factor * cts for (factor, cts) in zip(self.cts_to_pht, self.data.net_ctsspec_error)]
    
    
    @property
    def deconv_re_phtspec_error(self):
        
        return [factor * cts for (factor, cts) in zip(self.re_cts_to_pht, self.data.net_re_ctsspec_error)]
    
    
    @property
    def deconv_flxspec(self):
        
        return [factor * cts for (factor, cts) in zip(self.cts_to_flx, self.data.net_ctsspec)]
    
    
    @property
    def deconv_re_flxspec(self):
        
        return [factor * cts for (factor, cts) in zip(self.re_cts_to_flx, self.data.net_re_ctsspec)]
    
    
    @property
    def deconv_flxspec_error(self):
        
        return [factor * cts for (factor, cts) in zip(self.cts_to_flx, self.data.net_ctsspec_error)]
    
    
    @property
    def deconv_re_flxspec_error(self):
        
        return [factor * cts for (factor, cts) in zip(self.re_cts_to_flx, self.data.net_re_ctsspec_error)]
    
    
    @property
    def deconv_ergspec(self):
        
        return [factor * cts for (factor, cts) in zip(self.cts_to_erg, self.data.net_ctsspec)]
    
    
    @property
    def deconv_re_ergspec(self):
        
        return [factor * cts for (factor, cts) in zip(self.re_cts_to_erg, self.data.net_re_ctsspec)]
    
    
    @property
    def deconv_ergspec_error(self):
        
        return [factor * cts for (factor, cts) in zip(self.cts_to_erg, self.data.net_ctsspec_error)]
    
    
    @property
    def deconv_re_ergspec_error(self):
        
        return [factor * cts for (factor, cts) in zip(self.re_cts_to_erg, self.data.net_re_ctsspec_error)]


    @property
    def residual(self):
        
        return list(map(lambda oi, mi, si: (oi - mi) / si, 
                        self.data.net_ctsrate, 
                        self.conv_ctsrate, 
                        self.data.net_ctsrate_error))


    @property
    def re_residual(self):
        
        return list(map(lambda oi, mi, si: (oi - mi) / si, 
                        self.data.net_re_ctsrate, 
                        self.conv_re_ctsrate, 
                        self.data.net_re_ctsrate_error))


    @cached_property()
    def stat_func(self):
        
        return lambda S, B, m, ts, tb, sigma_S, sigma_B, stat: \
            np.inf if np.isnan(m).any() or np.isinf(m).any() else \
                self._allowed_stats[stat](S=S, B=B, m=m, ts=ts, tb=tb, \
                    sigma_S=sigma_S, sigma_B=sigma_B)


    def _stat_calculate(self):
        
        return np.array(list(map(self.stat_func, 
                                 self.data.src_counts_f64, 
                                 self.data.bkg_counts_f64, 
                                 self.model.conv_ctsrate_f64, 
                                 self.data.corr_src_efficiency_f64, 
                                 self.data.corr_bkg_efficiency_f64, 
                                 self.data.src_errors_f64, 
                                 self.data.bkg_errors_f64, 
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
