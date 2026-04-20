"""Binding between a :class:`Data` and a :class:`Model` for fit evaluation.

Holds the bound model–data pair, caches the forward convolution through
the detector response, exposes both observed and convolved count
rates/densities, and aggregates the per-unit likelihood statistic that
the inference layer minimises.
"""

import numpy as np

from .statistic import StatisticNB
from ..data.data import Data
from ..model.model import Model
from ..util.tools import cached_property, clear_cached_property



class Pair(object):
    """Binds one ``Data`` and one ``Model`` and evaluates their joint statistic.

    The class dispatches on each unit's statistic tag (``pgstat``,
    ``pstat``, ``cstat``, ``chi2``, etc.) via :attr:`_allowed_stats` and
    publishes convolved spectra, residuals, and the total log-likelihood.

    Most list-valued properties (``conv_ctsrate``, ``phtspec_at_rsp``,
    ``cts_to_pht``, ``deconv_*``, …) mirror the same-named properties on
    :class:`~bayspec.model.model.Model` but are computed from the paired
    ``data`` rather than ``model.fit_to``.

    Attributes:
        data: The bound :class:`~bayspec.data.data.Data`.
        model: The bound :class:`~bayspec.model.model.Model`.
    """

    _allowed_stats = {'gstat': StatisticNB.Gstat,
                      'chi2': StatisticNB.Gstat,
                      'pstat': StatisticNB.Pstat,
                      'ppstat': StatisticNB.PPstat,
                      'cstat': StatisticNB.PPstat,
                      'pgstat': StatisticNB.PGstat,
                      'ULppstat': StatisticNB.PPstat_UL,
                      'ULpgstat': StatisticNB.PGstat_UL}

    def __init__(self, data, model):
        """Pair ``data`` with ``model`` and wire up the ``fit_with`` reference.

        Args:
            data: ``Data`` container.
            model: ``Model`` instance.

        Raises:
            ValueError: If either argument is not of the expected type.
        """

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
        
        flat_phtflux = self.model.integ(self.data.egrid, self.data.tgrid, self.data.ngrid)
        phtflux = [flat_phtflux[i:j] for (i, j) in zip(self.data.bin_start, self.data.bin_stop)]
        ctsrate = [np.dot(pf, drm) for (pf, drm) in zip(phtflux, self.data.corr_rsp_drm)]
        
        return ctsrate
    
    
    def _convolve_f64(self):
        
        flat_phtflux = self.model.integ(self.data.egrid, self.data.tgrid, self.data.ngrid)
        phtflux = [flat_phtflux[i:j] for (i, j) in zip(self.data.bin_start, self.data.bin_stop)]
        ctsrate = [np.dot(pf, drm).astype(np.float64) for (pf, drm) in zip(phtflux, self.data.corr_rsp_drm)]
        
        return ctsrate
    
    
    def _re_convolve(self):
        
        flat_phtflux = self.model.integ(self.data.egrid, self.data.tgrid, self.data.ngrid)
        phtflux = [flat_phtflux[i:j] for (i, j) in zip(self.data.bin_start, self.data.bin_stop)]
        re_ctsrate = [np.dot(pf, drm) for (pf, drm) in zip(phtflux, self.data.corr_rsp_re_drm)]
        
        return re_ctsrate


    @property
    def conv_ctsrate(self):
        """Per-unit convolved count rate for the paired model.

        Every ``conv_*``, ``*_at_rsp``, ``cts_to_*``, ``deconv_*`` family
        on this class mirrors the same-named :class:`Model` property but
        uses the paired ``data`` so the values are consistent with the
        current ``Pair``.
        """

        return self._convolve()


    @property
    def conv_ctsrate_f64(self):
        """Same as :attr:`conv_ctsrate` but cast to ``float64`` for statistics."""

        return self._convolve_f64()


    @property
    def conv_re_ctsrate(self):
        """Convolved count rate on the re-binned detector response."""

        return self._re_convolve()


    @property
    def conv_ctsspec(self):
        """Convolved count density: :attr:`conv_ctsrate` divided by bin width."""

        return [cr / chw for (cr, chw) in zip(self.conv_ctsrate, self.data.rsp_chbin_width)]


    @property
    def conv_re_ctsspec(self):
        """Convolved count density on the re-binned response."""

        return [cr / chw for (cr, chw) in zip(self.conv_re_ctsrate, self.data.rsp_re_chbin_width)]


    @property
    def phtspec_at_rsp(self):
        """Photon spectrum sampled at the response channel midpoints."""

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
        """Per-unit ratio of integrated energy flux to observed net count rate."""

        ctsrate = [np.sum(cr) for cr in self.data.net_ctsrate]
        ergflux = [np.sum([self.model.ergflux(emin, emax, 1000, time) for emin, emax in notc])
                   for notc, time in zip(self.data.notcs, self.data.times)]

        return [flux / rate for (flux, rate) in zip(ergflux, ctsrate)]


    @property
    def conv_rate_to_flux(self):
        """Like :attr:`rate_to_flux` but using the convolved-model count rate."""

        ctsrate = [np.sum(cr) for cr in self.conv_ctsrate]
        ergflux = [np.sum([self.model.ergflux(emin, emax, 1000, time) for emin, emax in notc])
                   for notc, time in zip(self.data.notcs, self.data.times)]

        return [flux / rate for (flux, rate) in zip(ergflux, ctsrate)]


    def rate_to_fluxdensity(self, E=1, T=None, unit='fv'):
        """Per-unit conversion from net count rate to flux density at ``(E, T)``.

        Args:
            E: Energy at which the flux density is evaluated.
            T: Optional time.
            unit: ``'NE'`` (photon flux density), ``'Fv'`` (energy flux
                density), or ``'Jy'``.

        Returns:
            A list of per-unit conversion factors.

        Raises:
            ValueError: If ``unit`` is not recognized.
        """

        ctsrate = [np.sum(cr) for cr in self.data.net_ctsrate]
        if unit == 'NE':    # photons cm-2 s-1 keV-1
            fluxdensity = self.model.phtspec(E, T)
        elif unit == 'Fv':  # erg cm-2 s-1 keV-1
            fluxdensity = self.model.flxspec(E, T)
        elif unit == 'Jy':  # Jansky
            fluxdensity = self.model.flxspec(E, T) * 1e6 / 2.416
        else:
            raise ValueError(f'unsupported value of unit: {unit}')

        return [fluxdensity / rate for rate in ctsrate]


    def conv_rate_to_fluxdensity(self, E=1, T=None, unit='fv'):
        """Like :meth:`rate_to_fluxdensity` but using convolved count rates."""

        ctsrate = [np.sum(cr) for cr in self.conv_ctsrate]
        if unit == 'NE':    # photons cm-2 s-1 keV-1
            fluxdensity = self.model.phtspec(E, T)
        elif unit == 'Fv':  # erg cm-2 s-1 keV-1
            fluxdensity = self.model.flxspec(E, T)
        elif unit == 'Jy':  # Jansky
            fluxdensity = self.model.flxspec(E, T) * 1e6 / 2.416
        else:
            raise ValueError(f'unsupported value of unit: {unit}')
            
        return [fluxdensity / rate for rate in ctsrate]


    @property
    def deconv_phtspec(self):
        """Deconvolved photon spectrum: net counts × paired-model ``cts_to_pht``.

        Sibling ``deconv_*`` properties (``deconv_re_phtspec``,
        ``deconv_phtspec_error``, ``deconv_flxspec``, ``deconv_ergspec``
        and their ``_re_``/``_error`` variants) follow the same pattern
        for photon, flux, and energy spectra on the full or re-binned
        channel grids.
        """

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
        """Per-unit sigma residuals ``(observed - model) / error`` on the full grid."""

        return list(map(lambda oi, mi, si: (oi - mi) / si,
                        self.data.net_ctsrate,
                        self.conv_ctsrate,
                        self.data.net_ctsrate_error))


    @property
    def re_residual(self):
        """Per-unit sigma residuals on the re-binned grid."""

        return list(map(lambda oi, mi, si: (oi - mi) / si,
                        self.data.net_re_ctsrate,
                        self.conv_re_ctsrate,
                        self.data.net_re_ctsrate_error))


    @cached_property()
    def stat_func(self):
        """Closure returning the per-unit statistic; ``+inf`` when the model is non-finite."""

        return lambda S, B, m, ts, tb, sigma_S, sigma_B, stat: \
            np.inf if np.isnan(m).any() or np.isinf(m).any() else \
                self._allowed_stats[stat](S=S, B=B, m=m, ts=ts, tb=tb, \
                    sigma_S=sigma_S, sigma_B=sigma_B)[0]


    @cached_property()
    def pseudo_residual_func(self):
        """Closure returning the per-bin pseudo-residual from the chosen statistic."""

        return lambda S, B, m, ts, tb, sigma_S, sigma_B, stat: \
            np.ones_like(m) * np.inf if np.isnan(m).any() or np.isinf(m).any() else \
                self._allowed_stats[stat](S=S, B=B, m=m, ts=ts, tb=tb, \
                    sigma_S=sigma_S, sigma_B=sigma_B)[1]


    def _stat_calculate(self):
        
        return list(map(self.stat_func, 
                        self.data.src_counts_f64, 
                        self.data.bkg_counts_f64, 
                        self.model.conv_ctsrate_f64, 
                        self.data.corr_src_efficiency_f64, 
                        self.data.corr_bkg_efficiency_f64, 
                        self.data.src_errors_f64, 
                        self.data.bkg_errors_f64, 
                        self.data.stats))
        
        
    def _pseudo_residual_calculate(self):
        
        return list(map(self.pseudo_residual_func, 
                        self.data.src_counts_f64, 
                        self.data.bkg_counts_f64, 
                        self.model.conv_ctsrate_f64, 
                        self.data.corr_src_efficiency_f64, 
                        self.data.corr_bkg_efficiency_f64, 
                        self.data.src_errors_f64, 
                        self.data.bkg_errors_f64, 
                        self.data.stats))


    @property
    def stat_list(self):
        """Array of per-unit statistic values."""

        return np.array(self._stat_calculate())


    @property
    def pseudo_residual_list(self):
        """List of per-unit pseudo-residual arrays."""

        return self._pseudo_residual_calculate()


    @property
    def weight_list(self):
        """Per-unit likelihood weights taken from the paired ``Data``."""

        return self.data.weights


    @property
    def stat(self):
        """Total weighted statistic summed across all units."""

        return np.sum(self.stat_list * self.weight_list)


    @property
    def pseudo_residual(self):
        """Weight-scaled pseudo-residual vector concatenated across all units."""

        return np.concatenate([rd * np.sqrt(wt) for rd, wt in
                               zip(self.pseudo_residual_list, self.weight_list)])


    @property
    def loglike_list(self):
        """Per-unit log-likelihood, derived as ``-0.5 * stat_list``."""

        return -0.5 * self.stat_list


    @property
    def loglike(self):
        """Total log-likelihood, derived as ``-0.5 * stat``."""

        return -0.5 * self.stat


    @property
    def npoint_list(self):
        """Per-unit number of fitted data points."""

        return self.data.npoints


    @property
    def npoint(self):
        """Total number of fitted data points across all units."""

        return np.sum(self.npoint_list)
