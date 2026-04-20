"""Spectral model base classes and the composition algebra.

``Model`` defines the functional contract (``func(E, T, O)``), parameter
and config dictionaries, convolution with detector responses, and
posterior-aware evaluators. ``Additive``/``Multiplicative``/
``Mathematic`` are thin base classes that fix the ``type`` tag, and
``CompositeModel`` implements the arithmetic (+, -, *, /) plus the
convolution call operator (``conv(add)`` etc.) used to build
multi-component expressions.
"""

import os
import warnings
import numpy as np
from collections import OrderedDict

from ..util.info import Info
from ..util.prior import unif
from ..util.param import Par, Cfg
from ..util.tools import SuperDict
from ..util.tools import cached_property, json_dump, trapz_1d, trapz_2d



class Model(object):
    """Base class for a spectral model component.

    Subclasses override :meth:`func` with the model's analytical form and
    populate ``config`` (``Cfg`` entries — frozen values like redshift)
    and ``params`` (``Par`` entries — fittable quantities). The ``type``
    tag — ``'add'``, ``'mul'``, ``'conv'`` or ``'math'`` — drives which
    derived spectra (``phtspec``, ``flxspec``, ``ergspec``, ``nouspec``)
    are available and how composition is type-checked.

    Attributes:
        expr: Short expression used in tables and composite names.
        type: One of ``'add'``, ``'mul'``, ``'conv'``, ``'math'``.
        comment: Free-form description.
        config: ``OrderedDict`` of configuration ``Cfg`` entries.
        params: ``OrderedDict`` of fit ``Par`` entries.
    """

    _allowed_types = ('add', 'mul', 'conv', 'math')

    def __init__(self):
        """Initialise the default dummy component.

        Real subclasses override this to set ``expr``/``type``/``comment``
        and populate ``config``/``params`` with their own entries.
        """

        self.expr = 'model'
        self.type = 'add'
        self.comment = 'model base class'

        self.config = OrderedDict()
        self.config['cfg'] = Cfg(0)

        self.params = OrderedDict()
        self.params['par'] = Par(1, unif(0, 2))


    def func(self, E, T=None, O=None):
        """Evaluate the model at energies ``E`` and optional time ``T``.

        Additive models return a photon flux density in
        photons/cm²/s/keV; multiplicative and mathematical models return
        a dimensionless factor. ``O`` is a nested model argument used by
        convolution operators.

        Args:
            E: Scalar or array of energies in keV.
            T: Optional time or time array.
            O: Optional nested model passed down by a convolution.

        Returns:
            The model value at the given sampling; subclass-specific.
        """

        pass
        
        
    @property
    def mdicts(self):
        """Mapping from ``expr`` to component model; overridden by composites."""

        return OrderedDict([(self.expr, self)])


    @property
    def fdicts(self):
        """Mapping from ``expr`` to each component's ``func`` callable."""

        return OrderedDict([(ex, mo.func) for ex, mo in self.mdicts.items()])


    @property
    def cdicts(self):
        """Mapping from ``expr`` to each component's ``config`` dict."""

        return OrderedDict([(ex, mo.config) for ex, mo in self.mdicts.items()])


    @property
    def pdicts(self):
        """Mapping from ``expr`` to each component's ``params`` dict."""

        return OrderedDict([(ex, mo.params) for ex, mo in self.mdicts.items()])


    @property
    def cfg(self):
        """Flat :class:`SuperDict` of every config parameter across components."""

        cid = 0
        cfg = SuperDict()

        for config in self.cdicts.values():
            for cg in config.values():
                cid += 1
                cfg[str(cid)] = cg

        return cfg


    @property
    def par(self):
        """Flat :class:`SuperDict` of every fit parameter across components."""

        pid = 0
        par = SuperDict()

        for params in self.pdicts.values():
            for pr in params.values():
                pid += 1
                par[str(pid)] = pr

        return par


    @property
    def pvalues(self):
        """Tuple of current parameter values, preserving ``par`` order."""

        return tuple([pr.value for pr in self.par.values()])


    @property
    def all_config(self):
        """List of per-config rows with component, label, and value."""

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
        """List of per-parameter rows with value, prior, posterior, and frozen flag."""

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
        """Tabular :class:`Info` view of every configuration parameter."""

        all_config = self.all_config.copy()

        return Info.from_list_dict(all_config)


    @property
    def par_info(self):
        """Tabular :class:`Info` view of parameters with frozen ones tagged."""

        all_params = self.all_params.copy()

        for par in all_params:
            if par['Frozen']:
                par['Prior'] = 'frozen'

        all_params = Info.list_dict_to_dict(all_params)

        del all_params['Posterior']
        del all_params['Frozen']

        return Info.from_dict(all_params)


    def save(self, savepath):
        """Dump the config and parameter tables under ``savepath``.

        Args:
            savepath: Directory path. Created if missing.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.cfg_info.data_list_dict, savepath + '/model_cfg.json')
        json_dump(self.par_info.data_list_dict, savepath + '/model_par.json')


    @property
    def fit_to(self):
        """Return the ``Data`` this model is bound to, or raise if unset.

        Raises:
            AttributeError: If no data has been assigned.
        """

        try:
            return self._fit_to
        except AttributeError:
            raise AttributeError('no data fit to')


    @fit_to.setter
    def fit_to(self, new_data):
        """Bind a ``Data`` to this model and keep the back-reference in sync.

        Raises:
            ValueError: If ``new_data`` is not a ``Data``.
        """

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


    def integ(self, egrid, tgrid, ngrid):
        """Trapezoidal-integrate the model over each ``ngrid``-point bin.

        Args:
            egrid: Flattened energy grid, shaped ``(nbin * ngrid,)``.
            tgrid: Matching flattened time grid.
            ngrid: Sub-grid size per bin used to reshape ``egrid``.

        Returns:
            Integrated photon flux per bin.

        Raises:
            TypeError: If the model type is not ``'add'``.
        """

        fgrid = self.func(egrid, tgrid)

        if self.type == 'add':
            return trapz_2d(fgrid.reshape(-1, ngrid), egrid.reshape(-1, ngrid))

        else:
            msg = f'integ is invalid for {self.type} type model'
            raise TypeError(msg)


    def convolve_response(self, response, time=None):
        """Convolve the model with a single ``Response`` and return count-rate/spec.

        Args:
            response: A :class:`~bayspec.data.response.Response`.
            time: Optional time broadcast across photon bins.

        Returns:
            A ``(ctsrate, ctsspec)`` tuple.
        """

        ebin = response.phbin
        tarr = np.repeat(time, ebin.shape[0])
        
        ngrid = 5
        scale = np.linspace(0.0, 1.0, ngrid, dtype=float)
        
        egrid = ebin[:, [0]] + (ebin[:, [1]] - ebin[:, [0]]) * scale
        np.maximum(egrid, 1e-10, out=egrid)
        egrid = egrid.ravel()
        
        tgrid = np.repeat(tarr[:, None], ngrid, axis=1).ravel()
        
        phtflux = self.integ(egrid, tgrid, ngrid)
        ctsrate = np.dot(phtflux, response.drm)
        ctsspec = ctsrate / response.chbin_width
        
        return ctsrate, ctsspec
        
        
    def convolve_dataunit(self, dataunit):
        """Convolve the model with a single ``DataUnit``.

        Args:
            dataunit: A :class:`~bayspec.data.data.DataUnit`.

        Returns:
            A ``(ctsrate, ctsspec)`` tuple.
        """

        phtflux = self.integ(dataunit.egrid, dataunit.tgrid, dataunit.ngrid)
        ctsrate = np.dot(phtflux, dataunit.corr_rsp_drm)
        ctsspec = ctsrate / dataunit.rsp_chbin_width

        return ctsrate, ctsspec


    def convolve_data(self, data):
        """Convolve the model with every unit of a ``Data`` container.

        Args:
            data: A :class:`~bayspec.data.data.Data`.

        Returns:
            A ``(ctsrate_list, ctsspec_list)`` tuple, one entry per unit.
        """

        flat_phtflux = self.integ(data.egrid, data.tgrid, data.ngrid)
        phtflux = [flat_phtflux[i:j] for (i, j) in zip(data.bin_start, data.bin_stop)]
        ctsrate = [np.dot(pf, drm) for (pf, drm) in zip(phtflux, data.corr_rsp_drm)]
        ctsspec = [cr / chw for (cr, chw) in zip(ctsrate, data.rsp_chbin_width)]

        return ctsrate, ctsspec
        
        
    def _convolve(self):
        
        flat_phtflux = self.integ(self.fit_to.egrid, self.fit_to.tgrid, self.fit_to.ngrid)
        phtflux = [flat_phtflux[i:j] for (i, j) in zip(self.fit_to.bin_start, self.fit_to.bin_stop)]
        ctsrate = [np.dot(pf, drm) for (pf, drm) in zip(phtflux, self.fit_to.corr_rsp_drm)]
        
        return ctsrate
    
    
    def _convolve_f64(self):
        
        flat_phtflux = self.integ(self.fit_to.egrid, self.fit_to.tgrid, self.fit_to.ngrid)
        phtflux = [flat_phtflux[i:j] for (i, j) in zip(self.fit_to.bin_start, self.fit_to.bin_stop)]
        ctsrate = [np.dot(pf, drm).astype(np.float64) for (pf, drm) in zip(phtflux, self.fit_to.corr_rsp_drm)]
        
        return ctsrate
    
    
    def _re_convolve(self):
        
        flat_phtflux = self.integ(self.fit_to.egrid, self.fit_to.tgrid, self.fit_to.ngrid)
        phtflux = [flat_phtflux[i:j] for (i, j) in zip(self.fit_to.bin_start, self.fit_to.bin_stop)]
        re_ctsrate = [np.dot(pf, drm) for (pf, drm) in zip(phtflux, self.fit_to.corr_rsp_re_drm)]
        
        return re_ctsrate


    @property
    def conv_ctsrate(self):
        """Convolved count rate per unit against the bound ``fit_to`` data."""

        return self._convolve()


    @property
    def conv_ctsrate_f64(self):
        """Same as :attr:`conv_ctsrate` but cast to ``float64`` for statistics."""

        return self._convolve_f64()


    @property
    def conv_re_ctsrate(self):
        """Convolved count rate using the re-binned detector response."""

        return self._re_convolve()


    @property
    def conv_ctsspec(self):
        """Convolved count density: :attr:`conv_ctsrate` divided by bin width."""

        return [cr / chw for (cr, chw) in zip(self.conv_ctsrate, self.fit_to.rsp_chbin_width)]


    @property
    def conv_re_ctsspec(self):
        """Convolved count density on the re-binned response."""

        return [cr / chw for (cr, chw) in zip(self.conv_re_ctsrate, self.fit_to.rsp_re_chbin_width)]
    
    
    @property
    def phtspec_at_rsp(self):
        """Photon spectrum sampled at the response channel midpoints.

        Sibling properties ``re_phtspec_at_rsp``, ``flxspec_at_rsp``,
        ``re_flxspec_at_rsp``, ``ergspec_at_rsp``, ``re_ergspec_at_rsp``
        follow the same pattern, using either the full or re-binned
        channel grid.
        """

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
        """Per-bin conversion factor from convolved counts to photon density.

        ``re_cts_to_pht``, ``cts_to_flx``, ``re_cts_to_flx``, ``cts_to_erg``
        and ``re_cts_to_erg`` follow the same pattern for the photon/flux/
        energy spectra on the full or re-binned grids.
        """

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

        ctsrate = [np.sum(cr) for cr in self.fit_to.net_ctsrate]
        ergflux = [np.sum([self.ergflux(emin, emax, 1000, time) for emin, emax in notc])
                   for notc, time in zip(self.fit_to.notcs, self.fit_to.times)]

        return [flux / rate for (flux, rate) in zip(ergflux, ctsrate)]


    @property
    def conv_rate_to_flux(self):
        """Like :attr:`rate_to_flux` but using the convolved-model count rate."""

        ctsrate = [np.sum(cr) for cr in self.conv_ctsrate]
        ergflux = [np.sum([self.ergflux(emin, emax, 1000, time) for emin, emax in notc])
                   for notc, time in zip(self.fit_to.notcs, self.fit_to.times)]

        return [flux / rate for (flux, rate) in zip(ergflux, ctsrate)]


    def rate_to_fluxdensity(self, E=1, T=None, unit='Fv'):
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

        ctsrate = [np.sum(cr) for cr in self.fit_to.net_ctsrate]
        if unit == 'NE':    # photons cm-2 s-1 keV-1
            fluxdensity = self.phtspec(E, T)
        elif unit == 'Fv':  # erg cm-2 s-1 keV-1
            fluxdensity = self.flxspec(E, T)
        elif unit == 'Jy':  # Jansky
            fluxdensity = self.flxspec(E, T) * 1e6 / 2.416
        else:
            raise ValueError(f'unsupported value of unit: {unit}')

        return [fluxdensity / rate for rate in ctsrate]


    def conv_rate_to_fluxdensity(self, E=1, T=None, unit='Fv'):
        """Like :meth:`rate_to_fluxdensity` but using convolved count rates."""

        ctsrate = [np.sum(cr) for cr in self.conv_ctsrate]
        if unit == 'NE':    # photons cm-2 s-1 keV-1
            fluxdensity = self.phtspec(E, T)
        elif unit == 'Fv':  # erg cm-2 s-1 keV-1
            fluxdensity = self.flxspec(E, T)
        elif unit == 'Jy':  # Jansky
            fluxdensity = self.flxspec(E, T) * 1e6 / 2.416
        else:
            raise ValueError(f'unsupported value of unit: {unit}')
            
        return [fluxdensity / rate for rate in ctsrate]


    def phtspec(self, E, T=None):
        """Photon flux density :math:`N(E)` in photons/cm²/s/keV.

        Raises:
            TypeError: If the model type is not ``'add'``.
        """
        # NE in units of photons cm-2 s-1 keV-1

        if self.type not in ['add']:
            msg = f'phtspec is invalid for {self.type} type model'
            raise TypeError(msg)
        else:
            return self.func(E, T)


    def nouspec(self, E, T=None):
        """Dimensionless factor applied by multiplicative/mathematical models.

        Raises:
            TypeError: If the model type is not ``'mul'`` or ``'math'``.
        """
        # dimensionless

        if self.type not in ['mul', 'math']:
            msg = f'nouspec is invalid for {self.type} type model'
            raise TypeError(msg)
        else:
            return self.func(E, T)


    def flxspec(self, E, T=None):
        """Energy flux density :math:`F_\\nu = E \\, N(E)` in erg/cm²/s/keV.

        Raises:
            TypeError: If the model type is not ``'add'``.
        """
        # Fv in units of erg cm-2 s-1 keV-1

        if self.type not in ['add']:
            msg = f'flxspec is invalid for {self.type} type model'
            raise TypeError(msg)
        else:
            return 1.60218e-9 * E * self.phtspec(E, T)


    def ergspec(self, E, T=None):
        """Energy spectrum :math:`\\nu F_\\nu = E^2 \\, N(E)` in erg/cm²/s.

        Raises:
            TypeError: If the model type is not ``'add'``.
        """
        # vFv in units of erg cm-2 s-1

        if self.type not in ['add']:
            msg = f'ergspec is invalid for {self.type} type model'
            raise TypeError(msg)
        else:
            return 1.60218e-9 * E * E * self.phtspec(E, T)


    def phtflux(self, emin, emax, ngrid, time=None):
        """Integrated photon flux over ``[emin, emax]`` in photons/cm²/s.

        Args:
            emin: Lower energy bound in keV.
            emax: Upper energy bound in keV.
            ngrid: Number of log-spaced grid points.
            time: Optional time broadcast across the grid.

        Raises:
            TypeError: If the model type is not ``'add'``.
        """
        # integ(NE, E) in units of photons cm-2 s-1

        if self.type not in ['add']:
            msg = f'phtflux is invalid for {self.type} type model'
            raise TypeError(msg)
        else:
            egrid = np.logspace(np.log10(emin), np.log10(emax), ngrid)
            tgrid = np.repeat(time, ngrid)
            return trapz_1d(self.phtspec(egrid, tgrid), egrid)


    def ergflux(self, emin, emax, ngrid, time=None):
        """Integrated energy flux over ``[emin, emax]`` in erg/cm²/s.

        Raises:
            TypeError: If the model type is not ``'add'``.
        """
        # integ(Fv, E) in units of erg cm-2 s-1

        if self.type not in ['add']:
            msg = f'ergflux is invalid for {self.type} type model'
            raise TypeError(msg)
        else:
            egrid = np.logspace(np.log10(emin), np.log10(emax), ngrid)
            tgrid = np.repeat(time, ngrid)
            return trapz_1d(self.flxspec(egrid, tgrid), egrid)


    def at_par(self, theta):
        """Write every free parameter value from the 1-indexed sequence ``theta``."""

        for i, thi in enumerate(theta):
            self.par[i+1].val = thi


    @property
    def par_mean(self):
        """Per-parameter posterior mean (or frozen value when applicable)."""

        return [par.val if par.frozen else par.post.mean for par in self.par.values()]


    @property
    def par_median(self):
        """Per-parameter posterior median (or frozen value)."""

        return [par.val if par.frozen else par.post.median for par in self.par.values()]


    @property
    def par_best(self):
        """Per-parameter posterior best-fit sample (or frozen value)."""

        return [par.val if par.frozen else par.post.best for par in self.par.values()]


    @property
    def par_best_ci(self):
        """Per-parameter best-fit confidence bounds (or frozen value)."""

        return [par.val if par.frozen else par.post.best_ci for par in self.par.values()]


    @property
    def par_truth(self):
        """Per-parameter truth value (or frozen value); may contain ``None``."""

        return [par.val if par.frozen else par.post.truth for par in self.par.values()]


    def mean_phtspec(self, E, T=None):
        """Photon spectrum evaluated at the posterior mean parameter vector.

        The family ``{mean,median,best,best_ci,truth}_{pht,nou,flx,erg}spec``
        and ``_{pht,erg}flux`` all follow the same pattern: set ``par``
        from the named posterior summary, then evaluate the spectrum or
        flux. ``truth_*`` variants raise ``ValueError`` when any parameter
        lacks a truth value.
        """

        self.at_par(self.par_mean)

        return self.phtspec(E, T)
    
    
    def median_phtspec(self, E, T=None):
        
        self.at_par(self.par_median)
        
        return self.phtspec(E, T)


    def best_phtspec(self, E, T=None):
        
        self.at_par(self.par_best)
        
        return self.phtspec(E, T)


    def best_ci_phtspec(self, E, T=None):
        
        self.at_par(self.par_best_ci)
        
        return self.phtspec(E, T)
    
    
    def truth_phtspec(self, E, T=None):
        
        if None in self.par_truth:
            raise ValueError('no truth value for some parameters')
        
        self.at_par(self.par_truth)
        
        return self.phtspec(E, T)
    
    
    def mean_nouspec(self, E, T=None):
        
        self.at_par(self.par_mean)
        
        return self.nouspec(E, T)
    
    
    def median_nouspec(self, E, T=None):
        
        self.at_par(self.par_median)
        
        return self.nouspec(E, T)
    
    
    def best_nouspec(self, E, T=None):
        
        self.at_par(self.par_best)
        
        return self.nouspec(E, T)
    
    
    def best_ci_nouspec(self, E, T=None):
        
        self.at_par(self.par_best_ci)
        
        return self.nouspec(E, T)
    
    
    def truth_nouspec(self, E, T=None):
        
        if None in self.par_truth:
            raise ValueError('no truth value for some parameters')
        
        self.at_par(self.par_truth)
        
        return self.nouspec(E, T)
    
    
    def mean_flxspec(self, E, T=None):
        
        self.at_par(self.par_mean)
        
        return self.flxspec(E, T)
    
    
    def median_flxspec(self, E, T=None):
        
        self.at_par(self.par_median)
        
        return self.flxspec(E, T)
    
    
    def best_flxspec(self, E, T=None):
        
        self.at_par(self.par_best)
        
        return self.flxspec(E, T)
    
    
    def best_ci_flxspec(self, E, T=None):
        
        self.at_par(self.par_best_ci)
        
        return self.flxspec(E, T)
    
    
    def truth_flxspec(self, E, T=None):
        
        if None in self.par_truth:
            raise ValueError('no truth value for some parameters')
        
        self.at_par(self.par_truth)
        
        return self.flxspec(E, T)
    
    
    def mean_ergspec(self, E, T=None):
        
        self.at_par(self.par_mean)
        
        return self.ergspec(E, T)
    
    
    def median_ergspec(self, E, T=None):
        
        self.at_par(self.par_median)
        
        return self.ergspec(E, T)
    
    
    def best_ergspec(self, E, T=None):
        
        self.at_par(self.par_best)
        
        return self.ergspec(E, T)
    
    
    def best_ci_ergspec(self, E, T=None):
        
        self.at_par(self.par_best_ci)
        
        return self.ergspec(E, T)
    
    
    def truth_ergspec(self, E, T=None):
        
        if None in self.par_truth:
            raise ValueError('no truth value for some parameters')
        
        self.at_par(self.par_truth)
        
        return self.ergspec(E, T)

    
    def mean_phtflux(self, emin, emax, ngrid, time=None):
        
        self.at_par(self.par_mean)
        
        return self.phtflux(emin, emax, ngrid, time)
    
    
    def median_phtflux(self, emin, emax, ngrid, time=None):
        
        self.at_par(self.par_median)
        
        return self.phtflux(emin, emax, ngrid, time)

    
    def best_phtflux(self, emin, emax, ngrid, time=None):
        
        self.at_par(self.par_best)
        
        return self.phtflux(emin, emax, ngrid, time)
    
    
    def best_ci_phtflux(self, emin, emax, ngrid, time=None):
        
        self.at_par(self.par_best_ci)
        
        return self.phtflux(emin, emax, ngrid, time)
    
    
    def truth_phtflux(self, emin, emax, ngrid, time=None):
        
        if None in self.par_truth:
            raise ValueError('no truth value for some parameters')
        
        self.at_par(self.par_truth)
        
        return self.phtflux(emin, emax, ngrid, time)
    
    
    def mean_ergflux(self, emin, emax, ngrid, time=None):
        
        self.at_par(self.par_mean)
        
        return self.ergflux(emin, emax, ngrid, time)
    
    
    def median_ergflux(self, emin, emax, ngrid, time=None):
        
        self.at_par(self.par_median)
        
        return self.ergflux(emin, emax, ngrid, time)
    
    
    def best_ergflux(self, emin, emax, ngrid, time=None):
        
        self.at_par(self.par_best)
        
        return self.ergflux(emin, emax, ngrid, time)
    
    
    def best_ci_ergflux(self, emin, emax, ngrid, time=None):
        
        self.at_par(self.par_best_ci)
        
        return self.ergflux(emin, emax, ngrid, time)
    
    
    def truth_ergflux(self, emin, emax, ngrid, time=None):
        
        if None in self.par_truth:
            raise ValueError('no truth value for some parameters')
        
        self.at_par(self.par_truth)
        
        return self.ergflux(emin, emax, ngrid, time)
    
    
    def mean_phtflux_ratio(self, erange1, erange2, ngrid, time=None):
        
        self.at_par(self.par_mean)

        emin1, emax1 = erange1

        emin2, emax2 = erange2
        
        return self.phtflux(emin1, emax1, ngrid, time) / self.phtflux(emin2, emax2, ngrid, time)
    
    
    def median_phtflux_ratio(self, erange1, erange2, ngrid, time=None):
        
        self.at_par(self.par_median)

        emin1, emax1 = erange1

        emin2, emax2 = erange2
        
        return self.phtflux(emin1, emax1, ngrid, time) / self.phtflux(emin2, emax2, ngrid, time)
    
    
    def best_phtflux_ratio(self, erange1, erange2, ngrid, time=None):
        
        self.at_par(self.par_best)

        emin1, emax1 = erange1

        emin2, emax2 = erange2
        
        return self.phtflux(emin1, emax1, ngrid, time) / self.phtflux(emin2, emax2, ngrid, time)
    
    
    def best_ci_phtflux_ratio(self, erange1, erange2, ngrid, time=None):
        
        self.at_par(self.par_best_ci)

        emin1, emax1 = erange1

        emin2, emax2 = erange2
        
        return self.phtflux(emin1, emax1, ngrid, time) / self.phtflux(emin2, emax2, ngrid, time)
    

    def truth_phtflux_ratio(self, erange1, erange2, ngrid, time=None):
        
        if None in self.par_truth:
            raise ValueError('no truth value for some parameters')
        
        self.at_par(self.par_truth)

        emin1, emax1 = erange1

        emin2, emax2 = erange2
        
        return self.phtflux(emin1, emax1, ngrid, time) / self.phtflux(emin2, emax2, ngrid, time)
    
    
    def mean_ergflux_ratio(self, erange1, erange2, ngrid, time=None):
        
        self.at_par(self.par_mean)

        emin1, emax1 = erange1

        emin2, emax2 = erange2
        
        return self.ergflux(emin1, emax1, ngrid, time) / self.ergflux(emin2, emax2, ngrid, time)
    
    
    def median_ergflux_ratio(self, erange1, erange2, ngrid, time=None):
        
        self.at_par(self.par_median)

        emin1, emax1 = erange1

        emin2, emax2 = erange2
        
        return self.ergflux(emin1, emax1, ngrid, time) / self.ergflux(emin2, emax2, ngrid, time)
    

    def best_ergflux_ratio(self, erange1, erange2, ngrid, time=None):
        
        self.at_par(self.par_best)

        emin1, emax1 = erange1

        emin2, emax2 = erange2
        
        return self.ergflux(emin1, emax1, ngrid, time) / self.ergflux(emin2, emax2, ngrid, time)
    
    
    def best_ci_ergflux_ratio(self, erange1, erange2, ngrid, time=None):
        
        self.at_par(self.par_best_ci)

        emin1, emax1 = erange1

        emin2, emax2 = erange2
        
        return self.ergflux(emin1, emax1, ngrid, time) / self.ergflux(emin2, emax2, ngrid, time)
    
    
    def truth_ergflux_ratio(self, erange1, erange2, ngrid, time=None):
        
        if None in self.par_truth:
            raise ValueError('no truth value for some parameters')
        
        self.at_par(self.par_truth)

        emin1, emax1 = erange1

        emin2, emax2 = erange2
        
        return self.ergflux(emin1, emax1, ngrid, time) / self.ergflux(emin2, emax2, ngrid, time)


    @property
    def posterior_nsample(self):
        """Number of posterior draws; equals ``1`` when every parameter is frozen."""

        nsample = max([1 if par.frozen else par.post.nsample \
            for par in self.par.values()])

        return nsample


    @property
    def posterior_sample(self):
        """``(nsample, npar)`` matrix of posterior draws.

        Frozen parameters are filled with their fixed value so the matrix
        is rectangular.
        """

        sample = np.vstack([np.full(self.posterior_nsample, par.val) \
            if par.frozen else par.post.sample.copy() \
                for par in self.par.values()]).T

        return sample


    def sample_statistic(self, sample):
        """Summarize a draw matrix with mean, median, and 1/2/3-sigma intervals.

        Args:
            sample: ``(nsample, ...)`` array of draws.

        Returns:
            Dict with keys ``mean``, ``median``, ``Isigma``, ``IIsigma``,
            ``IIIsigma``.
        """

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
        """Summary statistics of the posterior parameter matrix."""

        return self.sample_statistic(self.posterior_sample)


    def phtspec_sample(self, E, T=None):
        """Return posterior summaries of the photon spectrum at ``(E, T)``.

        Sibling methods ``nouspec_sample``, ``flxspec_sample``,
        ``ergspec_sample``, ``phtflux_sample``, ``ergflux_sample``,
        ``phtflux_ratio_sample``, and ``ergflux_ratio_sample`` follow the
        same pattern on their respective quantities.
        """

        scalar = np.asarray(E).ndim == 0

        if scalar: sample = np.zeros(self.posterior_nsample, dtype=float)
        else: sample = np.zeros([self.posterior_nsample, len(E)], dtype=float)

        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.phtspec(E, T)

        return self.sample_statistic(sample)
    
    
    def nouspec_sample(self, E, T=None):
        
        scalar = np.asarray(E).ndim == 0
        
        if scalar: sample = np.zeros(self.posterior_nsample, dtype=float)
        else: sample = np.zeros([self.posterior_nsample, len(E)], dtype=float)
        
        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.nouspec(E, T)
            
        return self.sample_statistic(sample)
    
    
    def flxspec_sample(self, E, T=None):
        
        scalar = np.asarray(E).ndim == 0
        
        if scalar: sample = np.zeros(self.posterior_nsample, dtype=float)
        else: sample = np.zeros([self.posterior_nsample, len(E)], dtype=float)
        
        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.flxspec(E, T)
            
        return self.sample_statistic(sample)
    
    
    def ergspec_sample(self, E, T=None):
        
        scalar = np.asarray(E).ndim == 0
        
        if scalar: sample = np.zeros(self.posterior_nsample, dtype=float)
        else: sample = np.zeros([self.posterior_nsample, len(E)], dtype=float)
        
        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.ergspec(E, T)
            
        return self.sample_statistic(sample)
    
    
    def phtflux_sample(self, emin, emax, ngrid, time=None):
        
        sample = np.zeros(self.posterior_nsample, dtype=float)
        
        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.phtflux(emin, emax, ngrid, time)
            
        return self.sample_statistic(sample)
    
    
    def ergflux_sample(self, emin, emax, ngrid, time=None):
        
        sample = np.zeros(self.posterior_nsample, dtype=float)
        
        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.ergflux(emin, emax, ngrid, time)
            
        return self.sample_statistic(sample)
    
    
    def phtflux_ratio_sample(self, erange1, erange2, ngrid, time=None):
        
        sample = np.zeros(self.posterior_nsample, dtype=float)

        emin1, emax1 = erange1
        
        emin2, emax2 = erange2
        
        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.phtflux(emin1, emax1, ngrid, time) / self.phtflux(emin2, emax2, ngrid, time)
            
        return self.sample_statistic(sample)


    def ergflux_ratio_sample(self, erange1, erange2, ngrid, time=None):
        
        sample = np.zeros(self.posterior_nsample, dtype=float)

        emin1, emax1 = erange1
        
        emin2, emax2 = erange2
        
        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.ergflux(emin1, emax1, ngrid, time) / self.ergflux(emin2, emax2, ngrid, time)
            
        return self.sample_statistic(sample)


    def __add__(self, other):
        """Build a composite that sums this component with ``other``."""

        return CompositeModel(self, '+', other)


    def __radd__(self, other):
        """Right-side variant of :meth:`__add__`."""

        return self.__add__(other)


    def __sub__(self, other):
        """Build a composite that subtracts ``other`` from this component."""

        return CompositeModel(self, '-', other)


    def __rsub__(self, other):
        """Right-side variant of :meth:`__sub__`."""

        return CompositeModel(other, '-', self)


    def __mul__(self, other):
        """Build a composite that multiplies this component by ``other``."""

        return CompositeModel(self, '*', other)


    def __rmul__(self, other):
        """Right-side variant of :meth:`__mul__`."""

        return self.__mul__(other)


    def __truediv__(self, other):
        """Build a composite that divides this component by ``other``."""

        return CompositeModel(self, '/', other)


    def __rtruediv__(self, other):
        """Right-side variant of :meth:`__truediv__`."""

        return CompositeModel(other, '/', self)


    def __call__(self, other=None):
        """Build a convolution composite: ``self(other)`` passes ``other`` as ``O``."""

        return CompositeModel(self, '()', other)


    def __str__(self):
        
        return (
            f'*** Model ***\n'
            f'{self.expr} [{self.type}]\n'
            f'{self.comment}\n'
            f'*** Model Configurations ***\n'
            f'{self.cfg_info.text_table}\n'
            f'*** Model Parameters ***\n'
            f'{self.par_info.text_table}'
            )
        
    
    def __repr__(self):
        
        return self.__str__()
    
    
    def _repr_html_(self):
        
        return (
            f'{self.cfg_info.html_style}'
            f'<details open>'
            f'<summary style="margin-bottom: 10px;"><b>Model</b></summary>'
            f'<p><b>{self.expr} [{self.type}]</b></p>'
            f'<p style="white-space: pre-wrap;">{self.comment}</p>'
            f'<details open style="margin-top: 10px;">'
            f'<summary style="margin-bottom: 10px;"><b>Model Configurations</b></summary>'
            f'{self.cfg_info.html_table}'
            f'</details>'
            f'<details open style="margin-top: 10px;">'
            f'<summary style="margin-bottom: 10px;"><b>Model Parameters</b></summary>'
            f'{self.par_info.html_table}'
            f'</details>'
            f'</details>'
            )



class Additive(Model):
    """Base for additive photon-flux components; ``type`` is locked to ``'add'``."""

    @property
    def type(self):
        """Model type tag, always ``'add'``."""

        return 'add'


    @type.setter
    def type(self, new_type):
        """No-op; the type tag is fixed by subclassing :class:`Additive`."""

        pass



class Multiplicative(Model):
    """Base for dimensionless multiplicative components; ``type`` is ``'mul'``."""

    @property
    def type(self):
        """Model type tag, always ``'mul'``."""

        return 'mul'


    @type.setter
    def type(self, new_type):
        """No-op; the type tag is fixed by subclassing :class:`Multiplicative`."""

        pass



class Mathematic(Model):
    """Base for dimensionless mathematical components; ``type`` is ``'math'``."""

    @property
    def type(self):
        """Model type tag, always ``'math'``."""

        return 'math'


    @type.setter
    def type(self, new_type):
        """No-op; the type tag is fixed by subclassing :class:`Mathematic`."""

        pass



class FrozenConst(Mathematic):
    """Frozen scalar used to wrap numeric literals in composite expressions."""

    def __init__(self, value):
        """Hold ``value`` as a single frozen parameter.

        Args:
            value: The numeric constant.
        """

        self.expr = 'const'
        self.comment = f'constant model with value {value}'

        self.config = OrderedDict()

        self.params = OrderedDict()
        self.params['$C$'] = Par(value, frozen=True)


    def func(self, E=None, T=None, O=None):
        """Return the stored constant regardless of ``E``, ``T``, ``O``."""

        C = self.params['$C$'].value

        return C



class CompositeModel(Model):
    """Binary combination of two models under ``+``/``-``/``*``/``/``/``()``.

    The composite's type is inferred from ``type_operation`` via
    :attr:`tdict`; invalid combinations raise ``ValueError``. Duplicate
    component names are made unique with a numeric suffix.
    """

    def __init__(self, m1, op, m2):
        """Wrap two operands and normalize numeric literals to ``FrozenConst``.

        Args:
            m1: Left operand — a ``Model`` or numeric literal.
            op: One of ``'+'``, ``'-'``, ``'*'``, ``'/'``, ``'()'``.
            m2: Right operand — a ``Model`` or numeric literal.

        Raises:
            ValueError: If either operand has an unsupported type.
        """
        
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


    @property
    def expr(self):
        """Parenthesized expression assembled from the two operands."""

        if self.op == '()':
            if self.m2.expr[0] == '(' and self.m2.expr[-1] == ')':
                return f'{self.m1.expr}{self.m2.expr}'
            else:
                return f'{self.m1.expr}({self.m2.expr})'
        else:
            return f'({self.m1.expr}{self.op}{self.m2.expr})'


    @property
    def type(self):
        """Derived composite type looked up in :attr:`tdict`.

        Raises:
            ValueError: If the operand-type pair is not an allowed combination.
            AssertionError: If either operand has an unknown type tag.
        """

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


    @property
    def comment(self):
        """Concatenated per-component comments, one line per component."""

        return '\n'.join([f'{expr}: {mo.comment}' for expr, mo in self.mdicts.items()])


    def func(self, E, T=None, O=None):
        """Evaluate the composite by dispatching on :attr:`op`.

        For ``'()'``, ``m1`` is called with ``O=m2`` so convolution-style
        operators receive the nested model.

        Raises:
            ValueError: If ``op`` is not recognized.
        """

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
        
        
    @property
    def mdicts(self):
        """Merged component mapping from both operands."""

        return OrderedDict({**self.m1.mdicts, **self.m2.mdicts})


    @staticmethod
    def _generate_unique_name(name, family, number=2):
        """Return ``name`` suffixed with the first integer that is not in ``family``."""

        while True:
            new_name = f'{name}_{number}'
            if new_name in family:
                continue
            else:
                break
        return new_name


    def type_operation(self):
        """Return the resulting type for the current ``(m1, op, m2)`` triple.

        Raises:
            ValueError: If the operand-type pair is not a legal combination.
            AssertionError: If either operand has an unknown type tag.
        """

        assert self.m1.type in self._allowed_types, f'unsupported model.type: {self.m1.type}'
        assert self.m2.type in self._allowed_types, f'unsupported model.type: {self.m2.type}'
        
        types = (self.m1.type, self.m2.type)
        msg = f'unsupported model.type {types} for {self.op}'
        
        if self.op == '()':
            assert self.m1.type == 'conv', msg
            return self.m2.type
        elif self.op == '+' or self.op == '-':
            if set(types) < set(('add', 'math')):
                if 'add' in types:
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
            elif 'add' in types:
                return 'add'
            elif 'mul' in types:
                return 'mul'
            else:
                return 'math'


    @cached_property()
    def tdict(self):
        """Lookup table mapping ``'<t1><op><t2>'`` strings to the composite type.

        A value of ``False`` marks an illegal combination.
        """

        return {'add(add)': False,
                'add(mul)': False,
                'add(conv)': False,
                'add(math)': False,
                'mul(add)': False,
                'mul(mul)': False,
                'mul(conv)': False,
                'mul(math)': False,
                'conv(add)': 'add',
                'conv(mul)': 'mul',
                'conv(conv)': 'conv',
                'conv(math)': 'math',
                'math(add)': False,
                'math(mul)': False,
                'math(conv)': False,
                'math(math)': False,
                'add+add': 'add',
                'add+mul': False,
                'add+conv': False,
                'add+math': 'add',
                'mul+add': False,
                'mul+mul': 'mul',
                'mul+conv': False,
                'mul+math': 'mul',
                'conv+add': False,
                'conv+mul': False,
                'conv+conv': False,
                'conv+math': False,
                'math+add': 'add',
                'math+mul': 'mul',
                'math+conv': False,
                'math+math': 'math',
                'add-add': 'add',
                'add-mul': False,
                'add-conv': False,
                'add-math': 'add',
                'mul-add': False,
                'mul-mul': 'mul',
                'mul-conv': False,
                'mul-math': 'mul',
                'conv-add': False,
                'conv-mul': False,
                'conv-conv': False,
                'conv-math': False,
                'math-add': 'add',
                'math-mul': 'mul',
                'math-conv': False,
                'math-math': 'math',
                'add*add': False,
                'add*mul': 'add',
                'add*conv': False,
                'add*math': 'add',
                'mul*add': 'add',
                'mul*mul': 'mul',
                'mul*conv': False,
                'mul*math': 'mul',
                'conv*add': False,
                'conv*mul': False,
                'conv*conv': False,
                'conv*math': False,
                'math*add': 'add',
                'math*mul': 'mul',
                'math*conv': False,
                'math*math': 'math',
                'add/add': False,
                'add/mul': 'add',
                'add/conv': False,
                'add/math': 'add',
                'mul/add': False,
                'mul/mul': 'mul',
                'mul/conv': False,
                'mul/math': 'mul',
                'conv/add': False,
                'conv/mul': False,
                'conv/conv': False,
                'conv/math': False,
                'math/add': False,
                'math/mul': 'mul',
                'math/conv': False,
                'math/math': 'math'}
