"""Inference drivers: base aggregator, Bayesian samplers, and max-likelihood fits.

:class:`Infer` aggregates one or more ``(Data, Model)`` pairs, resolves
shared parameters through the linking machinery, and exposes the joint
prior, log-likelihood, and fit-statistic entry points that the samplers
consume. :class:`BayesInfer` adds driver methods for MultiNest and
emcee; :class:`MaxLikeFit` adds lmfit and iminuit minimisers plus
covariance-driven bootstrap sampling.
"""

import os
import json
import ctypes
import warnings
import numpy as np
from collections import OrderedDict
from collections.abc import Callable

from .pair import Pair
from ..data.data import Data
from ..model.model import Model
from ..util.info import Info
from ..util.tools import SuperDict, JsonEncoder, json_dump



class Infer(object):
    """Aggregate of one or more ``(Data, Model)`` pairs with shared parameters.

    Collects per-pair parameters into a single flat index, identifies
    which of them are free (after honouring ``Par.link``/``unlink``
    relations and the frozen flag), and publishes the pooled
    log-likelihood, log-prior, fit statistic, and residuals.

    **Group docstring for aggregator properties.** List-valued
    ``data_*`` properties flatten the same-named ``Data`` property over
    every bound ``Data``; ``model_*`` properties flatten the matching
    ``Model`` property over every bound ``Model``. The family covers the
    following patterns (the ``_re_`` infix uses re-binned channels):

    - ``data_chbin_mean``/``_re_``, ``data_chbin_width``/``_re_``
      (sourced from ``rsp_chbin_*``);
    - ``data_{ctsrate,ctsspec,phtspec,flxspec,ergspec}`` plus their
      ``_error``/``_re_`` variants (net counts and deconvolved spectra);
    - ``model_{ctsrate,ctsspec,phtspec,flxspec,ergspec}`` plus their
      ``_re_`` variants (convolved model spectra sampled at the data
      channels).

    Attributes:
        pairs: The raw list of ``(Data, Model)`` tuples.
        Data/Model/Pair: Per-pair unpacked containers.
        inference_type: Display label shown in :meth:`__str__`.
        loglike_func/logprior_func/prior_transform_func: Optional user
            overrides for the corresponding computations; ``None`` means
            use the built-in implementation.
    """

    def __init__(self, pairs=None):
        """Build an inference container with the given ``(Data, Model)`` pairs.

        Args:
            pairs: ``None`` or a list of ``(Data, Model)`` / ``(Model, Data)``
                tuples. The two orderings are both accepted.
        """

        self.pairs = pairs

        self.loglike_func = None
        self.logprior_func = None
        self.prior_transform_func = None

        self.inference_type = 'Inference'


    @property
    def pairs(self):
        
        return self._pairs
    
    
    @pairs.setter
    def pairs(self, new_pairs):
        """Replace all pairs from a list of tuples, then :meth:`_extract`.

        Raises:
            ValueError: If ``new_pairs`` is not ``None`` or a list.
        """

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
        """Append a ``(Data, Model)`` or ``(Model, Data)`` pair and re-extract.

        Args:
            *pair: Two positional arguments, one ``Data`` and one ``Model``
                in either order.
        """

        self._addpair(*pair)
        self._extract()


    def _extract(self):
        
        if self.pairs is None:
            raise ValueError('pairs is None')
        
        self._EXTRACT = object()
        
        self.nparis = len(self.pairs)
        
        self.Data = [pair[0] for pair in self.pairs]
        self.Model = [pair[1] for pair in self.pairs]
        self.Pair = [Pair(*pair) for pair in self.pairs]
        
        self.data_names = [key for data in self.Data for key in data.names]
        self.model_exprs = [model.expr for model in self.Model]
        
        self._you_free()


    @property
    def pdicts(self):
        """Mapping from every model/data expression to its ``pdicts`` dictionary."""

        return OrderedDict([(md.expr, md.pdicts) for md in (self.Model + self.Data)])


    @property
    def cdicts(self):
        """Mapping from every model expression to its ``cdicts`` dictionary."""

        return OrderedDict([(mo.expr, mo.cdicts) for mo in self.Model])


    @property
    def cfg(self):
        """Flat :class:`SuperDict` of every model config parameter."""

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
        """Flat :class:`SuperDict` of every model+data parameter (free or frozen)."""

        pid = 0
        par = SuperDict()
        
        for md in (self.Model + self.Data):
            for params in md.pdicts.values():
                for pr in params.values():
                    pid += 1
                    par[str(pid)] = pr
                
        return par
    
    
    @property
    def pvalues(self):
        """Tuple of current parameter values, preserving ``par`` order."""

        return tuple([pr.value for pr in self.par.values()])


    @staticmethod
    def foo(id):
        """Recover the Python object at address ``id`` via ``ctypes`` reflection."""

        return ctypes.cast(id, ctypes.py_object).value


    @property
    def idpid(self):
        """Map each ``id(Par)`` to the set of ``par#`` indices it occupies.

        Used to detect which parameter slots share the same underlying
        :class:`Par` instance (linked parameters).
        """

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
        """List of per-config rows tagged with component and class (``model``/``data``)."""

        cid = 0
        all_config = list()
        
        for i, md in enumerate(self.Model + self.Data):
            
            if i < self.nparis: cls = 'model'
            else: cls = 'data'
            
            for expr, config in md.cdicts.items():
                for cl, cg in config.items():
                    cid += 1
                    
                    all_config.append(
                        {'cfg#': str(cid), 
                         'Class': cls, 
                         'Expression': md.expr, 
                         'Component': expr, 
                         'Parameter': cl, 
                         'Value': cg.val})
                        
        return all_config


    @property
    def all_params(self):
        """List of per-parameter rows, with linked-parameter mates resolved."""

        pid = 0
        all_params = list()
        
        for i, md in enumerate(self.Model + self.Data):
            
            if i < self.nparis: cls = 'model'
            else: cls = 'data'
            
            for expr, params in md.pdicts.items():
                for pl, pr in params.items():
                    pid += 1
                    
                    self_id = self.idpid[str(id(pr))]
                    mate_id = [self.idpid[str(id(mate))] for mate in pr.mates]
                    mates = self_id.union(*mate_id)
                    mates.remove(str(pid))
                    
                    all_params.append(
                        {'par#': str(pid), 
                         'Class': cls, 
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
        """Rebuild the ``free_*`` caches by walking ``all_params`` once.

        A parameter is free if it is not ``frozen`` and no earlier slot
        that shares its underlying :class:`Par` has already claimed the
        free-parameter slot.
        """

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
        """Link every :class:`Par` in ``pids`` so they share value/prior/posterior.

        Args:
            pids: Iterable of ``par#`` indices (as strings or ints).
        """

        for i, ip in enumerate(pids):
            for j, jp in enumerate(pids):
                if j > i:
                    if id(self.par[ip]) != id(self.par[jp]):
                        self.par[ip].link(self.par[jp])

        self._you_free()


    def unlink(self, pids):
        """Undo any linking between every pair drawn from ``pids``.

        Args:
            pids: Iterable of ``par#`` indices.
        """

        for i, ip in enumerate(pids):
            for j, jp in enumerate(pids):
                if j > i:
                    if id(self.par[ip]) != id(self.par[jp]):
                        self.par[ip].unlink(self.par[jp])

        self._you_free()
        
        
    @property
    def free_par(self):
        """:class:`SuperDict` of the free :class:`Par` instances keyed by ``par#``."""

        return self._free_par


    @property
    def free_params(self):
        """Rows from :attr:`all_params` restricted to the free parameters."""

        return self._free_params


    @property
    def free_plabels(self):
        """LaTeX-decorated labels of the free parameters, in canonical order."""

        return self._free_plabels


    @property
    def clean_free_plabels(self):
        """:attr:`free_plabels` with LaTeX braces, dollars, and backslashes stripped."""

        return [pl.replace('$', '').replace('{', '').replace('}', '').replace('\\', '')
                for pl in self._free_plabels]


    @property
    def free_indexed_plabels(self):
        """Free-parameter labels prefixed with their ``par#`` index."""

        return [f'p{key}({label})' for label, key in zip(self.free_plabels, self.free_par.keys())]


    @property
    def clean_free_indexed_plabels(self):
        """Indexed labels with LaTeX markup removed."""

        return [f'p{key}({label})' for label, key in zip(self.clean_free_plabels, self.free_par.keys())]


    @property
    def free_pvalues(self):
        """Current values of every free parameter."""

        return self._free_pvalues


    @property
    def free_pranges(self):
        """Per-parameter ``(low, high)`` plausible ranges used by minimisers."""

        return self._free_pranges


    @property
    def free_nparams(self):
        """Number of free parameters."""

        return self._free_nparams
    
    
    @property
    def cfg_info(self):
        """Tabular :class:`Info` view of every configuration parameter."""

        all_config = self.all_config.copy()

        return Info.from_list_dict(all_config)


    @property
    def par_info(self):
        """Tabular parameter view tagging free slots with ``*`` and resolving linked aliases."""

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
        
        all_params = Info.list_dict_to_dict(all_params)
        
        del all_params['Posterior']
        del all_params['Mates']
        del all_params['Frozen']
        
        return Info.from_dict(all_params)


    @property
    def notable_par_info(self):
        """Parameter view like :attr:`par_info` but hides frozen data-level rows."""

        self._you_free()
        
        all_params = self.all_params.copy()
        notable_params = list()
        
        for par in all_params:
            if par['par#'] in self.free_par:
                par['par#'] = par['par#'] + '*'
            else: 
                if par['Frozen']:
                    par['Prior'] = 'frozen'
                    if par['Class'] == 'data': 
                        continue
                else:
                    par['Prior'] = '=par#{%s}'%(','.join(par['Mates']))
            notable_params.append(par)
        
        notable_params = Info.list_dict_to_dict(notable_params)
        
        del notable_params['Posterior']
        del notable_params['Mates']
        del notable_params['Frozen']
        
        return Info.from_dict(notable_params)
        
        
    @property
    def free_par_info(self):
        """Tabular :class:`Info` view restricted to the free parameters."""

        self._you_free()
        
        free_params = self.free_params.copy()
        
        free_params = Info.list_dict_to_dict(free_params)
        
        del free_params['Posterior']
        del free_params['Mates']
        del free_params['Frozen']
        
        return Info.from_dict(free_params)
    
    
    def save(self, savepath):
        """Dump config and parameter tables under ``savepath``.

        Args:
            savepath: Directory path. Created if missing.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.cfg_info.data_list_dict, savepath + '/infer_cfg.json')
        json_dump(self.par_info.data_list_dict, savepath + '/infer_par.json')


    @property
    def data_chbin_mean(self):
        """Concatenated per-channel midpoints from every bound ``Data``.

        Every ``data_*`` and ``model_*`` property on this class flattens
        the same-named list from the underlying ``Data``/``Model`` across
        every pair, so downstream code can treat the whole inference as a
        single long series.
        """

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
    def data_flxspec(self):
        
        return [value for data in self.Data for value in data.deconv_flxspec]
    
    
    @property
    def data_re_flxspec(self):
        
        return [value for data in self.Data for value in data.deconv_re_flxspec]
    
    
    @property
    def data_flxspec_error(self):
        
        return [value for data in self.Data for value in data.deconv_flxspec_error]
    
    
    @property
    def data_re_flxspec_error(self):
        
        return [value for data in self.Data for value in data.deconv_re_flxspec_error]
    
    
    @property
    def data_ergspec(self):
        
        return [value for data in self.Data for value in data.deconv_ergspec]
    
    
    @property
    def data_re_ergspec(self):
        
        return [value for data in self.Data for value in data.deconv_re_ergspec]
    
    
    @property
    def data_ergspec_error(self):
        
        return [value for data in self.Data for value in data.deconv_ergspec_error]
    
    
    @property
    def data_re_ergspec_error(self):
        
        return [value for data in self.Data for value in data.deconv_re_ergspec_error]
    
    
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
    def model_flxspec(self):
        
        return [value for model in self.Model for value in model.flxspec_at_rsp]
    
    
    @property
    def model_re_flxspec(self):
        
        return [value for model in self.Model for value in model.re_flxspec_at_rsp]
    
    
    @property
    def model_ergspec(self):
        
        return [value for model in self.Model for value in model.ergspec_at_rsp]
    
    
    @property
    def model_re_ergspec(self):
        
        return [value for model in self.Model for value in model.re_ergspec_at_rsp]


    @property
    def residual(self):
        """Per-unit sigma residuals aggregated across every pair."""

        return list(map(lambda oi, mi, si: (oi - mi) / si,
                        self.data_ctsrate,
                        self.model_ctsrate,
                        self.data_ctsrate_error))


    @property
    def re_residual(self):
        """Per-unit sigma residuals on the re-binned grid."""

        return list(map(lambda oi, mi, si: (oi - mi) / si,
                        self.data_re_ctsrate,
                        self.model_re_ctsrate,
                        self.data_re_ctsrate_error))


    @property
    def prior_list(self):
        """Per-parameter prior densities evaluated at the current free values."""

        return [par.prior.pdf(par.val) for par in self.free_par.values()]


    @property
    def prior(self):
        """Joint prior density as the product of :attr:`prior_list`."""

        return np.prod(self.prior_list)


    @property
    def logprior(self):
        """Joint log-prior; ``-inf`` when the prior vanishes."""

        if self.prior == 0:
            return -np.inf
        else:
            return np.log(self.prior)


    @property
    def stat_list(self):
        """Concatenated per-unit statistic across every pair."""

        return np.hstack([pair.stat_list for pair in self.Pair])


    @property
    def pseudo_residual_list(self):
        """Concatenated per-unit pseudo-residual arrays across every pair."""

        return [rd for pair in self.Pair for rd in pair.pseudo_residual_list]


    @property
    def weight_list(self):
        """Concatenated per-unit weights across every pair."""

        return np.hstack([pair.weight_list for pair in self.Pair])


    @property
    def stat(self):
        """Summed fit statistic across every pair."""

        return np.sum([pair.stat for pair in self.Pair])


    @property
    def pseudo_residual(self):
        """Concatenated weight-scaled pseudo-residual vector across every pair."""

        return np.hstack([pair.pseudo_residual for pair in self.Pair])


    @property
    def loglike_list(self):
        """Concatenated per-unit log-likelihood across every pair."""

        return np.hstack([pair.loglike_list for pair in self.Pair])


    @property
    def loglike(self):
        """Summed log-likelihood across every pair."""

        return np.sum([[pair.loglike for pair in self.Pair]])


    @property
    def npoint_list(self):
        """Concatenated per-unit point counts across every pair."""

        return np.hstack([pair.npoint_list for pair in self.Pair])


    @property
    def npoint(self):
        """Total number of fitted data points across every pair."""

        return np.sum([[pair.npoint for pair in self.Pair]])


    @property
    def dof(self):
        """Degrees of freedom: :attr:`npoint` minus the free-parameter count."""

        return self.npoint - self.free_nparams


    @property
    def all_stat(self):
        """Per-pair statistic summary plus a totals row, ready for ``Info.from_dict``."""

        all_stat = OrderedDict()
        all_stat['Data'] = ['Total']
        all_stat['Model'] = ['Total']
        all_stat['Statistic'] = ['stat/dof']
        all_stat['Value'] = ['{:.3f}/{:d}'.format(self.stat, self.dof)]
        all_stat['Bins'] = [self.npoint]

        for dt, mo in zip(self.Data, self.Model):
            mex = mo.expr
            for sex, stat in zip(dt.names, dt.stats):
                all_stat['Data'].insert(-1, sex)
                all_stat['Model'].insert(-1, mex)
                all_stat['Statistic'].insert(-1, stat)
                
        all_stat['Value'] = [stat for stat in self.stat_list] + all_stat['Value']
        all_stat['Bins'] = [point for point in self.npoint_list] + all_stat['Bins']

        return all_stat


    @property
    def stat_info(self):
        """Tabular :class:`Info` view of :attr:`all_stat`."""

        all_stat = self.all_stat.copy()

        return Info.from_dict(all_stat)


    def __str__(self):
        
        return (
            f'*** {self.inference_type} ***\n'
            f'*** Configurations ***\n'
            f'{self.cfg_info.text_table}\n'
            f'*** Parameters ***\n'
            f'{self.notable_par_info.text_table}'
            )
        
        
    def __repr__(self):
        
        return self.__str__()
    
    
    def _repr_html_(self):
        
        return (
            f'{self.cfg_info.html_style}'
            f'<details open>'
            f'<summary style="margin-bottom: 10px;"><b>{self.inference_type}</b></summary>'
            f'<details open style="margin-top: 10px;">'
            f'<summary style="margin-bottom: 10px;"><b>Configurations</b></summary>'
            f'{self.cfg_info.html_table}'
            f'</details>'
            f'<details open style="margin-top: 10px;">'
            f'<summary style="margin-bottom: 10px;"><b>Parameters</b></summary>'
            f'{self.notable_par_info.html_table}'
            f'</details>'
            f'</details>'
            )


    def at_par(self, theta):
        """Write free-parameter values from the 1-indexed sequence ``theta``."""

        for i, thi in enumerate(theta):
            self.free_par[i+1].val = thi


    @property
    def prior_transform_func(self):
        """Optional user override for the unit-cube to prior transform."""

        return self._prior_transform_func


    @prior_transform_func.setter
    def prior_transform_func(self, new_prior_transform_func):
        """Install a user-provided prior-transform callable or clear it with ``None``.

        Raises:
            ValueError: If the argument is neither callable nor ``None``.
        """

        if isinstance(new_prior_transform_func, (Callable, type(None))):
            self._prior_transform_func = new_prior_transform_func
        else:
            raise ValueError('prior_transform_func is expected to be Callable or None')


    @property
    def logprior_func(self):
        """Optional user override for the log-prior computation."""

        return self._logprior_func


    @logprior_func.setter
    def logprior_func(self, new_logprior_func):
        """Install a user-provided log-prior callable or clear it with ``None``."""

        if isinstance(new_logprior_func, (Callable, type(None))):
            self._logprior_func = new_logprior_func
        else:
            raise ValueError('logprior_func is expected to be Callable or None')


    @property
    def loglike_func(self):
        """Optional user override for the log-likelihood computation."""

        return self._loglike_func


    @loglike_func.setter
    def loglike_func(self, new_loglike_func):
        """Install a user-provided log-likelihood callable or clear it with ``None``."""

        if isinstance(new_loglike_func, (Callable, type(None))):
            self._loglike_func = new_loglike_func
        else:
            raise ValueError('loglike_func is expected to be Callable or None')


    def prior_transform(self, cube):
        """Transform a unit cube to parameter space via each prior's inverse CDF.

        Delegates to :attr:`prior_transform_func` when it has been set.

        Args:
            cube: Array-like of length :attr:`free_nparams` in ``[0, 1]``.

        Returns:
            Transformed parameter vector of the same length.
        """

        if self.prior_transform_func is None:

            theta = np.array(cube)

            for i, cui in enumerate(cube):
                theta[i] = self.free_par[i+1].prior.ppf(cui)

            return theta

        else:
            return self.prior_transform_func(self, cube)


    def calc_logprior(self, theta):
        """Apply ``theta`` and return the log-prior (or the user override)."""

        self.at_par(theta)

        if self.logprior_func is None:
            return self.logprior
        else:
            return self.logprior_func(self, theta)


    def calc_stat(self, theta):
        """Apply ``theta`` and return the summed fit statistic."""

        self.at_par(theta)

        return self.stat


    def calc_pseudo_residual(self, theta):
        """Apply ``theta`` and return the concatenated pseudo-residual vector."""

        self.at_par(theta)

        return self.pseudo_residual


    def calc_loglike(self, theta):
        """Apply ``theta`` and return the log-likelihood (or the user override)."""

        self.at_par(theta)

        if self.loglike_func is None:
            return self.loglike
        else:
            return self.loglike_func(self, theta)


    def calc_logprob(self, theta):
        """Return the unnormalised log-posterior; ``-inf`` outside the prior support."""

        lp = self.calc_logprior(theta)

        if not np.isfinite(lp):
            return -np.inf

        return lp + self.calc_loglike(theta)


    def calc_logprior_sample(self, theta_sample):
        """Vectorized log-prior over a sample matrix; returns ``-inf`` where it vanishes.

        Args:
            theta_sample: ``(nsample, nparams)`` array of draws.

        Returns:
            ``(nsample,)`` array of log-prior values.
        """

        prior_list_sample = np.zeros_like(theta_sample, dtype=float)
        
        for i in range(theta_sample.shape[1]):
            prior_list_sample[:, i] = self.free_par[i+1].prior.pdf(theta_sample[:, i])
            
        prior_sample = np.prod(prior_list_sample, axis=1)
        
        return np.where(prior_sample == 0, -np.inf, np.log(prior_sample))



class BayesInfer(Infer):
    """:class:`Infer` extension that wires up MultiNest and emcee drivers.

    Adds ``multinest`` and ``emcee`` methods that run the samplers,
    persist chains to ``savepath``, and return a :class:`Posterior` for
    downstream analysis.
    """

    def __init__(self, pairs=None):
        """Initialise like :class:`Infer` and tag the inference type."""
        super().__init__(pairs=pairs)

        self.inference_type = 'Bayesian Inference'


    def multinest_prior_transform(self, cube):
        """MultiNest-facing wrapper around :meth:`prior_transform`."""

        return self.prior_transform(cube)


    def multinest_calc_loglike(self, theta):
        """MultiNest-facing wrapper around :meth:`calc_loglike`."""

        return self.calc_loglike(theta)


    def multinest_safe_prior_transform(self, cube, ndim, nparams):
        """MultiNest C-callback wrapper that writes ``cube`` in place.

        Terminates the process on any Python-level exception so MultiNest
        does not silently swallow it.
        """

        try:
            cube_arr = np.array([cube[i] for i in range(ndim)])
            theta_arr = self.multinest_prior_transform(cube_arr)
            for i in range(ndim):
                cube[i] = theta_arr[i]
        except Exception as e:
            import sys
            sys.stderr.write('ERROR in prior: %s\n' % e)
            sys.exit(1)


    def multinest_safe_calc_loglike(self, cube, ndim, nparams, lnew):
        """MultiNest C-callback log-likelihood; returns ``-1e100`` when non-finite."""

        try:
            cube_arr = np.array([cube[i] for i in range(ndim)])
            ll = float(self.multinest_calc_loglike(cube_arr))
            if not np.isfinite(ll):
                return -1e100
            return ll
        except Exception as e:
            import sys
            sys.stderr.write('ERROR in loglikelihood: %s\n' % e)
            sys.exit(1)


    def multinest(self, nlive=500, resume=True, verbose=False, savepath='./'):
        """Run MultiNest and return a :class:`Posterior` wrapping the result.

        Args:
            nlive: Number of live points.
            resume: Resume from any chain already present under ``savepath``.
            verbose: Forward MultiNest's verbose flag.
            savepath: Directory for MultiNest outputs and cached samples.

        Returns:
            A :class:`~bayspec.infer.analyzer.Posterior`.
        """
        
        import pymultinest
        from .analyzer import Posterior
        
        self.sampler_type = 'nested'
        
        self._you_free()
        
        savepath_prefix = savepath + '/1-'
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        pymultinest.run(LogLikelihood=self.multinest_safe_calc_loglike, 
                        Prior=self.multinest_safe_prior_transform, 
                        n_dims=self.free_nparams, resume=resume, 
                        verbose=verbose, n_live_points=nlive, 
                        outputfiles_basename=savepath_prefix, sampling_efficiency=0.8, 
                        importance_nested_sampling=True, multimodal=True)

        multinest_analyzer = pymultinest.Analyzer(outputfiles_basename=savepath_prefix, n_params=self.free_nparams)
        
        posterior_stats = multinest_analyzer.get_stats()
        
        if (not resume) or (not os.path.exists(savepath_prefix + 'posterior_sample.txt')):
            self.posterior_sample = multinest_analyzer.get_equal_weighted_posterior()
            self.posterior_sample[:, -1] = self.posterior_sample[:, -1] + \
                self.calc_logprior_sample(self.posterior_sample[:, 0:-1])
            np.savetxt(savepath_prefix + 'posterior_sample.txt', self.posterior_sample)
        else:
            self.posterior_sample = np.loadtxt(savepath_prefix + 'posterior_sample.txt')

        self.logevidence = posterior_stats['nested importance sampling global log-evidence']
        
        json.dump(nlive, open(savepath_prefix + 'nlive.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(posterior_stats, open(savepath_prefix + 'posterior_stats.json', 'w'), indent=4, cls=JsonEncoder)

        return Posterior(self)


    def emcee_calc_logprob(self, theta):
        """emcee-facing wrapper around :meth:`calc_logprob`."""

        return self.calc_logprob(theta)


    def emcee(self, nstep=1000, discard=100, resume=True, savepath='./'):
        """Run emcee and return a :class:`Posterior` wrapping the flattened chain.

        Args:
            nstep: Number of MCMC steps per walker.
            discard: Burn-in steps discarded before flattening.
            resume: Reuse an existing chain cached under ``savepath``.
            savepath: Directory for chain outputs.

        Returns:
            A :class:`~bayspec.infer.analyzer.Posterior`.
        """

        import emcee
        from .analyzer import Posterior
        
        self.sampler_type = 'mcmc'
        
        self._you_free()
        
        savepath_prefix = savepath + '/1-'
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        np.random.seed(450001)
        ndim = self.free_nparams
        nwalkers = 32 if 2 * ndim < 32 else 2 * ndim
        pos = self.free_pvalues + 1e-4 * np.random.randn(nwalkers, ndim)

        if (not resume) or (not os.path.exists(savepath_prefix + '.npz')):
            emcee_sampler = emcee.EnsembleSampler(nwalkers, ndim, self.emcee_calc_logprob)
            emcee_sampler.run_mcmc(pos, nstep, progress=True)
            
            params_sample = emcee_sampler.get_chain()
            np.savez(savepath_prefix + '.npz', sample=params_sample)
            
            logprob_sample = emcee_sampler.get_log_prob()
            np.savetxt(savepath_prefix + 'logprob.dat', logprob_sample)
            
            try:
                autocorr_time = emcee_sampler.get_autocorr_time()
                json.dump(autocorr_time, open(savepath_prefix + 'autocorr_time.json', 'w'), 
                          indent=4, cls=JsonEncoder)
            except:
                pass

        params_sample = np.load(savepath_prefix + '.npz')['sample']
        logprob_sample = np.loadtxt(savepath_prefix + 'logprob.dat')
        
        flat_params_sample = params_sample[discard:, :, :].reshape(-1, ndim)
        flat_logprob_sample = logprob_sample[discard:, :].reshape(-1)
        
        self.posterior_sample = np.hstack((flat_params_sample, np.reshape(flat_logprob_sample, (-1, 1))))
        
        np.savetxt(savepath_prefix + 'posterior_sample.txt', self.posterior_sample)
        json.dump(nstep, open(savepath_prefix + 'nstep.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(discard, open(savepath_prefix + 'discard.json', 'w'), indent=4, cls=JsonEncoder)
        
        return Posterior(self)



class MaxLikeFit(Infer):
    """:class:`Infer` extension for maximum-likelihood fits with bootstrap sampling.

    Provides :meth:`lmfit` and :meth:`iminuit` drivers. Both run the
    minimiser, cache the best fit, build a covariance-driven bootstrap
    sample respecting the free-parameter ranges, and return a
    :class:`Bootstrap` for downstream analysis.
    """

    def __init__(self, pairs=None):
        """Initialise like :class:`Infer` and tag the inference type."""
        super().__init__(pairs=pairs)

        self.inference_type = 'Maximum Likelihood Estimation'


    def _make_bootstrap_sample(self, values, covar=None, errors=None, nsample=1000, random_state=450001):
        """Draw a covariance-respecting bootstrap sample and score each draw.

        Falls back to a diagonal covariance built from ``errors`` when
        ``covar`` is missing or non-finite. Draws are rejected if they
        fall outside any free parameter's range.

        Args:
            values: Best-fit free-parameter vector.
            covar: Optional parameter covariance matrix.
            errors: Optional per-parameter uncertainties, used for the
                fallback diagonal covariance.
            nsample: Target number of valid draws.
            random_state: Seed for reproducibility.
        """

        values = np.asarray(values, dtype=float)
        ndim = values.size
        
        nsample = max(int(nsample), 1)

        if covar is not None:
            covar = np.asarray(covar, dtype=float)

        if covar is None or covar.shape != (ndim, ndim) or (not np.isfinite(covar).all()):
            msg = 'Covariance matrix is not provided or invalid. \
                Using diagonal covariance with variances from errors or zeros.'
            warnings.warn(msg)
            err = np.zeros(ndim, dtype=float) if errors is None else np.asarray(errors, dtype=float)
            err = np.where(np.isfinite(err), np.abs(err), 0.0)
            covar = np.diag(err * err)
        
        covar = 0.5 * (covar + covar.T)
        eigval, eigvec = np.linalg.eigh(covar)
        scale = np.max(np.abs(eigval)) if eigval.size else 1.0
        floor = np.finfo(float).eps * (scale if scale > 0 else 1.0)
        eigval = np.clip(eigval, floor, None)
        covar = eigvec @ np.diag(eigval) @ eigvec.T
        
        lower = np.array([pr[0] for pr in self.free_pranges], dtype=float)
        upper = np.array([pr[1] for pr in self.free_pranges], dtype=float)
        
        rng = np.random.default_rng(random_state)
        
        param_sample = [values.copy()]
        tries = 0
        while len(param_sample) < nsample and tries < 10:
            batch_size = max(4 * (nsample - len(param_sample)), 128)
            draw = rng.multivariate_normal(values, covar, size=batch_size, check_valid='ignore')
            draw = np.atleast_2d(draw)

            inside = np.all((draw >= lower) & (draw <= upper), axis=1)
            param_sample.extend(draw[inside][:nsample - len(param_sample)])
            tries += 1
            
        if len(param_sample) < nsample:
            msg = f'Only {len(param_sample)} valid samples were generated after {tries} attempts.'
            warnings.warn(msg)
            param_sample = np.asarray(param_sample, dtype=float)
        else:
            param_sample = np.asarray(param_sample[:nsample], dtype=float)

        loglike_sample = np.array([self.calc_loglike(theta) for theta in param_sample], dtype=float)
        
        self.bootstrap_sample = np.hstack((param_sample, loglike_sample[:, None]))

        self.at_par(values)


    @staticmethod
    def _display_results(*objects):
        """Render each object with IPython when available, otherwise ``print`` it."""

        valid_objects = [obj for obj in objects if obj is not None]

        try:
            from IPython.display import HTML, display
        except ImportError:
            for obj in valid_objects:
                print(obj)
            return

        for obj in valid_objects:
            display(obj)


    def lmfit_residual(self, params):
        """lmfit-facing residual callback; delegates to :meth:`calc_pseudo_residual`."""

        theta = [params[pl] for pl in self.clean_free_plabels]

        return self.calc_pseudo_residual(theta)


    def lmfit(self, savepath=None):
        """Run ``lmfit.minimize`` on the pseudo-residuals and bootstrap the result.

        Args:
            savepath: Optional directory for persisted bootstrap samples
                and summary JSON; pass ``None`` to skip disk IO.

        Returns:
            A :class:`~bayspec.infer.analyzer.Bootstrap`.
        """
        
        import lmfit
        from .analyzer import Bootstrap
        
        self._you_free()
        
        lmfit_params = lmfit.Parameters()
        
        for pl, pv, pr in zip(self.clean_free_plabels, self.free_pvalues, self.free_pranges):
            lmfit_params.add(pl, value=pv, min=pr[0], max=pr[1], vary=True)
            
        lmfit_result = lmfit.minimize(self.lmfit_residual, lmfit_params)

        self._display_results(lmfit_result)
        
        values = np.array([lmfit_result.params[pl].value for pl in self.clean_free_plabels])
        errors = np.array([
            np.nan if lmfit_result.params[pl].stderr is None else lmfit_result.params[pl].stderr \
                for pl in self.clean_free_plabels])
        covar = getattr(lmfit_result, 'covar', None)
        
        self._make_bootstrap_sample(values, covar=covar, errors=errors)
        
        maxlike_res = {'values': values, 'errors': errors, 'covar': covar}
        
        if savepath is not None:
            savepath_prefix = savepath + '/1-'
            
            np.savetxt(savepath_prefix + 'bootstrap_sample.txt', self.bootstrap_sample)
            json.dump(maxlike_res, open(savepath_prefix + 'maxlike_res.json', 'w'), indent=4, cls=JsonEncoder)

        return Bootstrap(self)
    
    
    def iminuit_cost(self, *theta):
        """iminuit-facing cost function; returns ``1e100`` when the stat is non-finite."""

        cost = self.calc_stat(theta)

        if np.isfinite(cost):
            return float(cost)
        else:
            return 1e100


    def iminuit(self, savepath=None):
        """Run iminuit's ``migrad`` + ``hesse`` + ``minos`` and bootstrap the result.

        Args:
            savepath: Optional directory for persisted bootstrap samples
                and summary JSON.

        Returns:
            A :class:`~bayspec.infer.analyzer.Bootstrap`.
        """
        
        import iminuit
        from .analyzer import Bootstrap
        
        self._you_free()
        
        minuit = iminuit.Minuit(self.iminuit_cost, *self.free_pvalues, name=self.clean_free_indexed_plabels)
        minuit.errordef = 2 * iminuit.Minuit.LIKELIHOOD
        minuit.print_level = 0
        
        for pl, pr in zip(self.clean_free_indexed_plabels, self.free_pranges):
            minuit.limits[pl] = pr
        
        minuit.migrad()
        minuit.hesse()
        minuit.minos()
        
        self._display_results(minuit)
        
        values = np.array([par.value for par in minuit.params])
        errors = np.array([par.error for par in minuit.params])
        minos_errors = np.array([par.merror for par in minuit.params])
        covar = None if minuit.covariance is None else np.asarray(minuit.covariance)
        
        self._make_bootstrap_sample(values, covar=covar, errors=errors)
        
        maxlike_res = {'values': values, 'errors': errors, 'minos_errors': minos_errors, 'covar': covar}
        
        if savepath is not None:
            savepath_prefix = savepath + '/1-'
            
            np.savetxt(savepath_prefix + 'bootstrap_sample.txt', self.bootstrap_sample)
            json.dump(maxlike_res, open(savepath_prefix + 'maxlike_res.json', 'w'), indent=4, cls=JsonEncoder)

        return Bootstrap(self)