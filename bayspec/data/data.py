import inspect
import warnings
import numpy as np
from io import BytesIO
from scipy import special
from ..util.info import Info
from ..util.param import Par
from collections import OrderedDict
from .spectrum import Source, Background
from ..util.significance import pgsig, ppsig
from .response import Response, Redistribution, Auxiliary



class Data(object):
    
    def __init__(self, data=None):
        
        self.data = data
        

    @property
    def data(self):
        
        return self._data
    
    
    @data.setter
    def data(self, new_data):
        
        self._data = OrderedDict()

        if new_data is None:
            pass
            
        elif isinstance(new_data, list):
            for item in new_data:
                if isinstance(item, tuple):
                    self._setitem(*item)
                    
            self._extract()
                    
        elif isinstance(new_data, dict):
            for item in new_data.items():
                self._setitem(*item)
                
            self._extract()
            
        else:
            raise ValueError('unsupported data type')

        
    def _setitem(self, key, value):
        
        if not isinstance(value, DataUnit):
            raise ValueError('value parameter should be DataUnit type')
        
        if not value.completeness:
            raise ValueError('failed for completeness check for dataunit')
        
        value.name = key
        self._data[key] = value


    def _extract(self):
        
        if self.data is None:
            raise ValueError('data is None')
        
        self.exprs = [key for key in self.data.keys()]
        self.srcs = [unit.src_ins for unit in self.data.values()]
        self.bkgs = [unit.bkg_ins for unit in self.data.values()]
        self.rsps = [unit.rsp_ins for unit in self.data.values()]
        self.stats = [unit.stat for unit in self.data.values()]
        
        self.weights = np.array([unit.weight for unit in self.data.values()])
        self.npoints = np.array([unit.npoint for unit in self.data.values()])
        
        self.notcs = [unit.notc for unit in self.data.values()]
        
        self.rsp_factor = [unit.rsp_factor for unit in self.data.values()]
        self.src_factor = [unit.src_factor for unit in self.data.values()]
        self.bkg_factor = [unit.bkg_factor for unit in self.data.values()]
        
        self.src_efficiency = [unit.src_efficiency for unit in self.data.values()]
        self.bkg_efficiency = [unit.bkg_efficiency for unit in self.data.values()]

        self.src_counts = [src.counts for src in self.srcs]
        self.src_errors = [src.errors for src in self.srcs]
        
        self.bkg_counts = [bkg.counts for bkg in self.bkgs]
        self.bkg_errors = [bkg.errors for bkg in self.bkgs]
        
        self.rsp_chbin = [rsp.chbin for rsp in self.rsps]
        self.rsp_phbin = [rsp.phbin for rsp in self.rsps]
        self.rsp_drm = [rsp.drm for rsp in self.rsps]
        
        self.ebin = np.vstack([unit.ebin for unit in self.data.values()])
        self.tarr = np.hstack([unit.tarr for unit in self.data.values()])
        self.nbin = [unit.nbin for unit in self.data.values()]
        self.bin_start = np.cumsum([0] + self.nbin)[:-1]
        self.bin_stop = np.cumsum([0] + self.nbin)[1:]
        
        
    @property
    def fit_with(self):
        
        try:
            return self._fit_with
        except AttributeError:
            raise AttributeError('no model fit with')
    
    
    @fit_with.setter
    def fit_with(self, new_model):
        
        from ..model.model import Model
        
        self._fit_with = new_model
        
        if not isinstance(self._fit_with, Model): 
            raise ValueError('fit_with argument should be Model type!')
        
        try:
            self._fit_with.fit_to
        except AttributeError:
            self._fit_with.fit_to = self
        else:
            if self._fit_with.fit_to != self:
                self._fit_with.fit_to = self


    @property
    def corr_rsp_drm(self):
        
        return [unit.corr_rsp_drm for unit in self.data.values()]
    
    
    @property
    def corr_rsp_re_drm(self):
        
        return [unit.corr_rsp_re_drm for unit in self.data.values()]


    @property
    def corr_src_efficiency(self):
        
        return [unit.corr_src_efficiency for unit in self.data.values()]
    
    
    @property
    def corr_bkg_efficiency(self):
        
        return [unit.corr_bkg_efficiency for unit in self.data.values()]
    
    
    @property
    def rsp_chbin_mean(self):
        
        return [unit.rsp_chbin_mean for unit in self.data.values()]
    
    
    @property
    def rsp_re_chbin_mean(self):
        
        return [unit.rsp_re_chbin_mean for unit in self.data.values()]
    
    
    @property
    def rsp_chbin_width(self):
        
        return [unit.rsp_chbin_width for unit in self.data.values()]
    
    
    @property
    def rsp_re_chbin_width(self):
        
        return [unit.rsp_re_chbin_width for unit in self.data.values()]
    
    
    @property
    def rsp_chbin_tarr(self):
        
        return [unit.rsp_chbin_tarr for unit in self.data.values()]
    
    
    @property
    def rsp_re_chbin_tarr(self):
        
        return [unit.rsp_re_chbin_tarr for unit in self.data.values()]
    
    
    @property
    def src_ctsrate(self):
        
        return [unit.src_ctsrate for unit in self.data.values()]
    
    
    @property
    def src_re_ctsrate(self):
        
        return [unit.src_re_ctsrate for unit in self.data.values()]
    
    
    @property
    def src_ctsrate_error(self):
        
        return [unit.src_ctsrate_error for unit in self.data.values()]
    
    
    @property
    def src_re_ctsrate_error(self):
        
        return [unit.src_re_ctsrate_error for unit in self.data.values()]
    
    
    @property
    def src_ctsspec(self):
        
        return [unit.src_ctsspec for unit in self.data.values()]
    
    
    @property
    def src_re_ctsspec(self):
        
        return [unit.src_re_ctsspec for unit in self.data.values()]
    
    
    @property
    def src_ctsspec_error(self):
        
        return [unit.src_ctsspec_error for unit in self.data.values()]
    
    
    @property
    def src_re_ctsspec_error(self):
        
        return [unit.src_re_ctsspec_error for unit in self.data.values()]
    
    
    @property
    def bkg_ctsrate(self):
        
        return [unit.bkg_ctsrate for unit in self.data.values()]
    
    
    @property
    def bkg_re_ctsrate(self):
        
        return [unit.bkg_re_ctsrate for unit in self.data.values()]
    
    
    @property
    def bkg_ctsrate_error(self):
        
        return [unit.bkg_ctsrate_error for unit in self.data.values()]
    
    
    @property
    def bkg_re_ctsrate_error(self):
        
        return [unit.bkg_re_ctsrate_error for unit in self.data.values()]
    
    
    @property
    def bkg_ctsspec(self):
        
        return [unit.bkg_ctsspec for unit in self.data.values()]
    
    
    @property
    def bkg_re_ctsspec(self):
        
        return [unit.bkg_re_ctsspec for unit in self.data.values()]
    
    
    @property
    def bkg_ctsspec_error(self):
        
        return [unit.bkg_ctsspec_error for unit in self.data.values()]
    
    
    @property
    def bkg_re_ctsspec_error(self):
        
        return [unit.bkg_re_ctsspec_error for unit in self.data.values()]
    
    
    @property
    def net_ctsrate(self):
        
        return [unit.net_ctsrate for unit in self.data.values()]
    
    
    @property
    def net_re_ctsrate(self):
        
        return [unit.net_re_ctsrate for unit in self.data.values()]
    
    
    @property
    def net_ctsrate_error(self):
        
        return [unit.net_ctsrate_error for unit in self.data.values()]
    
    
    @property
    def net_re_ctsrate_error(self):
        
        return [unit.net_re_ctsrate_error for unit in self.data.values()]
    
    
    @property
    def net_ctsspec(self):
        
        return [unit.net_ctsspec for unit in self.data.values()]
    
    
    @property
    def net_re_ctsspec(self):
        
        return [unit.net_re_ctsspec for unit in self.data.values()]
    
    
    @property
    def net_ctsspec_error(self):
        
        return [unit.net_ctsspec_error for unit in self.data.values()]
    
    
    @property
    def net_re_ctsspec_error(self):
        
        return [unit.net_re_ctsspec_error for unit in self.data.values()]
    
    
    @property
    def deconv_phtspec(self):
        
        return [factor * cts for (factor, cts) in zip(self.fit_with.cts_to_pht, self.net_ctsspec)]
    
    
    @property
    def deconv_re_phtspec(self):
        
        return [factor * cts for (factor, cts) in zip(self.fit_with.re_cts_to_pht, self.net_re_ctsspec)]
    
    
    @property
    def deconv_phtspec_error(self):
        
        return [factor * cts for (factor, cts) in zip(self.fit_with.cts_to_pht, self.net_ctsspec_error)]
    
    
    @property
    def deconv_re_phtspec_error(self):
        
        return [factor * cts for (factor, cts) in zip(self.fit_with.re_cts_to_pht, self.net_re_ctsspec_error)]
    
    
    
    def net_counts_upperlimit(self, cl=0.9):
        
        return [unit.net_counts_upperlimit(cl) for unit in self.data.values()]
        
        
    def net_ctsrate_upperlimit(self, cl=0.9):
        
        return [unit.net_ctsrate_upperlimit(cl) for unit in self.data.values()]
    
    
    @property
    def pdicts(self):
        
        return OrderedDict([(key, unit.pdicts) for key, unit in self.data.items()])


    @property
    def info(self):
        
        info_dict = OrderedDict()
        info_dict['Name'] = [key for key in self.data.keys()]
        info_dict['Noticing'] = [unit.notc for unit in self.data.values()]
        info_dict['Statistic'] = [unit.stat for unit in self.data.values()]
        info_dict['Grouping'] = [unit.grpg for unit in self.data.values()]
        info_dict['Time'] = [unit.time for unit in self.data.values()]
        
        for key, values in info_dict.items():
            for i, value in enumerate(values):
                if value is None:
                    info_dict[key][i] = 'None'

        return Info.from_dict(info_dict)
                
                
    @property
    def expr(self):
        
        return self.get_obj_name()
                
                
    def get_obj_name(self):
        
        frame = inspect.currentframe()
        
        possible_var_names = []
        
        while frame:
            local_vars = frame.f_locals.items()
            var_names = [var_name for var_name, var_val in local_vars if var_val is self]
            if var_names:
                possible_var_names.extend(var_names)
            frame = frame.f_back
        
        if possible_var_names:
            return possible_var_names[-1]
        
        return None


    def __getitem__(self, key):
        
        return self._data[key]


    def __setitem__(self, key, value):
        
        self._setitem(key, value)
        self._extract()


    def __delitem__(self, key):
        
        del self._data[key]
        self._extract()


    def __contains__(self, key):
        
        return key in self._data


    def __str__(self):
        
        print(self.info.table)
        
        return ''



class DataUnit(object):
    
    def __init__(
        self, 
        src, 
        bkg=None, 
        rmf=None, 
        arf=None, 
        rsp=None, 
        stat='pgstat', 
        notc=None, 
        grpg=None, 
        rebn=None, 
        time=None, 
        weight=1, 
        rsp_factor=None, 
        src_factor=None, 
        bkg_factor=None
        ):
        
        self._src = src
        self._bkg = bkg
        self._rmf = rmf
        self._arf = arf
        self._rsp = rsp
        self._stat = stat
        self._notc = notc
        self._grpg = grpg
        self._rebn = rebn
        self._time = time
        
        self._update()
        
        self.weight = weight
        self.rsp_factor = rsp_factor
        self.src_factor = src_factor
        self.bkg_factor = bkg_factor


    @property
    def src(self):
        
        return self._src


    @src.setter
    def src(self, new_src):
        
        self._src = new_src
        self._update()
        
        
    def _update_src(self):
        
        if isinstance(self._src, tuple):
            self.src_file, self.src_ii = self._src
        else:
            self.src_file = self._src
            self.src_ii = None
            
        if isinstance(self.src_file, BytesIO):
            self.src_name = self.src_file.name
        elif isinstance(self.src_file, str):
            self.src_name = self.src_file.split('/')[-1]
        else:
            raise ValueError(f'unsupported src file type')
        
        self.src_ins = Source.from_plain(self.src_file, self.src_ii)


    @property
    def bkg(self):
        
        return self._bkg


    @bkg.setter
    def bkg(self, new_bkg):
        
        self._bkg = new_bkg
        self._update()
        
    
    def _update_bkg(self):
        
        if self._bkg is None:
            self.bkg_file = None
            self.bkg_ii = None
            self.bkg_name = None
            
            self.bkg_ins = Background.from_plain(self.src_file, self.src_ii)
            self.bkg_ins.set_zero()
            
        else:
            if isinstance(self._bkg, tuple):
                self.bkg_file, self.bkg_ii = self._bkg
            else:
                self.bkg_file = self._bkg
                self.bkg_ii = None
        
            if isinstance(self.bkg_file, BytesIO):
                self.bkg_name = self.bkg_file.name
            elif isinstance(self.bkg_file, str):
                self.bkg_name = self.bkg_file.split('/')[-1]
            else:
                raise ValueError(f'unsupported bkg file type')
            
            self.bkg_ins = Background.from_plain(self.bkg_file, self.bkg_ii)
            
            
    @property
    def rmf(self):
        
        return self._rmf
    
    
    @rmf.setter
    def rmf(self, new_rmf):
        
        self._rmf = new_rmf
        self._update()
        
        
    def _update_rmf(self):
        
        if self._rmf is None:
            self.rmf_file = None
            self.rmf_ii = None
            self.rmf_name = None
            self.rmf_ins = None
            
        else:
            if isinstance(self._rmf, tuple):
                self.rmf_file, self.rmf_ii = self._rmf
            else:
                self.rmf_file = self._rmf
                self.rmf_ii = None
                
            if isinstance(self.rmf_file, BytesIO):
                self.rmf_name = self.rmf_file.name
            elif isinstance(self.rmf_file, str):
                self.rmf_name = self.rmf_file.split('/')[-1]
            else:
                raise ValueError(f'unsupported rmf file type')
            
            self.rmf_ins = Redistribution.from_plain(self.rmf_file, self.rmf_ii)
            
            
    @property
    def arf(self):
        
        return self._arf
    
    
    @arf.setter
    def arf(self, new_arf):
        
        self._arf = new_arf
        self._update()
        
        
    def _update_arf(self):
        
        if self._arf is None:
            self.arf_file = None
            self.arf_ii = None
            self.arf_name = None
            self.arf_ins = None
            
        else:
            if isinstance(self._arf, tuple):
                self.arf_file, self.arf_ii = self._arf
            else:
                self.arf_file = self._arf
                self.arf_ii = None
                
            if isinstance(self.arf_file, BytesIO):
                self.arf_name = self.arf_file.name
            elif isinstance(self.arf_file, str):
                self.arf_name = self.arf_file.split('/')[-1]
            else:
                raise ValueError(f'unsupported arf file type')
            
            self.arf_ins = Auxiliary.from_plain(self.arf_file, self.arf_ii)


    @property
    def rsp(self):
        
        return self._rsp
    
    
    @rsp.setter
    def rsp(self, new_rsp):
        
        self._rsp = new_rsp
        self._update()
        

    def _update_rsp(self):
        
        if self._rsp is None:
            self.rsp_file = None
            self.rsp_ii = None
            self.rsp_name = None
            
            if self.rmf is not None and self.arf is not None:
                self.rsp_ins = self.rmf_ins * self.arf_ins
            else:
                self.rsp_ins = None
                warnings.warn(f'response is missing for spectrum {self.src_name}')

        else:
            if isinstance(self._rsp, tuple):
                self.rsp_file, self.rsp_ii = self._rsp
            else:
                self.rsp_file = self._rsp
                self.rsp_ii = None
                
            if isinstance(self.rsp_file, BytesIO):
                self.rsp_name = self.rsp_file.name
            elif isinstance(self.rsp_file, str):
                self.rsp_name = self.rsp_file.split('/')[-1]
            else:
                raise ValueError(f'unsupported rsp file type')
            
            self.rsp_ins = Response.from_plain(self.rsp_file, self.rsp_ii)
            
        if self.completeness:
            self.ebin = np.array(self.rsp_ins.phbin, dtype=float)
            self.nbin = self.rsp_ins.phbin.shape[0]
        else:
            self.ebin = self.nbin = None


    @property
    def stat(self):
        
        return self._stat
    
    
    @stat.setter
    def stat(self, new_stat):
        
        self._stat = new_stat
        self._update()
        
    
    def _update_stat(self):
        
        if not isinstance(self._stat, str):
            raise ValueError('<stat> parameter should be str')


    @property
    def qualifying(self):
        # quality flag:
        # qual == 0 if the data quality is good -> True
        # qual != 0 if the data quality is bad -> False
        
        return list(np.array(self.src_ins.quality) == 0)


    @property
    def notc(self):
        
        return self._notc
    
    
    @notc.setter
    def notc(self, new_notc):
        
        self._notc = new_notc
        self._update()
        
        
    def _update_notc(self):
        
        if self._notc is not None:
            if not isinstance(self._notc, (list, np.ndarray)):
                raise ValueError('<notc> parameter should be list or array')
            else:
                self._notc = list(self._notc)
                if type(self._notc[0]) is not list:
                    self._notc = [self._notc]
                else:
                    self._notc = self._union(self._notc)
        else:
            self._notc = [[np.min(self.rsp_ins._chbin), np.max(self.rsp_ins._chbin)]]
        
        if self.completeness:
            self.noticing = self._notice(self.rsp_ins._chbin, self._notc)
        else:
            self.noticing = None


    @property
    def grpg(self):
        
        return self._grpg


    @grpg.setter
    def grpg(self, new_grpg):
        
        self._grpg = new_grpg
        self._update()
        
        
    def _update_grpg(self):
        
        if self._grpg is not None:
            if not isinstance(self._grpg, dict):
                raise ValueError('<grpg> parameter should be dict')
            
        if self._grpg is None:
            self.grouping = self.src_ins.grouping
        else:
            gr_params = {'min_evt': None, 'min_sigma': None, 'max_bin': None}
            gr_params.update(self._grpg)
            
            if self.completeness:
                ini_flag = (np.array(self.qualifying) & np.array(self.noticing)).astype(int).tolist()
                
                self.grouping =  self._group(
                    self.src_ins._counts, 
                    self.bkg_ins._counts, 
                    self.bkg_ins._errors, 
                    self.src_ins.exposure, 
                    self.bkg_ins.exposure, 
                    self.src_ins.backscale, 
                    self.bkg_ins.backscale, 
                    min_evt=gr_params['min_evt'], 
                    min_sigma=gr_params['min_sigma'], 
                    max_bin=gr_params['max_bin'], 
                    stat=self.stat, 
                    ini_flag=ini_flag)
                
            else:
                self.grouping = None
                
                
    @property
    def rebn(self):
        
        return self._rebn
    
    
    @rebn.setter
    def rebn(self, new_rebn):
        
        self._rebn = new_rebn
        self._update()
        
        
    def _update_rebn(self):
        
        if self._rebn is not None:
            if not isinstance(self._rebn, dict):
                raise ValueError('<rebn> parameter should be dict')
            
        if self._rebn is None:
            self.rebining = self.grouping
        else:
            rb_params = {'min_evt': None, 'min_sigma': None, 'max_bin': None}
            rb_params.update(self._rebn)
            
            if self.completeness:
                ini_flag = (np.array(self.qualifying) & np.array(self.noticing)).astype(int).tolist()
                
                self.rebining =  self._rebin(
                    self.src_ins._counts, 
                    self.bkg_ins._counts, 
                    self.bkg_ins._errors, 
                    self.src_ins.exposure, 
                    self.bkg_ins.exposure, 
                    self.src_ins.backscale, 
                    self.bkg_ins.backscale, 
                    min_evt=rb_params['min_evt'], 
                    min_sigma=rb_params['min_sigma'], 
                    max_bin=rb_params['max_bin'], 
                    stat=self.stat, 
                    ini_flag=ini_flag)
                
            else:
                self.rebining = None


    def _update_data(self):
        
        if self.completeness:
        
            self.src_ins._update(self.qualifying, self.noticing, self.grouping, self.rebining)
            self.bkg_ins._update(self.qualifying, self.noticing, self.grouping, self.rebining)
            self.rsp_ins._update(self.qualifying, self.noticing, self.grouping, self.rebining)
            
        self.npoint = self.src_ins.counts.shape[0]
        
        self.src_efficiency = self.src_ins.exposure
        self.bkg_efficiency = self.bkg_ins.exposure * self.bkg_ins.backscale / self.src_ins.backscale
        
        self.alpha = self.src_efficiency / self.bkg_efficiency


    @property
    def time(self):
        
        return self._time
    
    
    @time.setter
    def time(self, new_time):
        
        self._time = new_time
        self._update()
        
        
    def _update_time(self):
        
        if self._time is not None:
            if not isinstance(self._time, (int, float)):
                raise ValueError('<time> parameter should be int or float')
        
        if self.completeness:
            self.tarr = np.repeat(self._time, self.nbin)
        else:
            self.tarr = None


    def _update(self):
        
        self._update_src()
        self._update_bkg()
        self._update_rmf()
        self._update_arf()
        self._update_rsp()
        self._update_stat()
        self._update_notc()
        self._update_grpg()
        self._update_rebn()
        self._update_data()
        self._update_time()
        
        
    @property
    def completeness(self):
        
        if self.src_ins is None or self.rsp_ins is None:
            return False
        else:
            return True


    @property
    def weight(self):
        
        return self._weight
    
    
    @weight.setter
    def weight(self, new_weight):
        
        self._weight = new_weight
        
        if not isinstance(self._weight, (int, float)):
            raise ValueError('<weight> parameter should be int or float')


    @property
    def rsp_factor(self):
        
        return self._rsp_factor.value
    
    
    @rsp_factor.setter
    def rsp_factor(self, new_rsp_factor):
        
        if new_rsp_factor is None:
            self._rsp_factor = Par(1, frozen=True)
        else:
            self._rsp_factor = new_rsp_factor
        
        if not isinstance(self._rsp_factor, Par):
            raise ValueError('<rsp_factor> parameter should be Param type')
        
        
    @property
    def src_factor(self):
        
        return self._src_factor.value
    
    
    @src_factor.setter
    def src_factor(self, new_src_factor):
        
        if new_src_factor is None:
            self._src_factor = Par(1, frozen=True)
        else:
            self._src_factor = new_src_factor
        
        if not isinstance(self._src_factor, Par):
            raise ValueError('<src_factor> parameter should be Param type')
        
        
    @property
    def bkg_factor(self):
        
        return self._bkg_factor.value
    
    
    @bkg_factor.setter
    def bkg_factor(self, new_bkg_factor):
        
        if new_bkg_factor is None:
            self._bkg_factor = Par(1, frozen=True)
        else:
            self._bkg_factor = new_bkg_factor
        
        if not isinstance(self._bkg_factor, Par):
            raise ValueError('<bkg_factor> parameter should be Param type')


    @property
    def corr_rsp_drm(self):
        
        return self.rsp_ins.drm * self.rsp_factor
    
    
    @property
    def corr_rsp_re_drm(self):
        
        return self.rsp_ins.re_drm * self.rsp_factor


    @property
    def corr_src_efficiency(self):
        
        return self.src_efficiency * self.src_factor


    @property
    def corr_bkg_efficiency(self):
        
        return self.bkg_efficiency * self.bkg_factor
    
    
    @property
    def rsp_chbin_mean(self):
        
        return np.mean(self.rsp_ins.chbin, axis=1)
    
    
    @property
    def rsp_re_chbin_mean(self):
        
        return np.mean(self.rsp_ins.re_chbin, axis=1)
    
    
    @property
    def rsp_chbin_width(self):
        
        return np.diff(self.rsp_ins.chbin, axis=1).reshape(1, -1)[0]
    
    
    @property
    def rsp_re_chbin_width(self):
        
        return np.diff(self.rsp_ins.re_chbin, axis=1).reshape(1, -1)[0]
    
    
    @property
    def rsp_chbin_tarr(self):
        
        return np.repeat(self._time, self.rsp_chbin_mean.shape[0])
    
    
    @property
    def rsp_re_chbin_tarr(self):
        
        return np.repeat(self._time, self.rsp_re_chbin_mean.shape[0])
    
    
    @property
    def src_ctsrate(self):
        
        return self.src_ins.counts / self.corr_src_efficiency
    
    
    @property
    def src_re_ctsrate(self):
        
        return self.src_ins.re_counts / self.corr_src_efficiency
    
    
    @property
    def src_ctsrate_error(self):
        
        return self.src_ins.errors  / self.corr_src_efficiency
    
    
    @property
    def src_re_ctsrate_error(self):
        
        return self.src_ins.re_errors  / self.corr_src_efficiency
    
    
    @property
    def src_ctsspec(self):
        
        return self.src_ctsrate / self.rsp_chbin_width
    
    
    @property
    def src_re_ctsspec(self):
        
        return self.src_re_ctsrate / self.rsp_re_chbin_width
    
    
    @property
    def src_ctsspec_error(self):
        
        return self.src_ctsrate_error / self.rsp_chbin_width
    
    
    @property
    def src_re_ctsspec_error(self):
        
        return self.src_re_ctsrate_error / self.rsp_re_chbin_width
    
    
    @property
    def bkg_ctsrate(self):
        
        return self.bkg_ins.counts  / self.corr_bkg_efficiency
    
    
    @property
    def bkg_re_ctsrate(self):
        
        return self.bkg_ins.re_counts  / self.corr_bkg_efficiency
    
    
    @property
    def bkg_ctsrate_error(self):
        
        return self.bkg_ins.errors  / self.corr_bkg_efficiency
    
    
    @property
    def bkg_re_ctsrate_error(self):
        
        return self.bkg_ins.re_errors  / self.corr_bkg_efficiency
    
    
    @property
    def bkg_ctsspec(self):
        
        return self.bkg_ctsrate / self.rsp_chbin_width
    
    
    @property
    def bkg_re_ctsspec(self):
        
        return self.bkg_re_ctsrate / self.rsp_re_chbin_width
    
    
    @property
    def bkg_ctsspec_error(self):
        
        return self.bkg_ctsrate_error / self.rsp_chbin_width
    
    
    @property
    def bkg_re_ctsspec_error(self):
        
        return self.bkg_re_ctsrate_error / self.rsp_re_chbin_width
    
    
    @property
    def net_ctsrate(self):
        
        return self.src_ctsrate - self.bkg_ctsrate
    
    
    @property
    def net_re_ctsrate(self):
        
        return self.src_re_ctsrate - self.bkg_re_ctsrate
    
    
    @property
    def net_ctsrate_error(self):
        
        return np.sqrt(self.src_ctsrate_error ** 2 + self.bkg_ctsrate_error ** 2)
    
    
    @property
    def net_re_ctsrate_error(self):
        
        return np.sqrt(self.src_re_ctsrate_error ** 2 + self.bkg_re_ctsrate_error ** 2)
    
    
    @property
    def net_ctsspec(self):
        
        return self.src_ctsspec - self.bkg_ctsspec
    
    
    @property
    def net_re_ctsspec(self):
        
        return self.src_re_ctsspec - self.bkg_re_ctsspec
    
    
    @property
    def net_ctsspec_error(self):
        
        return np.sqrt(self.src_ctsspec_error ** 2 + self.bkg_ctsspec_error ** 2)
    
    
    @property
    def net_re_ctsspec_error(self):
        
        return np.sqrt(self.src_re_ctsspec_error ** 2 + self.bkg_re_ctsspec_error ** 2)
    
    
    def net_counts_upperlimit(self, cl=0.9):
        
        N = np.sum(self.src_ins.counts)
        B = np.sum(self.bkg_ins.counts) * self.alpha
        
        return special.gammaincinv(N + 1, cl * special.gammaincc(N + 1, B) 
                                   + special.gammainc(N + 1, B)) - B
        
        
    def net_ctsrate_upperlimit(self, cl=0.9):
        
        return self.net_counts_upperlimit(cl) / self.corr_src_efficiency


    @property
    def name(self):
        
        try:
            return self._name
        except AttributeError:
            return self.get_obj_name()
    
    
    @name.setter
    def name(self, new_name):
        
        self._name = new_name
        
        
    @property
    def pdicts(self):
        
        pdicts = OrderedDict()
        
        if self._rsp_factor == Par(1, frozen=True):
            pdicts[f'rf@{self.name}'] = self._rsp_factor
            
        if self._src_factor == Par(1, frozen=True):
            pdicts[f'sf@{self.name}'] = self._src_factor
            
        if self._bkg_factor == Par(1, frozen=True):
            pdicts[f'bf@{self.name}'] = self._bkg_factor
        
        return pdicts


    @property
    def info(self):
        
        info_dict = OrderedDict()
        info_dict['src'] = self.src_name
        info_dict['bkg'] = self.bkg_name
        info_dict['rmf'] = self.rmf_name
        info_dict['arf'] = self.arf_name
        info_dict['rsp'] = self.rsp_name
        info_dict['notc'] = self.notc
        info_dict['stat'] = self.stat
        info_dict['grpg'] = self.grpg
        info_dict['time'] = self.time
        info_dict['weight'] = self.weight
        info_dict['rsp_factor'] = self.rsp_factor
        info_dict['src_factor'] = self.src_factor
        info_dict['bkg_factor'] = self.bkg_factor
        
        for key, value in info_dict.items():
            if value is None:
                info_dict[key] = 'None'

        info_dict = OrderedDict([('property', info_dict.keys()), 
                                 (self.name, info_dict.values())])
        
        return Info.from_dict(info_dict)

        
    def __str__(self):
        
        print(self.info.table)
        
        return ''


    def get_obj_name(self):
        
        frame = inspect.currentframe()
        
        possible_var_names = []
        
        while frame:
            local_vars = frame.f_locals.items()
            var_names = [var_name for var_name, var_val in local_vars if var_val is self]
            if var_names:
                possible_var_names.extend(var_names)
            frame = frame.f_back
        
        if possible_var_names:
            return possible_var_names[-1]
        
        return None
    
    
    @staticmethod
    def _union(bins):
        
        if len(bins) == 0:
            return []

        bins1 = np.array([bin_[0] for bin_ in bins])
        bins = np.array(bins)[np.argsort(bins1)]
        bins = bins.tolist()

        res = [bins[0]]
        for i in range(1, len(bins)):
            a1, a2 = res[-1][0], res[-1][1]
            b1, b2 = bins[i][0], bins[i][1]
            if b2 >= a1 and a2 >= b1:
                res[-1] = [min(a1, b1), max(a2, b2)]
            else: res.append(bins[i])

        return res


    @staticmethod
    def _notice(chbin, notc=None):
        
        if notc is None:
            notc = [[np.min(chbin), np.max(chbin)]]
        elif type(notc[0]) is not list:
            notc = [notc]
        else:
            notc = notc

        flag = [False] * len(chbin)
        for _, (low, upp) in enumerate(notc):
            flag_i = list(map(lambda l, u: low <= l and upp >= u, chbin[:, 0], chbin[:, 1]))
            flag = [pre or now for pre, now in zip(flag, flag_i)]
            
        return flag


    @staticmethod
    def _group(
        s, 
        b, 
        berr, 
        ts, 
        tb, 
        ss, 
        sb, 
        min_sigma=None, 
        min_evt=None, 
        max_bin=None, 
        stat=None, 
        ini_flag=None
        ):
        
        # grouping flag:
        # grpg = 0 if the channel is not allowed to group, including the not qualified noticed channels
        # grpg = +1 if the channel is the start of a new bin
        # grpg = -1 if the channel is part of a continuing bin
        
        if ini_flag is None:
            ini_flag = [1] * len(s)
            
        if min_sigma is None:
            min_sigma = -np.inf
        
        if min_evt is None:
            min_evt = 0
            
        if max_bin is None:
            max_bin = np.inf
            
        alpha = ts * ss / (tb * sb)

        flag, gs = [], []
        nowbin = False
        cs, cb, cberr, cp = 0, 0, 0, 0
        for i in range(len(s)):
            if ini_flag[i] != 1:
                flag.append(0)
                if nowbin:
                    if len(gs) < 2:
                        pass
                    else:
                        flag[gs[-1]] = -1
                nowbin = False
                cs, cb, cberr, cp = 0, 0, 0, 0
            else:
                if not nowbin:
                    flag.append(1)
                    gs.append(i)
                    cp = 1
                else:
                    flag.append(-1)
                    cp += 1

                si = s[i]
                bi = b[i]
                bierr = berr[i]
                cs += si
                cb += bi
                cberr = np.sqrt(cberr ** 2 + bierr ** 2)
                
                if stat is None: stat = 'pgstat'
                if stat in ['pstat', 'cstat', 'ppstat', 'Xppstat', 'Xcstat', 'ULppstat']:
                    if (cb < 0 or cs < 0) and (cb != cs):
                        sigma = 0
                    else:
                        sigma = ppsig(cs, cb, alpha)
                elif stat in ['gstat', 'chi2', 'pgstat', 'Xpgstat', 'ULpgstat']:
                    if cs <= 0 or cberr == 0:
                        sigma = 0
                    else:
                        sigma = pgsig(cs, cb * alpha, cberr * alpha)
                else:
                    raise AttributeError(f'unsupported stat: {stat}')
                
                evt = cs - cb * alpha
                
                if ((sigma >= min_sigma) and (evt >= min_evt)) or cp == max_bin:
                    nowbin = False
                    cs, cb, cberr, cp = 0, 0, 0, 0
                else:
                    nowbin = True

                if nowbin and i == (len(s) - 1):
                    if len(gs) < 2:
                        pass
                    else:
                        flag[gs[-1]] = -1

        return np.array(flag)


    @staticmethod
    def _rebin(
        s, 
        b, 
        berr, 
        ts, 
        tb, 
        ss, 
        sb, 
        min_sigma=None, 
        min_evt=None, 
        max_bin=None, 
        stat=None, 
        ini_flag=None
        ):
        
        return DataUnit._group(
            s, 
            b,
            berr, 
            ts, 
            tb, 
            ss, 
            sb, 
            min_sigma, 
            min_evt, 
            max_bin, 
            stat, 
            ini_flag)
