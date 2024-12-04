import inspect
import numpy as np
from io import BytesIO
from copy import deepcopy
from ..util.info import Info
import astropy.io.fits as fits
from collections import OrderedDict



class Response(object):
    
    def __init__(self, chbin, phbin, drm):
        
        if not (np.ndim(chbin) == np.ndim(phbin) == np.ndim(drm) == 2):
            raise ValueError('chbin, phbin and drm must be 2D arrays')
        
        if not (chbin.shape[0] == drm.shape[1] and chbin.shape[1] == 2):
            raise ValueError('chbin is 2-col array with rows same with cols of drm')
        
        if not (phbin.shape[0] == drm.shape[0] and phbin.shape[1] == 2):
            raise ValueError('phbin is 2-col array with rows same with rows of drm')
        
        self._chbin = self.chbin = chbin
        self._phbin = self.phbin = phbin
        self._drm = self.drm = drm
        
        
    @classmethod
    def from_rsp(cls, rsp_file):
        
        if isinstance(rsp_file, BytesIO):
            rsp_file = deepcopy(rsp_file)
        elif isinstance(rsp_file, str):
            pass
        else:
            raise ValueError(f'unsupported rsp_file type')

        rsp_hdu = fits.open(rsp_file, ignore_missing_simple=True)

        try:
            matExt = rsp_hdu["SPECRESP MATRIX"]
        except KeyError:
            matExt = rsp_hdu["MATRIX"]
        ebouExt = rsp_hdu['EBOUNDS']

        matHeader = matExt.header
        ebouHeader = ebouExt.header

        numEnerBins = matHeader['NAXIS2']
        numDetChans = ebouHeader['NAXIS2']

        matData = matExt.data
        ebouData = ebouExt.data

        rsp_hdu.close()
        
        ChanIndex = ebouData.field(0).astype(int)
        ChanBins = np.array(list(zip(ebouData.field(1), ebouData.field(2))))
        EnerBins = np.array(list(zip(matData.field(0), matData.field(1))))

        if matHeader['TFORM4'][0] == 'P':
            fchan = [fc for fc in matData.field(3)]
        else:
            fchan = [[fc] for fc in matData.field(3)]
        
        if matHeader['TFORM5'][0] == 'P':
            nchan = [nc for nc in matData.field(4)]
        else:
            nchan = [[nc] for nc in matData.field(4)]
            
        try:
            mchan = int(matHeader['TLMIN4'])
        except KeyError:
            mchan = 1
            
        matrix = matData.field(5)
        drm = np.zeros([numEnerBins, numDetChans])
        
        for fc, nc, i in zip(fchan, nchan, range(numEnerBins)):
            idx = []
            for fc_i, nc_i in zip(fc, nc):
                fc_i = int(fc_i)
                nc_i = int(nc_i)
                tc_i = fc_i + nc_i
                
                idx_i = np.arange(fc_i - mchan, tc_i - mchan).tolist()
                idx = idx + idx_i

            drm[i, idx] = matrix[i][:]
        drm = np.array(drm).astype(float)
        
        return cls(ChanBins, EnerBins, drm)
    
    
    @classmethod
    def from_rsp2(cls, rsp_file, ii=None):
        
        print('from_rsp2 is unfinished method')
        
        
    @classmethod
    def from_plain(cls, rsp_file, ii=None):
        
        if ii is not None:
            return cls.from_rsp2(rsp_file, ii)
        else:
            if isinstance(rsp_file, str) and ':' in rsp_file:
                return cls.from_rsp2(rsp_file)
            else:
                return cls.from_rsp(rsp_file)

        
    @classmethod
    def from_rmf_arf(cls, rmf, arf):
        
        chbin = rmf._chbin
        phbin = rmf._phbin
        drm = rmf._drm
        
        srp = arf._srp.reshape([-1, 1])
        
        if not (drm.shape[0] == srp.shape[0]):
            raise ValueError('drm and srp should have same rows')
        
        drm = drm * srp
        
        return cls(chbin, phbin, drm)


    def _update(self, qual, notc, grpg, rebn):
        
        self.qual = qual
        self.notc = notc
        self.grpg = grpg
        self.rebn = rebn
        
        new_chidx = 0
        new_chbin = []
        new_drm = []
        
        for i, (ql, nt, gr) in enumerate(zip(qual, notc, grpg)):
            if not (ql and nt):
                continue
            else:
                if gr == 0:
                    continue
                elif gr == 1:
                    new_chidx += 1
                    new_chbin.append(list(self._chbin[i]))
                    new_drm.append(self._drm[:, i].copy())
                elif gr == -1:
                    if new_chidx == 0:
                        new_chidx += 1
                        new_chbin.append(list(self._chbin[i]))
                        new_drm.append(self._drm[:, i].copy())
                    else:
                        new_chbin[-1][-1] = self._chbin[i][-1]
                        new_drm[-1] += self._drm[:, i].copy()
                    
        self.chbin = np.array(new_chbin)
        self.drm = np.column_stack(new_drm).astype(float)
        
        re_chidx = 0
        re_chbin = []
        re_drm = []
        
        for i, (ql, nt, rb) in enumerate(zip(qual, notc, rebn)):
            if not (ql and nt):
                continue
            else:
                if rb == 0:
                    continue
                elif rb == 1:
                    re_chidx += 1
                    re_chbin.append(list(self._chbin[i]))
                    re_drm.append(self._drm[:, i].copy())
                elif rb == -1:
                    if re_chidx == 0:
                        re_chidx += 1
                        re_chbin.append(list(self._chbin[i]))
                        re_drm.append(self._drm[:, i].copy())
                    else:
                        re_chbin[-1][-1] = self._chbin[i][-1]
                        re_drm[-1] += self._drm[:, i].copy()
                    
        self.re_chbin = np.array(re_chbin)
        self.re_drm = np.column_stack(re_drm).astype(float)
        
        
    @property
    def chbin_mean(self):
        
        return np.mean(self.chbin, axis=1)
    
    
    @property
    def re_chbin_mean(self):
        
        return np.mean(self.re_chbin, axis=1)
    
    
    @property
    def chbin_width(self):
        
        return np.diff(self.chbin, axis=1).reshape(1, -1)[0]
    
    
    @property
    def re_chbin_width(self):
        
        return np.diff(self.re_chbin, axis=1).reshape(1, -1)[0]


    @property
    def info(self):
        
        num_chbin = len(self._chbin)
        num_phbin = len(self._phbin)
        info_dict = OrderedDict([('Name', [self.name]), 
                                 ('Channel bins', [num_chbin]), 
                                 ('Photon bins', [num_phbin])])
        
        return Info.from_dict(info_dict)
    
    
    @property
    def name(self):
        
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


    @staticmethod
    def _intersection(A, B):

        A1 = np.array([i[-1] for i in A])
        B1 = np.array([i[-1] for i in B])
        A = np.array(A)[np.argsort(A1)]
        B = np.array(B)[np.argsort(B1)]

        i, j = 0, 0
        res = []
        while i < len(A) and j < len(B):
            a1, a2 = A[i][0], A[i][1]
            b1, b2 = B[j][0], B[j][1]
            if b2 > a1 and a2 > b1:
                res.append([max(a1, b1), min(a2, b2)])
            if b2 < a2: j += 1
            else: i += 1
            
        return res


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


    def __str__(self):
        
        print(self.info.table)
        
        return ''



class DisableMethodsMeta(type):
    
    def __new__(cls, name, bases, dct):
        
        cls_to_create = super().__new__(cls, name, bases, dct)
        
        methods_to_disable = dct.get('methods_to_disable', [])
        
        for method in methods_to_disable:
            if hasattr(cls_to_create, method):
                setattr(cls_to_create, method, cls.disable_method(method))
                
        return cls_to_create

    @staticmethod
    def disable_method(method_name):
        
        def method(*args, **kwargs):
            raise AttributeError(f"'{args[0].__class__.__name__}' object has no attribute '{method_name}'")
        
        return method



class Redistribution(Response, metaclass=DisableMethodsMeta):
    
    methods_to_disable = ['from_rmf_arf', '_update']
    
    def __init__(self, chbin, phbin, drm):
        
        super().__init__(chbin, phbin, drm)
        
        
    @classmethod
    def from_rmf(cls, rmf_file):
        
        return cls.from_rsp(rmf_file)
        
        
    @classmethod
    def from_rmf2(cls, rmf_file, ii=None):
        
        print('from_rmf2 is unfinished method')


    @classmethod
    def from_plain(cls, rmf_file, ii=None):
        
        if ii is not None:
            return cls.from_rmf2(rmf_file, ii)
        else:
            if isinstance(rmf_file, str) and ':' in rmf_file:
                return cls.from_rmf2(rmf_file)
            else:
                return cls.from_rmf(rmf_file)


    @property
    def info(self):
        
        num_chbin = len(self.chbin)
        num_phbin = len(self.phbin)
        info_dict = OrderedDict([('redistribution', [self.name]), 
                                 ('channel bins', [num_chbin]), 
                                 ('photon bins', [num_phbin])])
        
        return Info.from_dict(info_dict)

    
    def __mul__(self, arf):
        
        if not isinstance(arf, Auxiliary):
            raise ValueError('it should be Redistribution * Auxiliary')
        return Response.from_rmf_arf(self, arf)


    def __rmul__(self, arf):
        
        return self.__mul__(arf)



class Auxiliary(Response, metaclass=DisableMethodsMeta):
    
    methods_to_disable = ['from_rsp', 
                          'from_rsp2', 
                          'from_rmf_arf', 
                          '_update'
                          ]
    
    def __init__(self, phbin, srp):
        
        if not np.ndim(phbin) == 2:
            raise ValueError('phbin must be 2D array')
        
        if not np.ndim(srp) == 1:
            raise ValueError('srp must be 1D array')
        
        if not (phbin.shape[0] == srp.shape[0]):
            raise ValueError('phbin and srp should have same rows')
        
        self._phbin = self.phbin = phbin
        self._srp = self.srp = srp
        
        
    @classmethod
    def from_arf(cls, arf_file):
        
        if isinstance(arf_file, BytesIO):
            arf_file = deepcopy(arf_file)
        elif isinstance(arf_file, str):
            pass
        else:
            raise ValueError(f'unsupported arf_file type')
            
        arf_hdu = fits.open(arf_file, ignore_missing_simple=True)
        
        srpExt = arf_hdu['SPECRESP']
        
        srpData = srpExt.data
        
        arf_hdu.close()
        
        EnerBins = np.array(list(zip(srpData.field(0), srpData.field(1))))
        
        try:
            specresp = np.array(srpData['SPECRESP']).astype(float)
        except KeyError:
            specresp = np.array(srpData.field(2)).astype(float)
            
        return cls(EnerBins, specresp)
    
    
    @classmethod
    def from_arf2(cls, arf_file, ii=None):
        
        print('from_arf2 is unfinished method')


    @classmethod
    def from_plain(cls, arf_file, ii=None):
        
        if ii is not None:
            return cls.from_arf2(arf_file, ii)
        else:
            if isinstance(arf_file, str) and ':' in arf_file:
                return cls.from_arf2(arf_file)
            else:
                return cls.from_arf(arf_file)


    @property
    def info(self):
        
        num_phbin = len(self._phbin)
        info_dict = OrderedDict([('auxiliary', [self.name]), 
                                 ('photon bins', [num_phbin])])
        
        return Info.from_dict(info_dict)


    def __mul__(self, rmf):
        
        if not isinstance(rmf, Redistribution):
            raise ValueError('it should be Redistribution * Auxiliary')
        return Response.from_rmf_arf(rmf, self)


    def __rmul__(self, rmf):
        
        return self.__mul__(rmf)
