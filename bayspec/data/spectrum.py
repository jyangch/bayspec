import inspect
import numpy as np
from io import BytesIO
from copy import deepcopy
from ..util.info import Info
import astropy.io.fits as fits
from collections import OrderedDict



class Spectrum(object):
    
    def __init__(
        self, 
        counts, 
        errors, 
        exposure, 
        quality=None, 
        grouping=None,
        backscale=1.0
        ):
        
        if not (np.ndim(counts) == np.ndim(errors) == 1):
            raise ValueError('counts and errors must be 1D arrays')
        
        if not (np.shape(counts) == np.shape(errors)):
            raise ValueError('counts and errors must have the same shape')
        
        if not isinstance(exposure, (int, float, np.integer, np.floating)):
            raise ValueError('exposure must be int or float')
        
        if quality is None:
            quality = np.zeros(len(counts)).astype(int)
        else:
            if not (np.shape(quality) == np.shape(counts)):
                raise ValueError('quality must have the same shape with counts')
            
        if grouping is None:
            grouping = np.ones(len(counts)).astype(int)
        else:
            if not (np.shape(grouping) == np.shape(counts)):
                raise ValueError('grouping must have the same shape with counts')
        
        self._counts = self.counts = counts
        self._errors = self.errors = errors
        self._exposure = exposure
        self._quality = quality
        self._grouping = grouping
        self._backscale = backscale
        
        
    @property
    def exposure(self):
        
        return self._exposure
    
    
    @property
    def quality(self):
        
        return self._quality
    
    
    @property
    def grouping(self):
        
        return self._grouping
    
    
    @property
    def backscale(self):
        
        return self._backscale
    
    
    def set_zero(self):
        
        self._counts = self.counts = np.zeros_like(self.counts).astype(float)
        self._errors = self.errors = np.zeros_like(self.errors).astype(float)
    
    
    def _update(self, qual, notc, grpg, rebn):
        
        self.qual = qual
        self.notc = notc
        self.grpg = grpg
        self.rebn = rebn
        
        new_chidx = 0
        new_counts = []
        new_errors = []
        
        for i, (ql, nt, gr) in enumerate(zip(qual, notc, grpg)):
            if not (ql and nt):
                continue
            else:
                if gr == 0:
                    continue
                elif gr == 1:
                    new_chidx += 1
                    new_counts.append(self._counts[i])
                    new_errors.append(self._errors[i])
                elif gr == -1:
                    if new_chidx == 0:
                        new_chidx += 1
                        new_counts.append(self._counts[i])
                        new_errors.append(self._errors[i])
                    else:
                        new_counts[-1] += self._counts[i]
                        new_errors[-1] = np.sqrt(new_errors[-1] ** 2 + self._errors[i] ** 2)
                    
        self.counts = np.array(new_counts)
        self.errors = np.array(new_errors)
        
        re_chidx = 0
        re_counts = []
        re_errors = []
        
        for i, (ql, nt, rb) in enumerate(zip(qual, notc, rebn)):
            if not (ql and nt):
                continue
            else:
                if rb == 0:
                    continue
                elif rb == 1:
                    re_chidx += 1
                    re_counts.append(self._counts[i])
                    re_errors.append(self._errors[i])
                elif rb == -1:
                    if re_chidx == 0:
                        re_chidx += 1
                        re_counts.append(self._counts[i])
                        re_errors.append(self._errors[i])
                    else:
                        re_counts[-1] += self._counts[i]
                        re_errors[-1] = np.sqrt(re_errors[-1] ** 2 + self._errors[i] ** 2)
                    
        self.re_counts = np.array(re_counts)
        self.re_errors = np.array(re_errors)


    @property
    def info(self):
        
        num_channel = len(self._counts)
        num_counts = sum(self._counts)
        info_dict = OrderedDict([('Name', [self.name]), 
                                 ('Channels', [num_channel]), 
                                 ('Counts', [num_counts]), 
                                 ('Exposure', [self._exposure]), 
                                 ('Backscale', [self._backscale])])

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
    
    
    def __str__(self):
        
        print(self.info.table)
        
        return ''



class Source(Spectrum):
    
    @classmethod
    def from_src(cls, src_file):
        
        if isinstance(src_file, BytesIO):
            src_file = deepcopy(src_file)
        elif isinstance(src_file, str):
            pass
        else:
            raise ValueError(f'unsupported src_file type')
            
        src_hdu = fits.open(src_file, ignore_missing_simple=True)
        specExt = src_hdu["SPECTRUM"]

        specData = specExt.data

        SrcExpo = float(specExt.header['EXPOSURE'])
        try:
            SrcCounts = specData['COUNTS'].astype(float)
            try:
                Src_StatErr = specData['STAT_ERR'].astype(float)
            except KeyError:
                Src_StatErr = np.sqrt(SrcCounts)
            try:
                Src_SysErr = specData['SYS_ERR'].astype(float)
            except KeyError:
                Src_SysErr = 0
            SrcErr = np.sqrt(Src_StatErr ** 2 + (SrcCounts * Src_SysErr) ** 2)
        except KeyError:
            SrcCounts = specData['RATE'].astype(float) * SrcExpo
            try:
                Src_StatErr = specData['STAT_ERR'].astype(float) * SrcExpo
            except KeyError:
                Src_StatErr = np.sqrt(SrcCounts)
            try:
                Src_SysErr = specData['SYS_ERR'].astype(float)
            except KeyError:
                Src_SysErr = 0
            SrcErr = np.sqrt(Src_StatErr ** 2 + (SrcCounts * Src_SysErr) ** 2)

        try:
            SrcQual = specData['QUALITY'].astype(int)
        except KeyError:
            SrcQual = np.zeros(len(SrcCounts)).astype(int)

        try:
            SrcGrpg = specData['GROUPING'].astype(int)
        except KeyError:
            SrcGrpg = np.ones(len(SrcCounts)).astype(int)

        try:
            SrcBackSc = specData['BACKSCAL'].astype(float)
        except KeyError:
            try:
                SrcBackSc = float(specExt.header['BACKSCAL'])
            except KeyError:
                SrcBackSc = 1.0

        src_hdu.close()
        
        return cls(SrcCounts, SrcErr, SrcExpo, SrcQual, SrcGrpg, SrcBackSc)


    @classmethod
    def from_src2(cls, src_file, ii=None):
        
        if isinstance(src_file, BytesIO):
            src_file = deepcopy(src_file)
            assert isinstance(ii, int), 'ii should be int type'
        elif isinstance(src_file, str):
            if ':' in src_file:
                src_file, ii = src_file.split(':')
                src_file = src_file.strip()
                ii = int(ii.strip())
            else:
                assert isinstance(ii, int), 'ii should be int type'
        else:
            raise ValueError(f'unsupported src_file type')
            
        src_hdu = fits.open(src_file, ignore_missing_simple=True)
        specExt = src_hdu["SPECTRUM"]

        specData = specExt.data

        SrcExpo = specData['EXPOSURE'][ii].astype(float)
        try:
            SrcCounts = specData['COUNTS'][ii].astype(float)
            try:
                Src_StatErr = specData['STAT_ERR'][ii].astype(float)
            except KeyError:
                Src_StatErr = np.sqrt(SrcCounts)
            try:
                Src_SysErr = specData['SYS_ERR'][ii].astype(float)
            except KeyError:
                Src_SysErr = 0
            SrcErr = np.sqrt(Src_StatErr ** 2 + (SrcCounts * Src_SysErr) ** 2)
        except KeyError:
            SrcCounts = specData['RATE'][ii].astype(float) * SrcExpo
            try:
                Src_StatErr = specData['STAT_ERR'][ii].astype(float) * SrcExpo
            except KeyError:
                Src_StatErr = np.sqrt(SrcCounts)
            try:
                Src_SysErr = specData['SYS_ERR'][ii].astype(float)
            except KeyError:
                Src_SysErr = 0
            SrcErr = np.sqrt(Src_StatErr ** 2 + (SrcCounts * Src_SysErr) ** 2)

        try:
            SrcQual = specData['QUALITY'][ii].astype(int)
        except KeyError:
            SrcQual = np.zeros(len(SrcCounts)).astype(int)

        try:
            SrcGrpg = specData['GROUPING'][ii].astype(int)
        except KeyError:
            SrcGrpg = np.ones(len(SrcCounts)).astype(int)

        try:
            SrcBackSc = specData['BACKSCAL'][ii].astype(float)
        except KeyError:
            try:
                SrcBackSc = specExt.header['BACKSCAL']
            except KeyError:
                SrcBackSc = 1.0

        src_hdu.close()
        
        return cls(SrcCounts, SrcErr, SrcExpo, SrcQual, SrcGrpg, SrcBackSc)
    
    
    @classmethod
    def from_plain(cls, src_file, ii=None):
        
        if ii is not None:
            return cls.from_src2(src_file, ii)
        else:
            if isinstance(src_file, str) and ':' in src_file:
                return cls.from_src2(src_file)
            else:
                return cls.from_src(src_file)


    
class Background(Spectrum):
    
    @classmethod
    def from_bkg(cls, bkg_file):
        
        if isinstance(bkg_file, BytesIO):
            bkg_file = deepcopy(bkg_file)
        elif isinstance(bkg_file, str):
            pass
        else:
            raise ValueError(f'unsupported bkg_file type')
            
        bkg_hdu = fits.open(bkg_file, ignore_missing_simple=True)
        specExt = bkg_hdu["SPECTRUM"]

        specData = specExt.data

        BkgExpo = float(specExt.header['EXPOSURE'])
        try:
            BkgCounts = specData['COUNTS'].astype(float)
            try:
                Bkg_StatErr = specData['STAT_ERR'].astype(float)
            except KeyError:
                Bkg_StatErr = np.sqrt(BkgCounts)
            try:
                Bkg_SysErr = specData['SYS_ERR'].astype(float)
            except KeyError:
                Bkg_SysErr = 0
            BkgErr = np.sqrt(Bkg_StatErr ** 2 + (BkgCounts * Bkg_SysErr) ** 2)
        except KeyError:
            BkgCounts = specData['RATE'].astype(float) * BkgExpo
            try:
                Bkg_StatErr = specData['STAT_ERR'].astype(float) * BkgExpo
            except KeyError:
                Bkg_StatErr = np.sqrt(BkgCounts)
            try:
                Bkg_SysErr = specData['SYS_ERR'].astype(float)
            except KeyError:
                Bkg_SysErr = 0
            BkgErr = np.sqrt(Bkg_StatErr ** 2 + (BkgCounts * Bkg_SysErr) ** 2)

        try:
            BkgBackSc = specData['BACKSCAL'].astype(float)
        except KeyError:
            try:
                BkgBackSc = float(specExt.header['BACKSCAL'])
            except KeyError:
                BkgBackSc = 1.0

        bkg_hdu.close()
        
        return cls(BkgCounts, BkgErr, BkgExpo, None, None, BkgBackSc)
    
    
    @classmethod
    def from_bkg2(cls, bkg_file, ii=None):
        
        if isinstance(bkg_file, BytesIO):
            bkg_file = deepcopy(bkg_file)
            assert isinstance(ii, int), 'ii should be int type'
        elif isinstance(bkg_file, str):
            if ':' in bkg_file:
                bkg_file, ii = bkg_file.split(':')
                bkg_file = bkg_file.strip()
                ii = int(ii.strip())
            else:
                assert isinstance(ii, int), 'ii should be int type'
        else:
            raise ValueError(f'unsupported bkg_file type')
            
        bkg_hdu = fits.open(bkg_file, ignore_missing_simple=True)
        specExt = bkg_hdu["SPECTRUM"]

        specData = specExt.data

        BkgExpo = specData['EXPOSURE'][ii].astype(float)
        try:
            BkgCounts = specData['COUNTS'][ii].astype(float)
            try:
                Bkg_StatErr = specData['STAT_ERR'][ii].astype(float)
            except KeyError:
                Bkg_StatErr = np.sqrt(BkgCounts)
            try:
                Bkg_SysErr = specData['SYS_ERR'][ii].astype(float)
            except KeyError:
                Bkg_SysErr = 0
            BkgErr = np.sqrt(Bkg_StatErr ** 2 + (BkgCounts * Bkg_SysErr) ** 2)
        except KeyError:
            BkgCounts = specData['RATE'][ii].astype(float) * BkgExpo
            try:
                Bkg_StatErr = specData['STAT_ERR'][ii].astype(float) * BkgExpo
            except KeyError:
                Bkg_StatErr = np.sqrt(BkgCounts)
            try:
                Bkg_SysErr = specData['SYS_ERR'][ii].astype(float)
            except KeyError:
                Bkg_SysErr = 0
            BkgErr = np.sqrt(Bkg_StatErr ** 2 + (BkgCounts * Bkg_SysErr) ** 2)

        try:
            BkgBackSc = specData['BACKSCAL'][ii].astype(float)
        except KeyError:
            try:
                BkgBackSc = specExt.header['BACKSCAL']
            except KeyError:
                BkgBackSc = 1.0

        bkg_hdu.close()
        
        return cls(BkgCounts, BkgErr, BkgExpo, None, None, BkgBackSc)


    @classmethod
    def from_plain(cls, bkg_file, ii=None):
        
        if ii is not None:
            return cls.from_bkg2(bkg_file, ii)
        else:
            if isinstance(bkg_file, str) and ':' in bkg_file:
                return cls.from_bkg2(bkg_file)
            else:
                return cls.from_bkg(bkg_file)
