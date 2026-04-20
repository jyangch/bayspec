"""Detector response containers: full DRM, pure redistribution, and ARF.

Provides the base :class:`Response` (channel × photon redistribution
matrix), :class:`Redistribution` (RMF only), :class:`Auxiliary` (ARF
only), and the location-driven :class:`BalrogResponse` used for
Fermi/GBM localization-dependent responses. ``Redistribution * Auxiliary``
composes into a full :class:`Response`.
"""

import inspect
import numpy as np
from io import BytesIO
from copy import deepcopy
import astropy.io.fits as fits
from collections import OrderedDict

from ..util.info import Info
from ..util.param import Par
from ..util.prior import unif
from ..util.tools import cached_property



class Response(object):
    """Full detector response: channel bins, photon bins, and DRM.

    Attributes:
        chbin: ``(nchan, 2)`` array of channel energy bin edges.
        phbin: ``(nphot, 2)`` array of photon energy bin edges.
        drm: ``(nphot, nchan)`` detector redistribution matrix.
        ra: Optional right-ascension ``Par``.
        dec: Optional declination ``Par``.
        factor: Multiplicative ``Par`` applied during convolution.
    """

    def __init__(
        self,
        chbin,
        phbin,
        drm,
        ra=Par(0, frozen=True),
        dec=Par(0, frozen=True),
        factor=Par(1, frozen=True)
        ):
        """Build a response from channel/photon bins and the response matrix.

        Args:
            chbin: ``(nchan, 2)`` array of channel energy bin edges.
            phbin: ``(nphot, 2)`` array of photon energy bin edges.
            drm: ``(nphot, nchan)`` detector redistribution matrix.
            ra: Optional RA ``Par``.
            dec: Optional Dec ``Par``.
            factor: Multiplicative ``Par``.

        Raises:
            ValueError: If the shape invariants above are violated.
        """
        
        if not (np.ndim(chbin) == np.ndim(phbin) == np.ndim(drm) == 2):
            raise ValueError('chbin, phbin and drm must be 2D arrays')
        
        if not (chbin.shape[0] == drm.shape[1] and chbin.shape[1] == 2):
            raise ValueError('chbin is 2-col array with rows same with cols of drm')
        
        if not (phbin.shape[0] == drm.shape[0] and phbin.shape[1] == 2):
            raise ValueError('phbin is 2-col array with rows same with rows of drm')
        
        self._chbin = chbin
        self._phbin = phbin
        self._drm = drm
        self._ra = ra
        self._dec = dec
        self._factor = factor
        
        
    @classmethod
    def from_rsp(cls, rsp_file):
        """Load a single-extension OGIP RSP/RMF file.

        Supports both ``SPECRESP MATRIX`` and legacy ``MATRIX`` extension
        names. Handles both variable-length (``P``-form) and fixed-length
        ``FCHAN``/``NCHAN`` columns.

        Args:
            rsp_file: Path to the FITS file, or a ``BytesIO``.

        Returns:
            A new ``Response`` instance.

        Raises:
            ValueError: If ``rsp_file`` is neither a string nor ``BytesIO``.
        """

        if isinstance(rsp_file, BytesIO):
            rsp_file = deepcopy(rsp_file)
        elif isinstance(rsp_file, str):
            pass
        else:
            raise ValueError(f'unsupported rsp_file type')

        rsp_hdu = fits.open(rsp_file, ignore_missing_simple=True)

        try:
            matExt = rsp_hdu['SPECRESP MATRIX']
        except KeyError:
            matExt = rsp_hdu['MATRIX']
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
        
        if matrix[0].ndim == 0:
            matrix = matrix.reshape(-1, 1)
        
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
        """Load extension ``ii`` of a multi-extension RSP2 file.

        Accepts the row index via ``ii`` or via ``"path:ii"``.

        Args:
            rsp_file: Path or ``BytesIO`` to the RSP2 file.
            ii: Extension index within ``SPECRESP MATRIX``/``MATRIX``.

        Returns:
            A new ``Response`` instance.

        Raises:
            ValueError: If ``rsp_file`` is neither a string nor ``BytesIO``.
            AssertionError: If ``ii`` is not an ``int`` when required.
        """

        if isinstance(rsp_file, BytesIO):
            rsp_file = deepcopy(rsp_file)
            assert isinstance(ii, int), 'ii should be int type'
        elif isinstance(rsp_file, str):
            if ':' in rsp_file:
                rsp_file, ii = rsp_file.split(':')
                rsp_file = rsp_file.strip()
                ii = int(ii.strip())
            else:
                assert isinstance(ii, int), 'ii should be int type'
        else:
            raise ValueError(f'unsupported rsp_file type')
        
        rsp_hdu = fits.open(rsp_file, ignore_missing_simple=True)

        try:
            matExt = rsp_hdu['SPECRESP MATRIX', ii]
        except KeyError:
            matExt = rsp_hdu['MATRIX', ii]
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
        
        if matrix[0].ndim == 0:
            matrix = matrix.reshape(-1, 1)
        
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
    def from_plain(cls, rsp_file, ii=None):
        """Dispatch to ``from_rsp`` or ``from_rsp2`` based on ``rsp_file`` form.

        Args:
            rsp_file: Path (optionally ``"path:ii"``) or ``BytesIO``.
            ii: Extension index; forces the RSP2 loader when given.

        Returns:
            A new ``Response`` instance.
        """

        if ii is not None:
            return cls.from_rsp2(rsp_file, ii)
        else:
            if isinstance(rsp_file, str) and ':' in rsp_file:
                return cls.from_rsp2(rsp_file)
            else:
                return cls.from_rsp(rsp_file)


    @classmethod
    def from_rmf_arf(cls, rmf, arf):
        """Compose a full response from a redistribution matrix and ARF.

        Args:
            rmf: A :class:`Redistribution`-like object carrying ``chbin``,
                ``phbin``, and ``drm``.
            arf: An :class:`Auxiliary`-like object carrying ``srp``.

        Returns:
            A new ``Response`` whose DRM is ``rmf.drm * arf.srp``.

        Raises:
            ValueError: If the photon-bin dimensions disagree.
        """

        chbin = rmf._chbin
        phbin = rmf._phbin
        drm = rmf._drm
        
        srp = arf._srp.reshape([-1, 1])
        
        if not (drm.shape[0] == srp.shape[0]):
            raise ValueError('drm and srp should have same rows')
        
        drm = drm * srp
        
        return cls(chbin, phbin, drm)
    
    
    @property
    def chbin(self):
        
        return self._chbin
    
    
    @property
    def phbin(self):
        
        return self._phbin
    
    
    @property
    def drm(self):

        return self._drm
    
    
    @property
    def ra(self):
        
        return self._ra
    
    
    @ra.setter
    def ra(self, new_ra):
        """Set the RA ``Par``; ``None`` installs a uniform prior on ``[0, 360)``.

        Raises:
            ValueError: If the resolved value is not a ``Par``.
        """

        if new_ra is None:
            self._ra = Par(0, unif(0, 360))
        else:
            self._ra = new_ra

        if not isinstance(self._ra, Par):
            raise ValueError('<ra> parameter should be Param type')


    @property
    def dec(self):

        return self._dec


    @dec.setter
    def dec(self, new_dec):
        """Set the Dec ``Par``; ``None`` installs a uniform prior on ``[-90, 90]``.

        Raises:
            ValueError: If the resolved value is not a ``Par``.
        """

        if new_dec is None:
            self._dec = Par(0, unif(-90, 90))
        else:
            self._dec = new_dec

        if not isinstance(self._dec, Par):
            raise ValueError('<dec> parameter should be Param type')


    @property
    def factor(self):
        
        return self._factor
    
    
    @factor.setter
    def factor(self, new_factor):
        """Set the multiplicative ``Par``; ``None`` resets to a frozen unit factor.

        Raises:
            ValueError: If the resolved value is not a ``Par``.
        """

        if new_factor is None:
            self._factor = Par(1, frozen=True)
        else:
            self._factor = new_factor

        if not isinstance(self._factor, Par):
            raise ValueError('<factor> parameter should be Param type')


    @property
    def chbin_mean(self):
        """Per-channel midpoint energy, computed from ``chbin``."""

        return np.mean(self.chbin, axis=1)


    @property
    def chbin_width(self):
        """Per-channel bin width, computed from ``chbin``."""

        return np.diff(self.chbin, axis=1).reshape(1, -1)[0]


    @property
    def info(self):
        """Return a tabular :class:`Info` summary of bin counts."""

        num_chbin = len(self.chbin)
        num_phbin = len(self.phbin)
        info_dict = OrderedDict([('Name', [self.name]),
                                 ('Channel bins', [num_chbin]),
                                 ('Photon bins', [num_phbin])])

        return Info.from_dict(info_dict)


    @property
    def name(self):
        """Best-effort identifier inferred from the caller scope."""

        return self.get_obj_name()


    def get_obj_name(self):
        """Walk call frames and return the outermost local name bound to ``self``.

        Returns ``None`` if no binding is found.
        """

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
        
        return (
            f'*** Response ***\n'
            f'{self.info.text_table}'
            )
        
        
    def __repr__(self):
        
        return self.__str__()
    
    
    def _repr_html_(self):
        
        return (
            f'{self.info.html_style}'
            f'<details open>'
            f'<summary style="margin-bottom: 10px;"><b>Response</b></summary>'
            f'{self.info.html_table}'
            f'</details>'
            )



class DisableMethodsMeta(type):
    """Metaclass that hides inherited methods listed in ``methods_to_disable``.

    Each named method is replaced by a shim that raises ``AttributeError``,
    so the class still advertises the inherited API surface but callers
    are told the method does not apply to this subclass.
    """

    def __new__(cls, name, bases, dct):
        """Replace every method in ``methods_to_disable`` with a disabling shim."""

        cls_to_create = super().__new__(cls, name, bases, dct)

        methods_to_disable = dct.get('methods_to_disable', [])

        for method in methods_to_disable:
            if hasattr(cls_to_create, method):
                setattr(cls_to_create, method, cls.disable_method(method))

        return cls_to_create

    @staticmethod
    def disable_method(method_name):
        """Return a function that raises ``AttributeError`` for ``method_name``."""

        def method(*args, **kwargs):
            raise AttributeError(f"'{args[0].__class__.__name__}' object has no attribute '{method_name}'")

        return method



class BalrogResponse(Response, metaclass=DisableMethodsMeta):
    """Location-driven Fermi/GBM response backed by a balrog DRM.

    Recomputes the response matrix whenever ``ra`` or ``dec`` change. The
    file-based factory methods inherited from :class:`Response` are
    disabled since the DRM is produced by the balrog provider.
    """

    methods_to_disable = ['from_rsp',
                          'from_rsp2',
                          'from_plain',
                          'from_rmf_arf']

    def __init__(
        self,
        balrog_drm,
        ra=Par(0, unif(0, 360)),
        dec=Par(0, unif(-90, 90)),
        factor=Par(1, frozen=True)
        ):
        """Store the balrog provider and the sky location parameters.

        Args:
            balrog_drm: Balrog DRM provider exposing ``set_location``,
                ``matrix``, ``ebounds``, and ``monte_carlo_energies``.
            ra: RA ``Par``, by default uniform on ``[0, 360)``.
            dec: Dec ``Par``, by default uniform on ``[-90, 90]``.
            factor: Multiplicative ``Par``.
        """

        self._balrog_drm = balrog_drm
        self._ra = ra
        self._dec = dec
        self._factor = factor


    @property
    def balrog_drm(self):

        return self._balrog_drm
    
    
    @cached_property()
    def chbin(self):
        """Channel bins derived from the balrog ``ebounds`` array."""

        return np.vstack([self.balrog_drm.ebounds[:-1],
                          self.balrog_drm.ebounds[1:]]).T


    @cached_property()
    def phbin(self):
        """Photon bins derived from ``monte_carlo_energies``."""

        return np.vstack([self.balrog_drm.monte_carlo_energies[:-1],
                          self.balrog_drm.monte_carlo_energies[1:]]).T


    @cached_property(lambda self: (self.ra.value, self.dec.value))
    def drm(self):
        """Response matrix at the current ``(ra, dec)``; NaNs are coerced to 0."""

        self.balrog_drm.set_location(self.ra.value, self.dec.value)
        
        drm = self.balrog_drm.matrix

        if not np.all(np.isfinite(drm)):

            for i, j in zip(np.where(np.isnan(drm))[0], np.where(np.isnan(drm))[1]):

                drm[i, j] = 0.0

        return drm.T



class Redistribution(Response, metaclass=DisableMethodsMeta):
    """Pure redistribution matrix (RMF); combines with :class:`Auxiliary` via ``*``."""

    methods_to_disable = ['from_rmf_arf']

    def __init__(self, chbin, phbin, drm):
        """Build an RMF with the same shape invariants as :class:`Response`."""

        super().__init__(chbin, phbin, drm)


    @classmethod
    def from_rmf(cls, rmf_file):
        """Alias for :meth:`Response.from_rsp` that returns an RMF."""

        return cls.from_rsp(rmf_file)


    @classmethod
    def from_rmf2(cls, rmf_file, ii=None):
        """Alias for :meth:`Response.from_rsp2` that returns an RMF."""

        return cls.from_rsp2(rmf_file, ii)


    @classmethod
    def from_plain(cls, rmf_file, ii=None):
        """Dispatch to ``from_rmf`` or ``from_rmf2`` based on ``rmf_file`` form."""

        if ii is not None:
            return cls.from_rmf2(rmf_file, ii)
        else:
            if isinstance(rmf_file, str) and ':' in rmf_file:
                return cls.from_rmf2(rmf_file)
            else:
                return cls.from_rmf(rmf_file)


    @property
    def info(self):
        """Return a tabular :class:`Info` summary labelled as a redistribution."""

        num_chbin = len(self.chbin)
        num_phbin = len(self.phbin)
        info_dict = OrderedDict([('redistribution', [self.name]),
                                 ('channel bins', [num_chbin]),
                                 ('photon bins', [num_phbin])])

        return Info.from_dict(info_dict)


    def __mul__(self, arf):
        """Compose with an :class:`Auxiliary` to yield a full :class:`Response`."""

        if not isinstance(arf, Auxiliary):
            raise ValueError('it should be Redistribution * Auxiliary')
        return Response.from_rmf_arf(self, arf)


    def __rmul__(self, arf):
        """Right-side variant of :meth:`__mul__`."""

        return self.__mul__(arf)



class Auxiliary(Response, metaclass=DisableMethodsMeta):
    """Auxiliary response (ARF): effective area per photon bin.

    Attributes:
        phbin: ``(nphot, 2)`` array of photon energy bin edges.
        srp: 1D array of effective-area values aligned with ``phbin``.
    """

    methods_to_disable = ['from_rsp',
                          'from_rsp2',
                          'from_rmf_arf'
                          ]

    def __init__(self, phbin, srp):
        """Build an ARF from photon bins and the effective-area vector.

        Args:
            phbin: ``(nphot, 2)`` photon bin edges.
            srp: Effective area per photon bin.

        Raises:
            ValueError: If the shape invariants above are violated.
        """

        if not np.ndim(phbin) == 2:
            raise ValueError('phbin must be 2D array')

        if not np.ndim(srp) == 1:
            raise ValueError('srp must be 1D array')

        if not (phbin.shape[0] == srp.shape[0]):
            raise ValueError('phbin and srp should have same rows')

        self._phbin = phbin
        self._srp = srp


    @property
    def srp(self):
        
        return self._srp


    @classmethod
    def from_arf(cls, arf_file):
        """Load a single-extension OGIP ARF file.

        Args:
            arf_file: Path to the ARF FITS file, or a ``BytesIO``.

        Returns:
            A new ``Auxiliary`` instance.

        Raises:
            ValueError: If ``arf_file`` is neither a string nor ``BytesIO``.
        """

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
        """Load extension ``ii`` of a multi-extension ARF2 file.

        Args:
            arf_file: Path or ``BytesIO`` to the ARF2 file; may use the
                ``"path:ii"`` convention.
            ii: Extension index within ``SPECRESP``.

        Returns:
            A new ``Auxiliary`` instance.
        """

        if isinstance(arf_file, BytesIO):
            arf_file = deepcopy(arf_file)
            assert isinstance(ii, int), 'ii should be int type'
        elif isinstance(arf_file, str):
            if ':' in arf_file:
                arf_file, ii = arf_file.split(':')
                arf_file = arf_file.strip()
                ii = int(ii.strip())
            else:
                assert isinstance(ii, int), 'ii should be int type'
        else:
            raise ValueError(f'unsupported arf_file type')
        
        arf_hdu = fits.open(arf_file, ignore_missing_simple=True)
        
        srpExt = arf_hdu['SPECRESP', ii]
        
        srpData = srpExt.data
        
        arf_hdu.close()
        
        EnerBins = np.array(list(zip(srpData.field(0), srpData.field(1))))
        
        try:
            specresp = np.array(srpData['SPECRESP']).astype(float)
        except KeyError:
            specresp = np.array(srpData.field(2)).astype(float)
            
        return cls(EnerBins, specresp)


    @classmethod
    def from_plain(cls, arf_file, ii=None):
        """Dispatch to ``from_arf`` or ``from_arf2`` based on ``arf_file`` form."""

        if ii is not None:
            return cls.from_arf2(arf_file, ii)
        else:
            if isinstance(arf_file, str) and ':' in arf_file:
                return cls.from_arf2(arf_file)
            else:
                return cls.from_arf(arf_file)


    @property
    def info(self):
        """Return a tabular :class:`Info` summary labelled as an auxiliary."""

        num_phbin = len(self.phbin)
        info_dict = OrderedDict([('auxiliary', [self.name]),
                                 ('photon bins', [num_phbin])])

        return Info.from_dict(info_dict)


    def __mul__(self, rmf):
        """Compose with a :class:`Redistribution` to yield a full :class:`Response`."""

        if not isinstance(rmf, Redistribution):
            raise ValueError('it should be Redistribution * Auxiliary')
        return Response.from_rmf_arf(rmf, self)


    def __rmul__(self, rmf):
        """Right-side variant of :meth:`__mul__`."""

        return self.__mul__(rmf)
