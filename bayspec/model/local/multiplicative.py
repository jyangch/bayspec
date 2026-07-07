"""Dimensionless multiplicative components -- cutoffs and ISM absorbers.

The absorption models (``wabs``, ``phabs``, ``tbabs``, ``tinvabs``) load
their tabulated cross sections from the ``docs/xsect`` archive; the
cross-section lookup is memoized per instance for speed.
"""

from collections import OrderedDict
from os.path import abspath, dirname
from typing import ClassVar

import numpy as np

from ...util.param import Cfg, Par
from ...util.prior import unif
from ...util.tools import memoized
from ..model import Multiplicative

docs_path = dirname(dirname(dirname(abspath(__file__)))) + '/docs'


class hecut(Multiplicative):
    """High-energy exponential cutoff above a configurable threshold energy."""

    def __init__(self):
        r"""Initialise the cutoff with a log-cutoff parameter :math:`\log E_c`."""

        self.expr = 'hecut'
        self.comment = 'high-energy cutoff model'

        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)
        self.config['threshold_energy'] = Cfg(0.0)

        self.params = OrderedDict()
        self.params[r'log$E_c$'] = Par(2, unif(-1, 5))

    def func(self, E, T=None, O=None):  # noqa: E741
        """Return 1 below ``threshold_energy`` and ``exp((ethr - E) / Ec)`` above."""

        redshift = self.config['redshift'].value
        ethr = self.config['threshold_energy'].value

        logEc = self.params[r'log$E_c$'].value

        Ec = 10**logEc

        E = np.asarray(E)
        scalar = E.ndim == 0
        if scalar:
            E = E[np.newaxis]

        zi = 1 + redshift
        E = E * zi

        nouspec = np.zeros_like(E, dtype=float)

        i1 = ethr >= E
        i2 = ethr < E
        nouspec[i1] = 1.0
        nouspec[i2] = np.exp((ethr - E[i2]) / Ec)

        return nouspec[0] if scalar else nouspec


class wabs(Multiplicative):
    """Wisconsin ISM photoelectric absorption (``angr`` abundances)."""

    def __init__(self):
        """Initialise with the single parameter :math:`N_H` in ``1e22 cm^-2`` units."""

        self.expr = 'wabs'
        self.comment = 'Wisconsin ISM absorption model'

        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)

        self.params = OrderedDict()
        self.params[r'$N_H$'] = Par(1, unif(1e-4, 50))

        self.xsect_prefix = docs_path + '/xsect'
        self.xsect_dir = self.xsect_prefix + '/xsect_wabs_angr.npz'
        self.xsect_data = np.load(self.xsect_dir)
        self.xsect_energy = self.xsect_data['energy']
        self.xsect_sigma = self.xsect_data['sigma']

    def func(self, E, T=None, O=None):  # noqa: E741
        r"""Return :math:`\exp(-N_H \, \sigma(E))` using the tabulated cross section."""

        redshift = self.config['redshift'].value
        nh = self.params[r'$N_H$'].value

        sigma = self._get_cached_sigma(E, redshift)

        fracspec = np.exp(-nh * sigma)

        return fracspec

    @memoized()
    def _get_cached_sigma(self, E, redshift):
        """Interpolate the rest-frame cross section at redshift-corrected ``E``."""

        E = np.asarray(E, dtype=np.float64)
        scalar = E.ndim == 0
        if scalar:
            E = E[np.newaxis]

        zi = 1 + redshift
        E = E * zi

        sigma = np.interp(E, self.xsect_energy, self.xsect_sigma, right=0.0)

        return sigma[0] if scalar else sigma


class phabs(Multiplicative):
    """Photoelectric absorption using the ``aspl`` abundance table."""

    def __init__(self):
        """Initialise with the single parameter :math:`N_H` in ``1e22 cm^-2`` units."""

        self.expr = 'phabs'
        self.comment = 'photoelectric absorption model'

        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)

        self.params = OrderedDict()
        self.params[r'$N_H$'] = Par(1, unif(1e-4, 50))

        self.xsect_prefix = docs_path + '/xsect'
        self.xsect_dir = self.xsect_prefix + '/xsect_phabs_aspl.npz'
        self.xsect_data = np.load(self.xsect_dir)
        self.xsect_energy = self.xsect_data['energy']
        self.xsect_sigma = self.xsect_data['sigma']

    def func(self, E, T=None, O=None):  # noqa: E741
        r"""Return :math:`\exp(-N_H \, \sigma(E))` using the tabulated cross section."""

        redshift = self.config['redshift'].value
        nh = self.params[r'$N_H$'].value

        sigma = self._get_cached_sigma(E, redshift)

        fracspec = np.exp(-nh * sigma)

        return fracspec

    @memoized()
    def _get_cached_sigma(self, E, redshift):
        """Interpolate the rest-frame cross section at redshift-corrected ``E``."""

        E = np.asarray(E, dtype=np.float64)
        scalar = E.ndim == 0
        if scalar:
            E = E[np.newaxis]

        zi = 1 + redshift
        E = E * zi

        sigma = np.interp(E, self.xsect_energy, self.xsect_sigma, right=0.0)

        return sigma[0] if scalar else sigma


class tbabs(Multiplicative):
    """Tuebingen-Boulder ISM absorption using the ``wilm`` abundance table."""

    def __init__(self):
        """Initialise with the single parameter :math:`N_H` in ``1e22 cm^-2`` units."""

        self.expr = 'tbabs'
        self.comment = 'Tuebingen-Boulder ISM absorption model'

        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)

        self.params = OrderedDict()
        self.params[r'$N_H$'] = Par(1, unif(1e-4, 50))

        self.xsect_prefix = docs_path + '/xsect'
        self.xsect_dir = self.xsect_prefix + '/xsect_tbabs_wilm.npz'
        self.xsect_data = np.load(self.xsect_dir)
        self.xsect_energy = self.xsect_data['energy']
        self.xsect_sigma = self.xsect_data['sigma']

    def func(self, E, T=None, O=None):  # noqa: E741
        r"""Return :math:`\exp(-N_H \, \sigma(E))` using the tabulated cross section."""

        redshift = self.config['redshift'].value
        nh = self.params[r'$N_H$'].value

        sigma = self._get_cached_sigma(E, redshift)

        fracspec = np.exp(-nh * sigma)

        return fracspec

    @memoized()
    def _get_cached_sigma(self, E, redshift):
        """Interpolate the rest-frame cross section at redshift-corrected ``E``."""

        E = np.asarray(E, dtype=np.float64)
        scalar = E.ndim == 0
        if scalar:
            E = E[np.newaxis]

        zi = 1 + redshift
        E = E * zi

        sigma = np.interp(E, self.xsect_energy, self.xsect_sigma, right=0.0)

        return sigma[0] if scalar else sigma


class redden(Multiplicative):
    """Galactic IR/optical/UV dust extinction using CCM89."""

    _HC_KEV_ANGSTROM = 12.398419843320026

    def __init__(self):
        """Initialise with colour excess :math:`E(B-V)`."""

        self.expr = 'redden'
        self.comment = 'Galactic dust extinction model'

        self.config = OrderedDict()
        self.config[r'$R_V$'] = Cfg(3.1)
        self.config['lyman_limit'] = Cfg(912.0)

        self.params = OrderedDict()
        self.params[r'$E(B-V)$'] = Par(0.1, unif(0.0, 5.0))

    def func(self, E, T=None, O=None):  # noqa: E741
        """Return the CCM89 dust transmission at observed-frame energies ``E``."""

        rv = self.config[r'$R_V$'].value
        lyman_limit = self.config['lyman_limit'].value

        ebv = self.params[r'$E(B-V)$'].value

        E = np.asarray(E, dtype=np.float64)
        scalar = E.ndim == 0
        if scalar:
            E = E[np.newaxis]

        if ebv < 0 or rv <= 0 or lyman_limit <= 0:
            fracspec = np.full_like(E, np.nan, dtype=float)
            return fracspec[0] if scalar else fracspec

        lam_angstrom = redden._HC_KEV_ANGSTROM / E
        above_lyman_limit = lam_angstrom >= lyman_limit

        fracspec = np.ones_like(E, dtype=float)
        if not np.any(above_lyman_limit):
            return fracspec[0] if scalar else fracspec

        a, b = self._get_cached_ccm89_ab(lam_angstrom[above_lyman_limit])
        a_lambda = ebv * (rv * a + b)
        fracspec[above_lyman_limit] = 10.0 ** (-0.4 * a_lambda)

        return fracspec[0] if scalar else fracspec

    @memoized()
    def _get_cached_ccm89_ab(self, lam_angstrom):
        """Return CCM89 ``a``/``b`` coefficients for wavelengths in Angstrom."""

        inverse_micron = 1.0e4 / lam_angstrom
        inverse_micron = np.minimum(inverse_micron, 11.0)

        a = np.zeros_like(inverse_micron)
        b = np.zeros_like(inverse_micron)

        ir = inverse_micron < 1.1
        optical = (inverse_micron >= 1.1) & (inverse_micron < 3.3)
        uv = (inverse_micron >= 3.3) & (inverse_micron < 8.0)
        far_uv = inverse_micron >= 8.0

        a[ir] = 0.574 * inverse_micron[ir] ** 1.61
        b[ir] = -0.527 * inverse_micron[ir] ** 1.61

        optical_offset = inverse_micron[optical] - 1.82
        a[optical] = (
            1.0
            + 0.17699 * optical_offset
            - 0.50447 * optical_offset**2
            - 0.02427 * optical_offset**3
            + 0.72085 * optical_offset**4
            + 0.01979 * optical_offset**5
            - 0.77530 * optical_offset**6
            + 0.32999 * optical_offset**7
        )
        b[optical] = (
            1.41338 * optical_offset
            + 2.28305 * optical_offset**2
            + 1.07233 * optical_offset**3
            - 5.38434 * optical_offset**4
            - 0.62251 * optical_offset**5
            + 5.30260 * optical_offset**6
            - 2.09002 * optical_offset**7
        )

        uv_inverse_micron = inverse_micron[uv]
        uv_curvature_a = np.zeros_like(uv_inverse_micron)
        uv_curvature_b = np.zeros_like(uv_inverse_micron)
        far_uv_curvature = uv_inverse_micron >= 5.9
        uv_curvature_offset = uv_inverse_micron[far_uv_curvature] - 5.9
        uv_curvature_a[far_uv_curvature] = (
            -0.04473 * uv_curvature_offset**2 - 0.009779 * uv_curvature_offset**3
        )
        uv_curvature_b[far_uv_curvature] = (
            0.2130 * uv_curvature_offset**2 + 0.1207 * uv_curvature_offset**3
        )
        a[uv] = (
            1.752
            - 0.316 * uv_inverse_micron
            - 0.104 / ((uv_inverse_micron - 4.67) ** 2 + 0.341)
            + uv_curvature_a
        )
        b[uv] = (
            -3.090
            + 1.825 * uv_inverse_micron
            + 1.206 / ((uv_inverse_micron - 4.62) ** 2 + 0.263)
            + uv_curvature_b
        )

        far_uv_offset = inverse_micron[far_uv] - 8.0
        a[far_uv] = (
            -1.073 - 0.628 * far_uv_offset + 0.137 * far_uv_offset**2 - 0.070 * far_uv_offset**3
        )
        b[far_uv] = (
            13.670 + 4.257 * far_uv_offset - 0.420 * far_uv_offset**2 + 0.374 * far_uv_offset**3
        )

        return a, b


class zdust(Multiplicative):
    """Redshifted MW/LMC/SMC dust extinction using Pei92."""

    _HC_KEV_ANGSTROM = 12.398419843320026
    _ALLOWED_METHOD: ClassVar = ['MW', 'LMC', 'SMC']
    _STANDARD_RV: ClassVar = {'MW': 3.08, 'LMC': 3.16, 'SMC': 2.93}
    _PEI92_CURVES: ClassVar = {
        'MW': {
            'a': np.array([165.0, 14.0, 0.045, 0.002, 0.002, 0.012]),
            'lamb': np.array([0.047, 0.08, 0.22, 9.7, 18.0, 25.0]),
            'b': np.array([90.0, 4.0, -1.95, -1.95, -1.80, 0.0]),
            'n': np.array([2.0, 6.5, 2.0, 2.0, 2.0, 2.0]),
        },
        'LMC': {
            'a': np.array([175.0, 19.0, 0.023, 0.005, 0.006, 0.020]),
            'lamb': np.array([0.046, 0.08, 0.22, 9.7, 18.0, 25.0]),
            'b': np.array([90.0, 5.5, -1.95, -1.95, -1.80, 0.0]),
            'n': np.array([2.0, 4.5, 2.0, 2.0, 2.0, 2.0]),
        },
        'SMC': {
            'a': np.array([185.0, 27.0, 0.005, 0.010, 0.012, 0.030]),
            'lamb': np.array([0.042, 0.08, 0.22, 9.7, 18.0, 25.0]),
            'b': np.array([90.0, 5.5, -1.95, -1.95, -1.80, 0.0]),
            'n': np.array([2.0, 4.0, 2.0, 2.0, 2.0, 2.0]),
        },
    }

    def __init__(self):
        """Initialise with host-frame colour excess and redshifted dust law."""

        self.expr = 'zdust'
        self.comment = 'redshifted dust extinction model'

        self.config = OrderedDict()
        self.config['method'] = Cfg('MW')
        self.config['redshift'] = Cfg(0.0)
        self.config[r'$R_V$'] = Cfg(None)
        self.config['lyman_limit'] = Cfg(912.0)

        self.params = OrderedDict()
        self.params[r'$E(B-V)$'] = Par(0.1, unif(0.0, 100.0))

    def func(self, E, T=None, O=None):  # noqa: E741
        """Return Pei92 dust transmission at observed-frame energies ``E``."""

        method = str(self.config['method'].value).upper()
        redshift = self.config['redshift'].value
        rv = self.config[r'$R_V$'].value
        lyman_limit = self.config['lyman_limit'].value

        ebv = self.params[r'$E(B-V)$'].value

        E = np.asarray(E, dtype=np.float64)
        scalar = E.ndim == 0
        if scalar:
            E = E[np.newaxis]

        if method not in self._ALLOWED_METHOD:
            msg = f'method should be one of {self._ALLOWED_METHOD}, got {method!r}'
            raise ValueError(msg)

        rv = self._STANDARD_RV[method] if rv is None else rv

        if ebv < 0 or rv <= 0 or redshift < 0 or lyman_limit <= 0:
            fracspec = np.full_like(E, np.nan, dtype=float)
            return fracspec[0] if scalar else fracspec

        zi = 1 + redshift
        E = E * zi
        lam_angstrom = zdust._HC_KEV_ANGSTROM / E
        above_lyman_limit = lam_angstrom >= lyman_limit

        fracspec = np.ones_like(E, dtype=float)
        if not np.any(above_lyman_limit):
            return fracspec[0] if scalar else fracspec

        xi = self._get_cached_pei92_xi(lam_angstrom[above_lyman_limit], method)
        a_b = ebv * (1.0 + rv)
        a_lambda = a_b * xi
        fracspec[above_lyman_limit] = 10.0 ** (-0.4 * a_lambda)

        return fracspec[0] if scalar else fracspec

    @memoized()
    def _get_cached_pei92_xi(self, lam_angstrom, method):
        """Return Pei92 normalized extinction curve for wavelengths in Angstrom."""

        curve = self._PEI92_CURVES[method]
        lam_micron = lam_angstrom / 1.0e4
        lam_ratio = lam_micron[:, np.newaxis] / curve['lamb']

        term = lam_ratio ** curve['n']
        inverse_term = 1.0 / term
        denominator = term + inverse_term + curve['b']

        return np.sum(curve['a'] / denominator, axis=1)


class tinvabs(Multiplicative):
    r"""Exponentially-decaying absorption column :math:`N_H(T) = N_{H,0} e^{-T/\tau}`.

    Delegates the energy-dependent absorption to an inner :class:`tbabs`
    while updating its :math:`N_H` per unique ``T`` value.
    """

    def __init__(self):
        r"""Initialise with initial column :math:`N_{H,0}` and decay time :math:`\tau`."""

        self.expr = 'tinvabs'
        self.comment = 'time-involved absorption model'
        self.tbabs = tbabs()

        self.config = OrderedDict()
        self.config['redshift'] = Cfg(0.0)

        self.params = OrderedDict()
        self.params[r'$N_{H,0}$'] = Par(1, unif(1e-4, 50))
        self.params[r'$\tau$'] = Par(50, unif(0, 300))

    def func(self, E, T, O=None):  # noqa: E741
        """Absorb ``E`` by a column that decays with ``T``.

        Args:
            E: Energies in keV.
            T: Times; ``E`` and ``T`` must match in scalar-ness and shape.
            O: Unused.

        Returns:
            Per-sample attenuation factor.

        Raises:
            ValueError: If ``E`` and ``T`` shapes disagree.
        """

        redshift = self.config['redshift'].value

        NH0 = self.params[r'$N_{H,0}$'].value
        tau = self.params[r'$\tau$'].value

        self.tbabs.parameters['redshift'].value = redshift

        E = np.asarray(E)
        E_scalar = E.ndim == 0
        if E_scalar:
            E = E[np.newaxis]

        T = np.asarray(T)
        T_scalar = T.ndim == 0
        if T_scalar:
            T = T[np.newaxis]

        if E_scalar == T_scalar:
            if E.shape != T.shape:
                raise ValueError('E and T must have the same shape')
            else:
                scalar = E_scalar
        else:
            raise ValueError('E and T must both be scalars or both be arrays')

        fracspec = np.zeros_like(E, dtype=float)

        for Ti in set(T):
            idx = np.where(Ti == T)[0]
            NH = NH0 * np.exp(-Ti / tau)

            if NH < 1e-4:
                return np.nan if scalar else np.ones_like(E) * np.nan

            self.tbabs.params[r'$N_H$'].value = NH

            res = self.tbabs(np.array(E, dtype=float))
            fracspec[idx] = res

        return fracspec[0] if scalar else fracspec
