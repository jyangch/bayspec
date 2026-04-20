"""Template for user-defined spectral models.

Copy this file and customize the ``+++++++editable area+++++++`` blocks
to define new components; ``expr``, ``type``, ``config`` and ``params``
control how the model participates in composition and fitting.
"""

from collections import OrderedDict

from ..model import Model
from ...util.prior import unif
from ...util.param import Par, Cfg

#+++++++editable area+++++++
# import other package
#+++++++editable area+++++++


class user(Model):
    """Example user-defined model scaffold; edit to implement your spectrum."""

    def __init__(self):
        """Initialise the template with a single uniform parameter ``p1``."""
        super().__init__()

        #+++++++editable area+++++++
        self.expr = 'model'
        self.type = 'add'  # 'add' | 'mul' | 'tinv' | 'math'
        self.comment = 'user-defined model'
        #+++++++editable area+++++++

        self.config = OrderedDict()
        #+++++++editable area+++++++
        self.config['redshift'] = Cfg(0)
        #+++++++editable area+++++++

        self.params = OrderedDict()
        #+++++++editable area+++++++
        self.params['p1'] = Par(1, unif(0, 2))
        #+++++++editable area+++++++


    def func(self, E, T=None, O=None):
        """Evaluate the model spectrum at energies ``E``.

        Args:
            E: Energy grid in keV; redshift correction is applied internally.
            T: Time array in seconds; required for ``tinv``-type models.
            O: Nested model passed by a convolution operator; unused here.

        Returns:
            Photon spectrum N(E, T) in photons/cm²/s/keV for ``add``/``tinv``
            types, or dimensionless F(E, T) for ``mul``/``math`` types.
        """
        
        #+++++++editable area+++++++
        redshift = self.config['redshift'].value
        #+++++++editable area+++++++

        #+++++++editable area+++++++
        p1 = self.params['p1'].value
        #+++++++editable area+++++++

        zi = 1 + redshift
        E = E * zi

        #+++++++editable area+++++++
        phtspec = E ** p1 + T
        #+++++++editable area+++++++

        return phtspec