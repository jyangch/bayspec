"""Dimensionless mathematical components, evaluated independently of energy."""

from collections import OrderedDict

from ..model import Mathematic
from ...util.param import Par
from ...util.prior import unif



class const(Mathematic):
    """Scalar constant with a single free parameter ``C``."""

    def __init__(self):
        """Initialise the constant with ``C`` uniform on ``[-10, 10]``."""

        self.expr = 'const'
        self.comment = 'constant model'

        self.config = OrderedDict()

        self.params = OrderedDict()
        self.params[r'$C$'] = Par(0, unif(-10, 10))


    def func(self, E, T=None, O=None):
        """Return the current value of ``C`` regardless of ``E``, ``T``, ``O``."""

        C = self.params[r'$C$'].value

        return C
