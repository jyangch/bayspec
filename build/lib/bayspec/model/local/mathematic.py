from ...util.param import Par
from ..model import Mathematic
from ...util.prior import unif
from collections import OrderedDict



class Const(Mathematic):

    def __init__(self):
        super().__init__()
        
        self.expr = 'const'
        self.comment = 'constant model'

        self.params = OrderedDict()
        self.params[r'$C$'] = Par(0, unif(-10, 10))
        
    
    def func(self, E, T=None, O=None):
        C = self.params[r'$C$'].value
        return C
