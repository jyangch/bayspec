from collections import OrderedDict

from ..model import Model
from ...util.prior import unif
from ...util.param import Par, Cfg

#+++++++editable area+++++++
# import other package
#+++++++editable area+++++++


class user(Model):

    def __init__(self):
        super().__init__()
        
        #+++++++editable area+++++++
        # name of your model
        self.expr = 'model'
        # type of your model, optional
        # ('add', 'mul', 'tinv', 'math')
        self.type = 'add'
        # comment of your model
        self.comment = 'user-defined model'
        #+++++++editable area+++++++
        
        self.config = OrderedDict()
        #+++++++editable area+++++++
        # set your model configuration
        self.config['redshift'] = Cfg(0)
        # above sentence define redshift 
        # with value of 0
        #+++++++editable area+++++++
        
        self.params = OrderedDict()
        #+++++++editable area+++++++
        # set your model parameters
        self.params['p1'] = Par(1, unif(0, 2))
        # above sentence define a parameter p1 
        # with value of 1 
        # with prior of uniform within [0, 2]
        #+++++++editable area+++++++


    def func(self, E, T=None, O=None):
        """
        Parameters
        ----------
        E: energy array in keV
        T: time array in second
        O: useless now
        Returns
        -------
        if type is add or tinv:
        photon spectrum N(E, T) in photons/cm2/s/keV
        if type is mul or math:
        dimensionless F(E, T)
        """
        #+++++++editable area+++++++
        # get the value of model configuration
        redshift = self.config['redshift'].value
        #+++++++editable area+++++++
        
        #+++++++editable area+++++++
        # get the value of model parameter
        p1 = self.params['p1'].value
        #+++++++editable area+++++++

        # apply redshift correction
        zi = 1 + redshift
        E = E * zi

        #+++++++editable area+++++++
        # code your model
        phtspec = E ** p1 + T
        #+++++++editable area+++++++

        return phtspec