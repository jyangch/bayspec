import numpy as np
import astromodels
from collections import OrderedDict

from ..model import Model
from ...util.prior import unif
from ...util.param import Par, Cfg


__all__ = []

astromodel_dict = getattr(astromodels, 'list_functions')()

add_model_list = ['Blackbody', 
                  'ModifiedBlackbody', 
                  'NonDissipativePhotosphere', 
                  'NonDissipativePhotosphere_Deep', 
                  'Log_parabola', 
                  'Exponential_cutoff', 
                  'Powerlaw', 
                  'Powerlaw_flux', 
                  'Powerlaw_Eflux', 
                  'Cutoff_powerlaw', 
                  'Cutoff_powerlaw_Ep', 
                  'Inverse_cutoff_powerlaw', 
                  'Super_cutoff_powerlaw', 
                  'SmoothlyBrokenPowerLaw', 
                  'Broken_powerlaw', 
                  'Band', 
                  'Band_grbm', 
                  'Band_Calderone', 
                  'DoubleSmoothlyBrokenPowerlaw', 
                  'DMFitFunction', 
                  'DMSpectra']

add_model_classes = dict()
for name in add_model_list:
    try:
        add_model_classes[name] = getattr(astromodels, name)
    except AttributeError:
        pass
add_model_types = {name: 'add' for name in add_model_classes.keys()}

mul_model_list = ['PhAbs', 
                  'TbAbs', 
                  'WAbs', 
                  'EBLattenuation', 
                  'ZDust']

mul_model_classes = dict()
for name in mul_model_list:
    try:
        mul_model_classes[name] = getattr(astromodels, name)
    except AttributeError:
        pass
mul_model_types = {name: 'mul' for name in mul_model_classes.keys()}

math_model_list = ['Sin', 
                   'StepFunction', 
                   'StepFunctionUpper', 
                   'DiracDelta', 
                   'Line', 
                   'Quadratic', 
                   'Cubic', 
                   'Quartic']

math_model_classes = dict()
for name in math_model_list:
    try:
        math_model_classes[name] = getattr(astromodels, name)
    except AttributeError:
        pass
math_model_types = {name: 'math' for name in math_model_classes.keys()}

model_classes = dict(**add_model_classes, **mul_model_classes, **math_model_classes)
model_types = dict(**add_model_types, **mul_model_types, **math_model_types)


def list_astro_models():
    return [f'AS_{name}' for name in model_classes.keys()]
__all__.append('list_astro_models')


for name, cls in model_classes.items():
    
    def make_init(name, cls):
        
        def __init__(self):
            self.asexpr = name
            self.expr = f'AS_{name}'
            self.type = model_types[name]
            
            self.asmodel = cls()
            self.comment = self.asmodel.description
            
            self.params = OrderedDict()
            self.config = OrderedDict()
            
            for pl, pr in self.asmodel.parameters.items():
                if pr.free:
                    value = pr.value
                    min_value = pr.min_value
                    max_value = pr.max_value
                    if pr.is_normalization and pl != 'NH':
                        if min_value is None or min_value == 0: min_value = value / 1e5
                        if max_value is None: max_value = value * 1e5
                        self.params[f'log{pl}'] = Par(np.log10(value), unif(np.log10(min_value), np.log10(max_value)))
                    else:
                        if min_value is None: min_value = value / 10
                        if max_value is None: max_value = value * 10
                        if pl == 'NH':
                            self.params[pl] = Par(value, unif(min_value, 20))
                        else:
                            self.params[pl] = Par(value, unif(min_value, max_value))
                else:
                    value = pr.value
                    if pl == 'piv': pl = 'pivot_energy'
                    self.config[pl] = Cfg(value)
                    
            if 'redshift' not in self.asmodel.parameters:
                self.config['redshift'] = Cfg(0)
                
        return __init__

    
    def func(self, E, T=None, O=None):
        
        for pl, pr in self.asmodel.parameters.items():
            if pr.free:
                if pr.is_normalization and pl != 'NH':
                    pr.value = 10 ** self.params[f'log{pl}'].value
                else:
                    pr.value = self.params[pl].value
            else:
                if pl == 'piv': pl = 'pivot_energy'
                pr.value = self.config[pl].value
                
        if 'redshift' not in self.asmodel.parameters:
            redshift = self.config['redshift'].value
            zi = 1 + redshift
            E = E * zi
            
        asres = self.asmodel(np.array(E, dtype=float))
        return asres

    new_class = type(f'AS_{name}', (Model,), {'__init__': make_init(name, cls), 'func': func})
    
    globals()[f'AS_{name}'] = new_class
    
    __all__.append(f'AS_{name}')
    

astro_models = {name: cls for name, cls in globals().items() 
                if isinstance(cls, type) 
                and issubclass(cls, Model) 
                and name[:2] == 'AS'}
__all__.append('astro_models')
