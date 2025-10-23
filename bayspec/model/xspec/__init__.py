import numpy as np
import xspec_models_cxc as xsp
from collections import OrderedDict

from ..model import Model
from ...util.prior import unif
from ...util.param import Par, Cfg


__all__ = []

add_model_list = xsp.list_models(modeltype=xsp.ModelType.Add)
add_model_classes = {name: getattr(xsp, name) for name in add_model_list}
add_model_types = {name: 'add' for name in add_model_list}

mul_model_list = xsp.list_models(modeltype=xsp.ModelType.Mul)
mul_model_classes = {name: getattr(xsp, name) for name in mul_model_list}
mul_model_types = {name: 'mul' for name in mul_model_list}

conv_model_list = xsp.list_models(modeltype=xsp.ModelType.Con)
conv_model_classes = {name: getattr(xsp, name) for name in conv_model_list}
conv_model_types = {name: 'conv' for name in conv_model_list}

model_classes = dict(**add_model_classes, **mul_model_classes)
model_types = dict(**add_model_types, **mul_model_types)


def list_xspec_models():
    return [f'XS_{name}' for name in model_classes.keys()]
__all__.append('list_xspec_models')


def chatter(val=None):
    if val is None:
        return xsp.chatter()
    else:
        xsp.chatter(val)
__all__.append('chatter')


def abund(val=None):
    allowed_abund = ['angr', 'aspl', 'feld', 'aneb', 'grsa', 'wilm', 'lodd', 'lpgp']
    if val is None:
        return xsp.abundance()
    else:
        if val not in allowed_abund:
            msg = f'{val} is not allowed abundance'
            raise ValueError(msg)
        xsp.abundance(val)
__all__.append('abund')


def xsect(val=None):
    allowed_xsect = ['bcmc', 'obcm', 'vern']
    if val is None:
        return xsp.cross_section()
    else:
        if val not in allowed_xsect:
            msg = f'{val} is not allowed cross section'
            raise ValueError(msg)
        xsp.cross_section(val)
__all__.append('xsect')


def cosmo(val=None):
    allowed_key = ['h0', 'lambda0', 'q0']
    if val is None:
        return xsp.cosmology()
    else:
        if set(val.keys) != set(allowed_key):
            msg = f'{val} should be dict with keys h0, lambda0 and q0'
            raise ValueError(msg)
        xsp.cosmology(**val)
__all__.append('cosmo')


for name, cls in model_classes.items():
    
    def make_init(name, cls):
        
        def __init__(self):
            
            self.xsexpr = name
            self.expr = f'XS_{name}'
            self.type = model_types[name]
            self.comment = f'xspec model {name}'
            
            self.xsmodel = cls

            self.params = OrderedDict()
            self.config = OrderedDict()
            
            self.xsmodel_plabels = []
            for pr in xsp.info(name).parameters:
                pl = pr.name
                value = pr.default
                min_value = pr.softmin
                max_value = pr.softmax
                
                self.xsmodel_plabels.append(pl)
                if pl == 'Redshift':
                    self.config['redshift'] = Cfg(value)
                else:
                    self.params[pl] = Par(value, unif(min_value, max_value), frozen=pr.frozen)

            if self.type == 'add':
                self.params['logNorm'] = Par(0, unif(-10, 10))
                    
            if 'Redshift' not in self.xsmodel_plabels:
                self.config['redshift'] = Cfg(0)
                
        return __init__

    
    def func(self, E, T=None, O=None):
        
        pars = []
        for pr in xsp.info(self.xsexpr).parameters:
            pl = pr.name
            if pl == 'Redshift':
                pars.append(self.config['redshift'].value)
            else:
                pars.append(self.params[pl].value)
            
        if self.type == 'add':
            norm = 10 ** self.params['logNorm'].value
        else:
            norm = 1.0
                
        if 'Redshift' not in self.xsmodel_plabels:
            redshift = self.config['redshift'].value
            zi = 1 + redshift
            E = E * zi
            
        scale = 1e5
        ediff = E.reshape(-1, 1) * [1 - 1 / scale, 1, 1 + 1 / scale]
        integ = lambda egrid: np.mean(self.xsmodel(pars=pars, energies=egrid))
        
        if self.type == 'add':
            xsres = np.array(list(map(integ, ediff))) / (E / scale)
        elif self.type == 'mul':
            xsres = np.array(list(map(integ, ediff)))

        return norm * xsres

    new_class = type(f'XS_{name}', (Model,), {'__init__': make_init(name, cls), 'func': func})
    
    globals()[f'XS_{name}'] = new_class
    
    __all__.append(f'XS_{name}')


xspec_models = {name: cls for name, cls in globals().items() 
                if isinstance(cls, type) 
                and issubclass(cls, Model) 
                and name[:2] == 'XS'}
__all__.append('xspec_models')
