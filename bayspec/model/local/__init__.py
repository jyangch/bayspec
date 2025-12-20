from .additive import *
from .tinvolved import *
from .mathematic import *
from .multiplicative import *
from ..model import Model


local_models = {name: cls for name, cls in globals().items() 
                if isinstance(cls, type) 
                and issubclass(cls, Model)
                and name not in ['Model', 'Additive', 'Tinvolved', 'Multiplicative', 'Mathematic']}


def list_local_models():
    return list(local_models.keys())

__all__ = list(local_models.keys()) + ['list_local_models', 'local_models']
