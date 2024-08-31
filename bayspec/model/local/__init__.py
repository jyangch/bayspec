from .additive import *
from .tinvolved import *
from .mathematic import *
from ..model import Model


model_classes = {name: cls for name, cls in globals().items() 
                 if isinstance(cls, type) 
                 and issubclass(cls, Model)
                 and name not in ['Model', 'Additive', 'Tinvolved', 'Multiplicative', 'Mathematic']}


def list_local_models():
    return list(model_classes.keys())

__all__ = list(model_classes.keys()) + ['list_local_models']
