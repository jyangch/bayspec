import json
import numpy as np
from io import BytesIO
from collections import OrderedDict

from .param import Par


class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Par):
            return obj.todict()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, BytesIO):
            return obj.name
        else:
            return super(JsonEncoder, self).default(obj)



def json_dump(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, cls=JsonEncoder)



class SuperDict(OrderedDict):
    
    def __getitem__(self, key):
        
        if isinstance(key, int):
            if key < 1 or key > len(self):
                raise IndexError("index out of range")
            key = list(self.keys())[key - 1]
            
        return super().__getitem__(key)



_MISSING = object()

def cached_property(dep_getter=None, *, verbose=False):

    if dep_getter is None:
        dep_getter = lambda self: None

    def decorator(func):
        
        cache_attr = f"__cached_{func.__name__}"
        dep_attr = f"__cached_deps_{func.__name__}"

        @property
        def wrapper(self):
            current_dep = dep_getter(self)
            last_dep = getattr(self, dep_attr, _MISSING)

            if last_dep is _MISSING or last_dep != current_dep:
                if verbose:
                    print(f"[{func.__name__}] recompute (dep changed: {last_dep} -> {current_dep})")
                value = func(self)
                setattr(self, cache_attr, value)
                setattr(self, dep_attr, current_dep)
            else:
                if verbose:
                    print(f"[{func.__name__}] cache hit (dep={current_dep})")

            return getattr(self, cache_attr)

        return wrapper
    return decorator



def clear_cached_property(obj, *names):

    if names:
        for name in names:
            for attr in (f"__cached_{name}", f"__cached_deps_{name}"):
                if hasattr(obj, attr):
                    delattr(obj, attr)
    else:
        for attr in list(vars(obj).keys()):
            if attr.startswith("__cached_"):
                delattr(obj, attr)
