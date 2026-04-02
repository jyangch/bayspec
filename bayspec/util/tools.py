import json
import numpy as np
import numba as nb
from io import BytesIO
from pathlib import Path
from itertools import islice
from datetime import datetime, date
from collections import OrderedDict



class JsonEncoder(json.JSONEncoder):
    """
    Custom JSON encoder optimized for numpy data types and other common types.
    """
    
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, set):
            return list(obj)

        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        if hasattr(obj, 'todict') and callable(obj.todict):
            return obj.todict()
        
        if isinstance(obj, BytesIO):
            return obj.name

        return super().default(obj)



def json_dump(data, filepath, indent=4, ensure_ascii=False):
    """
    Dump data to a JSON file with support for numpy data types and datetime objects.
    
    Parameters:
    ----------
    data : any
        The data to be dumped to JSON. Can include numpy data types and datetime objects.
    filepath : str or Path
        The path to the JSON file where the data will be saved.
    indent : int, optional
        The number of spaces to use for indentation in the JSON file. Default is 4.
    ensure_ascii : bool, optional
        Whether to escape non-ASCII characters in the JSON output. Default is False. 
    """
    
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, cls=JsonEncoder)



class SuperDict(OrderedDict):
    """
    A dictionary that allows access to its elements using both keys and 1-based indices.
    """
    
    def __getitem__(self, key):
        
        if isinstance(key, int):
            real_index = key - 1
            
            if real_index < 0 or real_index >= len(self):
                raise IndexError("index out of range")
            
            actual_key = next(islice(self.keys(), real_index, None))
            
            return super().__getitem__(actual_key)
            
        return super().__getitem__(key)



_MISSING = object()

def cached_property(dep_getter=None, *, verbose=False):
    """
    A decorator that converts a method into a cached property. 
    
    Parameters:
    ----------
    dep_getter : callable, optional
        A function that takes the instance as an argument and returns a value representing 
        the dependencies of the property. If the returned value changes, the cached value will 
        be recomputed. If not provided, the property will be cached without any dependency tracking.
    verbose : bool, optional
        If True, print messages when the property is recomputed or when a cache hit occurs. 
        Default is False.
        
    Returns:
    -------
    property
        A property that caches its value and optionally tracks dependencies for cache invalidation.
    """

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
    """
    Clear cached properties of an object.

    Parameters:
    ----------
    obj : object
        The object whose cached properties are to be cleared.
    names : str, optional
        The names of the cached properties to clear. If not provided, all cached properties will be cleared.
    """

    if names:
        for name in names:
            for attr in (f"__cached_{name}", f"__cached_deps_{name}"):
                if hasattr(obj, attr):
                    delattr(obj, attr)
    else:
        for attr in list(vars(obj).keys()):
            if attr.startswith("__cached_"):
                delattr(obj, attr)



@nb.njit(fastmath=True, cache=True)
def jit_abs_eval(E, nh, redshift, xsect_energy, xsect_sigma):
    """
    JIT-compiled function to evaluate the absorption fraction.

    Parameters:
    ----------
    E : array_like
        The energy values.
    nh : float
        The column density.
    redshift : float
        The redshift value.
    xsect_energy : array_like
        The energy values for the cross-section.
    xsect_sigma : array_like
        The cross-section values.

    Returns:
    -------
    fracspec : ndarray
        The absorption fraction for each energy value.
    """
    
    n = len(E)
    fracspec = np.empty(n, dtype=np.float64)
    
    zi = 1 + redshift
    max_xsect_energy = xsect_energy[-1]
    
    for i in range(n):
        e_rest = E[i] * zi
        
        if e_rest > max_xsect_energy:
            sigma = 0.0
        else:
            sigma = np.interp(e_rest, xsect_energy, xsect_sigma)
            
        fracspec[i] = np.exp(-nh * sigma)
        
    return fracspec
