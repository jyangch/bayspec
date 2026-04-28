"""Small utilities shared across the package.

Hosts the numpy-aware JSON encoder, the index-plus-key dictionary, the
dependency-aware memoization decorators, and the numba-accelerated 1D/2D
trapezoidal integrators used in model evaluation.
"""

import json
import hashlib
import functools
import collections
import numpy as np
import numba as nb
from io import BytesIO
from pathlib import Path
from itertools import islice
from datetime import datetime, date
from collections import OrderedDict



class JsonEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy, set, datetime, and ``todict``-ables.

    Falls back to the default encoder for anything else, so ``TypeError``
    is still raised on unsupported objects.
    """

    def default(self, obj):
        """Serialize numpy scalars/arrays, sets, dates, ``todict``-ables, and ``BytesIO``."""

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
    """Write ``data`` to ``filepath`` as JSON using :class:`JsonEncoder`.

    Creates missing parent directories. Uses UTF-8 and leaves non-ASCII
    characters intact by default.

    Args:
        data: Serializable payload; may contain numpy and datetime values.
        filepath: Target path; parents are created if absent.
        indent: Indentation width for pretty-printing.
        ensure_ascii: When ``True``, escape non-ASCII characters.
    """

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, cls=JsonEncoder)


class SuperDict(OrderedDict):
    """``OrderedDict`` that also supports 1-based positional indexing.

    Integer keys are interpreted as ordinal positions; every other key
    type falls through to the underlying dictionary.
    """

    def __getitem__(self, key):
        """Look up by ordinal position when ``key`` is an ``int``, else by key.

        Raises:
            IndexError: If an integer ``key`` is outside ``[1, len(self)]``.
        """

        if isinstance(key, int):
            real_index = key - 1

            if real_index < 0 or real_index >= len(self):
                raise IndexError("index out of range")

            actual_key = next(islice(self.keys(), real_index, None))

            return super().__getitem__(actual_key)

        return super().__getitem__(key)


_WITH_MEMOIZATION = True
_DEFAULT_CACHE_SIZE = 10
_CACHE_ATTR_PREFIX = "_memoized_"

def get_fingerprint(x):
    """Recursively build a hashable fingerprint of ``x``.

    Used as cache-key material for :func:`memoized`. The output is a
    nested tuple containing only hashable leaves, so it can be used
    directly as a ``dict`` key.

    Handling by type:
        - ``np.ndarray``: ``(tag, shape, dtype, blake2b(content))``.
          Content hashing means identical buffers hit the cache and
          in-place modifications correctly invalidate it.
        - ``list`` / ``tuple``: recursed element-wise; the container
          type is preserved in the tag so a list and a tuple with the
          same items produce different fingerprints.
        - ``dict``: recursed value-wise and sorted by key, so key
          insertion order does not affect the fingerprint.
        - Anything else: returned as-is. The caller is responsible for
          ensuring it is hashable; unhashable values will raise when
          the fingerprint is used as a key.

    Args:
        x: Any object to fingerprint.

    Returns:
        A hashable (possibly nested) structure uniquely identifying
        ``x`` for caching purposes.
    """
    
    if isinstance(x, np.ndarray):
        return (
            "ndarray",
            x.shape,
            x.dtype.str,
            hashlib.blake2b(x.tobytes(), digest_size=16).digest()
        )
        
    if isinstance(x, (list, tuple)):
        return (
            type(x).__name__, 
            tuple(get_fingerprint(i) for i in x)
        )
        
    if isinstance(x, dict):
        return (
            "dict",
            tuple(sorted((k, get_fingerprint(v)) for k, v in x.items())),
        )
        
    return x


def memoized(dep_getter=None, *, cache_size=None, verbose=False):
    """Method-memoization decorator keyed on arguments and a dependency value.

    Each decorated method gets a per-instance bounded LRU cache keyed on
    a fingerprint of ``dep_getter(self)``, the positional arguments, and
    the keyword arguments. Numpy arrays are fingerprinted by content hash
    (BLAKE2b) along with shape and dtype, so identical contents hit the
    cache and in-place modifications correctly invalidate it.

    Args:
        dep_getter: Callable mapping ``self`` to the dependency value.
            When ``None``, dependencies are ignored.
        cache_size: Max entries per instance. ``None`` uses the global
            default (``_DEFAULT_CACHE_SIZE``).
        verbose: When ``True``, print one line on every hit or miss.

    Returns:
        A decorator that wraps a method with memoization.
    """

    if dep_getter is None:
        dep_getter = lambda self: None
        
    max_cache_size = cache_size if cache_size is not None else _DEFAULT_CACHE_SIZE

    def decorator(func):
        
        cache_attr = f"{_CACHE_ATTR_PREFIX}{func.__name__}"

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            if not _WITH_MEMOIZATION:
                return func(self, *args, **kwargs)
            
            fingerprint = (
                get_fingerprint(dep_getter(self)),
                tuple(get_fingerprint(a) for a in args),
                tuple(sorted((k, get_fingerprint(v)) for k, v in kwargs.items())),
            )
            
            cache = getattr(self, cache_attr, None)
            if cache is None:
                cache = collections.OrderedDict()
                setattr(self, cache_attr, cache)

            if fingerprint in cache:
                if verbose:
                    print(f"[{func.__name__}] hit")
                cache.move_to_end(fingerprint)
                return cache[fingerprint]
            
            if verbose:
                print(f"[{func.__name__}] recompute")
            result = func(self, *args, **kwargs)
            
            cache[fingerprint] = result
            if len(cache) > max_cache_size:
                cache.popitem(last=False)
                
            return result

        return wrapper

    return decorator


def clear_memoized(obj, *names):
    """Drop :func:`memoized` caches from ``obj``.

    Args:
        obj: Instance whose caches should be cleared.
        *names: Method names to clear; clears every memoized method when
            empty.
    """

    if names:
        for name in names:
            attr = f"{_CACHE_ATTR_PREFIX}{name}"
            if hasattr(obj, attr):
                delattr(obj, attr)
    else:
        for attr in list(vars(obj).keys()):
            if attr.startswith(_CACHE_ATTR_PREFIX):
                delattr(obj, attr)


def cached_property(dep_getter=None, *, verbose=False):
    """Per-instance cached-property decorator with optional dependency tracking.

    On each access, ``dep_getter(self)`` is reduced to a fingerprint via
    :func:`get_fingerprint` (numpy arrays are content-hashed by BLAKE2b
    along with shape and dtype). When the fingerprint differs from the
    last observed one, the cache is invalidated and the method is re-run.

    Cache state lives on the instance as ``_cached_<name>`` and
    ``_cached_dep_<name>``; drop it with :func:`clear_cached_property`.

    Args:
        dep_getter: Callable mapping ``self`` to the dependency value.
            When ``None``, the property caches forever after the first
            access.
        verbose: When ``True``, print one line on every hit or miss.

    Returns:
        A ``property`` whose getter memoizes the underlying method.
    """

    if dep_getter is None:
        dep_getter = lambda self: None
        
    def decorator(func):
        
        _MISSING = object()
        
        cache_attr = f"_cached_{func.__name__}"
        dep_attr = f"_cached_dep_{func.__name__}"
        
        @property
        @functools.wraps(func)
        def wrapper(self):
            current_dep = get_fingerprint(dep_getter(self))
            last_dep = getattr(self, dep_attr, _MISSING)
            
            if last_dep is _MISSING or last_dep != current_dep:
                if verbose:
                    print(f"[{func.__name__}] recompute")
                value = func(self)
                setattr(self, cache_attr, value)
                setattr(self, dep_attr, current_dep)
            elif verbose:
                print(f"[{func.__name__}] cache hit")

            return getattr(self, cache_attr)

        return wrapper

    return decorator


def clear_cached_property(obj, *names):
    """Drop :func:`cached_property` caches from ``obj``.

    Args:
        obj: Instance whose caches should be cleared.
        *names: Property names to clear; clears every cached property
            when empty.
    """

    if names:
        for name in names:
            for attr in (f"_cached_{name}", f"_cached_dep_{name}"):
                if hasattr(obj, attr):
                    delattr(obj, attr)
    else:
        for attr in list(vars(obj).keys()):
            if attr.startswith("_cached_"):
                delattr(obj, attr)


@nb.njit(fastmath=True, cache=True)
def trapz_1d(y, x):
    """Integrate ``y`` over ``x`` with the trapezoidal rule (numba-accelerated).

    Args:
        y: 1D array of integrand values.
        x: 1D array of sample points; must match ``y`` in length.

    Returns:
        The trapezoidal integral of ``y`` over ``x``.
    """

    acc = 0.0
    for i in range(len(y) - 1):
        acc += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])

    return acc


@nb.njit(fastmath=True, cache=True)
def trapz_2d(y, x):
    """Row-wise trapezoidal integration of a 2D array (numba-accelerated).

    Args:
        y: 2D array; integration runs along columns for each row.
        x: 2D array of matching shape holding the sample points.

    Returns:
        1D array of row integrals, length equal to ``y.shape[0]``.
    """

    nrow, ncol = y.shape
    out = np.empty(nrow, dtype=np.float64)

    for i in range(nrow):
        acc = 0.0
        for j in range(ncol - 1):
            acc += 0.5 * (y[i, j] + y[i, j + 1]) * (x[i, j + 1] - x[i, j])
        out[i] = acc

    return out
