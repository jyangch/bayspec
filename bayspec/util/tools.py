"""Small utilities shared across the package.

Hosts the numpy-aware JSON encoder, the index-plus-key dictionary, the
dependency-aware memoization decorators, and the numba-accelerated 1D/2D
trapezoidal integrators used in model evaluation.
"""

import json
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
_CACHE_SIZE = 10

def memoized(dep_getter=None, *, verbose=False):
    """Method-memoization decorator keyed on arguments and a dependency value.

    Each decorated method gets a per-instance bounded LRU cache keyed on
    a fingerprint of ``dep_getter(self)``, the positional arguments, and
    the keyword arguments. Numpy arrays are fingerprinted by ``id``,
    size, and min/max so identical buffers hit the cache.

    Args:
        dep_getter: Callable mapping ``self`` to the dependency value.
            When ``None``, dependencies are ignored.
        verbose: When ``True``, print one line on every hit or miss.

    Returns:
        A decorator that wraps a method with memoization.
    """

    if dep_getter is None:
        dep_getter = lambda self: None

    def decorator(func):

        cache_attr = f"_cache_{func.__name__}"

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            if not _WITH_MEMOIZATION:
                return func(self, *args, **kwargs)

            dep = dep_getter(self)
            dep_fprint = []
            if isinstance(dep, (list, tuple)):
                for d in dep:
                    if isinstance(d, np.ndarray):
                        dep_fprint.append((id(d), d.size, d.min(), d.max()))
                    else:
                        dep_fprint.append(d)
            elif isinstance(dep, np.ndarray):
                dep_fprint.append((id(dep), dep.size, dep.min(), dep.max()))
            else:
                dep_fprint.append(dep)

            arg_fprint = []
            for a in args:
                if isinstance(a, np.ndarray):
                    arg_fprint.append((id(a), a.size, a.min(), a.max()))
                else:
                    arg_fprint.append(a)

            fingerprint = hash((tuple(dep_fprint), tuple(arg_fprint), tuple(kwargs.items())))

            if not hasattr(self, cache_attr):
                setattr(self, cache_attr, collections.OrderedDict())
            cache = getattr(self, cache_attr)

            if fingerprint in cache:
                if verbose:
                    print(f"[{func.__name__}] hit")
                val = cache.pop(fingerprint)
                cache[fingerprint] = val
                return val

            if verbose:
                print(f"[{func.__name__}] recompute")
            result = func(self, *args, **kwargs)

            cache[fingerprint] = result
            if len(cache) > _CACHE_SIZE:
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
            attr = f"_cache_{name}"
            if hasattr(obj, attr):
                delattr(obj, attr)
    else:
        for attr in list(vars(obj).keys()):
            if attr.startswith("_cache_"):
                delattr(obj, attr)


_MISSING = object()

def cached_property(dep_getter=None, *, verbose=False):
    """Cached-property decorator with optional dependency tracking.

    The wrapped method is evaluated once and the result is cached; when
    ``dep_getter(self)`` returns a value that differs from the last
    observed one, the cache is invalidated and the method is re-run.

    Args:
        dep_getter: Callable mapping ``self`` to the dependency value.
            When ``None``, the property caches forever.
        verbose: When ``True``, print one line on every hit or miss.

    Returns:
        A ``property`` that memoizes the underlying method.
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
    """Drop :func:`cached_property` caches from ``obj``.

    Args:
        obj: Instance whose caches should be cleared.
        *names: Property names to clear; clears every cached property
            when empty.
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
