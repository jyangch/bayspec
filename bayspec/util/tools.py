"""Small utilities shared across the package.

Hosts the numpy-aware JSON encoder, the index-plus-key dictionary, the
dependency-aware memoization decorators, and the numba-accelerated 1D/2D
trapezoidal integrators used in model evaluation.
"""

import collections
from collections import OrderedDict
from collections.abc import Mapping
from datetime import date, datetime
import functools
import hashlib
import inspect
from io import BytesIO
from itertools import islice
import json
from numbers import Integral
from pathlib import Path
import warnings

from matplotlib import rcParams
import numba as nb
import numpy as np


def apply_plt_rcparams():
    """Apply the shared BaySpec matplotlib style (STIX serif fonts, PDF/PS-safe fonttype)."""

    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['STIX Two Text']
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.size'] = 12
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42


def apply_plt_legend(ax, inline_max=8, max_rows=40):
    """Keep the default in-axes legend when entries fit; once they exceed
    ``inline_max``, move the legend outside the right edge and wrap into
    extra columns (at most ``max_rows`` entries per column) so it never
    overflows onto the plot itself.
    """

    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return

    if len(labels) <= inline_max:
        ax.legend()
        return

    ncol = -(-len(labels) // max_rows)
    ax.legend(
        handles,
        labels,
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        ncol=ncol,
        fontsize='x-small',
        columnspacing=0.8,
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=0.0,
        frameon=False,
    )


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
                raise IndexError('index out of range')

            actual_key = next(islice(self.keys(), real_index, None))

            return super().__getitem__(actual_key)

        return super().__getitem__(key)


_WITH_MEMOIZATION = True
_DEFAULT_CACHE_SIZE = 10
_CACHE_ATTR_PREFIX = '_memoized_'


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
            'ndarray',
            x.shape,
            x.dtype.str,
            hashlib.blake2b(x.tobytes(), digest_size=16).digest(),
        )

    if isinstance(x, (list, tuple)):
        return (type(x).__name__, tuple(get_fingerprint(i) for i in x))

    if isinstance(x, dict):
        return (
            'dict',
            tuple(sorted((k, get_fingerprint(v)) for k, v in x.items())),
        )

    return x


def memoized(dep_getter=None, *, cache_size=None, verbose=False):
    """Method-memoization decorator keyed on arguments and a dependency value.

    Each decorated method gets a per-instance bounded LRU cache keyed on
    a fingerprint of ``dep_getter(self)`` and the arguments normalized by
    the method signature. Equivalent positional, keyword, and omitted-default
    calls therefore share one entry. Numpy arrays are fingerprinted by content
    hash (BLAKE2b) along with shape and dtype, so identical contents hit the
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

        def dep_getter(self):
            return None

    max_cache_size = cache_size if cache_size is not None else _DEFAULT_CACHE_SIZE

    def decorator(func):

        cache_attr = f'{_CACHE_ATTR_PREFIX}{func.__name__}'
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            if not _WITH_MEMOIZATION:
                return func(self, *args, **kwargs)

            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            call_fingerprint = tuple(
                (name, get_fingerprint(value)) for name, value in tuple(bound.arguments.items())[1:]
            )
            fingerprint = (get_fingerprint(dep_getter(self)), call_fingerprint)

            cache = getattr(self, cache_attr, None)
            if cache is None:
                cache = collections.OrderedDict()
                setattr(self, cache_attr, cache)

            if fingerprint in cache:
                if verbose:
                    print(f'[{func.__name__}] hit')
                cache.move_to_end(fingerprint)
                return cache[fingerprint]

            if verbose:
                print(f'[{func.__name__}] recompute')
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
            attr = f'{_CACHE_ATTR_PREFIX}{name}'
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

        def dep_getter(self):
            return None

    def decorator(func):

        _MISSING = object()

        cache_attr = f'_cached_{func.__name__}'
        dep_attr = f'_cached_dep_{func.__name__}'

        @property
        @functools.wraps(func)
        def wrapper(self):
            current_dep = get_fingerprint(dep_getter(self))
            last_dep = getattr(self, dep_attr, _MISSING)

            if last_dep is _MISSING or last_dep != current_dep:
                if verbose:
                    print(f'[{func.__name__}] recompute')
                value = func(self)
                setattr(self, cache_attr, value)
                setattr(self, dep_attr, current_dep)
            elif verbose:
                print(f'[{func.__name__}] cache hit')

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
            for attr in (f'_cached_{name}', f'_cached_dep_{name}'):
                if hasattr(obj, attr):
                    delattr(obj, attr)
    else:
        for attr in list(vars(obj).keys()):
            if attr.startswith('_cached_'):
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


def _ic_diagnostics(result, criterion, npoint):
    """Describe available predictive diagnostics without asserting convergence."""

    diagnostic = {'status': 'not_assessed', 'reasons': []}
    if criterion not in ('WAIC', 'LOOIC'):
        return diagnostic

    reasons = diagnostic['reasons']
    warning_flag = result.get('warning')
    incomplete = not isinstance(warning_flag, (bool, np.bool_))
    has_warning = bool(warning_flag) if not incomplete else False
    if incomplete:
        reasons.append('The criterion warning flag is missing or invalid.')
    if has_warning:
        reasons.append(f'{criterion} reports a diagnostic warning.')

    if criterion == 'LOOIC':
        masks = {}
        for name in ('nearly_constant', 'psis_failed'):
            mask = np.zeros(npoint, dtype=bool)
            if name in result:
                supplied = np.asarray(result[name])
                if supplied.shape == (npoint,) and supplied.dtype.kind == 'b':
                    mask = supplied
                else:
                    incomplete = True
                    reasons.append(f'{name} channel flags are invalid.')
            masks[name] = mask
            diagnostic[f'{name}_channels'] = np.flatnonzero(mask).tolist()
        if masks['nearly_constant'].any():
            reasons.append('Nearly constant channels used raw weights; no Pareto tail was fitted.')
        if masks['psis_failed'].any():
            has_warning = True
            reasons.append(
                'PSIS failed on nonconstant channels; raw-weight fallback is unreliable.'
            )

        try:
            pareto_k = np.asarray(result.get('pareto_k'), dtype=float)
            good_k = float(result.get('good_k'))
            k_metadata_valid = pareto_k.shape == (npoint,) and np.isfinite(good_k)
        except (TypeError, ValueError):
            k_metadata_valid = False
        if not k_metadata_valid:
            incomplete = True
            reasons.append('Pareto-k values or their diagnostic threshold are missing or invalid.')
        else:
            high_k = pareto_k > good_k
            unexplained_k = (np.isnan(pareto_k) | np.isneginf(pareto_k)) & ~(
                masks['nearly_constant'] | masks['psis_failed']
            )
            diagnostic['high_k_channels'] = np.flatnonzero(high_k).tolist()
            diagnostic['unexplained_k_channels'] = np.flatnonzero(unexplained_k).tolist()
            if high_k.any():
                has_warning = True
                reasons.append('Pareto-k exceeds good_k on one or more channels.')
            if unexplained_k.any():
                incomplete = True
                reasons.append('Undefined Pareto-k values have no recorded fallback explanation.')

    if has_warning:
        diagnostic['status'] = 'warning'
    elif incomplete:
        diagnostic['status'] = 'insufficient'
    else:
        diagnostic['status'] = 'no_warning'
    return diagnostic


def select_model(ic_by_model, criterion='WAIC', threshold=2.0, *, return_details=False):
    """Select a model from named ``ic_criteria`` bundles.

    Scores are criterion values when ``higher_is_better`` is true and
    their negatives otherwise, with no rescaling. Candidates must be
    strictly less than ``threshold`` below the global maximum score.
    Select the fewest parameters, then the highest score, then the
    lexicographically smallest model name. Input bundles are not modified.

    Args:
        ic_by_model: Mapping of unique model names to analyzer ``ic_criteria``
            dictionaries, either directly obtained or loaded from JSON.
        criterion: Criterion key, e.g. ``'BIC'``, ``'lnZ'``, or ``'WAIC'``.
        threshold: Positive finite difference in the criterion's native units.
        return_details: Return the selection, per-model diagnostics, and differences
            from both the selected model and the global highest-score model.

    Returns:
        The selected model name, or a dictionary containing ``best_model``,
        ``highest_score_model``, ``candidate_models``, ``criterion``, ``threshold``,
        per-model ``models`` records, and ``comparisons`` when ``return_details=True``.
        Comparisons are grouped as ``selected`` and ``highest_score``, each with
        its own ``reference_model`` and per-model ``models`` comparison records.

    Raises:
        ValueError: Empty/invalid input, missing or nonfinite criterion values,
            invalid parameter counts or threshold, or inconsistent point counts,
            optimization directions, or scales.

    Notes:
        Matching ``n_data_points``, optimization directions, and scales are
        required. Data metadata is not checked. Callers must ensure the fits
        use the same data, comparable likelihoods, and identical channel ordering.
        Criterion diagnostic warnings are emitted without excluding models.
        Only the requested criterion is required. Missing LOOIC does not affect
        comparisons using other criteria; requesting an absent criterion raises
        ValueError without silently dropping models or calculating it.
        Uncertainties and penalties do not enter this selection rule.
        Detailed WAIC/LOOIC comparisons require finite, aligned ``pointwise``
        arrays whose sums match their criterion values. ``delta`` is the
        reference score minus model score: positive means worse than the
        reference, negative means better. Only the highest-score group's
        differences are necessarily nonnegative. ``delta_error`` uses
        ``sqrt(N * var(pointwise_difference, ddof=0))``, following ArviZ.
        It is a data-based paired standard error, not Monte Carlo error or
        a correction for unreliable estimates. With fewer than two channels,
        or for nonpredictive criteria, no difference error is estimated.
        Diagnostic states describe available checks, not proof of reliability
        or statistical significance; no_warning does not establish convergence.
    """

    if not isinstance(ic_by_model, Mapping) or not ic_by_model:
        raise ValueError('ic_by_model must be a nonempty mapping of model names to IC bundles')
    if any(not isinstance(name, str) or not name for name in ic_by_model):
        raise ValueError('model names must be nonempty strings')
    if not isinstance(criterion, str) or not criterion:
        raise ValueError('criterion must be a nonempty string')
    try:
        if isinstance(threshold, (bool, np.bool_)):
            raise ValueError
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError('threshold must be positive and finite') from exc
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('threshold must be positive and finite')

    scores = {}
    nparams = {}
    n_data_points = None
    common_higher_is_better = None
    common_scale = None
    for model in sorted(ic_by_model):
        bundle = ic_by_model[model]

        try:
            nparam = bundle['n_params']
            model_n_data_points = bundle['n_data_points']
            result = bundle['criteria'][criterion]
            value = result['value']
            higher_is_better = result['higher_is_better']
            scale = result.get('scale')
        except (KeyError, TypeError) as exc:
            raise ValueError(f'{model}: missing or invalid {criterion} metadata: {exc}') from exc

        if not isinstance(higher_is_better, (bool, np.bool_)):
            raise ValueError(f'{model}: {criterion} higher_is_better must be boolean')
        if isinstance(nparam, bool) or not isinstance(nparam, Integral) or nparam < 0:
            raise ValueError(f'{model}: n_params must be a nonnegative integer')
        if (
            isinstance(model_n_data_points, bool)
            or not isinstance(model_n_data_points, Integral)
            or model_n_data_points < 0
        ):
            raise ValueError(f'{model}: n_data_points must be a nonnegative integer')
        try:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{model}: {criterion} value must be finite') from exc
        if not np.isfinite(value):
            raise ValueError(f'{model}: {criterion} value must be finite')

        if n_data_points is None:
            n_data_points = model_n_data_points
            common_higher_is_better = higher_is_better
            common_scale = scale
        else:
            if model_n_data_points != n_data_points:
                raise ValueError(f'{model}: n_data_points differs between models')
            if higher_is_better != common_higher_is_better:
                raise ValueError(f'{model}: {criterion} higher_is_better differs between models')
            if scale != common_scale:
                raise ValueError(f'{model}: {criterion} scale differs between models')

        if result.get('warning', False):
            warnings.warn(
                f'{model}: {criterion} has diagnostic warnings; retained for model selection.',
                UserWarning,
                stacklevel=2,
            )

        scores[model] = value if higher_is_better else -value
        nparams[model] = int(nparam)

    best_score = max(scores.values())
    candidates = [model for model in scores if best_score - scores[model] < threshold]

    best_model = min(candidates, key=lambda model: (nparams[model], -scores[model], model))
    if not return_details:
        return best_model

    highest_score_model = min(scores, key=lambda model: (-scores[model], model))
    predictive = criterion in ('WAIC', 'LOOIC')
    pointwise = {}
    models = {}
    for model in scores:
        result = ic_by_model[model]['criteria'][criterion]
        if predictive:
            try:
                values = np.asarray(result.get('pointwise'), dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{model}: invalid {criterion} pointwise values') from exc
            if values.shape != (n_data_points,) or not np.isfinite(values).all():
                raise ValueError(
                    f'{model}: {criterion} pointwise values must be finite and aligned'
                )
            if not np.isclose(values.sum(), float(result['value']), rtol=1e-10, atol=1e-10):
                raise ValueError(f'{model}: {criterion} pointwise sum differs from its value')
            pointwise[model] = values
        models[model] = {
            'value': float(result['value']),
            'score': scores[model],
            'n_params': nparams[model],
            'diagnostics': _ic_diagnostics(result, criterion, n_data_points),
        }

    comparisons = {}
    for name, reference_model in (('selected', best_model), ('highest_score', highest_score_model)):
        comparison_models = {}
        for model, record in models.items():
            status = 'not_assessed'
            error = None
            if predictive:
                statuses = (
                    record['diagnostics']['status'],
                    models[reference_model]['diagnostics']['status'],
                )
                if 'warning' in statuses:
                    status = 'warning'
                elif 'insufficient' in statuses or n_data_points < 2:
                    status = 'insufficient'
                else:
                    status = 'no_warning'
                if n_data_points >= 2:
                    with np.errstate(over='ignore', invalid='ignore'):
                        difference = pointwise[model] - pointwise[reference_model]
                        error = float(np.sqrt(n_data_points * np.var(difference, ddof=0)))
                    if not np.isfinite(error):
                        raise ValueError(
                            f'{model}: {criterion} pointwise difference error is nonfinite'
                        )
            comparison_models[model] = {
                'delta': scores[reference_model] - scores[model],
                'delta_error': error,
                'comparison_status': status,
            }
        comparisons[name] = {'reference_model': reference_model, 'models': comparison_models}

    return {
        'best_model': best_model,
        'candidate_models': candidates,
        'highest_score_model': highest_score_model,
        'criterion': criterion,
        'threshold': threshold,
        'models': models,
        'comparisons': comparisons,
    }
