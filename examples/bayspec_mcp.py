"""MCP server that exposes bayspec as tools for LLM-driven spectral fitting.

Covers the quickstart workflow (load data units, build a Data container,
pick a spectral model, run Bayesian inference, plot) plus the advancement
workflow (composite model algebra via `compose_model`, infer-level config
and parameter mutation, multi-model overlay plots, parameter linking).

Provides:
  - Granular tools backed by a session-scoped registry (data units, data,
    models, inferences, posteriors) so a caller can assemble a fit step by
    step.
  - `compose_model` for `+/-/*//` algebra on registered models, with
    optional `alias` to disambiguate same-kind instances.
  - Infer-level inspection/mutation: `describe_infer`, `set_infer_param`,
    `set_infer_cfg`, `link_params`/`unlink_params`.
  - Result tools: `evaluate_model`, `summarize_posterior`,
    `load_posterior`, `plot_infer`/`plot_corner`/`plot_model`/`plot_models`.
  - A `run_quickstart` convenience tool that reproduces the full
    `quickstart.py` workflow in one call and registers every intermediate
    object back into the session so the granular tools can take over.

Transport: stdio. Run with `python examples/bayspec_mcp.py`, or via an
MCP client config that launches this script.

Requires the `mcp` Python SDK (`pip install mcp`). All file paths passed to
the data-loading tools are resolved relative to the server's current
working directory (see `cwd`/`set_cwd`).
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    from bayspec import BayesInfer, Data, DataUnit, Plot
    from bayspec.infer.analyzer import Posterior
    from bayspec.model.local import local_models as _LOCAL_MODELS
    from bayspec.model.model import Model
    from bayspec.util.prior import all_priors as _ALL_PRIORS
except ImportError:
    import sys

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from bayspec import BayesInfer, Data, DataUnit, Plot
    from bayspec.infer.analyzer import Posterior
    from bayspec.model.local import local_models as _LOCAL_MODELS
    from bayspec.model.model import Model
    from bayspec.util.prior import all_priors as _ALL_PRIORS

try:
    from mcp.server.fastmcp import FastMCP, Image
except ImportError as exc:
    raise SystemExit("The 'mcp' package is required. Install it with 'pip install mcp'.") from exc


mcp = FastMCP('bayspec')


StatName = Literal['pgstat', 'pstat', 'cstat', 'gstat', 'chi2']
PlotStyle = Literal['CE', 'NE', 'Fv', 'vFv', 'CC', 'NoU']
SpecPloter = Literal['plotly', 'matplotlib']
CornerPloter = Literal['plotly', 'getdist', 'cornerpy']
SpecKind = Literal['phtspec', 'flxspec', 'ergspec', 'nouspec']
AtPoint = Literal['best', 'median', 'mean']
SamplerKind = Literal['nested', 'mcmc']


STATE: dict[str, dict[str, Any]] = {
    'data_units': {},
    'data': {},
    'models': {},
    'infers': {},
    'posteriors': {},
}


def _get(kind: str, name: str) -> Any:
    if name not in STATE[kind]:
        raise ValueError(
            f"No {kind[:-1]} registered under name '{name}'. Available: {sorted(STATE[kind])}"
        )
    return STATE[kind][name]


def _register(kind: str, name: str, obj: Any, overwrite: bool) -> None:
    if (not overwrite) and (name in STATE[kind]):
        raise ValueError(
            f"{kind[:-1]} '{name}' already registered. Pass overwrite=True to replace it."
        )
    STATE[kind][name] = obj


def _resolve_model_factory(kind: str):
    if kind not in _LOCAL_MODELS:
        raise ValueError(f"Unknown model '{kind}'. Available: {sorted(_LOCAL_MODELS)}")
    return _LOCAL_MODELS[kind]


_ALLOWED_AST_NODES: tuple[type, ...] = (
    ast.Expression,
    ast.Expr,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.UnaryOp,
    ast.USub,
    ast.UAdd,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
)


def _validate_model_expr_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"disallowed expression element: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("model calls must look like alias(other_alias)")
            if node.keywords or len(node.args) != 1:
                raise ValueError("model calls take exactly one positional argument")


def _eval_model_expr(expr: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(expr, mode='eval')
    _validate_model_expr_ast(tree)
    code = compile(tree, '<compose_model>', 'eval')
    return eval(code, {'__builtins__': {}}, namespace)


def _referenced_names(expr: str) -> set[str]:
    tree = ast.parse(expr, mode='eval')
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def _build_prior(spec: dict[str, Any] | None):
    if spec is None:
        return None
    if 'kind' not in spec:
        raise ValueError("prior spec must include 'kind' (e.g. {'kind':'unif','args':[0,10]})")
    name = spec['kind']
    if name not in _ALL_PRIORS:
        raise ValueError(f"Unknown prior '{name}'. Available: {sorted(_ALL_PRIORS)}")
    args = spec.get('args', [])
    return _ALL_PRIORS[name](*args)


def _ensure_dir(savepath: str) -> None:
    Path(savepath).mkdir(parents=True, exist_ok=True)


def _to_jsonable(v: Any) -> Any:
    """Recursively convert numpy / pandas values to plain Python types.

    Pydantic in FastMCP rejects raw ``numpy.int64`` / ``numpy.float64``
    even when nested inside dicts/lists, so we walk the structure.
    """
    if isinstance(v, dict):
        return {k: _to_jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if hasattr(v, 'tolist'):
        return v.tolist()
    return v


def _posterior_summary(p: Any) -> dict[str, Any]:
    fields = (
        'par_best',
        'par_median',
        'par_mean',
        'par_Isigma',
        'par_IIsigma',
        'max_loglike',
        'aic',
        'aicc',
        'bic',
        'lnZ',
    )
    summary: dict[str, Any] = {}
    for field in fields:
        try:
            summary[field] = _to_jsonable(getattr(p, field))
        except AttributeError:
            summary[field] = None
    for table in ('free_par_info', 'stat_info', 'IC_info'):
        owner = getattr(p, table, None)
        if owner is not None:
            summary[table] = _to_jsonable(getattr(owner, 'data_list_dict', None))
    return summary


def _maybe_inline_image(figwrap: Any, basename: str) -> Image | None:
    """Best-effort PNG snapshot of the plot, returned as MCP Image content.

    Returns ``None`` when the backend can't emit a PNG in the current
    environment (e.g. plotly without kaleido, or getdist).
    """
    png_path = f'{basename}.png'
    try:
        if figwrap.plotter == 'plotly':
            figwrap.fig.write_image(png_path, format='png')
        elif figwrap.plotter == 'matplotlib':
            figwrap.fig.savefig(png_path, dpi=150, bbox_inches='tight')
            # Defense-in-depth: bayspec's Figure.save() already closes the
            # figure on PDF write, but we may have written the PNG first or
            # the caller may skip save(); release the matplotlib resource
            # immediately so the figure doesn't accumulate in pyplot state.
            import matplotlib.pyplot as plt

            plt.close(figwrap.fig)
        else:
            return None
        return Image(path=png_path)
    except Exception:
        return None


def _state_snapshot() -> dict[str, list[str]]:
    return {kind: sorted(reg) for kind, reg in STATE.items()}


def _suggest_next(after: str) -> list[str]:
    """Return tool names that make sense to call next, given current STATE."""
    s = STATE
    if after == 'cwd' or after == 'set_cwd':
        return ['list_state', 'load_data_unit', 'run_quickstart']
    if after == 'list_models' or after == 'list_priors':
        return ['build_model', 'load_data_unit']
    if after == 'list_state':
        if not s['data_units']:
            return ['load_data_unit', 'run_quickstart']
        if not s['data']:
            return ['build_data', 'load_data_unit']
        if not s['models']:
            return ['list_models', 'build_model']
        if not s['infers']:
            return ['setup_infer', 'compose_model', 'set_param']
        if not s['posteriors']:
            return ['describe_infer', 'suggest_sampler', 'run_multinest', 'run_emcee']
        return ['summarize_posterior', 'plot_infer', 'plot_corner']
    if after == 'reset_session' or after == 'delete_object':
        return ['list_state']
    if after == 'load_data_unit':
        return ['load_data_unit', 'build_data']
    if after == 'build_data':
        if not s['models']:
            return ['list_models', 'build_model']
        return ['build_model', 'compose_model', 'setup_infer']
    if after == 'build_model':
        return ['describe_model', 'set_param', 'compose_model', 'build_model', 'setup_infer']
    if after == 'compose_model':
        return ['describe_model', 'set_param', 'setup_infer']
    if after == 'describe_model':
        return ['set_param', 'compose_model', 'setup_infer']
    if after == 'set_param':
        return ['describe_model', 'setup_infer']
    if after == 'setup_infer':
        return [
            'describe_infer',
            'set_infer_param',
            'link_params',
            'set_infer_cfg',
            'suggest_sampler',
        ]
    if after == 'describe_infer':
        return ['set_infer_param', 'link_params', 'set_infer_cfg', 'suggest_sampler']
    if after in ('set_infer_param', 'set_infer_cfg', 'link_params', 'unlink_params'):
        return ['describe_infer', 'suggest_sampler', 'run_multinest', 'run_emcee']
    if after == 'suggest_sampler':
        return ['run_multinest', 'run_emcee']
    if after in ('run_multinest', 'run_emcee', 'load_posterior'):
        return [
            'summarize_posterior',
            'plot_infer',
            'plot_corner',
            'plot_model',
            'evaluate_model',
        ]
    if after == 'summarize_posterior':
        return ['plot_infer', 'plot_corner', 'plot_model', 'evaluate_model']
    if after == 'evaluate_model':
        return ['plot_model', 'plot_models', 'summarize_posterior']
    if after in ('plot_infer', 'plot_corner', 'plot_model', 'plot_models'):
        return ['summarize_posterior', 'evaluate_model']
    if after == 'run_quickstart':
        return ['summarize_posterior', 'plot_infer', 'plot_corner', 'describe_infer']
    return ['list_state']


def _decorate(
    result: dict[str, Any] | None,
    after: str,
    hint: str | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Attach `state`, `ready_for`, optional `hint`/`warnings` to a tool's result dict.

    Tool-specific fields stay at the top level; the meta keys make the
    return self-describing for an LLM driver.
    """
    out: dict[str, Any] = {'ok': True}
    if result:
        out.update(result)
    out['state'] = _state_snapshot()
    out['ready_for'] = _suggest_next(after)
    if hint:
        out['hint'] = hint
    if warnings:
        out['warnings'] = warnings
    return out


# Hardcoded sanity rules keyed on lowercased parameter labels.
# `physical`: hard outer bounds beyond which the prior is almost certainly wrong.
# `typical`: looser-than-textbook range that covers >95% of real fits; outside
#   this we warn but accept.
# `positive`: parameter is physically positive — prior must not allow <=0.
SANITY_CHECKS: dict[str, dict[str, Any]] = {
    'alpha': {'physical': (-5.0, 5.0), 'typical': (-3.0, 3.0)},
    'phoindex': {'physical': (-5.0, 5.0), 'typical': (-3.0, 3.0)},
    'beta': {'physical': (-10.0, 5.0), 'typical': (-5.0, -1.0)},
    'ec': {'physical': (1e-2, 1e8), 'typical': (1.0, 1e5), 'positive': True},
    'epeak': {'physical': (1e-2, 1e8), 'typical': (1.0, 1e5), 'positive': True},
    'ep': {'physical': (1e-2, 1e8), 'typical': (1.0, 1e5), 'positive': True},
    'eb': {'physical': (1e-2, 1e8), 'typical': (1.0, 1e5), 'positive': True},
    'kt': {'physical': (1e-3, 1e4), 'typical': (0.1, 100.0), 'positive': True},
    'temperature': {'physical': (1e-3, 1e4), 'typical': (0.1, 100.0), 'positive': True},
    'k': {'positive': True},
    'norm': {'positive': True},
    'flux': {'positive': True},
    'nh': {'physical': (1e-4, 1e6), 'typical': (1e-3, 1e3), 'positive': True},
    'z': {'physical': (0.0, 20.0), 'typical': (0.0, 10.0)},
    'redshift': {'physical': (0.0, 20.0), 'typical': (0.0, 10.0)},
}


def _prior_bounds(spec: dict[str, Any]) -> tuple[float | None, float | None]:
    """Best-effort (low, high) extraction from a prior spec for sanity checking."""
    kind = spec.get('kind')
    args = spec.get('args') or []
    if kind in ('unif', 'logunif') and len(args) >= 2:
        return float(args[0]), float(args[1])
    if kind == 'truncnorm' and len(args) >= 4:
        return float(args[2]), float(args[3])
    if kind == 'beta' and len(args) >= 2:
        return 0.0, 1.0
    return None, None


def _param_label(model_obj: Any, par_id: str) -> str | None:
    """Find the textual label for a parameter id by scanning all_params."""
    for row in getattr(model_obj, 'all_params', []):
        if row.get('par#') == str(par_id):
            return row.get('Parameter')
    return None


def _normalize_label(label: str | None) -> tuple[str | None, bool]:
    """Strip LaTeX wrapping to get a SANITY_CHECKS key, plus a log-scale flag.

    bayspec parameter labels look like ``$\\alpha$``, ``log$E_p$``, ``log$A$``.
    A leading ``log`` means the parameter value is in log10 space — we
    convert prior bounds to linear before comparing against SANITY rules.
    """
    if not label:
        return None, False
    s = label.lower()
    for ch in ('$', '\\', '{', '}', '_', ' '):
        s = s.replace(ch, '')
    is_log = False
    if s.startswith('log'):
        is_log = True
        s = s[3:]
    return (s or None), is_log


def _sanity_check_prior(label: str | None, spec: dict[str, Any]) -> list[str]:
    """Return a list of warnings (possibly empty) for a prior on a labelled param."""
    name, is_log = _normalize_label(label)
    if not name:
        return []
    rules = SANITY_CHECKS.get(name)
    if not rules:
        return []
    low, high = _prior_bounds(spec)
    warnings: list[str] = []
    # Convert log-space bounds to linear for comparison against physical rules.
    if is_log and low is not None and high is not None:
        low_lin: float | None = 10 ** low
        high_lin: float | None = 10 ** high
    else:
        low_lin, high_lin = low, high
    # Positivity check: in log space the linear value is always > 0, so skip.
    if rules.get('positive') and not is_log and low_lin is not None and low_lin <= 0:
        warnings.append(
            f"prior on '{label}' allows non-positive values (low={low_lin}); this "
            'parameter must be positive — consider logunif over a positive range.'
        )
    if low_lin is not None and high_lin is not None:
        phys = rules.get('physical')
        if phys and (low_lin < phys[0] or high_lin > phys[1]):
            warnings.append(
                f"prior on '{label}' (linear=[{low_lin:.3g}, {high_lin:.3g}]) "
                f'is outside the physical range [{phys[0]:.3g}, {phys[1]:.3g}] — '
                'confirm with the user before fitting.'
            )
        else:
            typ = rules.get('typical')
            if typ and (low_lin < typ[0] or high_lin > typ[1]):
                warnings.append(
                    f"prior on '{label}' (linear=[{low_lin:.3g}, {high_lin:.3g}]) "
                    f'is wider than the typical range [{typ[0]:.3g}, {typ[1]:.3g}] — '
                    'fit may explore unphysical regions.'
                )
    return warnings


@mcp.tool()
def cwd() -> dict[str, Any]:
    """Return the server's current working directory."""
    return _decorate(
        {'cwd': os.getcwd()},
        after='cwd',
        hint='All file paths in subsequent tools are resolved relative to this directory.',
    )


@mcp.tool()
def set_cwd(path: str) -> dict[str, Any]:
    """Change the server's working directory; returns the new cwd."""
    os.chdir(path)
    return _decorate(
        {'cwd': os.getcwd()},
        after='set_cwd',
        hint='Working directory updated. Subsequent file paths are now relative to this directory.',
    )


@mcp.tool()
def list_models() -> dict[str, Any]:
    """List spectral models available under bayspec.model.local."""
    return _decorate(
        {'models': sorted(_LOCAL_MODELS)},
        after='list_models',
        hint='Pass any of these as `kind` to build_model. Ask the user which model fits the source physics before picking.',
    )


@mcp.tool()
def list_priors() -> dict[str, Any]:
    """List prior distributions accepted by `set_param`."""
    return _decorate(
        {'priors': sorted(_ALL_PRIORS)},
        after='list_priors',
        hint='Use as {"kind": "<name>", "args": [...]} when calling set_param/set_infer_param.',
    )


@mcp.tool()
def list_state() -> dict[str, Any]:
    """Return the names of all registered session objects, grouped by kind."""
    return _decorate(
        None,
        after='list_state',
        hint='`state` reflects what is currently loaded; `ready_for` is the suggested next step.',
    )


@mcp.tool()
def reset_session() -> dict[str, Any]:
    """Clear all session-scoped objects (data units, data, models, infers, posteriors)."""
    for reg in STATE.values():
        reg.clear()
    return _decorate(
        {'cleared': True},
        after='reset_session',
        hint='Session is empty. Start by loading data with load_data_unit or run_quickstart.',
    )


@mcp.tool()
def delete_object(
    kind: Literal['data_units', 'data', 'models', 'infers', 'posteriors'],
    name: str,
) -> dict[str, Any]:
    """Remove a single registered object from the session."""
    if name not in STATE[kind]:
        raise ValueError(f"No {kind[:-1]} registered under name '{name}'.")
    del STATE[kind][name]
    return _decorate(
        {'deleted': {'kind': kind, 'name': name}},
        after='delete_object',
        hint='Downstream objects (e.g. infers built on a deleted model) may now be stale.',
    )


@mcp.tool()
def load_data_unit(
    name: str,
    src: str,
    bkg: str | None = None,
    rsp: str | None = None,
    rmf: str | None = None,
    arf: str | None = None,
    notc: list[float] | list[list[float]] | None = None,
    stat: StatName = 'pgstat',
    rebn: dict[str, Any] | None = None,
    grpg: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Register a DataUnit (one detector's spectrum + background + response).

    Paths are resolved relative to the server's current working directory.
    `notc` is an energy noticing window — `[low, high]` or a list of pairs.
    `rebn`/`grpg` accept dicts like `{'min_sigma': 2, 'max_bin': 10}`.
    """
    unit = DataUnit(
        src=src,
        bkg=bkg,
        rsp=rsp,
        rmf=rmf,
        arf=arf,
        notc=notc,
        stat=stat,
        rebn=rebn,
        grpg=grpg,
    )
    _register('data_units', name, unit, overwrite)
    return _decorate(
        {'name': name, 'src': src, 'stat': stat, 'notc': notc},
        after='load_data_unit',
        hint=(
            'Repeat for each detector, then combine them with build_data. '
            'Confirm the energy notice (`notc`) matches the detector + source physics.'
        ),
    )


@mcp.tool()
def build_data(
    name: str,
    units: list[dict[str, str]],
    savepath: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Combine previously-registered DataUnits into a Data container.

    `units` is an ordered list like
    `[{"label": "nai", "unit": "me_unit"}, {"label": "bgo", "unit": "he_unit"}]`,
    where `label` is the name inside the Data container and `unit` references
    a name previously registered via `load_data_unit`. If `savepath` is set,
    the data table is written there.
    """
    items = [(u['label'], _get('data_units', u['unit'])) for u in units]
    data = Data(items)
    _register('data', name, data, overwrite)
    saved: str | None = None
    if savepath:
        _ensure_dir(savepath)
        data.save(savepath)
        saved = savepath
    return _decorate(
        {
            'name': name,
            'labels': [u['label'] for u in units],
            'saved_to': saved,
        },
        after='build_data',
        hint='Data ready. Next: build_model (or compose_model) and then setup_infer.',
    )


@mcp.tool()
def build_model(
    name: str,
    kind: str,
    savepath: str | None = None,
    alias: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Instantiate a spectral model by name from bayspec.model.local.

    Use `list_models` to see available kinds (e.g. 'cpl', 'pl', 'band').
    `alias` rewrites the component's display `expr` — set this when you'll
    use the same `kind` more than once inside a `compose_model` expression
    (e.g. two `tbabs` instances at different redshifts → alias them as
    `'tbabs'` and `'ztbabs'`) so bayspec doesn't auto-append numeric
    suffixes to disambiguate them.
    """
    factory = _resolve_model_factory(kind)
    model = factory()
    if alias is not None:
        model.expr = alias
    _register('models', name, model, overwrite)
    saved: str | None = None
    if savepath:
        _ensure_dir(savepath)
        model.save(savepath)
        saved = savepath
    return _decorate(
        {'name': name, 'kind': kind, 'alias': alias, 'saved_to': saved},
        after='build_model',
        hint=(
            'Call describe_model to see the parameter table; ask the user for '
            'prior bounds on each free parameter before fitting.'
        ),
    )


@mcp.tool()
def compose_model(
    name: str,
    expr: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compose a new model from registered models via +/-/*/() algebra.

    `expr` is a Python expression whose identifiers are registered model
    names (from `build_model`/`list_state`). Examples:

        'tbabs * ztbabs * (cpl + csbpl)'   # multiplicative absorption × additive emission
        'cpl + csbpl'                       # two additive components
        'tbabs(cpl)'                        # convolution via Model.__call__

    Allowed operators: `+ - * /` and `f(g)` (convolution). Numeric literals
    are wrapped in bayspec's FrozenConst automatically. No attribute
    access, no calls with multiple args, no other Python is permitted
    (the expression is AST-validated, not arbitrarily `eval`'d).

    Component sharing: if the same registered name appears in two
    `compose_model` calls, both resulting composites hold the **same**
    underlying instance, so their parameters are tied together. This is
    how the advancement workflow couples shared `cpl`/`csbpl` components
    across two (data, model) pairs.
    """
    referenced = _referenced_names(expr)
    missing = referenced - set(STATE['models'])
    if missing:
        raise ValueError(
            f"unknown model name(s) in expr: {sorted(missing)}. "
            f"Registered: {sorted(STATE['models'])}"
        )
    namespace = {n: STATE['models'][n] for n in referenced}
    composite = _eval_model_expr(expr, namespace)
    if not isinstance(composite, Model):
        raise ValueError(f"expression did not produce a Model (got {type(composite).__name__})")
    try:
        composite_type = composite.type
    except ValueError as exc:
        raise ValueError(
            f"invalid composition '{expr}': {exc}. "
            f"bayspec rejects this combination of component types; check the operators."
        ) from None
    _register('models', name, composite, overwrite)
    return _decorate(
        {
            'name': name,
            'expr': composite.expr,
            'type': composite_type,
            'components': sorted(referenced),
        },
        after='compose_model',
        hint=(
            'Composite registered. Components shared by multiple composites '
            'remain the same instance — their parameters stay tied. Use '
            'describe_model on the new name to see the merged parameter table.'
        ),
    )


@mcp.tool()
def describe_model(name: str) -> dict[str, Any]:
    """Inspect a registered model: type, fit-parameters (id/value/prior/frozen), and configs.

    Use the returned `par_id` strings with `set_param` to mutate values, priors,
    or freeze/thaw individual parameters.
    """
    m = _get('models', name)
    params = [
        {
            'par_id': row['par#'],
            'component': row['Component'],
            'label': row['Parameter'],
            'val': row['Value'],
            'prior': row['Prior'],
            'frozen': row['Frozen'],
            'posterior': row['Posterior'],
        }
        for row in m.all_params
    ]
    free_count = sum(1 for row in params if not row['frozen'])
    return _decorate(
        {
            'name': name,
            'type': getattr(m, 'type', None),
            'expr': getattr(m, 'expr', None),
            'params': params,
            'configs': list(m.all_config),
            'free_param_count': free_count,
        },
        after='describe_model',
        hint=(
            f'{free_count} free parameter(s). Use set_param(model, par_id, ...) '
            'to set priors or freeze values; ask the user for prior bounds.'
        ),
    )


def _apply_param_changes(
    par: Any,
    val: float | None,
    frozen: bool | None,
    prior: dict[str, Any] | None,
) -> None:
    if val is not None and frozen is True:
        par.frozen_at(val)
    else:
        if val is not None:
            par.val = val
        if frozen is not None:
            par.frozen = frozen
    if prior is not None:
        par.prior = _build_prior(prior)


@mcp.tool()
def set_param(
    model: str,
    par_id: str,
    val: float | None = None,
    frozen: bool | None = None,
    prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mutate a fit-parameter on a registered model.

    `par_id` is the `par#` from `describe_model` (1-based, as a string).
    Only the fields you pass change. `prior` is `{"kind": "unif",
    "args": [0, 10]}` etc — see `list_priors` for kinds and bayspec docs
    for each kind's argument order. Pass `frozen=True` with `val=...` to
    freeze at a specific value in one call.
    """
    m = _get('models', model)
    par_id = str(par_id)
    if par_id not in m.par:
        raise ValueError(f"par_id '{par_id}' not in model '{model}'. Available: {list(m.par)}")
    par = m.par[par_id]
    label = _param_label(m, par_id)
    warnings = _sanity_check_prior(label, prior) if prior is not None else []
    _apply_param_changes(par, val, frozen, prior)
    return _decorate(
        {
            'model': model,
            'par_id': par_id,
            'label': label,
            'val': par.val,
            'frozen': par.frozen,
            'prior': par.prior_info,
        },
        after='set_param',
        hint=(
            'Surface any warnings to the user before continuing — bad priors '
            'are the most common cause of failed fits.'
        )
        if warnings
        else None,
        warnings=warnings or None,
    )


@mcp.tool()
def describe_infer(name: str) -> dict[str, Any]:
    """Inspect a registered BayesInfer: configs, all params, and the free subset.

    `params[i].mates` lists the `par#` of every slot that shares the same
    underlying parameter (so changing one propagates). `free_params` is the
    canonical order MultiNest/emcee will see.
    """
    inf = _get('infers', name)
    inf._you_free()
    n_free = inf.free_nparams
    return _decorate(
        {
            'name': name,
            'configs': [
                {
                    'cfg_id': row['cfg#'],
                    'class': row['Class'],
                    'expression': row['Expression'],
                    'component': row['Component'],
                    'parameter': row['Parameter'],
                    'value': row['Value'],
                }
                for row in inf.all_config
            ],
            'params': [
                {
                    'par_id': row['par#'],
                    'class': row['Class'],
                    'expression': row['Expression'],
                    'component': row['Component'],
                    'parameter': row['Parameter'],
                    'val': row['Value'],
                    'prior': row['Prior'],
                    'frozen': row['Frozen'],
                    'mates': sorted(row['Mates']),
                    'posterior': row['Posterior'],
                }
                for row in inf.all_params
            ],
            'free_params': list(inf.free_par.keys()),
            'free_nparams': n_free,
            'free_plabels': inf.free_plabels,
        },
        after='describe_infer',
        hint=(
            f'{n_free} free parameter(s). Confirm this matches what the user '
            'expects, then call suggest_sampler before running a sampler.'
        ),
    )


@mcp.tool()
def set_infer_param(
    infer: str,
    par_id: str,
    val: float | None = None,
    frozen: bool | None = None,
    prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mutate a parameter via the infer-level `par#` (covers model + data params).

    Same semantics as `set_param`, but `par_id` is the index from
    `describe_infer` — needed when you want to mutate data-side params
    (sf/bf/rf/ra/dec) that don't appear in `describe_model`. Changes
    propagate to any linked mates automatically.
    """
    inf = _get('infers', infer)
    par_id = str(par_id)
    if par_id not in inf.par:
        raise ValueError(f"par_id '{par_id}' not in infer '{infer}'. Available: {list(inf.par)}")
    par = inf.par[par_id]
    label = _param_label(inf, par_id)
    warnings = _sanity_check_prior(label, prior) if prior is not None else []
    _apply_param_changes(par, val, frozen, prior)
    inf._you_free()
    return _decorate(
        {
            'infer': infer,
            'par_id': par_id,
            'label': label,
            'val': par.val,
            'frozen': par.frozen,
            'prior': par.prior_info,
            'free_nparams': inf.free_nparams,
        },
        after='set_infer_param',
        hint=(
            'Surface any warnings to the user before continuing — bad priors '
            'are the most common cause of failed fits.'
        )
        if warnings
        else None,
        warnings=warnings or None,
    )


@mcp.tool()
def set_infer_cfg(
    infer: str,
    cfg_id: str,
    val: Any,
) -> dict[str, Any]:
    """Set a config value on a registered BayesInfer.

    Use `describe_infer` to discover `cfg#` indices and value types
    (e.g. redshift is float, `vfv_peak` is bool, `pivot_energy` is float).
    """
    inf = _get('infers', infer)
    cfg_id = str(cfg_id)
    if cfg_id not in inf.cfg:
        raise ValueError(f"cfg_id '{cfg_id}' not in infer '{infer}'. Available: {list(inf.cfg)}")
    inf.cfg[cfg_id].val = val
    return _decorate(
        {'infer': infer, 'cfg_id': cfg_id, 'val': inf.cfg[cfg_id].val},
        after='set_infer_cfg',
    )


@mcp.tool()
def link_params(infer: str, par_ids: list[str]) -> dict[str, Any]:
    """Link parameter slots so they share value/prior/posterior."""
    inf = _get('infers', infer)
    for pid in par_ids:
        if str(pid) not in inf.par:
            raise ValueError(f"par_id '{pid}' not in infer '{infer}'.")
    inf.link([str(p) for p in par_ids])
    return _decorate(
        {'infer': infer, 'linked': [str(p) for p in par_ids], 'free_nparams': inf.free_nparams},
        after='link_params',
        hint='Linked params share value/prior/posterior. Re-check free_nparams before sampling.',
    )


@mcp.tool()
def unlink_params(infer: str, par_ids: list[str]) -> dict[str, Any]:
    """Undo any links between every pair drawn from `par_ids`."""
    inf = _get('infers', infer)
    for pid in par_ids:
        if str(pid) not in inf.par:
            raise ValueError(f"par_id '{pid}' not in infer '{infer}'.")
    inf.unlink([str(p) for p in par_ids])
    return _decorate(
        {'infer': infer, 'unlinked': [str(p) for p in par_ids], 'free_nparams': inf.free_nparams},
        after='unlink_params',
    )


@mcp.tool()
def setup_infer(
    name: str,
    pairs: list[dict[str, str]],
    savepath: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a BayesInfer from a list of (data, model) pairs.

    `pairs` looks like `[{"data": "d1", "model": "m1"}, ...]`. Names must
    refer to objects previously registered via `build_data` / `build_model`.
    """
    pair_objs = [(_get('data', p['data']), _get('models', p['model'])) for p in pairs]
    infer = BayesInfer(pair_objs)
    _register('infers', name, infer, overwrite)
    saved: str | None = None
    if savepath:
        _ensure_dir(savepath)
        infer.save(savepath)
        saved = savepath
    return _decorate(
        {
            'name': name,
            'pairs': pairs,
            'saved_to': saved,
            'free_nparams': infer.free_nparams,
        },
        after='setup_infer',
        hint=(
            'Call describe_infer to confirm the free-parameter set, then '
            'suggest_sampler before running.'
        ),
    )


@mcp.tool()
def suggest_sampler(
    n_free_params: int,
    expect_multimodal: bool = False,
    need_evidence: bool = False,
) -> dict[str, Any]:
    """Recommend `run_multinest` vs `run_emcee` for the current problem shape.

    Decision tree:
      - need_evidence (Bayes factor / model selection)         -> multinest (only path to lnZ)
      - expect_multimodal=True OR n_free_params > 10           -> multinest
      - otherwise                                              -> emcee (faster, well-tested)

    Returns a recommendation plus a sampler-specific config hint
    (`nlive` for multinest, `nstep`/`discard`/`walkers_default` for emcee).
    The caller (LLM) should confirm with the user before invoking the
    sampler — we don't pick silently.
    """
    if need_evidence:
        return _decorate(
            {
                'recommendation': 'multinest',
                'reason': (
                    'Only nested sampling produces a marginal-likelihood (lnZ) '
                    'estimate suitable for Bayes-factor model comparison.'
                ),
                'config_hint': {'nlive': max(400, 50 * n_free_params)},
            },
            after='suggest_sampler',
            hint='Confirm with the user, then call run_multinest with the suggested nlive.',
        )
    if expect_multimodal or n_free_params > 10:
        return _decorate(
            {
                'recommendation': 'multinest',
                'reason': (
                    f'Multinest handles multimodal / high-dim posteriors '
                    f'({n_free_params} free params, multimodal={expect_multimodal}) '
                    'more reliably than emcee, at ~5-10x the wallclock cost.'
                ),
                'config_hint': {'nlive': max(400, 50 * n_free_params)},
            },
            after='suggest_sampler',
            hint='Confirm with the user, then call run_multinest with the suggested nlive.',
        )
    return _decorate(
        {
            'recommendation': 'emcee',
            'reason': (
                f'emcee is fast and well-suited to {n_free_params}-D unimodal '
                'posteriors. Switch to multinest if you suspect multimodality '
                'or care about lnZ.'
            ),
            'config_hint': {
                'nstep': 5000,
                'discard': 500,
                'walkers_default': max(2 * n_free_params, 32),
            },
        },
        after='suggest_sampler',
        hint='Confirm with the user, then call run_emcee with the suggested nstep.',
    )


@mcp.tool()
def run_multinest(
    infer: str,
    post: str,
    nlive: int = 400,
    resume: bool = True,
    savepath: str = './quickstart',
    random_seed: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run MultiNest on a registered BayesInfer; store the Posterior under `post`."""
    inf = _get('infers', infer)
    _ensure_dir(savepath)
    posterior = inf.multinest(
        nlive=nlive,
        resume=resume,
        savepath=savepath,
        random_seed=random_seed,
    )
    posterior.save(savepath)
    _register('posteriors', post, posterior, overwrite)
    summary = _posterior_summary(posterior)
    summary['post'] = post
    summary['savepath'] = savepath
    return _decorate(
        summary,
        after='run_multinest',
        hint=(
            'Sampling done. Inspect par_best/par_Isigma and lnZ; if anything '
            'looks off, report diagnostics to the user before re-running. '
            'Then plot_infer + plot_corner for the user.'
        ),
    )


@mcp.tool()
def run_emcee(
    infer: str,
    post: str,
    nstep: int = 1000,
    discard: int = 100,
    resume: bool = True,
    savepath: str = './quickstart_emcee',
    random_seed: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run emcee on a registered BayesInfer; store the Posterior under `post`.

    Default `savepath` differs from `run_multinest` so the two samplers'
    artifacts don't end up in the same directory.
    """
    inf = _get('infers', infer)
    _ensure_dir(savepath)
    posterior = inf.emcee(
        nstep=nstep,
        discard=discard,
        resume=resume,
        savepath=savepath,
        random_seed=random_seed,
    )
    posterior.save(savepath)
    _register('posteriors', post, posterior, overwrite)
    summary = _posterior_summary(posterior)
    summary['post'] = post
    summary['savepath'] = savepath
    return _decorate(
        summary,
        after='run_emcee',
        hint=(
            'Sampling done. Inspect par_best/par_Isigma; if walkers do not look '
            'mixed (or autocorr time is large), report to the user before '
            're-running. Then plot_infer + plot_corner.'
        ),
    )


@mcp.tool()
def load_posterior(
    infer: str,
    post: str,
    savepath: str,
    sampler: SamplerKind = 'nested',
    overwrite: bool = False,
) -> dict[str, Any]:
    """Rebuild a Posterior from a previous run's `1-posterior_sample.txt`.

    The supplied `infer` must reflect the same model + data layout used for
    the original run (so the column count matches `free_nparams + 1`).
    For `sampler='nested'`, `logevidence` is read from
    `1-posterior_stats.json` if present, so `lnZ` survives the round-trip.
    """
    inf = _get('infers', infer)
    sample_path = Path(savepath) / '1-posterior_sample.txt'
    if not sample_path.exists():
        raise FileNotFoundError(f"No posterior sample at {sample_path}")
    inf.sampler_type = sampler
    inf._you_free()
    inf.posterior_sample = np.loadtxt(sample_path)
    if sampler == 'nested':
        stats_path = Path(savepath) / '1-posterior_stats.json'
        if stats_path.exists():
            with stats_path.open() as f:
                stats = json.load(f)
            key = 'nested importance sampling global log-evidence'
            if key in stats:
                inf.logevidence = stats[key]
    posterior = Posterior(inf)
    _register('posteriors', post, posterior, overwrite)
    summary = _posterior_summary(posterior)
    summary['post'] = post
    summary['loaded_from'] = str(sample_path)
    return _decorate(
        summary,
        after='load_posterior',
        hint='Posterior reconstructed without re-sampling. Plot or evaluate as usual.',
    )


@mcp.tool()
def summarize_posterior(post: str) -> dict[str, Any]:
    """Return best-fit params, credible intervals, and information criteria."""
    summary = _posterior_summary(_get('posteriors', post))
    summary['post'] = post
    return _decorate(
        summary,
        after='summarize_posterior',
        hint='Report par_best/par_Isigma and the IC tables to the user before plotting.',
    )


@mcp.tool()
def evaluate_model(
    model: str,
    kind: SpecKind = 'ergspec',
    e_min: float = 10.0,
    e_max: float = 1000.0,
    e_npts: int = 100,
    post: str | None = None,
    at: AtPoint = 'best',
) -> dict[str, Any]:
    """Evaluate a model's spectrum on a log-spaced energy grid.

    `kind`: `phtspec` (N(E), add-type), `flxspec` (F_nu = E·N(E)),
    `ergspec` (nuF_nu = E^2·N(E)), or `nouspec` (mul/math models).

    When `post` is given, the model is first set to that posterior's
    `best`/`median`/`mean` parameters (the model must have been fit).
    Returns numeric arrays so callers can compare/analyse without plotting.
    """
    m = _get('models', model)
    earr = np.logspace(np.log10(e_min), np.log10(e_max), e_npts)
    if post is not None:
        p = _get('posteriors', post)
        if m not in getattr(p, 'Model', []):
            raise ValueError(f"Model '{model}' is not part of posterior '{post}'.")
        if at == 'best':
            p.at_par(p.par_best)
        elif at == 'median':
            p.at_par(p.par_median)
        elif at == 'mean':
            p.at_par(p.par_mean)
    spec_fn = getattr(m, kind, None)
    if spec_fn is None:
        raise ValueError(f"Model has no '{kind}' method.")
    y = spec_fn(earr)
    return _decorate(
        {
            'model': model,
            'kind': kind,
            'at': at if post is not None else None,
            'post': post,
            'E': earr.tolist(),
            'Y': np.asarray(y).tolist(),
        },
        after='evaluate_model',
        hint='Numeric arrays only — call plot_model / plot_models to visualise.',
    )


@mcp.tool()
def plot_infer(
    post: str,
    save: str,
    style: PlotStyle = 'CE',
    ploter: SpecPloter = 'plotly',
    inline: bool = False,
) -> list[Any] | dict[str, Any]:
    """Plot data + best-fit + residuals from a posterior.

    `style`: 'CE' counts, 'NE' photon, 'Fv' flux, 'vFv' nuFnu, plus 'CC'
    and 'NoU' diagnostics. `save` is the output basename — the ploter picks
    the extension (plotly emits .html and .pdf; matplotlib emits .pdf).
    When `inline=True`, also writes a PNG and returns it as MCP Image
    content (best-effort: needs kaleido for plotly).

    **Naming convention** (matches examples/quickstart.py): pass
    `save='{savepath}/ctsspec'` for style='CE',  '/phtspec' for 'NE',
    '/flxspec' for 'Fv', '/ergspec' for 'vFv'. Do NOT invent custom names
    like 'fit_CE' — the canonical basenames keep results comparable across
    runs and consistent with the example notebooks.
    """
    p = _get('posteriors', post)
    fig = Plot.infer(p, style=style, ploter=ploter)
    img = _maybe_inline_image(fig, save) if inline else None
    fig.save(save)
    meta = _decorate(
        {'saved': save, 'style': style, 'ploter': ploter, 'post': post},
        after='plot_infer',
        hint='Show the saved file path to the user; mention the style (CE/NE/Fv/vFv).',
    )
    return [meta, img] if img is not None else meta


@mcp.tool()
def plot_corner(
    post: str,
    save: str,
    ploter: CornerPloter = 'plotly',
    inline: bool = False,
) -> list[Any] | dict[str, Any]:
    """Corner plot of posterior samples. `ploter` is 'plotly', 'matplotlib', or 'getdist'.

    **Naming convention**: pass `save='{savepath}/corner'` (matches
    examples/quickstart.py).
    """
    p = _get('posteriors', post)
    fig = Plot.post_corner(p, ploter=ploter)
    img = _maybe_inline_image(fig, save) if inline else None
    fig.save(save)
    meta = _decorate(
        {'saved': save, 'ploter': ploter, 'post': post},
        after='plot_corner',
        hint='Show the saved corner-plot path to the user.',
    )
    return [meta, img] if img is not None else meta


@mcp.tool()
def plot_model(
    model: str,
    save: str,
    style: PlotStyle = 'vFv',
    ploter: SpecPloter = 'plotly',
    e_min: float = 10.0,
    e_max: float = 1000.0,
    e_npts: int = 100,
    with_posterior: bool = False,
    post: str | None = None,
    inline: bool = False,
) -> list[Any] | dict[str, Any]:
    """Plot a registered model over a log-spaced energy grid (keV).

    With `with_posterior=True` you MUST also pass `post` — the name of a
    Posterior produced from a fit that included this very model — so that
    posterior-sampled bands can be drawn around the point estimate.

    **Naming convention**: pass `save='{savepath}/model'` (matches
    examples/quickstart.py).
    """
    m = _get('models', model)
    if with_posterior:
        if post is None:
            raise ValueError("with_posterior=True requires `post` (a registered Posterior name).")
        p = _get('posteriors', post)
        if m not in getattr(p, 'Model', []):
            raise ValueError(
                f"Model '{model}' was not part of posterior '{post}'. "
                "Pass a matching posterior or set with_posterior=False."
            )
    earr = np.logspace(np.log10(e_min), np.log10(e_max), e_npts)
    modelplot = Plot.model(ploter=ploter, style=style, post=with_posterior)
    modelplot.add_model(m, E=earr)
    fig = modelplot.get_fig()
    img = _maybe_inline_image(fig, save) if inline else None
    fig.save(save)
    meta = _decorate(
        {
            'saved': save,
            'style': style,
            'ploter': ploter,
            'e_range': [e_min, e_max],
            'e_npts': e_npts,
            'with_posterior': with_posterior,
            'model': model,
        },
        after='plot_model',
        hint='Show the saved file path to the user.',
    )
    return [meta, img] if img is not None else meta


@mcp.tool()
def plot_models(
    models: list[str],
    save: str,
    style: PlotStyle = 'vFv',
    ploter: SpecPloter = 'plotly',
    e_min: float = 10.0,
    e_max: float = 1000.0,
    e_npts: int = 100,
    with_posterior: bool = False,
    post: str | None = None,
    inline: bool = False,
) -> list[Any] | dict[str, Any]:
    """Overlay multiple registered models on a single energy grid.

    With `with_posterior=True` you MUST pass `post` (a Posterior name),
    AND every entry in `models` must be one of that posterior's models —
    otherwise the band-drawing call falls back to an uninitialised state.
    Mirrors advancement.ipynb's loop that adds `model1`, `model2`, plus
    the shared `cpl`/`csbpl` components to one figure.

    **Naming convention**: pass `save='{savepath}/model'` for the standard
    overlay plot (matches examples/quickstart.py).
    """
    if not models:
        raise ValueError("models must contain at least one registered name")
    ms = [_get('models', n) for n in models]
    if with_posterior:
        if post is None:
            raise ValueError("with_posterior=True requires `post` (a registered Posterior name).")
        p = _get('posteriors', post)
        post_models = getattr(p, 'Model', [])
        for n, m in zip(models, ms, strict=True):
            if m not in post_models:
                raise ValueError(f"Model '{n}' was not part of posterior '{post}'.")
    earr = np.logspace(np.log10(e_min), np.log10(e_max), e_npts)
    modelplot = Plot.model(ploter=ploter, style=style, post=with_posterior)
    for m in ms:
        modelplot.add_model(m, E=earr)
    fig = modelplot.get_fig()
    img = _maybe_inline_image(fig, save) if inline else None
    fig.save(save)
    meta = _decorate(
        {
            'saved': save,
            'style': style,
            'ploter': ploter,
            'e_range': [e_min, e_max],
            'e_npts': e_npts,
            'with_posterior': with_posterior,
            'models': list(models),
        },
        after='plot_models',
        hint='Show the saved file path to the user.',
    )
    return [meta, img] if img is not None else meta


@mcp.tool()
def run_quickstart(
    data_units: list[dict[str, Any]],
    savepath: str = './quickstart',
    model: str = 'cpl',
    nlive: int = 400,
    resume: bool = True,
    random_seed: int | None = None,
    make_plots: bool = True,
    prefix: str = 'qs',
) -> dict[str, Any]:
    """One-shot reproduction of examples/quickstart.py.

    `data_units` is a list of dicts, each describing one detector. Example:

        [
            {"label": "nai",
             "src": "./ME/me.src", "bkg": "./ME/me.bkg", "rsp": "./ME/me.rsp",
             "notc": [8, 900],
             "rebn": {"min_sigma": 2, "max_bin": 10}},
            {"label": "bgo",
             "src": "./HE/he.src", "bkg": "./HE/he.bkg", "rsp": "./HE/he.rsp",
             "notc": [300, 38000],
             "rebn": {"min_sigma": 2, "max_bin": 10}}
        ]

    Every intermediate object is registered in the session so the granular
    tools (`plot_infer`, `plot_model`, `summarize_posterior`, ...) can take
    over. Names are `<prefix>_<label>` for units and `<prefix>_data`,
    `<prefix>_model`, `<prefix>_infer`, `<prefix>_post` for the rest. Reuses
    existing slot names (overwrite=True), since this is the one-shot entry.

    `model` is any name from `list_models` (default 'cpl').
    """
    factory = _resolve_model_factory(model)
    _ensure_dir(savepath)

    units: list[tuple[str, DataUnit]] = []
    for u in data_units:
        unit = DataUnit(
            src=u['src'],
            bkg=u.get('bkg'),
            rsp=u.get('rsp'),
            rmf=u.get('rmf'),
            arf=u.get('arf'),
            notc=u.get('notc'),
            stat=u.get('stat', 'pgstat'),
            rebn=u.get('rebn'),
            grpg=u.get('grpg'),
        )
        _register('data_units', f'{prefix}_{u["label"]}', unit, overwrite=True)
        units.append((u['label'], unit))

    data = Data(units)
    data.save(savepath)
    _register('data', f'{prefix}_data', data, overwrite=True)

    mdl = factory()
    mdl.save(savepath)
    _register('models', f'{prefix}_model', mdl, overwrite=True)

    inf = BayesInfer([(data, mdl)])
    inf.save(savepath)
    _register('infers', f'{prefix}_infer', inf, overwrite=True)

    posterior = inf.multinest(
        nlive=nlive,
        resume=resume,
        savepath=savepath,
        random_seed=random_seed,
    )
    posterior.save(savepath)
    _register('posteriors', f'{prefix}_post', posterior, overwrite=True)

    artifacts: list[str] = []
    if make_plots:
        # Plotly's save() writes both .html and .pdf, so even within one style
        # we suffix the basename with the ploter to avoid clobbering.
        spec_plots = [
            ('CE', 'plotly', 'ctsspec'),
            ('CE', 'matplotlib', 'ctsspec'),
            ('NE', 'plotly', 'phtspec'),
            ('NE', 'matplotlib', 'phtspec'),
            ('Fv', 'plotly', 'flxspec'),
            ('Fv', 'matplotlib', 'flxspec'),
            ('vFv', 'plotly', 'ergspec'),
            ('vFv', 'matplotlib', 'ergspec'),
        ]
        for style, ploter, basename in spec_plots:
            fig = Plot.infer(posterior, style=style, ploter=ploter)
            target = f'{savepath}/{basename}_{ploter}'
            fig.save(target)
            artifacts.append(target)

        for ploter in ('plotly', 'getdist'):
            fig = Plot.post_corner(posterior, ploter=ploter)
            target = f'{savepath}/corner_{ploter}'
            fig.save(target)
            artifacts.append(target)

        earr = np.logspace(1, 3, 100)
        for ploter in ('plotly', 'matplotlib'):
            modelplot = Plot.model(ploter=ploter, style='vFv', post=True)
            modelplot.add_model(mdl, E=earr)
            fig = modelplot.get_fig()
            target = f'{savepath}/model_{ploter}'
            fig.save(target)
            artifacts.append(target)

    summary = _posterior_summary(posterior)
    summary.update(
        {
            'savepath': savepath,
            'model': model,
            'nlive': nlive,
            'labels': [u['label'] for u in data_units],
            'registered': {
                'data_units': [f'{prefix}_{u["label"]}' for u in data_units],
                'data': f'{prefix}_data',
                'model': f'{prefix}_model',
                'infer': f'{prefix}_infer',
                'post': f'{prefix}_post',
            },
            'artifacts': artifacts,
        }
    )
    return _decorate(
        summary,
        after='run_quickstart',
        hint=(
            'Quickstart finished. Every intermediate object is registered, '
            'so the granular tools (plot_*, summarize_posterior, ...) can take '
            'over from here. Note: this convenience tool picks priors and '
            'sampler defaults silently — for non-trivial sources prefer the '
            'half-automatic flow.'
        ),
    )


@mcp.prompt()
def bayspec_workflow() -> str:
    """Standard playbook for guiding a user through a bayspec spectral fit.

    Pin this when starting a fit so the assistant follows the half-automatic
    protocol (user decides; assistant executes).
    """
    return (
        'You are helping the user fit an astronomical spectrum with bayspec via this MCP server.\n'
        '\n'
        '# Half-automatic protocol\n'
        '\n'
        'You execute, the user decides. Do NOT silently choose:\n'
        '  - the energy range to fit (detector + source physics);\n'
        '  - which spectral model to use;\n'
        '  - prior bounds on physical parameters;\n'
        '  - the sampler (emcee vs multinest);\n'
        '  - the result-save directory (savepath) — always ask the user;\n'
        '  - whether a non-converged fit should be re-run.\n'
        '\n'
        'When any of these is undetermined, ASK THE USER instead of guessing.\n'
        '\n'
        '# Default workflow\n'
        '\n'
        '1. `list_state` first — tells you what is already loaded.\n'
        '2. Resolve paths via `cwd` / `set_cwd` if the user mentions a working directory.\n'
        '3. Load data:\n'
        '     - one detector → `load_data_unit` then `build_data` with one unit;\n'
        '     - several detectors → loop `load_data_unit` then a single `build_data`.\n'
        '4. Build model:\n'
        '     - simple → `build_model("cpl")` etc.;\n'
        '     - composite → `build_model` each piece, then `compose_model("m", "tbabs * (cpl + bbody)")`.\n'
        '5. Discuss priors with the user. Call `describe_model` and propose sensible bounds.\n'
        '   When `set_param` returns warnings, surface them — do not bury them.\n'
        '6. `setup_infer` — then `describe_infer` to confirm what is free vs frozen.\n'
        '7. Pick sampler: call `suggest_sampler(n_free_params, expect_multimodal, need_evidence)`.\n'
        '   Confirm the recommendation with the user before running.\n'
        '8. **Ask the user for `savepath`** — the directory where the sampler dump\n'
        '   and plot files will go. Do not default silently to "./quickstart".\n'
        '9. Run the sampler. Do NOT silently rerun if it does not converge — report\n'
        '   diagnostics to the user and let them decide.\n'
        '10. After fitting:\n'
        '     - `summarize_posterior` for the table;\n'
        '     - `plot_infer(style="CE", save="{savepath}/ctsspec")`;\n'
        '     - `plot_infer(style="NE", save="{savepath}/phtspec")`;\n'
        '     - `plot_infer(style="Fv", save="{savepath}/flxspec")`;\n'
        '     - `plot_infer(style="vFv", save="{savepath}/ergspec")` (the SED);\n'
        '     - `plot_corner(save="{savepath}/corner")`.\n'
        '\n'
        '# File-naming convention\n'
        '\n'
        'Match `examples/quickstart.py`. Always use these basenames inside the\n'
        'user-chosen savepath directory; do NOT invent custom names like\n'
        '`fit_CE` or `my_corner`:\n'
        '\n'
        '  | plot                     | save argument                |\n'
        '  | ------------------------ | ---------------------------- |\n'
        '  | plot_infer style="CE"    | `{savepath}/ctsspec`         |\n'
        '  | plot_infer style="NE"    | `{savepath}/phtspec`         |\n'
        '  | plot_infer style="Fv"    | `{savepath}/flxspec`         |\n'
        '  | plot_infer style="vFv"   | `{savepath}/ergspec`         |\n'
        '  | plot_corner              | `{savepath}/corner`          |\n'
        '  | plot_model / plot_models | `{savepath}/model`           |\n'
        '\n'
        'When a single style is plotted with multiple ploters (plotly +\n'
        'matplotlib), keep the same basename — the ploter writes different\n'
        'extensions. If you genuinely need both files side by side, suffix\n'
        'with `_<ploter>` (e.g. `ctsspec_matplotlib`); never rename the stem.\n'
        '\n'
        '# Reading tool returns\n'
        '\n'
        'Every tool returns a dict with:\n'
        '  - tool-specific fields (`saved`, `par_best`, …);\n'
        '  - `state`: snapshot of registered objects;\n'
        '  - `ready_for`: tool names that make sense as the next step;\n'
        '  - `hint`: one-line guidance;\n'
        '  - `warnings`: present only when the tool flagged something (most often a\n'
        '    prior outside the typical range).\n'
        '\n'
        'Trust `ready_for` as your menu. If a tool errors with "object not\n'
        'registered", call `list_state` first.\n'
        '\n'
        '# When to push back on the user\n'
        '\n'
        '  - prior allows `low <= 0` on a positive-only quantity (norm, kT, Ec, NH);\n'
        '  - prior outside the physical range (e.g. PhoIndex < -5 or > 5);\n'
        '  - >10 free parameters with no opinion on multimodality — recommend\n'
        '    `suggest_sampler` first.\n'
        '\n'
        '# Convenience tool\n'
        '\n'
        '`run_quickstart` is end-to-end with default priors and multinest. Only use\n'
        'it for a known-easy case the user has explicitly approved; otherwise the\n'
        'half-automatic flow above is safer.\n'
    )


if __name__ == '__main__':
    mcp.run()
