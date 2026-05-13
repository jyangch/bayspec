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
CornerPloter = Literal['plotly', 'matplotlib', 'getdist']
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
            summary[table] = getattr(owner, 'data_list_dict', None)
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
        else:
            return None
        return Image(path=png_path)
    except Exception:
        return None


@mcp.tool()
def cwd() -> str:
    """Return the server's current working directory."""
    return os.getcwd()


@mcp.tool()
def set_cwd(path: str) -> str:
    """Change the server's working directory; returns the new cwd."""
    os.chdir(path)
    return os.getcwd()


@mcp.tool()
def list_models() -> list[str]:
    """List spectral models available under bayspec.model.local."""
    return sorted(_LOCAL_MODELS)


@mcp.tool()
def list_priors() -> list[str]:
    """List prior distributions accepted by `set_param`."""
    return sorted(_ALL_PRIORS)


@mcp.tool()
def list_state() -> dict[str, list[str]]:
    """Return the names of all registered session objects, grouped by kind."""
    return {kind: sorted(reg) for kind, reg in STATE.items()}


@mcp.tool()
def reset_session() -> str:
    """Clear all session-scoped objects (data units, data, models, infers, posteriors)."""
    for reg in STATE.values():
        reg.clear()
    return 'session cleared'


@mcp.tool()
def delete_object(kind: Literal['data_units', 'data', 'models', 'infers', 'posteriors'], name: str) -> str:
    """Remove a single registered object from the session."""
    if name not in STATE[kind]:
        raise ValueError(f"No {kind[:-1]} registered under name '{name}'.")
    del STATE[kind][name]
    return f'deleted {kind[:-1]} {name!r}'


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
    return {'name': name, 'src': src, 'stat': stat, 'notc': notc}


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
    return {
        'name': name,
        'labels': [u['label'] for u in units],
        'saved_to': saved,
    }


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
    return {'name': name, 'kind': kind, 'alias': alias, 'saved_to': saved}


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
    return {
        'name': name,
        'expr': composite.expr,
        'type': composite_type,
        'components': sorted(referenced),
    }


@mcp.tool()
def describe_model(name: str) -> dict[str, Any]:
    """Inspect a registered model: type, fit-parameters (id/value/prior/frozen), and configs.

    Use the returned `par_id` strings with `set_param` to mutate values, priors,
    or freeze/thaw individual parameters.
    """
    m = _get('models', name)
    return {
        'name': name,
        'type': getattr(m, 'type', None),
        'expr': getattr(m, 'expr', None),
        'params': [
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
        ],
        'configs': list(m.all_config),
    }


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
    _apply_param_changes(par, val, frozen, prior)
    return {
        'model': model,
        'par_id': par_id,
        'val': par.val,
        'frozen': par.frozen,
        'prior': par.prior_info,
    }


@mcp.tool()
def describe_infer(name: str) -> dict[str, Any]:
    """Inspect a registered BayesInfer: configs, all params, and the free subset.

    `params[i].mates` lists the `par#` of every slot that shares the same
    underlying parameter (so changing one propagates). `free_params` is the
    canonical order MultiNest/emcee will see.
    """
    inf = _get('infers', name)
    inf._you_free()
    return {
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
        'free_nparams': inf.free_nparams,
        'free_plabels': inf.free_plabels,
    }


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
    _apply_param_changes(par, val, frozen, prior)
    inf._you_free()
    return {
        'infer': infer,
        'par_id': par_id,
        'val': par.val,
        'frozen': par.frozen,
        'prior': par.prior_info,
        'free_nparams': inf.free_nparams,
    }


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
    return {'infer': infer, 'cfg_id': cfg_id, 'val': inf.cfg[cfg_id].val}


@mcp.tool()
def link_params(infer: str, par_ids: list[str]) -> dict[str, Any]:
    """Link parameter slots so they share value/prior/posterior."""
    inf = _get('infers', infer)
    for pid in par_ids:
        if str(pid) not in inf.par:
            raise ValueError(f"par_id '{pid}' not in infer '{infer}'.")
    inf.link([str(p) for p in par_ids])
    return {'infer': infer, 'linked': [str(p) for p in par_ids], 'free_nparams': inf.free_nparams}


@mcp.tool()
def unlink_params(infer: str, par_ids: list[str]) -> dict[str, Any]:
    """Undo any links between every pair drawn from `par_ids`."""
    inf = _get('infers', infer)
    for pid in par_ids:
        if str(pid) not in inf.par:
            raise ValueError(f"par_id '{pid}' not in infer '{infer}'.")
    inf.unlink([str(p) for p in par_ids])
    return {'infer': infer, 'unlinked': [str(p) for p in par_ids], 'free_nparams': inf.free_nparams}


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
    return {'name': name, 'pairs': pairs, 'saved_to': saved}


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
    return summary


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
    return summary


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
    return summary


@mcp.tool()
def summarize_posterior(post: str) -> dict[str, Any]:
    """Return best-fit params, credible intervals, and information criteria."""
    return _posterior_summary(_get('posteriors', post))


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
    return {
        'model': model,
        'kind': kind,
        'at': at if post is not None else None,
        'post': post,
        'E': earr.tolist(),
        'Y': np.asarray(y).tolist(),
    }


@mcp.tool()
def plot_infer(
    post: str,
    save: str,
    style: PlotStyle = 'CE',
    ploter: SpecPloter = 'plotly',
    inline: bool = False,
) -> list[Any] | dict[str, str]:
    """Plot data + best-fit + residuals from a posterior.

    `style`: 'CE' counts, 'NE' photon, 'Fv' flux, 'vFv' nuFnu, plus 'CC'
    and 'NoU' diagnostics. `save` is the output basename — the ploter picks
    the extension (plotly emits .html and .pdf; matplotlib emits .pdf).
    When `inline=True`, also writes a PNG and returns it as MCP Image
    content (best-effort: needs kaleido for plotly).
    """
    p = _get('posteriors', post)
    fig = Plot.infer(p, style=style, ploter=ploter)
    img = _maybe_inline_image(fig, save) if inline else None
    fig.save(save)
    meta = {'saved': save, 'style': style, 'ploter': ploter}
    return [meta, img] if img is not None else meta


@mcp.tool()
def plot_corner(
    post: str,
    save: str,
    ploter: CornerPloter = 'plotly',
    inline: bool = False,
) -> list[Any] | dict[str, str]:
    """Corner plot of posterior samples. `ploter` is 'plotly', 'matplotlib', or 'getdist'."""
    p = _get('posteriors', post)
    fig = Plot.post_corner(p, ploter=ploter)
    img = _maybe_inline_image(fig, save) if inline else None
    fig.save(save)
    meta = {'saved': save, 'ploter': ploter}
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
    meta = {
        'saved': save,
        'style': style,
        'ploter': ploter,
        'e_range': [e_min, e_max],
        'e_npts': e_npts,
        'with_posterior': with_posterior,
    }
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
    meta = {
        'saved': save,
        'style': style,
        'ploter': ploter,
        'e_range': [e_min, e_max],
        'e_npts': e_npts,
        'with_posterior': with_posterior,
        'models': list(models),
    }
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
    return summary


if __name__ == '__main__':
    mcp.run()
