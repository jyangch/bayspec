bayspec.infer package
=====================

Statistic results
-----------------

Starting with BaySpec 0.4.0, every low-level statistic function returns a
``StatisticResult`` instead of a ``(stat, residual)`` tuple:

.. code:: python

   from bayspec import Statistic

   result = Statistic.PGstat(
       S=source_counts,
       B=background_counts,
       m=model_rate,
       ts=source_exposure,
       tb=background_exposure,
       sigma_S=source_error,
       sigma_B=background_error,
   )

   print(result.stat)
   print(result.residual)
   print(result.pointwise_loglike)
   print(result.pointwise_sign)

``pointwise_loglike`` and ``pointwise_sign`` are the primitive channel-wise
outputs. ``pointwise_stat``, ``stat``, ``loglike``, and ``residual`` are
derived from them.

Likewise, a custom ``Infer.loglike_func`` must return one log-likelihood value
per fitted channel rather than a summed scalar. ``calc_loglike(theta)`` derives
the total by summing that vector. This guarantees that posterior sampling and
WAIC/LOO use the same likelihood definition.

``posterior_sample`` and ``bootstrap_sample`` contain parameter draws only,
with shape ``(nsample, nfree)``. Constructing ``Posterior`` or ``Bootstrap``
eagerly evaluates and exposes the corresponding derived arrays:

.. code:: python

   analyzer.pointwise_loglike_sample  # (nsample, npoint)
   analyzer.loglike_sample            # (nsample,)
   analyzer.logprior_sample           # (nsample,)
   analyzer.logprob_sample            # (nsample,)

Posterior draws are ranked by ``logprob_sample``; bootstrap draws are ranked
by ``loglike_sample``. Sample files written by older versions include a final
score column and must be regenerated or converted before loading.

Constructing ``Posterior`` also calculates and caches its default WAIC result.
PSIS-LOO is computed only on an explicit ``post.loo()`` call, not during
initialization, display, or saving. ``Posterior.IC_info`` reports the
lower-is-better deviance-scale ``WAIC`` with its standard error; it never
includes LOOIC, even after an explicit LOO calculation.
Nested-sampling evidence is likewise shown as ``lnZ ± lnZ_err``. The numeric
values remain available from ``post.waic()``, ``post.loo()``, ``post.lnZ``, and
``post.lnZ_err``. The explicit pointwise LOO result includes ``loo_i``,
``pareto_k``, ``good_k``, ``nearly_constant``, and ``psis_failed``. The latter
two boolean arrays distinguish near-constant likelihood channels from actual
PSIS failures that used raw-weight fallback. LOO results retain their existing
argument-aware caching and are invalidated when samples are reloaded.
ArviZ's WAIC-variance warnings are suppressed during initialization; inspect
``waic.warning`` explicitly. Runtime warnings matching
``overflow encountered in ...`` are also suppressed during this eager
calculation; other runtime warnings remain visible.

Machine-readable information criteria
-------------------------------------

``analyzer.ic_criteria`` packages unrounded information criteria in a plain
dictionary. ``save`` writes the bundle alongside the display tables:

.. code:: python

   post.ic_criteria["criteria"]["BIC"]["value"]
   post.ic_criteria["criteria"]["WAIC"]["value"]
   post.ic_criteria["criteria"]["lnZ"]["value"]
   post.save("results/pl")

   import json

   with open("results/pl/post_ic_criteria.json", encoding="utf-8") as stream:
       pl_ic = json.load(stream)
   with open("results/cpl/post_ic_criteria.json", encoding="utf-8") as stream:
       cpl_ic = json.load(stream)

The dictionaries can be passed directly to a separately defined comparison
function. ``Posterior`` bundles contain ``AIC``, ``AICc``, ``BIC``, ``WAIC``,
and ``lnZ`` under ``criteria``; ``Bootstrap`` bundles contain only
``AIC``, ``AICc``, and ``BIC``. Each criterion has a numeric ``value`` and a
``higher_is_better`` flag. No model-selection threshold is applied on export.

LOOIC is always omitted from ``ic_criteria`` and its saved JSON, even after
``post.loo()`` has been called. Explicit LOO results are returned to the caller
only; no cached LOO result is automatically inserted into the export.

For WAIC, ``value``, ``error``, and ``pointwise`` are all on the
lower-is-better deviance scale (``-2 * ELPD``). The bundle also preserves
the effective parameter count as ``penalty`` and the ``warning`` flag.
The pointwise arrays allow a comparison function to calculate
paired differences and their standard error without loading posterior draws.
The standard error of a difference must be calculated from these paired
contributions, not by combining the two individual standard errors.

``penalty`` is the unscaled ``p_waic``, the sum of posterior log-likelihood
variances over observations. With ``lppd`` denoting the sum of log
posterior-mean likelihoods for the fitted observations, WAIC satisfies
``value = -2 * lppd + 2 * penalty``. This correction for training-data
optimism is often interpreted as an effective number of parameters; it
need not equal the actual parameter count and can be non-integer.
It is already included in ``value`` and must not be added again.

Evidence is stored as ``lnZ.value`` and ``lnZ.error``, both on natural-log
scale. ``lnZ.error`` is the nested-sampling uncertainty, whereas predictive
criteria's ``error`` describes uncertainty across observations. Larger ``lnZ``
is preferred. Unavailable evidence (for example, after emcee) remains
``null`` and must not be treated as zero.

The JSON also records its analyzer/sampler type, sample
and parameter counts, model expressions, and ordered data units. Each unit
contains its statistic, weight, channel energy bins in keV, and a half-open
``slice`` into the pointwise arrays. This assists channel alignment but does
not prove that two files describe identical observations: the comparison
function's caller must ensure the same input data, likelihood convention,
and channel selection/grouping were used.

The bundle uses standard JSON: missing/undefined numeric entries are
``null``, and positive/negative infinities are the strings ``"Infinity"``
and ``"-Infinity"``. For numeric diagnostic arrays,
``np.asarray(values, dtype=float)`` restores ``null`` as NaN and the infinity
strings as floating-point infinities. Comparison functions must check for
unavailable/non-finite scores and diagnostic warnings before ranking models.

``post.save(directory)`` additionally writes ``post_ic_criteria.json``;
``bootstrap.save(directory)`` writes ``boot_ic_criteria.json``. The existing
``post_IC.json`` / ``boot_IC.json`` files retain their formatted table layout.

Submodules
----------

bayspec.infer.infer module
--------------------------

.. automodule:: bayspec.infer.infer
   :members:
   :undoc-members:
   :show-inheritance:

bayspec.infer.pair module
-------------------------

.. automodule:: bayspec.infer.pair
   :members:
   :undoc-members:
   :show-inheritance:

bayspec.infer.analyzer module
-----------------------------

.. automodule:: bayspec.infer.analyzer
   :members:
   :undoc-members:
   :show-inheritance:

bayspec.infer.statistic module
------------------------------

.. automodule:: bayspec.infer.statistic
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: bayspec.infer
