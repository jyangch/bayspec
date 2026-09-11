bayspec.util package
====================

Submodules
----------

bayspec.util.corner module
--------------------------

.. automodule:: bayspec.util.corner
   :members:
   :undoc-members:
   :show-inheritance:

bayspec.util.group module
--------------------------

.. automodule:: bayspec.util.group
   :members:
   :undoc-members:
   :show-inheritance:

bayspec.util.info module
------------------------

.. automodule:: bayspec.util.info
   :members:
   :undoc-members:
   :show-inheritance:

bayspec.util.param module
-------------------------

.. automodule:: bayspec.util.param
   :members:
   :undoc-members:
   :show-inheritance:

bayspec.util.plot module
------------------------

.. automodule:: bayspec.util.plot
   :members:
   :undoc-members:
   :show-inheritance:

bayspec.util.post module
------------------------

.. automodule:: bayspec.util.post
   :members:
   :undoc-members:
   :show-inheritance:

bayspec.util.prior module
-------------------------

.. automodule:: bayspec.util.prior
   :members:
   :undoc-members:
   :show-inheritance:

bayspec.util.significance module
--------------------------------

``pgsig_inv`` and ``ppsig_inv`` invert a requested signed significance to the
corresponding observed count.  They return the continuous real-valued root by
default; pass ``integer=True`` to obtain the first integer count reaching the
target on the selected branch::

   from bayspec.util.significance import pgsig_inv, ppsig_inv

   gaussian_count = pgsig_inv(5, b=10, sigma=2)
   poisson_count = ppsig_inv(5, b=20, alpha=0.5, integer=True)

For the bounded-systematic form (Vianello 2018 eq. 7), positive targets use
the high-count branch beginning at ``alpha * (1 + k) * b``.  Negative targets
use the low-count branch below ``alpha * b``; targets in the discontinuity
between those branches are not reachable and raise ``ValueError``.

.. automodule:: bayspec.util.significance
   :members:
   :undoc-members:
   :show-inheritance:

bayspec.util.tools module
-------------------------

Model selection from exported information criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``select_model`` accepts a mapping from model names to ``ic_criteria`` bundles,
including dictionaries loaded from ``post_ic_criteria.json``:

.. code-block:: python

   import json
   from bayspec.util.tools import select_model

   ic_by_model = {}
   for model in ('PL', 'CPL'):
       with open(f'{model}/post_ic_criteria.json', encoding='utf-8') as stream:
           ic_by_model[model] = json.load(stream)

   comparison = select_model(ic_by_model, criterion='WAIC')
   best_model = comparison['best_model']

Only the requested criterion must be present in every model's bundle.
Current Analyzer exports omit LOOIC, even after explicit ``post.loo()`` calls;
WAIC, BIC, lnZ, and other available criteria can be compared without it.
LOOIC comparison remains supported for legacy or caller-supplied bundles
containing that criterion. Explicitly requesting LOOIC when any model lacks
it raises ``ValueError``; no LOO calculation is triggered, no model is silently
excluded, and no alternative criterion is substituted.

The score is the criterion value when larger is better (e.g. lnZ), or its
negative when smaller is better (e.g. BIC, WAIC, LOOIC). No rescaling is applied:
the threshold is in the chosen criterion's native units. ``threshold=None``
selects these defaults; an explicit positive finite value overrides them:

.. list-table:: Default selection tolerances
   :header-rows: 1

   * - Criterion
     - Threshold
     - Difference error
   * - AIC / AICc
     - 2
     - Not used
   * - BIC
     - ``ln(10)`` (about 2.30)
     - Not used
   * - lnZ
     - ``ln(10) / 2`` (about 1.15)
     - Independent evidence integration errors
   * - WAIC / LOOIC
     - 8 on deviance scale
     - Paired pointwise error

Other criteria require an explicit threshold. The automatic WAIC/LOOIC
thresholds require ``scale='deviance'`` and ``higher_is_better=False``;
the automatic lnZ threshold requires ``scale='log'`` and
``higher_is_better=True``.

The lnZ tolerance is the lower boundary of Jeffreys' "substantial" evidence
category; BIC uses the corresponding doubled value under the Bayes-factor
approximation. The LOOIC tolerance converts the ``loo`` FAQ's small-difference
heuristic of 4 ELPD units to 8 deviance units. WAIC adopts that tolerance on
the same scale. This is an engineering default for WAIC, not a separately
validated universal cutoff or an ArviZ-prescribed significance threshold.
See the `Jeffreys scale table
<https://revbayes.github.io/tutorials/model_selection_bayes_factors/bf_intro.html>`_
and the `loo cross-validation FAQ
<https://mc-stan.org/loo/articles/online-only/faq.html>`_.

First compare every model with the global highest-score model. A model enters
the candidate set if either of these conditions holds:

.. code-block:: python

   delta < threshold
   delta <= sigma * delta_error  # WAIC, LOOIC, and lnZ only

Here ``delta = highest_score - model_score``. ``sigma`` defaults to 2
and must be positive and finite. The fixed-threshold boundary is strict;
the error boundary is inclusive. Models without error estimates use only
the fixed threshold. Among candidates, select the fewest parameters, then
the highest score, then the lexicographically smallest model name,
independently of input order. Neither criterion values nor scores are
modified by the errors; the errors change candidate membership.

Missing or nonfinite values, mismatched data-point counts, optimization
directions, or scales raise ``ValueError``. Data-unit metadata is not checked,
including when it is present in older bundles. Identical input data, comparable
likelihoods, and matching channel ordering remain the caller's responsibility.
Diagnostic warnings or incomplete predictive diagnostics are reported without
automatically excluding models. They set ``selection_status='provisional'``
for the overall selection, even if the affected model is not selected.
Otherwise the status is ``'selected'``: this is not a reliability guarantee.
Penalties are already included in criterion values and are not added again.

``select_model`` always returns a dictionary containing the selected model,
diagnostics, and both sets of comparisons. Read ``best_model`` when only the
selected model name is needed:

.. code-block:: python

   comparison = select_model(ic_by_model, 'WAIC')
   comparison['best_model']
   comparison['highest_score_model']
   comparison['candidate_models']
   comparison['threshold']
   comparison['sigma']
   comparison['selection_status']
   comparison['models']['CPL']['diagnostics']

   selected = comparison['comparisons']['selected']
   highest_score = comparison['comparisons']['highest_score']
   selected['reference_model']
   selected['models']['CPL']['delta']
   selected['models']['CPL']['delta_error']
   selected['models']['CPL']['comparison_status']
   highest_score['models']['CPL']['delta']

The ``selected`` group uses the finally selected, potentially simpler model
as its reference. The ``highest_score`` group uses the global highest-score
model (name breaks exact ties). Each group compares all input models, including
those outside the candidate set. Per-model values, parameter counts, scores,
and diagnostics remain in the top-level ``models`` mapping and are not duplicated
in these comparison groups.

In both groups, ``delta = reference_score - model_score`` in native criterion
units: positive means worse than the reference, negative means better. Only
the highest-score group's differences are necessarily nonnegative, and these
are the differences used to form the candidate set. Each group's errors and
comparison statuses use its own reference. When the two references are the
same, both groups contain equal results. For WAIC/LOOIC, the difference
error is ``sqrt(N * var(model_pointwise - reference_pointwise, ddof=0))``,
using the same variance convention as ArviZ. The saved pointwise values are
already on deviance scale; no further factor of two is applied. This is a
paired, data-based standard error, not the quadrature sum of the individual
criterion errors, not a Monte Carlo error, and not a correction for bias.
Missing/nonfinite/misaligned pointwise arrays, inconsistent pointwise sums,
or fewer than two channels raise ``ValueError`` for WAIC/LOOIC selection,
whether or not detailed output is requested. The individual criterion's
``error`` field is not substituted for the paired difference error.

For lnZ, each bundle must provide a finite, nonnegative ``error``. Assuming
independent evidence calculations, the difference error is
``hypot(error_model, error_reference)``. This is a numerical integration
uncertainty, not uncertainty across observations and not prior sensitivity.
Correlated evidence estimates require covariance information that this
interface does not accept. A comparison of a model with itself has zero
error, not ``sqrt(2)`` times its evidence error. Missing or invalid errors
raise ``ValueError`` rather than silently becoming zero. For AIC, AICc, BIC,
and custom criteria, no difference error is estimated.

Diagnostic ``status`` is ``no_warning``, ``warning``, or ``insufficient`` for
predictive criteria, and ``not_assessed`` otherwise. ``reasons`` explains the
findings. ``no_warning`` means only that the available checks raised no warning;
it does not prove convergence or reliable predictive inference. WAIC uses its
stored warning flag. LOOIC additionally checks Pareto-k against ``good_k`` and
reports zero-based channel indices for high k, nearly constant likelihoods,
PSIS failures, and unexplained undefined k (``unexplained_k_channels``).
Near-constant raw-weight channels are not treated as failed PSIS fits.
Legacy bundles with unexplained null k
values are marked as diagnostically insufficient, unless another diagnostic
already warrants a warning.

``comparison_status`` incorporates diagnostics from both models in a pair.
An error can still be calculated when that status is ``warning`` or
``insufficient``, but is not thereby certified reliable. It is not a
significance decision: small samples, outliers, and closely matched models
can make normal approximations misleading. Retaining a model because its
difference is small relative to its error means it is not clearly separated
by this rule; it does not establish model equivalence. Choosing the simpler
candidate is a decision preference. For lnZ, ``comparison_status`` remains
``not_assessed`` because predictive reliability diagnostics do not apply.

.. automodule:: bayspec.util.tools
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: bayspec.util
