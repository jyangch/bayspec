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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``select_model`` accepts a mapping from model names to ``ic_criteria`` bundles,
including dictionaries loaded from ``post_ic_criteria.json``:

.. code-block:: python

   import json
   from bayspec.util.tools import select_model

   ic_by_model = {}
   for model in ('PL', 'CPL'):
       with open(f'{model}/post_ic_criteria.json', encoding='utf-8') as stream:
           ic_by_model[model] = json.load(stream)

   best_model = select_model(ic_by_model, criterion='WAIC', threshold=2.0)

Only the requested criterion must be present in every model's bundle.
Current Analyzer exports omit LOOIC, even after explicit ``post.loo()`` calls;
WAIC, BIC, lnZ, and other available criteria can be compared without it.
LOOIC comparison remains supported for legacy or caller-supplied bundles
containing that criterion. Explicitly requesting LOOIC when any model lacks
it raises ``ValueError``; no LOO calculation is triggered, no model is silently
excluded, and no alternative criterion is substituted.

The score is the criterion value when larger is better (e.g. lnZ), or its
negative when smaller is better (e.g. BIC, WAIC, LOOIC). No rescaling is applied:
the threshold is in the chosen criterion's native units. Candidates are
strictly less than the threshold below the global best score. The selected
candidate has the fewest parameters; ties prefer the highest score, then the
lexicographically smallest model name, independently of input order.

Missing or nonfinite values, mismatched data-point counts, optimization
directions, or scales raise ``ValueError``. Data-unit metadata is not checked,
including when it is present in older bundles. Identical input data, comparable
likelihoods, and matching channel ordering remain the caller's responsibility.
Diagnostic warnings are reported without excluding models. Neither
uncertainties nor penalties are added to the selection score.

Use ``return_details=True`` to inspect predictive diagnostics and paired
differences without changing that selection rule:

.. code-block:: python

   comparison = select_model(ic_by_model, 'WAIC', threshold=2.0, return_details=True)
   comparison['best_model']
   comparison['highest_score_model']
   comparison['candidate_models']
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
Missing/nonfinite/misaligned pointwise arrays or inconsistent pointwise sums
raise ``ValueError`` in detailed predictive comparisons. With fewer than two
channels, ``delta_error`` is ``None`` and the comparison lacks enough information
to assess uncertainty. For BIC, lnZ, and other nonpredictive criteria, no
difference error is estimated.

Diagnostic ``status`` is ``no_warning``, ``warning``, or ``insufficient`` for
predictive criteria, and ``not_assessed`` otherwise. ``reasons`` explains the
findings. ``no_warning`` means only that the available checks raised no warning;
it does not prove convergence or reliable predictive inference. WAIC uses its
stored warning flag. LOOIC additionally checks Pareto-k against ``good_k`` and
reports zero-based channel indices for high k, nearly constant likelihoods,
PSIS failures, and unexplained undefined k (``unexplained_k_channels``).
Near-constant raw-weight channels
are not treated as failed PSIS fits. Legacy bundles with unexplained null k
values are marked as diagnostically insufficient, unless another diagnostic
already warrants a warning.

``comparison_status`` incorporates diagnostics from both models in a pair.
An error can still be calculated when that status is ``warning`` or
``insufficient``, but is not thereby certified reliable. It is not a
significance decision: small samples, outliers, and closely matched models
can make normal approximations misleading. The selection threshold is never
replaced by a difference/error ratio.

.. automodule:: bayspec.util.tools
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: bayspec.util
