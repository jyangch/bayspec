bayspec.infer package
=====================

Statistic results
-----------------

Starting with BaySpec 0.4.0, every low-level statistic function returns a
``StatisticResult`` instead of a ``(stat, residual)`` tuple:

.. code:: python

   from bayspec import StatisticNB

   result = StatisticNB.PGstat(
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

Constructing ``Posterior`` also calculates and caches its default WAIC and
PSIS-LOO results. ``Posterior.IC_info`` reports their lower-is-better deviance
forms as ``WAIC`` and ``LOOIC``, formatted together with their standard errors.
Nested-sampling evidence is likewise shown as ``lnZ ± lnZ_err``. The numeric
values remain available from ``post.waic()``, ``post.loo()``, ``post.lnZ``, and
``post.lnZ_err``; the ArviZ results also include pointwise values and Pareto-k
diagnostics. ArviZ's WAIC-variance and Pareto-k warnings are suppressed during
this eager calculation; inspect ``waic.warning``, ``loo.warning``, and
``loo.pareto_k`` explicitly. Runtime warnings matching
``overflow encountered in ...`` are also suppressed during this eager
calculation; other runtime warnings remain visible.

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
