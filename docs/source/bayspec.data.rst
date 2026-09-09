bayspec.data package
====================

Submodules
----------

bayspec.data.data module
------------------------

.. automodule:: bayspec.data.data
   :members:
   :undoc-members:
   :show-inheritance:

Count significance limits
~~~~~~~~~~~~~~~~~~~~~~~~~

``DataUnit.net_counts_significance_limit(sig=3)`` returns the continuous net
count threshold at which the source-region observation reaches the requested
significance. ``pgstat`` uses ``pgsig_inv`` with the background counts and
quadrature background error scaled into the source region. ``ppstat`` and
``cstat`` use ``ppsig_inv`` with the raw background-region counts and the
source-to-background efficiency ratio ``alpha``. ``pstat`` is rejected because
it does not supply a background measurement for either significance model;
Gaussian-source statistics (``gstat`` and ``chi2``) are rejected because these
inverse functions assume Poisson source counts.

The corresponding count-rate threshold is available from
``DataUnit.net_ctsrate_significance_limit``. A ``Data`` container exposes both
methods and returns one result per unit::

   net_count_limits = data.net_counts_significance_limit(sig=3)
   net_rate_limits = data.net_ctsrate_significance_limit(sig=3)

These are detection thresholds, not confidence or credible upper limits. The
existing ``net_counts_upperlimit(cl=...)`` and
``net_ctsrate_upperlimit(cl=...)`` methods retain their Poisson credible-limit
definition. Neither kind of count limit is a flux limit without an assumed
spectrum, absorption model, and detector response.

bayspec.data.response module
----------------------------

.. automodule:: bayspec.data.response
   :members:
   :undoc-members:
   :show-inheritance:

bayspec.data.spectrum module
----------------------------

.. automodule:: bayspec.data.spectrum
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: bayspec.data
