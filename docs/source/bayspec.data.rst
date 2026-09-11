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

Optimal grouping
~~~~~~~~~~~~~~~~

``DataUnit`` supports Kaastra--Bleeker optimal grouping based on the
HEASP/``ftgrouppha`` width calculation::

   unit = DataUnit(src=src, rsp=rsp, grpg={'method': 'optimal'})

The calculation uses the observed source-region counts and the response FWHM;
the background is not subtracted when determining the optimal width. Quality
and noticing selection is applied before estimating widths. All selected,
quality-good channels in a unit share the resolution-element estimate
``R = 1 + sum(1 / FWHM)``. Each local FWHM count window is clipped to its
contiguous valid interval, retaining original channel coordinates; excluded
counts cannot affect widths or grouping. The factor ``1.314`` is retained at
clipped boundaries as an approximation. Count and window-boundary rounding
remain half-away-from-zero, as in HEASP.

Groups cannot cross quality or noticing gaps, and short boundary groups are
retained. Changing ``notc`` recalculates automatic grouping. With every channel
valid, the result matches HEASP; selected-range widths can differ because
HEASP estimates widths using the full observed channel array. The response
FWHM itself is still calculated from the full response.

BaySpec thresholds can be applied on top of the optimal width::

   unit = DataUnit(
       src=src,
       bkg=bkg,
       rsp=rsp,
       grpg={'method': 'optimal', 'min_evt': 20, 'min_sigma': 3},
   )

The Kaastra--Bleeker width is the minimum allowed width. Thresholds extend a
bin to the right one channel at a time until they are met, ``max_bin`` is
reached, or the selected segment ends. Reaching ``max_bin`` closes the bin even
when a threshold remains unmet. A final under-threshold tail is merged backward
only while the merged bin remains within ``max_bin``; otherwise it is retained
as a separate bin. A ``max_bin`` smaller than the initial optimal width is
rejected because the two width requirements conflict. Responses containing a
non-positive or non-finite channel FWHM are also rejected explicitly.

For ``BalrogResponse``, ``method='optimal'`` requires both ``ra`` and ``dec``
to be frozen because changing either coordinate changes the response on which
the grouping is based. Use ``method='threshold'`` when fitting sky position.

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
