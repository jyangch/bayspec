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

.. automodule:: bayspec.util.tools
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: bayspec.util
