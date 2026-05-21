Installation
============

*BaySpec* is available on PyPI:

.. code:: bash

   pip install bayspec


Optional: ``MultiNest`` sampler
-------------------------------

To enable `MultiNest <https://github.com/rjw57/MultiNest>`_ for nested
sampling, follow the
`pymultinest <https://johannesbuchner.github.io/PyMultiNest/>`_ install
guide.


Optional: ``astromodels`` components
------------------------------------

To pull components from
`astromodels <https://astromodels.readthedocs.io/en/latest/notebooks/function_list.html>`_,
install ``astromodels`` in your Python environment.


Optional: ``Xspec`` components
------------------------------

To pull components from
`Xspec <https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/Models.html>`_:

1. Install ``HEASoft`` and ``Xspec v12.12.1+``.
2. After initialising the ``HEASoft`` environment, install
   `xspec-models-cxc <https://github.com/cxcsds/xspec-models-cxc>`_.

.. note::

   *BaySpec* currently supports only ``Additive`` and ``Multiplicative``
   ``Xspec`` models.
