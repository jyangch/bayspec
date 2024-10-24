Installation
==================

*BaySpec* is available via pip:

.. code:: bash

   $ pip install bayspec

Utilize *multinest* sampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use
`Multinest <https://github.com/rjw57/MultiNest>`_ for Bayesian
inference, you can follow the instructions in the
`pymultinest <https://johannesbuchner.github.io/PyMultiNest/>`_
documentation to install it.

Access *Astromodels* models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To utilize models from
`Astromodels <https://astromodels.readthedocs.io/en/latest/notebooks/function_list.html#>`_,
ensure that *Astromodels* is installed on your system.

Access *Xspec* models
~~~~~~~~~~~~~~~~~~~~~~~

To utilize models from
`Xspec <https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/Models.html>`_,
ensure that both *HEASoft* and *Xspec v12.12.1+* are installed on
your system. After confirming that *HEASoft* environment is properly
initialized, then you need install
`xspec-models-cxc <https://github.com/cxcsds/xspec-models-cxc>`_.

**NOTE**: *BaySpec* currently only supports *Additive* and
*Multiplicative* models in *Xspec*.
