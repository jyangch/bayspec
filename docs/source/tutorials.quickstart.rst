A Quick Start Tutorial
======================

This tutorial offers a quick-start guide for using *BaySpec* to fit a
spectral model to gamma-ray data. It can be broadly divided into the
following three sections:

- Data: The spectra from Fermi/GBM’s NaI detector and BGO detector.
- Model: A simple cutoff power-law function.
- Fitting: Bayesian inference implemented using *multinest*.

.. code:: ipython3

    import numpy as np
    from bayspec.model.local import *
    from bayspec import DataUnit, Data, Infer, Plot

1. Load spectra and response data.

.. code:: ipython3

    nai = DataUnit(
        src='./ME/me.src', 
        bkg='./ME/me.bkg', 
        rsp='./ME/me.rsp', 
        notc=[8, 900], 
        stat='pgstat', 
        grpg={'min_sigma': 3, 'max_bin': 10})
    
    bgo = DataUnit(
        src='./HE/he.src', 
        bkg='./HE/he.bkg', 
        rsp='./HE/he.rsp', 
        notc=[300, 38000], 
        stat='pgstat', 
        grpg={'min_sigma': 3, 'max_bin': 10})
    
    data = Data([('nai', nai), 
                 ('bgo', bgo)])
    
    print(data)


.. parsed-literal::

    ╒════════╤════════════════╤═════════════╤═════════════════════════════════╤════════╕
    │  Name  │    Noticing    │  Statistic  │            Grouping             │  Time  │
    ╞════════╪════════════════╪═════════════╪═════════════════════════════════╪════════╡
    │  nai   │   [[8, 900]]   │   pgstat    │ {'min_sigma': 3, 'max_bin': 10} │  None  │
    ├────────┼────────────────┼─────────────┼─────────────────────────────────┼────────┤
    │  bgo   │ [[300, 38000]] │   pgstat    │ {'min_sigma': 3, 'max_bin': 10} │  None  │
    ╘════════╧════════════════╧═════════════╧═════════════════════════════════╧════════╛
    


2.Define spectral model.

.. code:: ipython3

    model = cpl()
    print(model)


.. parsed-literal::

    cpl [add]
    cutoff power law model
    ╒════════╤═════════════╤═════════════╤═════════╕
    │  cfg#  │  Component  │  Parameter  │  Value  │
    ╞════════╪═════════════╪═════════════╪═════════╡
    │   1    │     cpl     │  redshift   │    0    │
    ╘════════╧═════════════╧═════════════╧═════════╛
    ╒════════╤═════════════╤═════════════╤═════════╤═════════════╤══════════╕
    │  par#  │  Component  │  Parameter  │  Value  │    Prior    │  Frozen  │
    ╞════════╪═════════════╪═════════════╪═════════╪═════════════╪══════════╡
    │   1    │     cpl     │  $\\alpha$   │   -1    │ unif(-8, 4) │  False   │
    ├────────┼─────────────┼─────────────┼─────────┼─────────────┼──────────┤
    │   2    │     cpl     │ log$E_{c}$  │    2    │ unif(0, 4)  │  False   │
    ├────────┼─────────────┼─────────────┼─────────┼─────────────┼──────────┤
    │   3    │     cpl     │   log$A$    │   -1    │ unif(-6, 5) │  False   │
    ╘════════╧═════════════╧═════════════╧═════════╧═════════════╧══════════╛
    


3.Run Bayesian inference.

.. code:: ipython3

    infer = Infer([(data, model)])
    print(infer)


.. parsed-literal::

    ╒════════╤══════════════╤═════════════╤═════════════╤═════════╕
    │  cfg#  │  Expression  │  Component  │  Parameter  │  Value  │
    ╞════════╪══════════════╪═════════════╪═════════════╪═════════╡
    │   1    │     cpl      │     cpl     │  redshift   │    0    │
    ╘════════╧══════════════╧═════════════╧═════════════╧═════════╛
    ╒════════╤══════════════╤═════════════╤═════════════╤═════════╤═════════════╕
    │  par#  │  Expression  │  Component  │  Parameter  │  Value  │    Prior    │
    ╞════════╪══════════════╪═════════════╪═════════════╪═════════╪═════════════╡
    │   1*   │     cpl      │     cpl     │  $\\alpha$   │   -1    │ unif(-8, 4) │
    ├────────┼──────────────┼─────────────┼─────────────┼─────────┼─────────────┤
    │   2*   │     cpl      │     cpl     │ log$E_{c}$  │    2    │ unif(0, 4)  │
    ├────────┼──────────────┼─────────────┼─────────────┼─────────┼─────────────┤
    │   3*   │     cpl      │     cpl     │   log$A$    │   -1    │ unif(-6, 5) │
    ╘════════╧══════════════╧═════════════╧═════════════╧═════════╧═════════════╛
    


.. code:: ipython3

    post = infer.multinest(nlive=300, resume=True, savepath='./multinest')
    print(post)


.. parsed-literal::

     *****************************************************
     MultiNest v3.10
     Copyright Farhan Feroz & Mike Hobson
     Release Jul 2015
    
     no. of live points =  300
     dimensionality =    3
     resuming from previous job
     *****************************************************
     Starting MultiNest
    Acceptance Rate:                        0.562397
    Replacements:                               6165
    Total Samples:                             10962
    Nested Sampling ln(Z):               -229.153983
    Importance Nested Sampling ln(Z):    -229.261712 +/-  0.022377
      analysing data from ./multinest/1-.txt
     ln(ev)=  -228.86015520780347      +/-  0.23816955101640733     
     Total Likelihood Evaluations:        10962
     Sampling finished. Exiting MultiNest
    ╒════════╤══════════════╤═════════════╤═════════════╤════════╤══════════╤════════╤══════════════════╕
    │  par#  │  Expression  │  Component  │  Parameter  │  Mean  │  Median  │  Best  │    1sigma CI     │
    ╞════════╪══════════════╪═════════════╪═════════════╪════════╪══════════╪════════╪══════════════════╡
    │   1    │     cpl      │     cpl     │  $\\alpha$   │ -1.563 │  -1.562  │ -1.562 │ [-1.573, -1.552] │
    ├────────┼──────────────┼─────────────┼─────────────┼────────┼──────────┼────────┼──────────────────┤
    │   2    │     cpl      │     cpl     │ log$E_{c}$  │ 2.691  │   2.69   │  2.69  │  [2.673, 2.709]  │
    ├────────┼──────────────┼─────────────┼─────────────┼────────┼──────────┼────────┼──────────────────┤
    │   3    │     cpl      │     cpl     │   log$A$    │ -0.771 │  -0.771  │ -0.771 │ [-0.778, -0.765] │
    ╘════════╧══════════════╧═════════════╧═════════════╧════════╧══════════╧════════╧══════════════════╛
    ╒════════╤═════════╤═════════════╤════════════╤════════╕
    │  Data  │  Model  │  Statistic  │   Value    │  Bins  │
    ╞════════╪═════════╪═════════════╪════════════╪════════╡
    │  nai   │   cpl   │   pgstat    │   388.43   │  106   │
    ├────────┼─────────┼─────────────┼────────────┼────────┤
    │  bgo   │   cpl   │   pgstat    │   32.18    │   26   │
    ├────────┼─────────┼─────────────┼────────────┼────────┤
    │ Total  │  Total  │  stat/dof   │ 420.61/129 │  132   │
    ╘════════╧═════════╧═════════════╧════════════╧════════╛
    ╒════════╤════════╤════════╤═════════╕
    │  AIC   │  AICc  │  BIC   │   lnZ   │
    ╞════════╪════════╪════════╪═════════╡
    │ 426.61 │ 426.8  │ 435.26 │ -229.26 │
    ╘════════╧════════╧════════╧═════════╛
    


.. code:: ipython3

    fig = Plot.infer_ctsspec(post, style='CE', show=True)



.. raw:: html

    <iframe src="_static/qs_ctsspec.html" width="100%" height="600px" frameborder="0"></iframe>



.. code:: ipython3

    fig = Plot.post_corner(post, show=True)



.. raw:: html

    <iframe src="_static/qs_corner.html" width="100%" height="600px" frameborder="0"></iframe>



.. code:: ipython3

    earr = np.logspace(np.log10(0.5), 3, 100)
    
    modelplot = Plot.model(ploter='plotly', style='vFv', CI=True)
    fig = modelplot.add_model(model, E=earr, show=True)



.. raw:: html

    <iframe src="_static/qs_model.html" width="100%" height="600px" frameborder="0"></iframe>
    