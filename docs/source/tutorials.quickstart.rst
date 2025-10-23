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

.. code:: ipython3

    savepath = 'quickstart'

1. Load spectra and response data.

.. code:: ipython3

    nai = DataUnit(
        src='./ME/me.src', 
        bkg='./ME/me.bkg', 
        rsp='./ME/me.rsp', 
        notc=[8, 900], 
        stat='pgstat', 
        rebn={'min_sigma': 2, 'max_bin': 10})
    
    bgo = DataUnit(
        src='./HE/he.src', 
        bkg='./HE/he.bkg', 
        rsp='./HE/he.rsp', 
        notc=[300, 38000], 
        stat='pgstat', 
        rebn={'min_sigma': 2, 'max_bin': 10})
    
    data = Data([('nai', nai), 
                 ('bgo', bgo)])
    
    data.save(savepath)
    print(data)


.. parsed-literal::

    ╒════════╤════════════════╤═════════════╤════════════╤════════╕
    │  Name  │    Noticing    │  Statistic  │  Grouping  │  Time  │
    ╞════════╪════════════════╪═════════════╪════════════╪════════╡
    │  nai   │   [[8, 900]]   │   pgstat    │    None    │  None  │
    ├────────┼────────────────┼─────────────┼────────────┼────────┤
    │  bgo   │ [[300, 38000]] │   pgstat    │    None    │  None  │
    ╘════════╧════════════════╧═════════════╧════════════╧════════╛

    ╒════════╤═════════════╤═════════════╤═════════╤═════════╤══════════╕
    │  par#  │  Component  │  Parameter  │  Value  │  Prior  │  Frozen  │
    ╞════════╪═════════════╪═════════════╪═════════╪═════════╪══════════╡
    │   1    │     nai     │     sf      │    1    │  None   │   True   │
    ├────────┼─────────────┼─────────────┼─────────┼─────────┼──────────┤
    │   2    │     nai     │     bf      │    1    │  None   │   True   │
    ├────────┼─────────────┼─────────────┼─────────┼─────────┼──────────┤
    │   3    │     nai     │     rf      │    1    │  None   │   True   │
    ├────────┼─────────────┼─────────────┼─────────┼─────────┼──────────┤
    │   4    │     nai     │     ra      │    0    │  None   │   True   │
    ├────────┼─────────────┼─────────────┼─────────┼─────────┼──────────┤
    │   5    │     nai     │     dec     │    0    │  None   │   True   │
    ├────────┼─────────────┼─────────────┼─────────┼─────────┼──────────┤
    │   6    │     bgo     │     sf      │    1    │  None   │   True   │
    ├────────┼─────────────┼─────────────┼─────────┼─────────┼──────────┤
    │   7    │     bgo     │     bf      │    1    │  None   │   True   │
    ├────────┼─────────────┼─────────────┼─────────┼─────────┼──────────┤
    │   8    │     bgo     │     rf      │    1    │  None   │   True   │
    ├────────┼─────────────┼─────────────┼─────────┼─────────┼──────────┤
    │   9    │     bgo     │     ra      │    0    │  None   │   True   │
    ├────────┼─────────────┼─────────────┼─────────┼─────────┼──────────┤
    │   10   │     bgo     │     dec     │    0    │  None   │   True   │
    ╘════════╧═════════════╧═════════════╧═════════╧═════════╧══════════╛
    


2.Define spectral model.

.. code:: ipython3

    model = cpl()
    model.save(savepath)
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
    infer.save(savepath)
    print(infer)


.. parsed-literal::

    ╒════════╤═════════╤══════════════╤═════════════╤═════════════╤═════════╕
    │  cfg#  │  Class  │  Expression  │  Component  │  Parameter  │  Value  │
    ╞════════╪═════════╪══════════════╪═════════════╪═════════════╪═════════╡
    │   1    │  model  │     cpl      │     cpl     │  redshift   │    0    │
    ╘════════╧═════════╧══════════════╧═════════════╧═════════════╧═════════╛
    ╒════════╤═════════╤══════════════╤═════════════╤═════════════╤═════════╤═════════════╕
    │  par#  │  Class  │  Expression  │  Component  │  Parameter  │  Value  │    Prior    │
    ╞════════╪═════════╪══════════════╪═════════════╪═════════════╪═════════╪═════════════╡
    │   1*   │  model  │     cpl      │     cpl     │  $\\alpha$   │   -1    │ unif(-8, 4) │
    ├────────┼─────────┼──────────────┼─────────────┼─────────────┼─────────┼─────────────┤
    │   2*   │  model  │     cpl      │     cpl     │ log$E_{c}$  │    2    │ unif(0, 4)  │
    ├────────┼─────────┼──────────────┼─────────────┼─────────────┼─────────┼─────────────┤
    │   3*   │  model  │     cpl      │     cpl     │   log$A$    │   -1    │ unif(-6, 5) │
    ╘════════╧═════════╧══════════════╧═════════════╧═════════════╧═════════╧═════════════╛
    


.. code:: ipython3

    post = infer.emcee(nstep=1000, resume=True, savepath=savepath)
    post.save(savepath)
    print(post)


.. parsed-literal::

    ╒════════╤═════════╤══════════════╤═════════════╤═════════════╤════════╤══════════╤════════╤══════════════════╕
    │  par#  │  Class  │  Expression  │  Component  │  Parameter  │  Mean  │  Median  │  Best  │    1sigma CI     │
    ╞════════╪═════════╪══════════════╪═════════════╪═════════════╪════════╪══════════╪════════╪══════════════════╡
    │   1    │  model  │     cpl      │     cpl     │  $\\alpha$   │ -1.562 │  -1.562  │ -1.563 │ [-1.572, -1.552] │
    ├────────┼─────────┼──────────────┼─────────────┼─────────────┼────────┼──────────┼────────┼──────────────────┤
    │   2    │  model  │     cpl      │     cpl     │ log$E_{c}$  │  2.69  │  2.689   │  2.69  │  [2.672, 2.708]  │
    ├────────┼─────────┼──────────────┼─────────────┼─────────────┼────────┼──────────┼────────┼──────────────────┤
    │   3    │  model  │     cpl      │     cpl     │   log$A$    │ -0.771 │  -0.771  │ -0.771 │ [-0.777, -0.764] │
    ╘════════╧═════════╧══════════════╧═════════════╧═════════════╧════════╧══════════╧════════╧══════════════════╛
    ╒════════╤═════════╤═════════════╤════════════╤════════╕
    │  Data  │  Model  │  Statistic  │   Value    │  Bins  │
    ╞════════╪═════════╪═════════════╪════════════╪════════╡
    │  nai   │   cpl   │   pgstat    │   405.02   │  119   │
    ├────────┼─────────┼─────────────┼────────────┼────────┤
    │  bgo   │   cpl   │   pgstat    │   149.73   │  117   │
    ├────────┼─────────┼─────────────┼────────────┼────────┤
    │ Total  │  Total  │  stat/dof   │ 554.75/233 │  236   │
    ╘════════╧═════════╧═════════════╧════════════╧════════╛
    ╒════════╤════════╤════════╤═════════╕
    │  AIC   │  AICc  │  BIC   │   lnZ   │
    ╞════════╪════════╪════════╪═════════╡
    │ 560.75 │ 560.85 │ 571.14 │ -296.34 │
    ╘════════╧════════╧════════╧═════════╛
    


.. code:: ipython3

    fig = Plot.infer_ctsspec(post, style='CE')
    fig.save(f'{savepath}/ctsspec')



.. raw:: html

    <iframe src="_static/qs_ctsspec.html"></iframe>



.. code:: ipython3

    fig = Plot.post_corner(post)
    fig.save(f'{savepath}/corner')



.. raw:: html

    <iframe src="_static/qs_corner.html"></iframe>



.. code:: ipython3

    earr = np.logspace(1, 3, 100)

    modelplot = Plot.model(ploter='plotly', style='vFv', CI=True)
    modelplot.add_model(model, E=earr)
    fig = modelplot.get_fig()
    fig.save(f'{savepath}/model')



.. raw:: html

    <iframe src="_static/qs_model.html"></iframe>
    