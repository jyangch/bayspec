A Quick Start Tutorial
======================

This tutorial offers a quick-start guide for using *BaySpec* to fit a
spectral model to gamma-ray data. It can be broadly divided into the
following three sections:

- Data: The spectra from Fermi/GBM’s NaI detector and BGO detector.
- Model: A simple cutoff power-law function.
- Fitting: Bayesian inference implemented using *multinest* (or
  *emcee*).

.. code:: ipython3

    import numpy as np
    from bayspec.model.local import *
    from bayspec import DataUnit, Data, BayesInfer, Plot, MaxLikeFit

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
    data




.. raw:: html

    
                <style>
                .my-table {
                    border-collapse: collapse;
                    font-family: sans-serif;
                }
                .my-table th, .my-table td {
                    padding-left: 12px;
                    padding-right: 12px;
    
                    padding-top: 8px;
                    padding-bottom: 8px;
    
                    text-align: center
                    border: none;
                }
                </style>
                <details open><summary style="margin-bottom: 10px;"><b>Data</b></summary><table class="my-table">
    <thead>
    <tr><th style="text-align: center;"> Name </th><th style="text-align: center;">   Noticing   </th><th style="text-align: center;"> Statistic </th><th style="text-align: center;"> Grouping </th><th style="text-align: center;"> Time </th></tr>
    </thead>
    <tbody>
    <tr><td style="text-align: center;"> nai  </td><td style="text-align: center;">  [[8, 900]]  </td><td style="text-align: center;">  pgstat   </td><td style="text-align: center;">   None   </td><td style="text-align: center;"> None </td></tr>
    <tr><td style="text-align: center;"> bgo  </td><td style="text-align: center;">[[300, 38000]]</td><td style="text-align: center;">  pgstat   </td><td style="text-align: center;">   None   </td><td style="text-align: center;"> None </td></tr>
    </tbody>
    </table><details open style="margin-top: 10px;"><summary style="margin-bottom: 10px;"><b>Data Parameters</b></summary><table class="my-table">
    <thead>
    <tr><th style="text-align: center;"> par# </th><th style="text-align: center;"> Component </th><th style="text-align: center;"> Parameter </th><th style="text-align: center;"> Value </th><th style="text-align: center;"> Prior </th><th style="text-align: center;"> Frozen </th></tr>
    </thead>
    <tbody>
    <tr><td style="text-align: center;">  1   </td><td style="text-align: center;">    nai    </td><td style="text-align: center;">    sf     </td><td style="text-align: center;">   1   </td><td style="text-align: center;"> None  </td><td style="text-align: center;">  True  </td></tr>
    <tr><td style="text-align: center;">  2   </td><td style="text-align: center;">    nai    </td><td style="text-align: center;">    bf     </td><td style="text-align: center;">   1   </td><td style="text-align: center;"> None  </td><td style="text-align: center;">  True  </td></tr>
    <tr><td style="text-align: center;">  3   </td><td style="text-align: center;">    nai    </td><td style="text-align: center;">    rf     </td><td style="text-align: center;">   1   </td><td style="text-align: center;"> None  </td><td style="text-align: center;">  True  </td></tr>
    <tr><td style="text-align: center;">  4   </td><td style="text-align: center;">    nai    </td><td style="text-align: center;">    ra     </td><td style="text-align: center;">   0   </td><td style="text-align: center;"> None  </td><td style="text-align: center;">  True  </td></tr>
    <tr><td style="text-align: center;">  5   </td><td style="text-align: center;">    nai    </td><td style="text-align: center;">    dec    </td><td style="text-align: center;">   0   </td><td style="text-align: center;"> None  </td><td style="text-align: center;">  True  </td></tr>
    <tr><td style="text-align: center;">  6   </td><td style="text-align: center;">    bgo    </td><td style="text-align: center;">    sf     </td><td style="text-align: center;">   1   </td><td style="text-align: center;"> None  </td><td style="text-align: center;">  True  </td></tr>
    <tr><td style="text-align: center;">  7   </td><td style="text-align: center;">    bgo    </td><td style="text-align: center;">    bf     </td><td style="text-align: center;">   1   </td><td style="text-align: center;"> None  </td><td style="text-align: center;">  True  </td></tr>
    <tr><td style="text-align: center;">  8   </td><td style="text-align: center;">    bgo    </td><td style="text-align: center;">    rf     </td><td style="text-align: center;">   1   </td><td style="text-align: center;"> None  </td><td style="text-align: center;">  True  </td></tr>
    <tr><td style="text-align: center;">  9   </td><td style="text-align: center;">    bgo    </td><td style="text-align: center;">    ra     </td><td style="text-align: center;">   0   </td><td style="text-align: center;"> None  </td><td style="text-align: center;">  True  </td></tr>
    <tr><td style="text-align: center;">  10  </td><td style="text-align: center;">    bgo    </td><td style="text-align: center;">    dec    </td><td style="text-align: center;">   0   </td><td style="text-align: center;"> None  </td><td style="text-align: center;">  True  </td></tr>
    </tbody>
    </table></details></details>



2.Define spectral model.

.. code:: ipython3

    model = cpl()
    model.save(savepath)
    model




.. raw:: html

    
                <style>
                .my-table {
                    border-collapse: collapse;
                    font-family: sans-serif;
                }
                .my-table th, .my-table td {
                    padding-left: 12px;
                    padding-right: 12px;
    
                    padding-top: 8px;
                    padding-bottom: 8px;
    
                    text-align: center
                    border: none;
                }
                </style>
                <details open><summary style="margin-bottom: 10px;"><b>Model</b></summary><p><b>cpl [add]</b></p><p style="white-space: pre-wrap;">power-law model with high-energy cutoff</p><details open style="margin-top: 10px;"><summary style="margin-bottom: 10px;"><b>Model Configurations</b></summary><table class="my-table">
    <thead>
    <tr><th style="text-align: center;"> cfg# </th><th style="text-align: center;"> Component </th><th style="text-align: center;"> Parameter  </th><th style="text-align: center;"> Value </th></tr>
    </thead>
    <tbody>
    <tr><td style="text-align: center;">  1   </td><td style="text-align: center;">    cpl    </td><td style="text-align: center;">  redshift  </td><td style="text-align: center;"> 0.000 </td></tr>
    <tr><td style="text-align: center;">  2   </td><td style="text-align: center;">    cpl    </td><td style="text-align: center;">pivot_energy</td><td style="text-align: center;"> 1.000 </td></tr>
    <tr><td style="text-align: center;">  3   </td><td style="text-align: center;">    cpl    </td><td style="text-align: center;">  vfv_peak  </td><td style="text-align: center;"> True  </td></tr>
    </tbody>
    </table></details><details open style="margin-top: 10px;"><summary style="margin-bottom: 10px;"><b>Model Parameters</b></summary><table class="my-table">
    <thead>
    <tr><th style="text-align: center;"> par# </th><th style="text-align: center;"> Component </th><th style="text-align: center;"> Parameter </th><th style="text-align: center;"> Value </th><th style="text-align: center;">    Prior    </th></tr>
    </thead>
    <tbody>
    <tr><td style="text-align: center;">  1   </td><td style="text-align: center;">    cpl    </td><td style="text-align: center;"> $\alpha$  </td><td style="text-align: center;">  -1   </td><td style="text-align: center;"> unif(-2, 2) </td></tr>
    <tr><td style="text-align: center;">  2   </td><td style="text-align: center;">    cpl    </td><td style="text-align: center;"> log$E_p$  </td><td style="text-align: center;">   2   </td><td style="text-align: center;"> unif(0, 4)  </td></tr>
    <tr><td style="text-align: center;">  3   </td><td style="text-align: center;">    cpl    </td><td style="text-align: center;">  log$A$   </td><td style="text-align: center;">   0   </td><td style="text-align: center;">unif(-10, 10)</td></tr>
    </tbody>
    </table></details></details>



3.Run Bayesian inference.

.. code:: ipython3

    infer = BayesInfer([(data, model)])
    infer.save(savepath)
    infer




.. raw:: html

    
                <style>
                .my-table {
                    border-collapse: collapse;
                    font-family: sans-serif;
                }
                .my-table th, .my-table td {
                    padding-left: 12px;
                    padding-right: 12px;
    
                    padding-top: 8px;
                    padding-bottom: 8px;
    
                    text-align: center
                    border: none;
                }
                </style>
                <details open><summary style="margin-bottom: 10px;"><b>Bayesian Inference</b></summary><details open style="margin-top: 10px;"><summary style="margin-bottom: 10px;"><b>Configurations</b></summary><table class="my-table">
    <thead>
    <tr><th style="text-align: center;"> cfg# </th><th style="text-align: center;"> Class </th><th style="text-align: center;"> Expression </th><th style="text-align: center;"> Component </th><th style="text-align: center;"> Parameter  </th><th style="text-align: center;"> Value </th></tr>
    </thead>
    <tbody>
    <tr><td style="text-align: center;">  1   </td><td style="text-align: center;"> model </td><td style="text-align: center;">    cpl     </td><td style="text-align: center;">    cpl    </td><td style="text-align: center;">  redshift  </td><td style="text-align: center;"> 0.000 </td></tr>
    <tr><td style="text-align: center;">  2   </td><td style="text-align: center;"> model </td><td style="text-align: center;">    cpl     </td><td style="text-align: center;">    cpl    </td><td style="text-align: center;">pivot_energy</td><td style="text-align: center;"> 1.000 </td></tr>
    <tr><td style="text-align: center;">  3   </td><td style="text-align: center;"> model </td><td style="text-align: center;">    cpl     </td><td style="text-align: center;">    cpl    </td><td style="text-align: center;">  vfv_peak  </td><td style="text-align: center;"> True  </td></tr>
    </tbody>
    </table></details><details open style="margin-top: 10px;"><summary style="margin-bottom: 10px;"><b>Parameters</b></summary><table class="my-table">
    <thead>
    <tr><th style="text-align: center;"> par# </th><th style="text-align: center;"> Class </th><th style="text-align: center;"> Expression </th><th style="text-align: center;"> Component </th><th style="text-align: center;"> Parameter </th><th style="text-align: center;"> Value </th><th style="text-align: center;">    Prior    </th></tr>
    </thead>
    <tbody>
    <tr><td style="text-align: center;">  1*  </td><td style="text-align: center;"> model </td><td style="text-align: center;">    cpl     </td><td style="text-align: center;">    cpl    </td><td style="text-align: center;"> $\alpha$  </td><td style="text-align: center;">  -1   </td><td style="text-align: center;"> unif(-2, 2) </td></tr>
    <tr><td style="text-align: center;">  2*  </td><td style="text-align: center;"> model </td><td style="text-align: center;">    cpl     </td><td style="text-align: center;">    cpl    </td><td style="text-align: center;"> log$E_p$  </td><td style="text-align: center;">   2   </td><td style="text-align: center;"> unif(0, 4)  </td></tr>
    <tr><td style="text-align: center;">  3*  </td><td style="text-align: center;"> model </td><td style="text-align: center;">    cpl     </td><td style="text-align: center;">    cpl    </td><td style="text-align: center;">  log$A$   </td><td style="text-align: center;">   0   </td><td style="text-align: center;">unif(-10, 10)</td></tr>
    </tbody>
    </table></details></details>



.. code:: ipython3

    post = infer.multinest(nlive=400, resume=True, verbose=False, savepath=savepath)
    post.save(savepath)
    post




.. raw:: html

    
                <style>
                .my-table {
                    border-collapse: collapse;
                    font-family: sans-serif;
                }
                .my-table th, .my-table td {
                    padding-left: 12px;
                    padding-right: 12px;
    
                    padding-top: 8px;
                    padding-bottom: 8px;
    
                    text-align: center
                    border: none;
                }
                </style>
                <details open><summary style="margin-bottom: 10px;"><b>Posterior Results</b></summary><details open style="margin-top: 10px;"><summary style="margin-bottom: 10px;"><b>Parameters</b></summary><table class="my-table">
    <thead>
    <tr><th style="text-align: center;"> par# </th><th style="text-align: center;"> Class </th><th style="text-align: center;"> Expression </th><th style="text-align: center;"> Component </th><th style="text-align: center;"> Parameter </th><th style="text-align: center;"> Mean </th><th style="text-align: center;"> Median </th><th style="text-align: center;"> Best </th><th style="text-align: center;"> 1sigma Best </th><th style="text-align: center;">   1sigma CI    </th></tr>
    </thead>
    <tbody>
    <tr><td style="text-align: center;">  1   </td><td style="text-align: center;"> model </td><td style="text-align: center;">    cpl     </td><td style="text-align: center;">    cpl    </td><td style="text-align: center;"> $\alpha$  </td><td style="text-align: center;">-1.562</td><td style="text-align: center;"> -1.562 </td><td style="text-align: center;">-1.561</td><td style="text-align: center;">   -1.561    </td><td style="text-align: center;">[-1.572, -1.551]</td></tr>
    <tr><td style="text-align: center;">  2   </td><td style="text-align: center;"> model </td><td style="text-align: center;">    cpl     </td><td style="text-align: center;">    cpl    </td><td style="text-align: center;"> log$E_p$  </td><td style="text-align: center;">2.331 </td><td style="text-align: center;"> 2.331  </td><td style="text-align: center;">2.330 </td><td style="text-align: center;">    2.330    </td><td style="text-align: center;"> [2.321, 2.341] </td></tr>
    <tr><td style="text-align: center;">  3   </td><td style="text-align: center;"> model </td><td style="text-align: center;">    cpl     </td><td style="text-align: center;">    cpl    </td><td style="text-align: center;">  log$A$   </td><td style="text-align: center;">2.353 </td><td style="text-align: center;"> 2.354  </td><td style="text-align: center;">2.352 </td><td style="text-align: center;">    2.352    </td><td style="text-align: center;"> [2.338, 2.368] </td></tr>
    </tbody>
    </table></details><details open style="margin-top: 10px;"><summary style="margin-bottom: 10px;"><b>Statistics</b></summary><table class="my-table">
    <thead>
    <tr><th style="text-align: center;"> Data </th><th style="text-align: center;"> Model </th><th style="text-align: center;"> Statistic </th><th style="text-align: center;">   Value   </th><th style="text-align: center;"> Bins </th></tr>
    </thead>
    <tbody>
    <tr><td style="text-align: center;"> nai  </td><td style="text-align: center;">  cpl  </td><td style="text-align: center;">  pgstat   </td><td style="text-align: center;">  405.171  </td><td style="text-align: center;"> 119  </td></tr>
    <tr><td style="text-align: center;"> bgo  </td><td style="text-align: center;">  cpl  </td><td style="text-align: center;">  pgstat   </td><td style="text-align: center;">  149.560  </td><td style="text-align: center;"> 117  </td></tr>
    <tr><td style="text-align: center;">Total </td><td style="text-align: center;"> Total </td><td style="text-align: center;"> stat/dof  </td><td style="text-align: center;">554.731/233</td><td style="text-align: center;"> 236  </td></tr>
    </tbody>
    </table></details><details open style="margin-top: 10px;"><summary style="margin-bottom: 10px;"><b>Information Criterias</b></summary><table class="my-table">
    <thead>
    <tr><th style="text-align: center;">  AIC  </th><th style="text-align: center;"> AICc  </th><th style="text-align: center;">  BIC  </th><th style="text-align: center;">  lnZ   </th></tr>
    </thead>
    <tbody>
    <tr><td style="text-align: center;">560.731</td><td style="text-align: center;">560.834</td><td style="text-align: center;">571.123</td><td style="text-align: center;">-295.844</td></tr>
    </tbody>
    </table></details>



.. code:: ipython3

    fig = Plot.infer(post, style='CE')
    fig.save(f'{savepath}/ctsspec')



.. raw:: html

    <iframe src="_static/qs_ctsspec.html"></iframe>



.. code:: ipython3

    fig = Plot.post_corner(post, ploter='plotly')
    fig.save(f'{savepath}/corner')



.. raw:: html

    <iframe src="_static/qs_corner.html"></iframe>



.. code:: ipython3

    earr = np.logspace(1, 3, 100)
    
    modelplot = Plot.model(ploter='plotly', style='vFv', post=True)
    modelplot.add_model(model, E=earr)
    fig = modelplot.get_fig()
    fig.save(f'{savepath}/model')



.. raw:: html

    <iframe src="_static/qs_model.html"></iframe>



.. code:: ipython3

    ergflux = model.best_ergflux(emin=10, emax=1000, ngrid=1000)
    ergflux_sample = model.ergflux_sample(emin=10, emax=1000, ngrid=1000)
    
    print(ergflux)
    print(ergflux_sample)


.. parsed-literal::

    8.367943905909297e-06
    {'mean': 8.369501130327865e-06, 'median': 8.369420051840691e-06, 'Isigma': array([8.31271099e-06, 8.42362544e-06]), 'IIsigma': array([8.25891689e-06, 8.48377819e-06]), 'IIIsigma': array([8.20528996e-06, 8.54385474e-06])}
