# *Welcome* *To* *BAYSPEC* 👋

### A Bayesian Inference-based Spectral Fitting Tool for High-energy Astrophysical Data.

[![PyPI - Version](https://img.shields.io/pypi/v/bayspec?color=blue&logo=PyPI&logoColor=white&style=for-the-badge)](https://pypi.org/project/bayspec/)
[![License: GPL v3](https://img.shields.io/github/license/jyangch/bayspec?color=blue&logo=open-source-initiative&logoColor=white&style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)


## Features
- **Bayesian inference-based**: implemented by MCMC (e.g., emcee) or nested sampling (e.g., multinest)
- **Multi-dimensional**: enabling the fit of time-evolving spectra with time-involved physical models
- **Multi-wavelength**: supporting for the joint fitting to multi-wavelength astronomical spectra
- **Friendly interface**: easy-to-use web application developed with streamlit
- **Others**: simultaneous fitting of multi-spectra and multi-models, freely combining available models and add new model


## Installation

_BaySpec_ is available via `pip`:
```bash
$ pip install bayspec
```

### Utilize `multinest` sampler
If you want to use [`Multinest`](https://github.com/rjw57/MultiNest) for Bayesian inference, you can follow the instructions in the [`pymultinest`](https://johannesbuchner.github.io/PyMultiNest/) documentation to install it.

### Access `Astromodels` models
To utilize models from [`Astromodels`](https://astromodels.readthedocs.io/en/latest/notebooks/function_list.html#), ensure that `Astromodels` is installed on your system.

### Access `Xspec` models
To utilize models from [`Xspec`](https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/Models.html), ensure that both `HEASoft` and `Xspec v12.12.1+` are installed on your system. After confirming that `HEASoft` environment is properly initialized, then you need install [`xspec-models-cxc`](https://github.com/cxcsds/xspec-models-cxc).

**NOTE**: _BaySpec_ currently only supports `Additive` and `Multiplicative` models in `Xspec`.


## BaySpec App

[_BaySpec App_](https://github.com/jyangch/bayspec_app) provides an easy-to-use web application implemented using [`streamlit`](https://streamlit.io/).

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bayspec.streamlit.app)


## Documentation

If you wish to learn about the usage, you may check the [`examples`](https://github.com/jyangch/bayspec/tree/main/examples) or read the [`documentation`](https://bayspec.readthedocs.io).


## License

_BaySpec_ is distributed under the terms of the [`GPL-3.0`](https://www.gnu.org/licenses/gpl-3.0-standalone.html) license.
