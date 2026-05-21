<p align="center">
  <a href="https://pypi.org/project/bayspec/">
    <img src="https://raw.githubusercontent.com/jyangch/bayspec/main/docs/source/_static/logo.svg" alt="bayspec" height="72">
  </a>
</p>

<p align="center">
  <strong>Bayesian inference-based spectral fitting for high-energy astrophysical data.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/bayspec/"><img alt="PyPI" src="https://img.shields.io/pypi/v/bayspec?color=4F46E5&labelColor=1f2937&logo=pypi&logoColor=white"></a>
  <a href="https://bayspec.readthedocs.io/"><img alt="Documentation" src="https://img.shields.io/badge/docs-readthedocs-06B6D4?labelColor=1f2937"></a>
  <a href="https://huggingface.co/spaces/jyangch/bayspec"><img alt="BaySpec App on Hugging Face Spaces" src="https://img.shields.io/badge/web%20UI-BaySpec%20App-FFD21E?labelColor=1f2937"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img alt="License GPL-3.0" src="https://img.shields.io/badge/license-GPL--3.0-9CA3AF?labelColor=1f2937"></a>
</p>

---

`BaySpec` is a Python library for Bayesian inference on high-energy
astrophysical spectra. It pairs MCMC and nested-sampling backends with
multi-spectrum, multi-model fitting machinery, reads OGIP FITS data out
of the box, and bridges to local, [`astromodels`](https://astromodels.readthedocs.io/),
and [`Xspec`](https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/Models.html)
model libraries.


## Features

- **Inference backends.** Posterior sampling via MCMC
  ([`emcee`](https://emcee.readthedocs.io/)) or nested sampling
  ([`MultiNest`](https://github.com/rjw57/MultiNest) via
  [`pymultinest`](https://johannesbuchner.github.io/PyMultiNest/));
  maximum-likelihood fits via
  [`lmfit`](https://lmfit.github.io/lmfit-py/) or
  [`iminuit`](https://iminuit.readthedocs.io/) for quick checks.
- **OGIP Type I & II FITS.** Source and background `PHA`/`PHA2`
  spectra, plus `RMF`, `RSP`/`RSP2`, and `ARF`/`ARF2` responses —
  multi-extension archives are row-indexed so individual spectra plug in
  directly.
- **Multi-dimensional.** Time-evolving spectra fit with time-involved
  physical models.
- **Multi-wavelength.** Joint fitting across spectra at different
  wavelengths.
- **Multi-spectrum, multi-model.** Simultaneously fit any number of
  `(data, model)` pairs; combine, freeze, or link parameters across
  pairs.
- **Pluggable model libraries.** A local set plus optional
  `astromodels` and `Xspec` bridges. Register your own Python functions
  as new components.
- **Browser workbench (optional).** [BaySpec App](#bayspec-app) is a
  FastAPI + HTMX front-end that exposes the same fitting machinery
  through a web UI.


## Installation

`BaySpec` is available on PyPI:

```bash
pip install bayspec
```

### Optional: `MultiNest` sampler

To enable [`MultiNest`](https://github.com/rjw57/MultiNest) for nested
sampling, follow the
[`pymultinest`](https://johannesbuchner.github.io/PyMultiNest/) install
guide.

### Optional: `astromodels` components

To pull components from
[`astromodels`](https://astromodels.readthedocs.io/en/latest/notebooks/function_list.html),
install `astromodels` in your Python environment.

### Optional: `Xspec` components

To pull components from
[`Xspec`](https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/Models.html):

1. Install `HEASoft` and `Xspec v12.12.1+`.
2. After initialising the `HEASoft` environment, install
   [`xspec-models-cxc`](https://github.com/cxcsds/xspec-models-cxc).

> Note: `bayspec` currently supports only `Additive` and
> `Multiplicative` `Xspec` models.


## BaySpec App

[**BaySpec App**](https://github.com/jyangch/bayspec_app) is the
browser front-end for `bayspec` — a FastAPI + HTMX workbench that loads
OGIP spectra, composes models, and runs inference without notebook glue.
A public, hosted deployment runs on Hugging Face Spaces:

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/jyangch/bayspec)


## Documentation

Read the [full documentation](https://bayspec.readthedocs.io) or browse
the [examples](https://github.com/jyangch/bayspec/tree/main/examples)
for typical workflows end to end.


## License

`BaySpec` is distributed under the
[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0-standalone.html) license.
