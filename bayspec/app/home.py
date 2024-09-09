import streamlit as st


css='''
<style>
    section.main > div {max-width:75rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

st.write("# Welcome to :rainbow[*BAYSPEC*] 👋")

st.markdown(
    """
    BaySpec is a Bayesian inference-based spectral fitting tool for multi-dimensional (time and energy) 
    and multi-wavelength (X-ray and gamma-ray) astrophysical data.
    ### Features:
    - Bayesian inference-based: implemented by MCMC (e.g., emcee) or nested sampling (e.g., multinest)
    - Multi-dimensional: enabling the fit of time-evolving spectra with time-involved physical models
    - Multi-wavelength: supporting for the joint fitting to multi-wavelength astronomical spectra
    - Others: simultaneous fitting of multi-spectra and multi-models, freely combining available models and add new model
    ### Available models:
    - ***local*** models
    - ***astromodels*** models
    - ***Xspec*** models
    """
    )
