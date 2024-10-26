from setuptools import setup, find_packages
from bayspec import __version__


setup(
    name="bayspec",
    version=__version__,
    description="A Bayesian inference-based spectral fitting tool for multi-dimensional and multi-wavelength astrophysical data.",
    author="Jun Yang",
    author_email="jyang@smail.nju.edu.cn",
    url="https://github.com/jyangch/bayspec",
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.1.0",
        "scipy>=1.11.2",
        "toml>=0.10.2",
        "tabulate>=0.9.0",
        "plotly>=5.22.0",
        "matplotlib>=3.9.2",
        "corner>=2.2.2",
        "astropy>=5.3.2",
        "emcee>=3.1.6",
        "pymultinest>=2.12",
        "streamlit>=1.36.0",
        "st-pages>=0.4.5",
        "streamlit_code_editor>=0.1.20"
    ],
    packages=find_packages(),
    include_package_data=True,
)
