from setuptools import setup, find_packages


_info_ = {}
with open("bayspec/__info__.py", "r") as f:
    exec(f.read(), _info_)


setup(
    name="bayspec",
    version=_info_['__version__'],
    description="Astronomical spectrum fitting tool",
    long_description="A Bayesian inference-based spectral fitting tool for multi-dimensional and multi-wavelength astrophysical data",
    author="Jun Yang",
    author_email="jyang@smail.nju.edu.cn",
    url="https://github.com/jyangch/bayspec",
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Topic :: Scientific/Engineering :: Astronomy",
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
        "pymultinest>=2.12"
    ],
    packages=find_packages(exclude=["examples*", "docs*"]),
    include_package_data=True,
    project_urls={
        "Source Code": "https://github.com/jyangch/bayspec",
        "Documentation": "https://bayspec.readthedocs.io"}
)
