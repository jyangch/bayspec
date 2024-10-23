from setuptools import setup

setup(
    name='bayspec',
    version='0.1.0',
    packages=['bayspec', 'bayspec.data', 'bayspec.util', 'bayspec.infer', 'bayspec.model', 'bayspec.model.user', 'bayspec.model.astro', 'bayspec.model.local', 'bayspec.model.xspec'],
    url='https://github.com/jyangch/bayspec',
    license='GPLv3',
    author='jyang',
    author_email='jyang@smail.nju.edu.cn',
    description='A Bayesian inference-based spectral fitting tool for multi-dimensional and multi-wavelength astrophysical data.'
)
