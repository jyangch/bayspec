import toml
import numpy as np
import subprocess as sp
import astropy.units as u
from ..model import Tinvolved
from ...util.prior import unif
from ...util.param import Par, Cfg
from collections import OrderedDict
from os.path import dirname, abspath
from astropy.cosmology import Planck18
docs_path = dirname(dirname(dirname(abspath(__file__)))) + '/docs'



class hlecpl(Tinvolved):
    # 10.1088/0004-637X/690/1/L10
    
    def __init__(self):
        super().__init__()
        
        self.expr = 'hlecpl'
        self.comment = 'curvature effect model for cpl function'

        self.params = OrderedDict()
        self.params[r'$\alpha$'] = Par(-1, unif(-2, 2))
        self.params[r'log$E_{p,c}$'] = Par(2, unif(0, 4))
        self.params[r'log$A_{c}$'] = Par(0, unif(-6, 6))
        self.params[r'$t_{0}$'] = Par(0, unif(-20, 20))
        self.params[r'$t_{c}$'] = Par(10, unif(0, 50))


    def func(self, E, T, O=None):
        alpha = self.params[r'$\alpha$'].value
        logEpc = self.params[r'log$E_{p,c}$'].value
        logAc = self.params[r'log$A_{c}$'].value
        t0 = self.params[r'$t_{0}$'].value
        tc = self.params[r'$t_{c}$'].value

        Epc = 10 ** logEpc
        Ac = 10 ** logAc

        if tc <= t0 or tc > np.min(T):
            return np.ones_like(E) * np.nan

        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        Ept = Epc * ((T - t0) / (tc - t0)) ** (- 1)
        At = Ac * ((T - t0) / (tc - t0)) ** (alpha - 1)

        phtspec = At * (E / 100) ** alpha * np.exp(-(2 + alpha) * E / Ept)
        return phtspec



class hleband(Tinvolved):
    # 10.1088/0004-637X/690/1/L10
    
    def __init__(self):
        super().__init__()
        
        self.expr = 'hleband'
        self.comment = 'curvature effect model for band function'

        self.params = OrderedDict()
        self.params[r'$\alpha$'] = Par(-1, unif(-2, 2))
        self.params[r'$\beta$'] = Par(-4, unif(-6, -2))
        self.params[r'log$E_{p,c}$'] = Par(2, unif(0, 4))
        self.params[r'log$A_{c}$'] = Par(0, unif(-6, 6))
        self.params[r'$t_{0}$'] = Par(0, unif(-20, 20))
        self.params[r'$t_{c}$'] = Par(10, unif(0, 50))


    def func(self, E, T, O=None):
        alpha = self.params[r'$\alpha$'].value
        beta = self.params[r'$\beta$'].value
        logEpc = self.params[r'log$E_{p,c}$'].value
        logAc = self.params[r'log$A_{c}$'].value
        t0 = self.params[r'$t_{0}$'].value
        tc = self.params[r'$t_{c}$'].value

        Epc = 10 ** logEpc
        Ac = 10 ** logAc

        if tc <= t0 or tc > np.min(T):
            return np.ones_like(E) * np.nan

        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        Ept = Epc * ((T - t0) / (tc - t0)) ** (- 1)
        At = Ac * ((T - t0) / (tc - t0)) ** (alpha - 1)
        Ebt = (alpha - beta) / (alpha + 2) * Ept
        phtspec = np.zeros_like(E, dtype=float)

        i1 = E < Ebt; i2 = E >= Ebt
        phtspec[i1] = At[i1] * (E[i1] / 100) ** alpha * np.exp(-(2 + alpha) * E[i1] / Ept[i1])
        phtspec[i2] = At[i2] * (Ebt[i2] / 100) ** (alpha - beta) * (E[i2] / 100) ** beta * np.exp(beta - alpha)
        return phtspec



"""
class katu(Tinvolved):

    def __init__(self):
        super().__init__()
        
        self.expr = 'katu'
        self.comment = 'katu model'
        
        self.mo_prefix = docs_path + '/Katu'

        ############ path of katu ############
        self.mo_dir = self.mo_prefix + '/katu.sh'
        ############ path of katu ############

        ############ path of katu configuration ############
        self.mo_cfg_dir = self.mo_prefix + '/config.toml'
        ############ path of katu configuration ############
        
        # load the default configuration file
        with open(self.mo_cfg_dir, 'r') as f_obj:
            self.mo_cfg = toml.load(f_obj)

        ############ model parameters ############
        # model parameters and their ranges
        # the first one is log$B_0$, its range is [0, 2]
        # ......
        # the last one is log$E_0$, its range is [40, 60]
        self.params = OrderedDict()
        self.params['log$B_0$'] = Par(1, unif(0, 2))
        self.params['$\\alpha$'] = Par(1, unif(0, 3))
        self.params['$p$'] = Par(2, unif(1.5, 3.5))
        self.params['log$R_0$'] = Par(15, unif(12, 17))
        self.params['log$E_0$'] = Par(50, unif(40, 60))
        ############ model parameters ############


    def func(self, E, T, O=None):
        ########################
        # This function is used to calculate the fv for 
        # given model parameters (see __init__), observed time (T) and energy list (E).
        # katu can be a black box, which can take in energy, parameters, and time, 
        # and output the corresponding spectrum.
        ########################

        ############ inputs ############
        # E: energy (or frequency) array in unit of keV, for example, [1, 10, 100, 1000]
        # T: the observation time array in unit of second, for example, [2, 5]
        ############ inputs ############

        # take out parameters from theta
        logB0 = self.params['log$B_0$'].value
        alpha = self.params['$\\alpha$'].value
        p = self.params['$p$'].value
        logR0 = self.params['log$R_0$'].value
        logE0 = self.params['log$E_0$'].value

        B_0 = 10 ** logB0
        R_0 = 10 ** logR0
        E_0 = 10 ** logE0

        redshift = self.config['redshift'].value

        zi = 1 + redshift
        E = E * zi

        # update configuration file based on parameters
        self.set_cfg('Prompt.B_0', B_0)
        self.set_cfg('Prompt.alpha', alpha)
        self.set_cfg('Zone.p', p)
        self.set_cfg('Prompt.R_0', R_0)
        self.set_cfg('Jet.E_0', E_0)

        # save the updated configuration file
        mo_cfg_dir = self.mo_prefix + '/config_%d.toml' % (np.random.uniform() * 1e10)
        with open(mo_cfg_dir, 'w') as f_obj:
            toml.dump(self.mo_cfg, f_obj)

        # frenquency and time list
        E_str = ' '.join([str(Ei) for Ei in E])
        T_str = ' '.join([str(Ti) for Ti in T])

        ############ command that runs in terminal ############
        # katu running command: path_of_katu path_of_toml --t_obs t_list --nu_obs nu_list
        cmd = self.mo_dir + ' ' + mo_cfg_dir + ' --t_obs ' + T_str + ' --nu_obs ' + E_str
        ############ command that runs in terminal ############

        ############ run command in terminal ############
        process = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
        ############ run command in terminal ############

        ############ get outputs from terminal ############
        # output should be 2D fv (erg/s/cm^2/keV) or NE (photons/s/cm^2/keV) grid 
        # at given t_list and nu_list
        (out, err) = process.communicate()
        ############ get outputs from terminal ############

        if out == '':
            print('+++++ Error Message +++++')
            print('out: ', out)
            print('err: ', err)
            print('cmd: ', cmd)
            print('+++++ +++++++++++++ +++++')
            
        phtspec = out.split()
        return phtspec


    def set_cfg(self, key, value):
        # this function is used to update configuration
        # 1) based on the free parameters generated by sampler
        # 2) bsed on the settings provided by user, like:
        # General.t_obs_end, default 100
        # General.IC_switch, default True
        # General.IC_KN, default True
        # Jet.jet_type, default Top_hat
        # Prompt.Enable_prompt, default True
        # Flux.z, default 1
        key_list = key.split('.')
        n_key = len(key_list)

        cfg_ = self.mo_cfg
        for i, key_i in enumerate(key_list):
            assert key_i in cfg_, 'no key (%s) in cfg' % key_i
            if i < n_key - 1:
                cfg_ = cfg_[key_i]
            else:
                cfg_[key_i] = value
"""