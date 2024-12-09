import os
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


    
class zxhsync(Tinvolved):

    def __init__(self):
        super().__init__()
        
        self.expr = 'zxhsync'
        self.comment = "zxh's synchrotron model"
        
        self.mo_prefix = docs_path + '/ZXHSYNC'
        
        self.mo_dir = self.mo_prefix + '/spec_lc_ele_z_dL_gm_gmax_injpl_v2.o'

        self.params = OrderedDict()
        self.params['log$B_0$'] = Par(1, unif(0, 2))
        self.params['$\\alpha_B$'] = Par(2, unif(0, 3))
        self.params['log$\\gamma_{min}$'] = Par(5, unif(3, 7))
        self.params['log$\\gamma_{max}$'] = Par(7, unif(5, 9))
        self.params['log$\\Gamma$'] = Par(2, unif(1, 3))
        self.params['$p$'] = Par(2, unif(1.5, 3.5))
        self.params['$t_{inj}$'] = Par(10, unif(-0.5, 26.00))
        self.params['$q$'] = Par(5, unif(0, 10))
        self.params['log$R_0$'] = Par(15, unif(12, 17))
        self.params['log$Q_0$'] = Par(40, unif(30, 50))
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(1.0)
        self.config['max_time'] = Cfg(26.0)
        self.config['zero_offset'] = Cfg(0.5)
        self.config['spec_prec'] = Cfg(2)
        self.config['temp_prec'] = Cfg(25)


    def func(self, E, T, O=None):
        logB0 = self.params['log$B_0$'].value
        alphaB = self.params['$\\alpha_B$'].value
        loggamma_min = self.params['log$\\gamma_{min}$'].value
        loggamma_max = self.params['log$\\gamma_{max}$'].value
        logGamma = self.params['log$\\Gamma$'].value
        p = self.params['$p$'].value
        tinj = self.params['$t_{inj}$'].value
        q = self.params['$q$'].value
        logR0 = self.params['log$R_0$'].value
        logQ0 = self.params['log$Q_0$'].value   # in unit of s^-1

        B0 = 10 ** logB0
        gamma_min = 10 ** loggamma_min
        gamma_max = 10 ** loggamma_max
        Gamma = 10 ** logGamma
        R0 = 10 ** logR0
        Q0 = 10 ** logQ0

        B0_str = str(B0)
        alphaB_str = str(alphaB)
        gamma_min_str = str(gamma_min)
        gamma_max_str = str(gamma_max)
        Gamma_str = str(Gamma)
        p_str = str(p)
        
        zero_dt_str = '%.4f'%self.config['zero_offset'].value
        tinj_str = str(tinj)
        q_str = str(q)
        max_time_str = '%.4f'%self.config['max_time'].value
        R0_str = str(R0)
        Q0_str = str(Q0)
        z_str = '%.4f'%self.config['redshift'].value
        dL_str = '%.4e'%self.luminosity_distance
        spec_prec_str = '%.2f'%self.config['spec_prec'].value
        temp_prec_str = '%.2f' % self.config['temp_prec'].value

        phtspec = np.zeros_like(E)
        
        for Ti in set(T):
            idx = np.where(T == Ti)[0]
            t_obs_str = str(Ti)
            E_str = ' '.join([str(Ei) for Ei in E[idx]])
            n_str = str(len(E[idx]))
            it_str = '0'
            ielec_str = '0'

            cmd = self.mo_dir + ' ' + self.mo_prefix + ' ' + B0_str + ' ' + alphaB_str + ' ' + gamma_min_str \
                + ' ' + gamma_max_str + ' ' + Gamma_str + ' ' + p_str + ' ' + t_obs_str + ' ' + zero_dt_str \
                + ' ' + tinj_str + ' ' + q_str + ' ' + max_time_str + ' ' + R0_str + ' ' + Q0_str + ' ' + z_str \
                + ' ' + dL_str + ' ' + n_str + ' ' + it_str + ' ' + ielec_str + ' ' + spec_prec_str \
                + ' ' + temp_prec_str + ' ' + E_str

            process = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
            (out, err) = process.communicate()
            Fd_str = out.split()
            if out == '' or len(Fd_str) != len(E[idx]):
                print('+++++ Error Message +++++')
                print('out: ', out)
                print('err: ', err)
                print('cmd: ', cmd)
                print('+++++ +++++++++++++ +++++')
            Fd = np.array([float(Fdi) for Fdi in Fd_str])   # in unit of mJy
            Fv = Fd / (E[idx] * 1.6022e-9) / (6.62607e-34 * 6.2415e15) / 1.e26   # in unit of photons/s/cm^2/keV
            phtspec[idx] = Fv
        return phtspec
    
    
    @property
    def luminosity_distance(self):
        return Planck18.luminosity_distance(self.config['redshift'].value).to(u.cm).value



class katu(Tinvolved):

    def __init__(self):
        super().__init__()
        
        self.expr = 'katu'
        self.comment = 'katu model'
        
        self.mo_prefix = docs_path + '/Katu'

        os.chdir(self.mo_prefix)

        self.mo_dir = self.mo_prefix + '/GRB_MZ'
        self.mo_cfg_dir = self.mo_prefix + '/prompt.toml'
        
        # load the default configuration file
        with open(self.mo_cfg_dir, 'r') as f_obj:
            self.mo_cfg = toml.load(f_obj)
            
        self.params = OrderedDict()
        self.params['log$B_0$'] = Par(1, unif(0, 2))
        self.params['$\\alpha_B$'] = Par(1, unif(0, 3))
        self.params['log$\\gamma_{min}$'] = Par(5, unif(3, 7))
        self.params['log$\\Gamma$'] = Par(2, unif(1, 3))
        self.params['$p$'] = Par(3, unif(1.5, 3.5))
        self.params['$t_{inj}$'] = Par(10, unif(0, 20))
        self.params['log$R_0$'] = Par(15, unif(12, 17))
        self.params['log$Q_0$'] = Par(40, unif(30, 50))
        
        self.config = OrderedDict()
        self.config['redshift'] = Cfg(1.0)
        self.config['t_obs_start'] = Cfg(0)
        self.config['t_obs_end'] = Cfg(20)


    def func(self, E, T, O=None):
        logB0 = self.params['log$B_0$'].value
        alphaB = self.params['$\\alpha_B$'].value
        loggamma_min = self.params['log$\\gamma_{min}$'].value
        logGamma = self.params['log$\\Gamma$'].value
        p = self.params['$p$'].value
        tinj = self.params['$t_{inj}$'].value
        logR0 = self.params['log$R_0$'].value
        logQ0 = self.params['log$Q_0$'].value

        self.set_cfg('Prompt.B_0', logB0)
        self.set_cfg('Prompt.alpha', alphaB)
        self.set_cfg('Prompt.prompt_gmin', loggamma_min)
        self.set_cfg('Prompt.Gamma_init', logGamma)
        self.set_cfg('Prompt.prompt_p', p)
        self.set_cfg('Prompt.t_inj', tinj)
        self.set_cfg('Prompt.R_0', logR0)
        self.set_cfg('Prompt.Q_0', logQ0)
        
        redshift = self.config['redshift'].value
        t_obs_start = self.config['t_obs_start'].value
        t_obs_end = self.config['t_obs_end'].value
        
        self.set_cfg('Flux.z', redshift)
        self.set_cfg('General.t_obs_start', t_obs_start)
        self.set_cfg('General.t_obs_end', t_obs_end)

        # save the updated configuration file
        with open(self.mo_cfg_dir, 'w') as f_obj:
            toml.dump(self.mo_cfg, f_obj)

        E_str = ' '.join([str(Ei) for Ei in E])
        T_str = ' '.join([str(Ti) for Ti in T])

        cmd = self.mo_dir + ' prompt.toml ' + '--energy ' + E_str + ' --time ' + T_str

        process = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
        (out, err) = process.communicate()
        if out == '':
            print('+++++ Error Message +++++')
            print('out: ', out)
            print('err: ', err)
            print('cmd: ', cmd)
            print('+++++ +++++++++++++ +++++')
        phtspec = np.array([float(ps) for ps in out.split()[3::4]])
        return phtspec


    def set_cfg(self, key, value):
        key_list = key.split('.')
        n_key = len(key_list)

        cfg_ = self.mo_cfg
        for i, key_i in enumerate(key_list):
            assert key_i in cfg_, 'no key (%s) in cfg' % key_i
            if i < n_key - 1:
                cfg_ = cfg_[key_i]
            else:
                cfg_[key_i] = value
