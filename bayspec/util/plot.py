import corner
import numpy as np
import matplotlib as mpl
from itertools import chain
import plotly.express as px
from ..infer.pair import Pair
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ..model.model import Model
from ..infer.infer import Infer
from ..data.spectrum import Spectrum
from ..data.data import Data, DataUnit
from ..infer.posterior import Posterior
from plotly.subplots import make_subplots
from ..data.response import Response, Auxiliary



class Plot(object):
    
    def __init__(self, cls):
        
        self.cls = cls
        
        
    @classmethod
    def from_spectrum(cls, spectrum):
        
        if isinstance(spectrum, Spectrum):
            return cls(spectrum)
        
        else:
            raise ValueError('spectrum argument should be Spectrum type')
        
        
    @classmethod
    def from_response(cls, response):
        
        if isinstance(response, Response):
            return cls(response)
        
        else:
            raise ValueError('response argument should be Response type')
        
        
    @classmethod
    def from_dataunit(cls, dataunit):
        
        if isinstance(dataunit, DataUnit):
            return cls(dataunit)
        
        else:
            raise ValueError('dataunit argument should be DataUnit type')
        
        
    @classmethod
    def from_data(cls, data):
        
        if isinstance(data, Data):
            return cls(data)
        
        else:
            raise ValueError('data argument should be Data type')
        
        
    @classmethod
    def from_model(cls, model):
        
        if isinstance(model, Model):
            return cls([model])
        
        elif isinstance(model, list) and isinstance(model[0], Model):
            return cls(model)
        
        else:
            raise ValueError('model argument should be Model type or Model list')
        
        
    @classmethod
    def from_pair(cls, pair):
        
        if isinstance(pair, Pair):
            return cls(pair)
        
        else:
            raise ValueError('pair argument should be Pair type')
        
        
    @classmethod
    def from_infer(cls, infer):
        
        if isinstance(infer, Infer):
            return cls(infer)
        
        else:
            raise ValueError('infer argument should be Infer type')
        
        
    @classmethod
    def from_posterior(cls, posterior):
        
        if isinstance(posterior, Posterior):
            return cls(posterior)
        
        else:
            raise ValueError('posterior argument should be Posterior type')
        
        
    @staticmethod
    def get_rgb(color, opacity=1.0):
        
        rgba = mpl.colors.to_rgba(color)
        rgb = [int(x * 255) for x in rgba[:3]] + [opacity]
        
        return 'rgba(%d, %d, %d, %f)' % tuple(rgb)

    
    @property
    def cls(self):
        
        return self._cls


    @cls.setter
    def cls(self, new_cls):
        
        self._cls = new_cls

    
    def spectrum(self, ploter='plotly'):
        
        if not isinstance(self.cls, Spectrum):
            raise TypeError('cls is not Spectrum type, cannot call spectrum method')
        
        if ploter == 'plotly':
            self.fig = go.Figure()
            obs = go.Scatter(x=np.arange(len(self.cls._counts)), 
                             y=self.cls._counts.astype(float), 
                             mode='lines', 
                             showlegend=False, 
                             error_y=dict(
                                 type='data',
                                 array=self.cls._errors.astype(float),
                                 thickness=1.5,
                                 width=0)
                             )
            self.fig.add_trace(obs)
            
            self.fig.update_xaxes(title_text='Channel')
            self.fig.update_yaxes(title_text='Counts', type='log')
            self.fig.update_layout(template='plotly_white', height=600, width=800)
            
            self.fig.show()
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            self.fig = plt.figure(figsize=(8, 6))
            gs = self.fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = self.fig.add_subplot(gs[0, 0])

            ax.errorbar(np.arange(len(self.cls._counts)), self.cls._counts, 
                        yerr=self.cls._errors, fmt='-', lw=1.0, elinewidth=1.0, capsize=0)
            ax.set_yscale('log')
            ax.set_xlabel('Channel')
            ax.set_ylabel('Counts')
            ax.autoscale(axis='x', tight=True)
            ax.autoscale(axis='y', tight=True)
            ax.minorticks_on()
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(which='major', width=0.5, length=3)
            ax.tick_params(which='minor', width=0.5, length=2)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            plt.show()
            
            
    def response(self, ploter='plotly', ch_range=None, ph_range=None):
        
        if not isinstance(self.cls, Response):
            raise TypeError('cls is not Response type, cannot call response method')
        
        if isinstance(self.cls, Auxiliary):
            raise TypeError('cls is Auxiliary type, cannot call response method')
        
        ch_mean = np.mean(self.cls._chbin, axis=1)
        ph_mean = np.mean(self.cls._phbin, axis=1)
        
        if ch_range is None:
            ch_idx = np.arange(len(ch_mean))
        else:
            ch_idx = np.where((ch_mean >= ch_range[0]) & (ch_mean <= ch_range[1]))[0]
            
        if ph_range is None:
            ph_idx = np.arange(len(ph_mean))
        else:
            ph_idx = np.where((ph_mean >= ph_range[0]) & (ph_mean <= ph_range[1]))[0]
        
        if ploter == 'plotly':
            self.fig = go.Figure()
            rsp = go.Contour(z=self.cls._drm[ph_idx, :][:, ch_idx].astype(float), 
                             x=ch_mean[ch_idx].astype(float), 
                             y=ph_mean[ph_idx].astype(float), 
                             colorscale='Jet'
                             )
            self.fig.add_trace(rsp)
            
            self.fig.update_xaxes(title_text='Channel energy (keV)', type='log')
            self.fig.update_yaxes(title_text='Photon energy (keV)', type='log')
            self.fig.update_layout(template='plotly_white', height=600, width=800)
            
            self.fig.show()
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            X, Y = np.meshgrid(ch_mean[ch_idx], ph_mean[ph_idx])
            
            self.fig = plt.figure(figsize=(8, 6))
            gs = self.fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = self.fig.add_subplot(gs[0, 0])

            c = ax.contourf(X, Y, self.cls._drm[ph_idx, :][:, ch_idx], cmap='jet')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Channel energy (keV)')
            ax.set_ylabel('Photon energy (keV)')
            self.fig.colorbar(c, orientation='vertical')

            plt.show()
            
            
    def response_photon(self, ploter='plotly', ph_range=None):
        
        if not isinstance(self.cls, Response):
            raise TypeError('cls is not Response type, cannot call response_photon method')
        
        ph_mean = np.mean(self.cls._phbin, axis=1)
        
        if ph_range is None:
            ph_idx = np.arange(len(ph_mean))
        else:
            ph_idx = np.where((ph_mean >= ph_range[0]) & (ph_mean <= ph_range[1]))[0]
        
        x = ph_mean[ph_idx].astype(float)
        
        if isinstance(self.cls, Auxiliary):
            y = self.cls._srp[ph_idx].astype(float)
        else:
            y = np.sum(self.cls._drm[ph_idx, :], axis=1).astype(float)
            
        
        if ploter == 'plotly':
            self.fig = go.Figure()
            obs = go.Scatter(x=x, 
                             y=y, 
                             mode='lines', 
                             showlegend=False
                             )
            self.fig.add_trace(obs)
            
            self.fig.update_xaxes(title_text='Photon energy (keV)', type='log')
            self.fig.update_yaxes(type='log')
            self.fig.update_layout(template='plotly_white', height=600, width=800)
            
            self.fig.show()
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            self.fig = plt.figure(figsize=(8, 6))
            gs = self.fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = self.fig.add_subplot(gs[0, 0])

            ax.plot(x, y, lw=1.0)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Photon energy (keV)')
            ax.autoscale(axis='x', tight=True)
            ax.autoscale(axis='y', tight=True)
            ax.minorticks_on()
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(which='major', width=0.5, length=3)
            ax.tick_params(which='minor', width=0.5, length=2)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            plt.show()


    def response_channel(self, ploter='plotly', ch_range=None):
        
        if not isinstance(self.cls, Response):
            raise TypeError('cls is not Response type, cannot call response_channel method')
        
        if isinstance(self.cls, Auxiliary):
            raise TypeError('cls is Auxiliary type, cannot call response_channel method')
        
        ch_mean = np.mean(self.cls._chbin, axis=1)
        
        if ch_range is None:
            ch_idx = np.arange(len(ch_mean))
        else:
            ch_idx = np.where((ch_mean >= ch_range[0]) & (ch_mean <= ch_range[1]))[0]
        
        if ploter == 'plotly':
            self.fig = go.Figure()
            obs = go.Scatter(x=ch_mean[ch_idx].astype(float), 
                             y=np.sum(self.cls._drm[:, ch_idx], axis=0).astype(float), 
                             mode='lines', 
                             showlegend=False
                             )
            self.fig.add_trace(obs)
            
            self.fig.update_xaxes(title_text='Channel energy (keV)', type='log')
            self.fig.update_yaxes(type='log')
            self.fig.update_layout(template='plotly_white', height=600, width=800)
            
            self.fig.show()
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            self.fig = plt.figure(figsize=(8, 6))
            gs = self.fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = self.fig.add_subplot(gs[0, 0])

            ax.plot(ch_mean[ch_idx], np.sum(self.cls._drm[:, ch_idx], axis=0), lw=1.0)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Channel energy (keV)')
            ax.autoscale(axis='x', tight=True)
            ax.autoscale(axis='y', tight=True)
            ax.minorticks_on()
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(which='major', width=0.5, length=3)
            ax.tick_params(which='minor', width=0.5, length=2)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            plt.show()
            
            
    def dataunit(self, ploter='plotly', style='CE'):
        
        if not isinstance(self.cls, DataUnit):
            raise TypeError('cls is not DataUnit type, cannot call dataunit method')
        
        x = self.cls.rsp_chbin_mean.astype(float)
        x_le = self.cls.rsp_chbin_width.astype(float) / 2
        x_he = self.cls.rsp_chbin_width.astype(float) / 2
        
        if style == 'CC':
            src_y = self.cls.src_ctsrate.astype(float)
            src_y_err = self.cls.src_ctsrate_error.astype(float)
            
            bkg_y = self.cls.bkg_ctsrate.astype(float)
            bkg_y_err = self.cls.bkg_ctsrate_error.astype(float)
            
            net_y = self.cls.net_ctsrate.astype(float)
            net_y_err = self.cls.net_ctsrate_error.astype(float)
            
            ylabel = 'Counts/s/channel'
        
        elif style == 'CE':
            src_y = self.cls.src_ctsspec.astype(float)
            src_y_err = self.cls.src_ctsspec_error.astype(float)
            
            bkg_y = self.cls.bkg_ctsspec.astype(float)
            bkg_y_err = self.cls.bkg_ctsspec_error.astype(float)
            
            net_y = self.cls.net_ctsspec.astype(float)
            net_y_err = self.cls.net_ctsspec_error.astype(float)
            
            ylabel = 'Counts/s/keV'
            
        else:
            raise ValueError(f'unsupported style argument: {style}')
            
        if ploter == 'plotly':
            self.fig = go.Figure()
            src = go.Scatter(x=x, 
                             y=src_y, 
                             mode='markers', 
                             name='src', 
                             showlegend=True, 
                             error_x=dict(
                                 type='data',
                                 symmetric=False,
                                 array=x_he,
                                 arrayminus=x_le,
                                 thickness=1.5,
                                 width=0),
                             error_y=dict(
                                 type='data',
                                 array=src_y_err,
                                 thickness=1.5,
                                 width=0), 
                             marker=dict(symbol='circle', size=3))
            
            bkg = go.Scatter(x=x, 
                             y=bkg_y, 
                             mode='markers', 
                             name='bkg', 
                             showlegend=True, 
                             error_x=dict(
                                 type='data',
                                 symmetric=False,
                                 array=x_he,
                                 arrayminus=x_le,
                                 thickness=1.5,
                                 width=0),
                             error_y=dict(
                                 type='data',
                                 array=bkg_y_err,
                                 thickness=1.5,
                                 width=0), 
                             marker=dict(symbol='circle', size=3))
            
            net = go.Scatter(x=x, 
                             y=net_y, 
                             mode='markers', 
                             name='net', 
                             showlegend=True, 
                             error_x=dict(
                                 type='data',
                                 symmetric=False,
                                 array=x_he,
                                 arrayminus=x_le,
                                 thickness=1.5,
                                 width=0),
                             error_y=dict(
                                 type='data',
                                 array=net_y_err,
                                 thickness=1.5,
                                 width=0), 
                             marker=dict(symbol='circle', size=3))
            
            self.fig.add_trace(src)
            self.fig.add_trace(bkg)
            self.fig.add_trace(net)
            
            self.fig.update_xaxes(title_text='Energy (keV)', type='log')
            self.fig.update_yaxes(title_text=ylabel, type='log')
            self.fig.update_layout(template='plotly_white', height=600, width=800)
            
            self.fig.show()
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            self.fig = plt.figure(figsize=(8, 6))
            gs = self.fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = self.fig.add_subplot(gs[0, 0])
            
            ax.errorbar(x, src_y, xerr = [x_le, x_he], yerr=src_y_err, fmt='none', 
                        ecolor='m', elinewidth=1.0, capsize=0, label='src')
            ax.errorbar(x, bkg_y, xerr = [x_le, x_he], yerr=bkg_y_err, fmt='none', 
                        ecolor='b', elinewidth=1.0, capsize=0, label='bkg')
            ax.errorbar(x, net_y, xerr = [x_le, x_he], yerr=net_y_err, fmt='none', 
                        ecolor='c', elinewidth=1.0, capsize=0, label='net')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Energy (keV)')
            ax.set_ylabel(ylabel)
            ax.autoscale(axis='x', tight=True)
            ax.autoscale(axis='y', tight=True)
            ax.minorticks_on()
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(which='major', width=0.5, length=3)
            ax.tick_params(which='minor', width=0.5, length=2)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.legend()
            
            plt.show()


    def data(self, ploter='plotly', style='CE'):
        
        if not isinstance(self.cls, Data):
            raise TypeError('cls is not Data type, cannot call data method')
        
        if len(self.cls.exprs) <= 10:
            self.colors = dict(zip(self.cls.exprs, px.colors.qualitative.Plotly))
        elif 10 < len(self.cls.exprs) <= 24:
            self.colors = dict(zip(self.cls.exprs, px.colors.qualitative.Dark24))
        else:
            self.colors = dict(zip(self.cls.exprs, mpl.colormaps['rainbow'](np.linspace(0, 1, len(self.cls.exprs)))))
        
        if ploter == 'plotly':
            self.fig = go.Figure()
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42
            self.fig = plt.figure(figsize=(8, 6))
            gs = self.fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = self.fig.add_subplot(gs[0, 0])
            
        x = self.cls.rsp_chbin_mean
        x_le = [chw / 2 for chw in self.cls.rsp_chbin_width]
        x_he = [chw / 2 for chw in self.cls.rsp_chbin_width]
        
        if style == 'CC':
            y = self.cls.net_ctsrate
            y_err = self.cls.net_ctsrate_error
            
            ylabel = 'Counts/s/channel'
            
        elif style == 'CE':
            y = self.cls.net_ctsspec
            y_err = self.cls.net_ctsspec_error
            
            ylabel = 'Counts/s/keV'
            
        else:
            raise ValueError(f'unsupported style argument: {style}')
            
        for i, expr in enumerate(self.cls.exprs):
                
            if ploter == 'plotly':
                obs = go.Scatter(x=x[i].astype(float), 
                                 y=y[i].astype(float), 
                                 mode='markers', 
                                 name=expr, 
                                 showlegend=True, 
                                 error_x=dict(
                                     type='data',
                                     symmetric=False,
                                     array=x_he[i].astype(float),
                                     arrayminus=x_le[i].astype(float),
                                     color=self.colors[expr],
                                     thickness=1.5,
                                     width=0),
                                 error_y=dict(
                                     type='data',
                                     array=y_err[i].astype(float),
                                     color=self.colors[expr],
                                     thickness=1.5,
                                     width=0), 
                                 marker=dict(symbol='circle', size=3, color=self.colors[expr]))
                self.fig.add_trace(obs)
                
            elif ploter == 'matplotlib':
                ax.errorbar(x[i], y[i], xerr=[x_le[i], x_he[i]], yerr=y_err[i], fmt='none', 
                            ecolor=self.colors[expr], elinewidth=0.8, capsize=0, capthick=0, label=expr)
                
        if ploter == 'plotly':
            self.fig.update_xaxes(title_text='Energy (keV)', type='log')
            self.fig.update_yaxes(title_text=ylabel, type='log')
            self.fig.update_layout(template='plotly_white', height=600, width=800)
            self.fig.show()
            
        elif ploter == 'matplotlib':
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Energy (keV)')
            ax.set_ylabel(ylabel)
            ax.autoscale(axis='x', tight=True)
            ax.autoscale(axis='y', tight=True)
            ax.minorticks_on()
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(which='major', width=0.5, length=3)
            ax.tick_params(which='minor', width=0.5, length=2)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.legend()
            plt.show()
            

    def model(self, E, T=None, ploter='plotly', style='NE', CI=False, yrange=None):
        
        if not isinstance(self.cls[0], Model):
            raise TypeError('cls is not Model list, cannot call model method')
        
        if len(self.cls) <= 10:
            self.colors = px.colors.qualitative.Plotly
        elif 10 < len(self.cls) <= 24:
            self.colors = px.colors.qualitative.Dark24
        else:
            raise ValueError('too much models')
        
        if ploter == 'plotly':
            self.fig = go.Figure()
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            self.fig = plt.figure(figsize=(8, 6))
            gs = self.fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = self.fig.add_subplot(gs[0, 0])
        
        x = np.array(E).astype(float)
        
        for i, model in enumerate(self.cls):
        
            if style == 'NE':
                ylabel = 'Photons/cm^2/s/keV'
                y = model.phtspec(E, T).astype(float)
                if CI:
                    y_ci = model.phtspec_sample(E, T)['Isigma'].astype(float)
                    
            elif style == 'Fv':
                ylabel = 'erg/cm^2/s/keV'
                y = model.flxspec(E, T).astype(float)
                if CI:
                    y_ci = model.flxspec_sample(E, T)['Isigma'].astype(float)
                    
            elif style == 'vFv':
                ylabel = 'erg/cm^2/s'
                y = model.ergspec(E, T).astype(float)
                if CI:
                    y_ci = model.ergspec_sample(E, T)['Isigma'].astype(float)
                    
            elif style == 'Fr':
                ylabel = 'Fraction'
                y = model.fracspec(E).astype(float)
                if CI:
                    y_ci = model.fracspec_sample(E)['Isigma'].astype(float)
                    
            else:
                raise ValueError(f'unsupported style argument: {style}')
        
            if ploter == 'plotly':
                mo = go.Scatter(x=x, 
                                y=y, 
                                mode='lines', 
                                name=model.expr, 
                                showlegend=True, 
                                line=dict(width=2, color=self.colors[i]))
                self.fig.add_trace(mo)
                
                if CI:
                    low = go.Scatter(x=x, 
                                    y=y_ci[0], 
                                    mode='lines', 
                                    name=f'{model.expr} lower', 
                                    fill=None, 
                                    line_color='rgba(0,0,0,0)', 
                                    showlegend=False)
                    self.fig.add_trace(low)
                    
                    upp = go.Scatter(x=x, 
                                    y=y_ci[1], 
                                    mode='lines', 
                                    name=f'{model.expr} upper', 
                                    fill='tonexty', 
                                    line_color='rgba(0,0,0,0)', 
                                    fillcolor=Plot.get_rgb(self.colors[i], 0.5), 
                                    showlegend=False)
                    self.fig.add_trace(upp)

            elif ploter == 'matplotlib':
                ax.plot(x, y, lw=1.0, color=self.colors[i])
                if CI: ax.fill_between(x, y_ci[0], y_ci[1], fc=self.colors[i], alpha=0.5)
                
        if ploter == 'plotly':
                
            self.fig.update_xaxes(title_text='Energy (keV)', type='log')
            self.fig.update_yaxes(title_text=ylabel, type='log')
            if yrange is not None: self.fig.update_yaxes(range=[np.log10(yrange[0]), np.log10(yrange[1])])
            self.fig.update_layout(template='plotly_white', height=600, width=800)
            
            self.fig.show()
                
        elif ploter == 'matplotlib':
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Energy (keV)')
            ax.set_ylabel(ylabel)
            if yrange is not None: ax.set_ylim(yrange)
            ax.minorticks_on()
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(which='major', width=0.5, length=3)
            ax.tick_params(which='minor', width=0.5, length=2)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            plt.show()
            

    def pair(self, ploter='plotly', style='CE'):
        
        if not isinstance(self.cls, Pair):
            raise TypeError('cls is not Pair type, cannot call pair method')
        
        if len(self.cls.data.exprs) <= 10:
            self.colors = dict(zip(self.cls.data.exprs, px.colors.qualitative.Plotly))
        elif 10 < len(self.cls.data.exprs) <= 24:
            self.colors = dict(zip(self.cls.data.exprs, px.colors.qualitative.Dark24))
        else:
            self.colors = dict(zip(self.cls.data.exprs, mpl.colormaps['rainbow'](np.linspace(0, 1, len(self.cls.data.exprs)))))
        
        if ploter == 'plotly':
            self.fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.75, 0.25], 
                shared_xaxes=True,
                horizontal_spacing=0,
                vertical_spacing=0.02)
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42
            self.fig = plt.figure(figsize=(6, 8))
            gs = self.fig.add_gridspec(4, 1, wspace=0, hspace=0)
            ax1 = self.fig.add_subplot(gs[0:3, 0])
            ax2 = self.fig.add_subplot(gs[3, 0], sharex=ax1)
            
        obs_x = self.cls.data.rsp_chbin_mean
        obs_x_le = [chw / 2 for chw in self.cls.data.rsp_chbin_width]
        obs_x_he = [chw / 2 for chw in self.cls.data.rsp_chbin_width]
        
        if style == 'CC':
            ylabel = 'Counts/s/channel'
            
            obs_y = self.cls.data.net_ctsrate
            obs_y_err = self.cls.data.net_ctsrate_error
            mo_y = self.cls.model.ctsrate
            res_y = list(map(lambda oi, mi, si: (oi - mi) / si, obs_y, mo_y, obs_y_err))
            
        elif style == 'CE':
            ylabel = 'Counts/s/keV'
            
            obs_y = self.cls.data.net_ctsspec
            obs_y_err = self.cls.data.net_ctsspec_error
            mo_y = self.cls.model.ctsspec
            res_y = list(map(lambda oi, mi, si: (oi - mi) / si, obs_y, mo_y, obs_y_err))
            
        else:
            raise ValueError(f'unsupported style argument: {style}')
        
        ymin = np.quantile(list(chain.from_iterable(obs_y)), 0.05).astype(float)
        ymax = 2 * np.max(list(chain.from_iterable(obs_y))).astype(float)
            
        for i, expr in enumerate(self.cls.data.exprs):
                
            if ploter == 'plotly':
                obs = go.Scatter(x=obs_x[i].astype(float), 
                                 y=obs_y[i].astype(float), 
                                 mode='markers', 
                                 name=f'obs of {expr}', 
                                 showlegend=False, 
                                 error_x=dict(
                                     type='data',
                                     symmetric=False,
                                     array=obs_x_he[i].astype(float),
                                     arrayminus=obs_x_le[i].astype(float),
                                     color=self.colors[expr],
                                     thickness=1.5,
                                     width=0),
                                 error_y=dict(
                                     type='data',
                                     array=obs_y_err[i].astype(float),
                                     color=self.colors[expr],
                                     thickness=1.5,
                                     width=0), 
                                 marker=dict(symbol='cross-thin', size=0, color=self.colors[expr]))
                mo = go.Scatter(x=obs_x[i].astype(float), 
                                y=mo_y[i].astype(float), 
                                name=expr, 
                                showlegend=True, 
                                mode='lines', 
                                line=dict(width=2, color=self.colors[expr]))
                res = go.Scatter(x=obs_x[i].astype(float), 
                                 y=res_y[i].astype(float), 
                                 name=f'res of {expr}', 
                                 showlegend=False, 
                                 mode='markers', 
                                 marker=dict(symbol='cross-thin', size=10, color=self.colors[expr], 
                                             line=dict(width=1.5, color=self.colors[expr])))
                
                self.fig.add_trace(obs, row=1, col=1)
                self.fig.add_trace(mo, row=1, col=1)
                self.fig.add_trace(res, row=2, col=1)
                
            elif ploter == 'matplotlib':
                ax1.errorbar(obs_x[i], obs_y[i], xerr = [obs_x_le[i], obs_x_he[i]], yerr=obs_y_err[i], 
                             fmt='none', ecolor=self.colors[expr], elinewidth=0.8, capsize=0, capthick=0, label=expr)
                ax1.plot(obs_x[i], mo_y[i], color=self.colors[expr], lw=1.0)
                ax2.scatter(obs_x[i], res_y[i], marker='+', color=self.colors[expr], s=40, linewidths=0.8)
                
        if ploter == 'plotly':
            self.fig.update_xaxes(title_text='', row=1, col=1, type='log')
            self.fig.update_xaxes(title_text='Energy (keV)', row=2, col=1, type='log')
            self.fig.update_yaxes(title_text=ylabel, row=1, col=1, type='log', range=[np.log10(ymin), np.log10(ymax)])
            self.fig.update_yaxes(title_text='Sigma', showgrid=False, range=[-3.5, 3.5], row=2, col=1)
            self.fig.update_layout(template='plotly_white', height=700, width=700)
            self.fig.show()
            
        elif ploter == 'matplotlib':
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylabel(ylabel)
            ax1.set_ylim([ymin, ymax])
            ax1.minorticks_on()
            ax1.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax1.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax1.tick_params(which='major', width=1.0, length=5)
            ax1.tick_params(which='minor', width=1.0, length=3)
            ax1.xaxis.set_ticks_position('both')
            ax1.yaxis.set_ticks_position('both')
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax1.spines['bottom'].set_linewidth(1.0)
            ax1.spines['top'].set_linewidth(1.0)
            ax1.spines['left'].set_linewidth(1.0)
            ax1.spines['right'].set_linewidth(1.0)
            ax1.legend(frameon=True)
            ax2.axhline(0, c='grey', lw=1, ls='--')
            ax2.set_xlabel('Energy (keV)')
            ax2.set_ylabel('Sigma')
            ax2.set_ylim([-3.5, 3.5])
            ax2.minorticks_on()
            ax2.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax2.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax2.tick_params(which='major', width=1.0, length=5)
            ax2.tick_params(which='minor', width=1.0, length=3)
            ax2.yaxis.set_ticks_position('both')
            ax2.spines['bottom'].set_linewidth(1.0)
            ax2.spines['top'].set_linewidth(1.0)
            ax2.spines['left'].set_linewidth(1.0)
            ax2.spines['right'].set_linewidth(1.0)
            plt.show()
            
            
    def walker(self):
        
        if not isinstance(self.cls, (Infer, Posterior)):
            raise TypeError('cls is not Infer or Posterior type, cannot call walker method')
        
        self.fig, axes = plt.subplots(self.cls.free_nparams, figsize=(10, 2 * self.cls.free_nparams), sharex='all')
        for i in range(self.cls.free_nparams):
            ax = axes[i]
            ax.plot(self.cls.samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(self.cls.samples))
            ax.set_ylabel(self.cls.free_plabels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        
        plt.show()
            
            
    def ctsspec(self, ploter='plotly', style='CE'):
        
        if not isinstance(self.cls, (Infer, Posterior)):
            raise TypeError('cls is not Infer or Posterior type, cannot call infer method')
        
        if isinstance(self.cls, Posterior):
            self.cls.at_par(self.cls.par_best_ci())
        
        if len(self.cls.data_exprs) <= 10:
            self.colors = dict(zip(self.cls.data_exprs, px.colors.qualitative.Plotly))
        elif 10 < len(self.cls.data_exprs) <= 24:
            self.colors = dict(zip(self.cls.data_exprs, px.colors.qualitative.Dark24))
        else:
            self.colors = dict(zip(self.cls.data_exprs, mpl.colormaps['rainbow'](np.linspace(0, 1, len(self.cls.data_exprs)))))
        
        if ploter == 'plotly':
            self.fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.75, 0.25], 
                shared_xaxes=True,
                horizontal_spacing=0,
                vertical_spacing=0.02)
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42
            self.fig = plt.figure(figsize=(6, 8))
            gs = self.fig.add_gridspec(4, 1, wspace=0, hspace=0)
            ax1 = self.fig.add_subplot(gs[0:3, 0])
            ax2 = self.fig.add_subplot(gs[3, 0], sharex=ax1)
            
        obs_x = self.cls.data_chbin_mean
        obs_x_le = [chw / 2 for chw in self.cls.data_chbin_width]
        obs_x_he = [chw / 2 for chw in self.cls.data_chbin_width]
        
        if style == 'CC':
            ylabel = 'Counts/s/channel'
            
            obs_y = self.cls.data_ctsrate
            obs_y_err = self.cls.data_ctsrate_error
            mo_y = self.cls.model_ctsrate
            res_y = list(map(lambda oi, mi, si: (oi - mi) / si, obs_y, mo_y, obs_y_err))
            
        elif style == 'CE':
            ylabel = 'Counts/s/keV'
            
            obs_y = self.cls.data_ctsspec
            obs_y_err = self.cls.data_ctsspec_error
            mo_y = self.cls.model_ctsspec
            res_y = list(map(lambda oi, mi, si: (oi - mi) / si, obs_y, mo_y, obs_y_err))
            
        else:
            raise ValueError(f'unsupported style argument: {style}')
        
        ymin = np.quantile(list(chain.from_iterable(obs_y)), 0.05).astype(float)
        ymax = 2 * np.max(list(chain.from_iterable(obs_y))).astype(float)
            
        for i, expr in enumerate(self.cls.data_exprs):
                
            if ploter == 'plotly':
                obs = go.Scatter(x=obs_x[i].astype(float), 
                                 y=obs_y[i].astype(float), 
                                 mode='markers', 
                                 name=f'obs of {expr}', 
                                 showlegend=False, 
                                 error_x=dict(
                                     type='data',
                                     symmetric=False,
                                     array=obs_x_he[i].astype(float),
                                     arrayminus=obs_x_le[i].astype(float),
                                     color=self.colors[expr],
                                     thickness=1.5,
                                     width=0),
                                 error_y=dict(
                                     type='data',
                                     array=obs_y_err[i].astype(float),
                                     color=self.colors[expr],
                                     thickness=1.5,
                                     width=0), 
                                 marker=dict(symbol='cross-thin', size=0, color=self.colors[expr]))
                mo = go.Scatter(x=obs_x[i].astype(float), 
                                y=mo_y[i].astype(float), 
                                name=expr, 
                                showlegend=True, 
                                mode='lines', 
                                line=dict(width=2, color=self.colors[expr]))
                res = go.Scatter(x=obs_x[i].astype(float), 
                                 y=res_y[i].astype(float), 
                                 name=f'res of {expr}', 
                                 showlegend=False, 
                                 mode='markers', 
                                 marker=dict(symbol='cross-thin', size=10, color=self.colors[expr], 
                                             line=dict(width=1.5, color=self.colors[expr])))
                
                self.fig.add_trace(obs, row=1, col=1)
                self.fig.add_trace(mo, row=1, col=1)
                self.fig.add_trace(res, row=2, col=1)
                
            elif ploter == 'matplotlib':
                ax1.errorbar(obs_x[i], obs_y[i], xerr = [obs_x_le[i], obs_x_he[i]], yerr=obs_y_err[i], 
                             fmt='none', ecolor=self.colors[expr], elinewidth=0.8, capsize=0, capthick=0, label=expr)
                ax1.plot(obs_x[i], mo_y[i], color=self.colors[expr], lw=1.0)
                ax2.scatter(obs_x[i], res_y[i], marker='+', color=self.colors[expr], s=40, linewidths=0.8)
                
        if ploter == 'plotly':
            self.fig.update_xaxes(title_text='', row=1, col=1, type='log')
            self.fig.update_xaxes(title_text='Energy (keV)', row=2, col=1, type='log')
            self.fig.update_yaxes(title_text=ylabel, row=1, col=1, type='log', range=[np.log10(ymin), np.log10(ymax)])
            self.fig.update_yaxes(title_text='Sigma', showgrid=False, range=[-3.5, 3.5], row=2, col=1)
            self.fig.update_layout(template='plotly_white', height=700, width=700)
            self.fig.show()
            
        elif ploter == 'matplotlib':
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylabel(ylabel)
            ax1.set_ylim([ymin, ymax])
            ax1.minorticks_on()
            ax1.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax1.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax1.tick_params(which='major', width=1.0, length=5)
            ax1.tick_params(which='minor', width=1.0, length=3)
            ax1.xaxis.set_ticks_position('both')
            ax1.yaxis.set_ticks_position('both')
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax1.spines['bottom'].set_linewidth(1.0)
            ax1.spines['top'].set_linewidth(1.0)
            ax1.spines['left'].set_linewidth(1.0)
            ax1.spines['right'].set_linewidth(1.0)
            ax1.legend(frameon=True)
            ax2.axhline(0, c='grey', lw=1, ls='--')
            ax2.set_xlabel('Energy (keV)')
            ax2.set_ylabel('Sigma')
            ax2.set_ylim([-3.5, 3.5])
            ax2.minorticks_on()
            ax2.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax2.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax2.tick_params(which='major', width=1.0, length=5)
            ax2.tick_params(which='minor', width=1.0, length=3)
            ax2.yaxis.set_ticks_position('both')
            ax2.spines['bottom'].set_linewidth(1.0)
            ax2.spines['top'].set_linewidth(1.0)
            ax2.spines['left'].set_linewidth(1.0)
            ax2.spines['right'].set_linewidth(1.0)
            plt.show()
            
            
    def corner(self):
        
        if not isinstance(self.cls, Posterior):
            raise TypeError('cls is not Posterior type, cannot call corner method')
        
        data = self.cls.posterior_sample[:, :-1].copy()
        weights = np.ones(self.cls.posterior_sample.shape[0]) / self.cls.posterior_sample.shape[0]
        levels = 1.0 - np.exp(-0.5 * np.array([1, 1.5, 2]) ** 2)
        
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.size'] = 12
        rcParams['pdf.fonttype'] = 42
        
        self.fig = corner.corner(
            data, 
            bins=30, 
            color='blue', 
            weights=weights, 
            labels=self.cls.free_plabels, 
            show_titles=True, 
            use_math_text=True, 
            smooth1d=2, 
            smooth=2, 
            levels=levels, 
            plot_datapoints=True, 
            plot_density=True, 
            plot_contours=True, 
            fill_contours=False, 
            no_fill_contours=False)

        axes = np.array(self.fig.axes).reshape((self.cls.free_nparams, self.cls.free_nparams))
        
        title_fmt = '%s = $%.2f_{-%.2f}^{+%.2f}$'
        plabel = self.cls.free_plabels
        value = self.cls.par_best_ci()
        error = self.cls.par_error(value)
        
        for i in range(self.cls.free_nparams):
            ax = axes[i, i]
            ax.set_title(title_fmt % (plabel[i], value[i], error[i][0], error[i][1]))
            ax.errorbar(value[i], 0.005, 
                        xerr=[[error[i][0]], [error[i][1]]], 
                        fmt='or', ms=2, ecolor='r', elinewidth=1)
            
        for yi in range(self.cls.free_nparams):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.errorbar(value[xi], value[yi], 
                            xerr=[[error[xi][0]], [error[xi][1]]],
                            yerr=[[error[yi][0]], [error[yi][1]]],
                            fmt='or', ms=2, ecolor='r', elinewidth=1)
        
        plt.show()
