import sys
import corner
import numpy as np
import matplotlib as mpl
from itertools import chain
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib import rcParams
from getdist import plots, MCSamples
from plotly.subplots import make_subplots

from .tools import json_dump
from ..infer.pair import Pair
from ..model.model import Model
from ..infer.infer import Infer
from .corner import corner_plotly
from ..data.spectrum import Spectrum
from ..data.data import Data, DataUnit
from ..infer.posterior import Posterior
from ..data.response import Response, Auxiliary



class Plot(object):
    
    colors = px.colors.qualitative.Plotly \
        + px.colors.qualitative.D3 \
            + px.colors.qualitative.G10 \
                + px.colors.qualitative.T10 \
                    + px.colors.qualitative.Alphabet

    @staticmethod
    def get_rgb(color, opacity=1.0):
        
        rgba = mpl.colors.to_rgba(color)
        rgb = [int(x * 255) for x in rgba[:3]] + [opacity]
        
        return 'rgba(%d, %d, %d, %f)' % tuple(rgb)
    
    
    @staticmethod
    def spectrum(cls, ploter='plotly'):
        
        if not isinstance(cls, Spectrum):
            raise TypeError('cls is not Spectrum type, cannot call spectrum method')
        
        x = np.arange(len(cls.counts))
        y = cls.counts.astype(float)
        y_e = cls.errors.astype(float)
        
        if ploter == 'plotly':
            fig = go.Figure()
            spec = go.Scatter(
                x=x, 
                y=y, 
                mode='lines', 
                showlegend=False, 
                error_y=dict(
                    type='data',
                    array=y_e,
                    thickness=1.5,
                    width=0))
            fig.add_trace(spec)
            
            fig.update_xaxes(title_text='Channel')
            fig.update_yaxes(title_text='Counts', type='log')
            fig.update_layout(template='plotly_white', height=600, width=600)
            fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])

            ax.errorbar(x, y, yerr=y_e, fmt='-', lw=1.0, elinewidth=1.0, capsize=0)
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
            
        fig_data = {'spec': {'x': x, 'y': y, 'y_e': y_e}}
            
        return Figure(fig, fig_data, ploter)


    @staticmethod
    def response(cls, ploter='plotly', ch_range=None, ph_range=None):
        
        if not isinstance(cls, Response):
            raise TypeError('cls is not Response type, cannot call response method')
        
        if isinstance(cls, Auxiliary):
            raise TypeError('cls is Auxiliary type, cannot call response method')
        
        ch_mean = np.mean(cls.chbin, axis=1)
        ph_mean = np.mean(cls.phbin, axis=1)
        
        if ch_range is None:
            ch_idx = np.arange(len(ch_mean))
        else:
            ch_idx = np.where((ch_mean >= ch_range[0]) & (ch_mean <= ch_range[1]))[0]
            
        if ph_range is None:
            ph_idx = np.arange(len(ph_mean))
        else:
            ph_idx = np.where((ph_mean >= ph_range[0]) & (ph_mean <= ph_range[1]))[0]
            
        x = ch_mean[ch_idx].astype(float)
        y = ph_mean[ph_idx].astype(float)
        z = cls.drm[ph_idx, :][:, ch_idx].astype(float)
        
        if ploter == 'plotly':
            fig = go.Figure()
            resp = go.Contour(
                z=z, 
                x=x, 
                y=y, 
                colorscale='Jet')
            fig.add_trace(resp)
            
            fig.update_xaxes(title_text='Channel energy (keV)', type='log')
            fig.update_yaxes(title_text='Photon energy (keV)', type='log')
            fig.update_layout(template='plotly_white', height=600, width=600)
            fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            X, Y = np.meshgrid(x, y)
            
            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])

            c = ax.contourf(X, Y, z, cmap='jet')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Channel energy (keV)')
            ax.set_ylabel('Photon energy (keV)')
            fig.colorbar(c, orientation='vertical')
            
        fig_data = {'resp': {'x': x, 'y': y, 'z': z}}
            
        return Figure(fig, fig_data, ploter)


    @staticmethod
    def response_photon(cls, ploter='plotly', ph_range=None):
        
        if not isinstance(cls, Response):
            raise TypeError('cls is not Response type, cannot call response_photon method')
        
        ph_mean = np.mean(cls.phbin, axis=1)
        
        if ph_range is None:
            ph_idx = np.arange(len(ph_mean))
        else:
            ph_idx = np.where((ph_mean >= ph_range[0]) & (ph_mean <= ph_range[1]))[0]
        
        x = ph_mean[ph_idx].astype(float)
        
        if isinstance(cls, Auxiliary):
            y = cls.srp[ph_idx].astype(float)
        else:
            y = np.sum(cls.drm[ph_idx, :], axis=1).astype(float)

        if ploter == 'plotly':
            fig = go.Figure()
            resp = go.Scatter(
                x=x, 
                y=y, 
                mode='lines', 
                showlegend=False)
            fig.add_trace(resp)
            
            fig.update_xaxes(title_text='Photon energy (keV)', type='log')
            fig.update_yaxes(type='log')
            fig.update_layout(template='plotly_white', height=600, width=600)
            fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])

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

        fig_data = {'resp': {'x': x, 'y': y}}
            
        return Figure(fig, fig_data, ploter)


    @staticmethod
    def response_channel(cls, ploter='plotly', ch_range=None):
        
        if not isinstance(cls, Response):
            raise TypeError('cls is not Response type, cannot call response_channel method')
        
        if isinstance(cls, Auxiliary):
            raise TypeError('cls is Auxiliary type, cannot call response_channel method')
        
        ch_mean = np.mean(cls.chbin, axis=1)
        
        if ch_range is None:
            ch_idx = np.arange(len(ch_mean))
        else:
            ch_idx = np.where((ch_mean >= ch_range[0]) & (ch_mean <= ch_range[1]))[0]
            
        x = ch_mean[ch_idx].astype(float)
        y = np.sum(cls.drm[:, ch_idx], axis=0).astype(float)
        
        if ploter == 'plotly':
            fig = go.Figure()
            obs = go.Scatter(
                x=x, 
                y=y, 
                mode='lines', 
                showlegend=False)
            fig.add_trace(obs)
            
            fig.update_xaxes(title_text='Channel energy (keV)', type='log')
            fig.update_yaxes(type='log')
            fig.update_layout(template='plotly_white', height=600, width=600)
            fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])

            ax.plot(x, y, lw=1.0)
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
            
        fig_data = {'resp': {'x': x, 'y': y}}
            
        return Figure(fig, fig_data, ploter)


    @staticmethod
    def dataunit(cls, ploter='plotly', style='CE'):
        
        if not isinstance(cls, DataUnit):
            raise TypeError('cls is not DataUnit type, cannot call dataunit method')
        
        if not cls.completeness:
            raise AttributeError('failed for completeness check')
        
        x = cls.rsp_chbin_mean.astype(float)
        x_le = cls.rsp_chbin_width.astype(float) / 2
        x_he = cls.rsp_chbin_width.astype(float) / 2
        
        if style == 'CC':
            src_y = cls.src_ctsrate.astype(float)
            src_y_e = cls.src_ctsrate_error.astype(float)
            
            bkg_y = cls.bkg_ctsrate.astype(float)
            bkg_y_e = cls.bkg_ctsrate_error.astype(float)
            
            net_y = cls.net_ctsrate.astype(float)
            net_y_e = cls.net_ctsrate_error.astype(float)
            
            ylabel = 'Counts/s/channel'
        
        elif style == 'CE':
            src_y = cls.src_ctsspec.astype(float)
            src_y_e = cls.src_ctsspec_error.astype(float)
            
            bkg_y = cls.bkg_ctsspec.astype(float)
            bkg_y_e = cls.bkg_ctsspec_error.astype(float)
            
            net_y = cls.net_ctsspec.astype(float)
            net_y_e = cls.net_ctsspec_error.astype(float)
            
            ylabel = 'Counts/s/keV'
            
        else:
            raise ValueError(f'unsupported style argument: {style}')
            
        if ploter == 'plotly':
            fig = go.Figure()
            src = go.Scatter(
                x=x, 
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
                    array=src_y_e,
                    thickness=1.5,
                    width=0), 
                marker=dict(symbol='circle', size=3))
            
            bkg = go.Scatter(
                x=x, 
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
                    array=bkg_y_e,
                    thickness=1.5,
                    width=0), 
                marker=dict(symbol='circle', size=3))
            
            net = go.Scatter(
                x=x, 
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
                    array=net_y_e,
                    thickness=1.5,
                    width=0), 
                marker=dict(symbol='circle', size=3))
            
            fig.add_trace(src)
            fig.add_trace(bkg)
            fig.add_trace(net)
            
            fig.update_xaxes(title_text='Energy (keV)', type='log')
            fig.update_yaxes(title_text=ylabel, type='log')
            fig.update_layout(template='plotly_white', height=600, width=600)
            fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])
            
            ax.errorbar(x, src_y, xerr = [x_le, x_he], yerr=src_y_e, fmt='none', 
                        ecolor='m', elinewidth=1.0, capsize=0, label='src')
            ax.errorbar(x, bkg_y, xerr = [x_le, x_he], yerr=bkg_y_e, fmt='none', 
                        ecolor='b', elinewidth=1.0, capsize=0, label='bkg')
            ax.errorbar(x, net_y, xerr = [x_le, x_he], yerr=net_y_e, fmt='none', 
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
            
        fig_data = {
            'src': {'x': x, 'y': src_y, 'x_le': x_le, 'x_he': x_he, 'y_e': src_y_e}, 
            'bkg': {'x': x, 'y': bkg_y, 'x_le': x_le, 'x_he': x_he, 'y_e': bkg_y_e}, 
            'net': {'x': x, 'y': net_y, 'x_le': x_le, 'x_he': x_he, 'y_e': net_y_e}}
            
        return Figure(fig, fig_data, ploter)


    @staticmethod
    def data(cls, ploter='plotly', style='CE'):
        
        if not isinstance(cls, Data):
            raise TypeError('cls is not Data type, cannot call data method')
        
        if ploter == 'plotly':
            fig = go.Figure()
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42
            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])
            
        x = cls.rsp_chbin_mean
        x_le = [chw / 2 for chw in cls.rsp_chbin_width]
        x_he = [chw / 2 for chw in cls.rsp_chbin_width]
        
        if style == 'CC':
            y = cls.net_ctsrate
            y_e = cls.net_ctsrate_error
            
            ylabel = 'Counts/s/channel'
            
        elif style == 'CE':
            y = cls.net_ctsspec
            y_e = cls.net_ctsspec_error
            
            ylabel = 'Counts/s/keV'
            
        else:
            raise ValueError(f'unsupported style argument: {style}')
            
        fig_data = {}
            
        for i, name in enumerate(cls.names):
                
            if ploter == 'plotly':
                obs = go.Scatter(
                    x=x[i].astype(float), 
                    y=y[i].astype(float), 
                    mode='markers', 
                    name=f'obs of {name}', 
                    showlegend=True, 
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=x_he[i].astype(float),
                        arrayminus=x_le[i].astype(float),
                        color=Plot.colors[i],
                        thickness=1.5,
                        width=0),
                    error_y=dict(
                        type='data',
                        array=y_e[i].astype(float),
                        color=Plot.colors[i],
                        thickness=1.5,
                        width=0), 
                    marker=dict(symbol='circle', size=3, color=Plot.colors[i]))
                fig.add_trace(obs)
                
            elif ploter == 'matplotlib':
                ax.errorbar(x[i], y[i], xerr=[x_le[i], x_he[i]], yerr=y_e[i], fmt='none', 
                            ecolor=Plot.colors[i], elinewidth=0.8, capsize=0, capthick=0, label=name)
                
            fig_data[name] = {
                'obs': {'x': x[i], 'y': y[i], 'x_le': x_le[i], 'x_he': x_he[i], 'y_e': y_e[i]}}
                
        if ploter == 'plotly':
            fig.update_xaxes(title_text='Energy (keV)', type='log')
            fig.update_yaxes(title_text=ylabel, type='log')
            fig.update_layout(template='plotly_white', height=600, width=600)
            fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
            
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
            
        return Figure(fig, fig_data, ploter)


    @staticmethod
    def model(ploter='plotly', style='NE', post=False, yrange=None):

        modelplot = ModelPlot(ploter=ploter, style=style, post=post, yrange=yrange)

        return modelplot


    @staticmethod
    def pair(cls, ploter='plotly', style='CE'):
        
        if not isinstance(cls, Pair):
            raise TypeError('cls is not Pair type, cannot call pair method')
        
        if ploter == 'plotly':
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.75, 0.25], 
                shared_xaxes=True,
                horizontal_spacing=0,
                vertical_spacing=0.02)
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42
            fig = plt.figure(figsize=(6, 8))
            gs = fig.add_gridspec(4, 1, wspace=0, hspace=0)
            ax1 = fig.add_subplot(gs[0:3, 0])
            ax2 = fig.add_subplot(gs[3, 0], sharex=ax1)
            
        obs_x = cls.data.rsp_chbin_mean
        obs_x_le = [chw / 2 for chw in cls.data.rsp_chbin_width]
        obs_x_he = [chw / 2 for chw in cls.data.rsp_chbin_width]
        
        if style == 'CC':
            ylabel = 'Counts/s/channel'
            
            obs_y = cls.data.net_ctsrate
            obs_y_e = cls.data.net_ctsrate_error
            mo_y = cls.model.conv_ctsrate
            res_y = list(map(lambda oi, mi, si: (oi - mi) / si, obs_y, mo_y, obs_y_e))
            
        elif style == 'CE':
            ylabel = 'Counts/s/keV'
            
            obs_y = cls.data.net_ctsspec
            obs_y_e = cls.data.net_ctsspec_error
            mo_y = cls.model.conv_ctsspec
            res_y = list(map(lambda oi, mi, si: (oi - mi) / si, obs_y, mo_y, obs_y_e))
            
        elif style == 'NE':
            ylabel = 'Photons/cm2/s/keV'
            
            obs_y = cls.deconv_phtspec
            obs_y_e = cls.deconv_phtspec_error
            mo_y = cls.phtspec_at_rsp
            res_y = list(map(lambda oi, mi, si: (oi - mi) / si, obs_y, mo_y, obs_y_e))
            
        elif style == 'Fv' or style == 'ENE':
            ylabel = 'erg/cm2/s/keV'
            
            obs_y = cls.deconv_flxspec
            obs_y_e = cls.deconv_flxspec_error
            mo_y = cls.flxspec_at_rsp
            res_y = list(map(lambda oi, mi, si: (oi - mi) / si, obs_y, mo_y, obs_y_e))
            
        elif style == 'vFv' or style == 'EENE':
            ylabel = 'erg/cm2/s'
            
            obs_y = cls.deconv_ergspec
            obs_y_e = cls.deconv_ergspec_error
            mo_y = cls.ergspec_at_rsp
            res_y = list(map(lambda oi, mi, si: (oi - mi) / si, obs_y, mo_y, obs_y_e))
            
        else:
            raise ValueError(f'unsupported style argument: {style}')
        
        yall = np.array(list(chain.from_iterable(obs_y)))
        ymin = 0.5 * np.min(yall[yall > 0]).astype(float)
        ymax = 2 * np.max(yall[yall > 0]).astype(float)
            
        fig_data = {}    
        
        for i, name in enumerate(cls.data.names):
                
            if ploter == 'plotly':
                obs = go.Scatter(
                    x=obs_x[i].astype(float), 
                    y=obs_y[i].astype(float), 
                    mode='markers', 
                    name=f'obs of {name}', 
                    showlegend=False, 
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=obs_x_he[i].astype(float),
                        arrayminus=obs_x_le[i].astype(float),
                        color=Plot.colors[i],
                        thickness=1.5,
                        width=0),
                    error_y=dict(
                        type='data',
                        array=obs_y_e[i].astype(float),
                        color=Plot.colors[i],
                        thickness=1.5,
                        width=0), 
                    marker=dict(symbol='cross-thin', size=0, color=Plot.colors[i]))
                mo = go.Scatter(
                    x=obs_x[i].astype(float), 
                    y=mo_y[i].astype(float), 
                    name=name, 
                    showlegend=True, 
                    mode='lines', 
                    line=dict(width=2, color=Plot.colors[i]))
                res = go.Scatter(
                    x=obs_x[i].astype(float), 
                    y=res_y[i].astype(float), 
                    name=f'res of {name}', 
                    showlegend=False, 
                    mode='markers', 
                    marker=dict(symbol='cross-thin', size=10, color=Plot.colors[i], 
                                line=dict(width=1.5, color=Plot.colors[i])))
                
                fig.add_trace(obs, row=1, col=1)
                fig.add_trace(mo, row=1, col=1)
                fig.add_trace(res, row=2, col=1)
                
            elif ploter == 'matplotlib':
                ax1.errorbar(obs_x[i], obs_y[i], xerr = [obs_x_le[i], obs_x_he[i]], yerr=obs_y_e[i], 
                             fmt='none', ecolor=Plot.colors[i], elinewidth=0.8, capsize=0, capthick=0, label=name)
                ax1.plot(obs_x[i], mo_y[i], color=Plot.colors[i], lw=1.0)
                ax2.scatter(obs_x[i], res_y[i], marker='+', color=Plot.colors[i], s=40, linewidths=0.8)
                
            fig_data[name] = {
                'obs': {'x': obs_x[i], 'y': obs_y[i], 'x_le': obs_x_le[i], 'x_he': obs_x_he[i], 'y_e': obs_y_e[i]}, 
                'mo': {'x': obs_x[i], 'y': mo_y[i]}, 
                'res': {'x': obs_x[i], 'y': res_y[i]}}
                
        if ploter == 'plotly':
            fig.update_xaxes(title_text='', row=1, col=1, type='log')
            fig.update_xaxes(title_text='Energy (keV)', row=2, col=1, type='log')
            fig.update_yaxes(title_text=ylabel, row=1, col=1, type='log')
            fig.update_yaxes(title_text=ylabel, row=1, col=1, type='log', range=[np.log10(ymin), np.log10(ymax)])
            fig.update_yaxes(title_text='Sigma', showgrid=False, range=[-3.5, 3.5], row=2, col=1)
            fig.update_layout(template='plotly_white', height=700, width=600)
            fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
            
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
            
        return Figure(fig, fig_data, ploter)
            
    
    @staticmethod
    def emcee_walker(cls):
        
        if not isinstance(cls, (Infer, Posterior)):
            raise TypeError('cls is not Infer or Posterior type, cannot call walker method')
        
        fig, axes = plt.subplots(cls.free_nparams, figsize=(10, 2 * cls.free_nparams), sharex='all')
        for i in range(cls.free_nparams):
            ax = axes[i]
            ax.plot(cls.params_samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(cls.params_samples))
            ax.set_ylabel(cls.free_plabels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        
        fig_data = None
        
        return Figure(fig, fig_data, 'matplotlib')
            
    
    @staticmethod
    def infer(cls, ploter='plotly', style='CE', rebin=True, at_par='best'):
        
        if not isinstance(cls, (Infer, Posterior)):
            raise TypeError('cls is not Infer or Posterior type, cannot call infer method')
        
        if isinstance(cls, Posterior):
            if at_par == 'best': cls.at_par(cls.par_best)
            elif at_par == 'best-ci': cls.at_par(cls.par_best_ci)
            elif at_par == 'median': cls.at_par(cls.par_median)
            elif at_par == 'mean': cls.at_par(cls.par_mean)
            else: raise ValueError(f'unsupported at_par argument: {at_par}')
        
        if ploter == 'plotly':
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.75, 0.25], 
                shared_xaxes=True,
                horizontal_spacing=0,
                vertical_spacing=0.02)
            
        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42
            fig = plt.figure(figsize=(6, 8))
            gs = fig.add_gridspec(4, 1, wspace=0, hspace=0)
            ax1 = fig.add_subplot(gs[0:3, 0])
            ax2 = fig.add_subplot(gs[3, 0], sharex=ax1)
            
        if not rebin:
            obs_x = cls.data_chbin_mean
            obs_x_le = [chw / 2 for chw in cls.data_chbin_width]
            obs_x_he = [chw / 2 for chw in cls.data_chbin_width]
            res_y = cls.residual
        else:
            obs_x = cls.data_re_chbin_mean
            obs_x_le = [chw / 2 for chw in cls.data_re_chbin_width]
            obs_x_he = [chw / 2 for chw in cls.data_re_chbin_width]
            res_y = cls.re_residual
        
        if style == 'CC':
            ylabel = 'Counts/s/channel'
            
            if not rebin:
                obs_y = cls.data_ctsrate
                obs_y_e = cls.data_ctsrate_error
                mo_y = cls.model_ctsrate
            else:
                obs_y = cls.data_re_ctsrate
                obs_y_e = cls.data_re_ctsrate_error
                mo_y = cls.model_re_ctsrate
            
        elif style == 'CE':
            ylabel = 'Counts/s/keV'
            
            if not rebin:
                obs_y = cls.data_ctsspec
                obs_y_e = cls.data_ctsspec_error
                mo_y = cls.model_ctsspec
            else:
                obs_y = cls.data_re_ctsspec
                obs_y_e = cls.data_re_ctsspec_error
                mo_y = cls.model_re_ctsspec
                
        elif style == 'NE':
            ylabel = 'Photons/cm2/s/keV'
            
            if not rebin:
                obs_y = cls.data_phtspec
                obs_y_e = cls.data_phtspec_error
                mo_y = cls.model_phtspec
            else:
                obs_y = cls.data_re_phtspec
                obs_y_e = cls.data_re_phtspec_error
                mo_y = cls.model_re_phtspec
                
        elif style == 'Fv' or style == 'ENE':
            ylabel = 'erg/cm2/s/keV'
            
            if not rebin:
                obs_y = cls.data_flxspec
                obs_y_e = cls.data_flxspec_error
                mo_y = cls.model_flxspec
            else:
                obs_y = cls.data_re_flxspec
                obs_y_e = cls.data_re_flxspec_error
                mo_y = cls.model_re_flxspec
                
        elif style == 'vFv' or style == 'EENE':
            ylabel = 'erg/cm2/s'
            
            if not rebin:
                obs_y = cls.data_ergspec
                obs_y_e = cls.data_ergspec_error
                mo_y = cls.model_ergspec
            else:
                obs_y = cls.data_re_ergspec
                obs_y_e = cls.data_re_ergspec_error
                mo_y = cls.model_re_ergspec
            
        else:
            raise ValueError(f'unsupported style argument: {style}')
        
        yall = np.array(list(chain.from_iterable(obs_y)))
        ymin = 0.5 * np.min(yall[yall > 0]).astype(float)
        ymax = 2 * np.max(yall[yall > 0]).astype(float)
        
        fig_data = {}
            
        for i, name in enumerate(cls.data_names):
                
            if ploter == 'plotly':
                obs = go.Scatter(
                    x=obs_x[i].astype(float), 
                    y=obs_y[i].astype(float), 
                    mode='markers', 
                    name=f'obs of {name}', 
                    showlegend=False, 
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=obs_x_he[i].astype(float),
                        arrayminus=obs_x_le[i].astype(float),
                        color=Plot.colors[i],
                        thickness=1.5,
                        width=0),
                    error_y=dict(
                        type='data',
                        array=obs_y_e[i].astype(float),
                        color=Plot.colors[i],
                        thickness=1.5,
                        width=0), 
                    marker=dict(symbol='cross-thin', size=0, color=Plot.colors[i]))
                mo = go.Scatter(
                    x=obs_x[i].astype(float), 
                    y=mo_y[i].astype(float), 
                    name=name, 
                    showlegend=True, 
                    mode='lines', 
                    line=dict(width=2, color=Plot.colors[i]))
                res = go.Scatter(
                    x=obs_x[i].astype(float), 
                    y=res_y[i].astype(float), 
                    name=f'res of {name}', 
                    showlegend=False, 
                    mode='markers', 
                    marker=dict(symbol='cross-thin', size=10, color=Plot.colors[i], 
                                line=dict(width=1.5, color=Plot.colors[i])))
                
                fig.add_trace(obs, row=1, col=1)
                fig.add_trace(mo, row=1, col=1)
                fig.add_trace(res, row=2, col=1)
                
            elif ploter == 'matplotlib':
                ax1.errorbar(obs_x[i], obs_y[i], xerr = [obs_x_le[i], obs_x_he[i]], yerr=obs_y_e[i], 
                             fmt='none', ecolor=Plot.colors[i], elinewidth=0.8, capsize=0, capthick=0, label=name)
                ax1.plot(obs_x[i], mo_y[i], color=Plot.colors[i], lw=1.0)
                ax2.scatter(obs_x[i], res_y[i], marker='+', color=Plot.colors[i], s=40, linewidths=0.8)
                
            fig_data[name] = {
                'obs': {'x': obs_x[i], 'y': obs_y[i], 'x_le': obs_x_le[i], 'x_he': obs_x_he[i], 'y_e': obs_y_e[i]}, 
                'mo': {'x': obs_x[i], 'y': mo_y[i]}, 
                'res': {'x': obs_x[i], 'y': res_y[i]}}
                
        if ploter == 'plotly':
            fig.update_xaxes(title_text='', row=1, col=1, type='log')
            fig.update_xaxes(title_text='Energy (keV)', row=2, col=1, type='log')
            fig.update_yaxes(title_text=ylabel, row=1, col=1, type='log')
            fig.update_yaxes(title_text=ylabel, row=1, col=1, type='log', range=[np.log10(ymin), np.log10(ymax)])
            fig.update_yaxes(title_text='Sigma', showgrid=False, range=[-3.5, 3.5], row=2, col=1)
            fig.update_layout(template='plotly_white', height=700, width=600)
            fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
            
        elif ploter == 'matplotlib':
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylabel(f'$\\rm {ylabel}$')
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
            
        return Figure(fig, fig_data, ploter)
            
        
    @staticmethod
    def post_corner(cls, ploter='plotly', at_par='best'):
        
        if not isinstance(cls, Posterior):
            raise TypeError('cls is not Posterior type, cannot call corner method')
        
        data = cls.posterior_sample[:, :-1].copy()
        weights = np.ones(cls.posterior_sample.shape[0]) / cls.posterior_sample.shape[0]

        title_fmt = '$%.2f_{-%.2f}^{+%.2f}~(%.2f)$'
        plabels = [f'p{key}({label})' for label, key in zip(cls.free_plabels, cls.free_par.keys())]
        
        if at_par == 'best': truth = cls.par_best
        elif at_par == 'best-ci': truth = cls.par_best_ci
        elif at_par == 'median': truth = cls.par_median
        elif at_par == 'mean': truth = cls.par_mean
        else: raise ValueError(f'unsupported at_par argument: {at_par}')
        
        median = cls.par_median
        error = cls.par_error(median)
        
        if ploter == 'plotly':
            
            levels = 1.0 - np.exp(-0.5 * np.array([1, 2]) ** 2)
            
            fig = corner_plotly(
                data, 
                bins=30, 
                weights=weights, 
                smooth1d=2, 
                smooth=2, 
                labels=plabels, 
                levels=levels)
            
            for i in range(cls.free_nparams):
                fig.add_trace(
                    go.Scatter(
                        x=[median[i]],
                        y=[0.01],
                        mode='markers',
                        name=plabels[i],
                        showlegend=False, 
                        error_x=dict(
                            type='data',
                            symmetric=False,
                            array=[error[i][1]], 
                            arrayminus=[error[i][0]], 
                            color='#FF0092',
                            thickness=1,
                            width=0),
                        marker=dict(symbol='circle', size=5, color='#FF0092')), 
                    row=i + 1, col=i + 1)
                
            for yi in range(cls.free_nparams):
                for xi in range(yi):
                    fig.add_vline(truth[xi], line_width=1, line_color='#FF0092', row=yi + 1, col=xi + 1)
                    fig.add_hline(truth[yi], line_width=1, line_color='#FF0092', row=yi + 1, col=xi + 1)
                    fig.add_trace(
                        go.Scatter(
                            x=[truth[xi]], 
                            y=[truth[yi]], 
                            mode='markers', 
                            name=f'{plabels[xi]}&{plabels[yi]}', 
                            showlegend=False, 
                            marker=dict(symbol='square', size=5, color='#FF0092')),
                        row=yi + 1, col=xi + 1)

        elif ploter == 'getdist':
            
            fig = plots.get_subplot_plotter()
            fig.settings.num_plot_contours = 2
            fig.settings.num_shades = 30
            fig.settings.title_limit_fontsize = 10

            mcsample = MCSamples(samples=data, names=plabels, sampler=cls.sampler)
            mcsample.updateSettings({"contours": [0.6827, 0.9545, 0.9973]})
            
            fig.triangle_plot(mcsample, plabels, shaded=True)
            
            for i in range(cls.free_nparams):
                ax = fig.subplots[i, i]
                ax.set_title(title_fmt % (median[i], error[i][0], error[i][1], truth[i]), 
                             math_fontfamily='stix')
                ax.errorbar(median[i], 0.05, xerr=[[error[i][0]], [error[i][1]]], 
                            fmt='or', ms=2, ecolor='r', elinewidth=0.7)
                ax.tick_params(axis='both', which='both', zorder=10)
                
            for yi in range(cls.free_nparams):
                for xi in range(yi):
                    ax = fig.subplots[yi, xi]
                    ax.axvline(truth[xi], color='r', lw=0.7, ls='-')
                    ax.axhline(truth[yi], color='r', lw=0.7, ls='-')
                    ax.scatter(truth[xi], truth[yi], marker='s', color='r', s=10, linewidths=0, zorder=10)
                    ax.tick_params(axis='both', which='both', zorder=10)

        elif ploter == 'cornerpy':
        
            levels = 1.0 - np.exp(-0.5 * np.array([1, 2]) ** 2)
            
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            fig = corner.corner(
                data, 
                bins=30, 
                color='#08519c', 
                weights=weights, 
                labels=plabels, 
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

            axes = np.array(fig.axes).reshape((cls.free_nparams, cls.free_nparams))
            
            for i in range(cls.free_nparams):
                ax = axes[i, i]
                ax.set_title(title_fmt % (median[i], error[i][0], error[i][1], truth[i]), 
                             math_fontfamily='stix')
                ax.errorbar(median[i], 0.005, xerr=[[error[i][0]], [error[i][1]]], 
                            fmt='or', ms=2, ecolor='r', elinewidth=1)
                
            for yi in range(cls.free_nparams):
                for xi in range(yi):
                    ax = axes[yi, xi]
                    ax.axvline(truth[xi], color='r', lw=1, ls='-')
                    ax.axhline(truth[yi], color='r', lw=1, ls='-')
                    ax.scatter(truth[xi], truth[yi], marker='s', color='r', s=20, linewidths=0)
            
        fig_data = None
            
        return Figure(fig, fig_data, ploter)



class ModelPlot(object):
    
    colors = px.colors.qualitative.Plotly \
        + px.colors.qualitative.D3 \
            + px.colors.qualitative.G10 \
                + px.colors.qualitative.T10 \
                    + px.colors.qualitative.Alphabet
    
    def __init__(self, ploter='plotly', style='NE', post=False, yrange=None):
        
        self.ploter = ploter
        self.style = style
        self.post = post
        self.yrange = yrange
        
        if self.style == 'NE':
            ylabel = 'Photons/cm2/s/keV'
        elif self.style == 'Fv' or self.style == 'ENE':
            ylabel = 'erg/cm2/s/keV'
        elif self.style == 'vFv' or self.style == 'EENE':
            ylabel = 'erg/cm2/s'
        elif self.style == 'NoU':
            ylabel = 'dimensionless'
        else:
            raise ValueError(f'unsupported style argument: {self.style}')
        
        if self.ploter == 'plotly':
            self.fig = go.Figure()
            
            self.fig.update_xaxes(title_text='Energy (keV)', type='log')
            self.fig.update_yaxes(title_text=ylabel, type='log')
            if yrange is not None: self.fig.update_yaxes(range=[np.log10(yrange[0]), np.log10(yrange[1])])
            self.fig.update_layout(template='plotly_white', height=600, width=600)
            self.fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
            
        elif self.ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            self.fig = plt.figure(figsize=(8, 6))
            gs = self.fig.add_gridspec(1, 1, wspace=0, hspace=0)
            self.ax = self.fig.add_subplot(gs[0, 0])
            
            self.ax.set_xscale('log')
            self.ax.set_yscale('log')
            self.ax.set_xlabel('Energy (keV)')
            self.ax.set_ylabel(ylabel)
            if yrange is not None: self.ax.set_ylim(yrange)
            self.ax.minorticks_on()
            self.ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            self.ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            self.ax.tick_params(which='major', width=0.5, length=3)
            self.ax.tick_params(which='minor', width=0.5, length=2)
            self.ax.spines['bottom'].set_linewidth(0.5)
            self.ax.spines['left'].set_linewidth(0.5)
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['top'].set_visible(False)
            
        self.fig_data = {}
            
        self.model_index = -1
        
        
    @staticmethod
    def get_rgb(color, opacity=1.0):
        
        rgba = mpl.colors.to_rgba(color)
        rgb = [int(x * 255) for x in rgba[:3]] + [opacity]
        
        return 'rgba(%d, %d, %d, %f)' % tuple(rgb)
        
        
    def add_model(self, model, E, T=None, post=None, at_par='best'):
        
        if not isinstance(model, Model):
            raise TypeError('model is not Model type, cannot call add_model method')
        
        if post is None:
            post = self.post
        
        self.model_index += 1
        
        x = np.array(E).astype(float)
        
        if self.style == 'NE':
            if model.type not in ['add', 'tinv']:
                raise AttributeError(f'{self.style} is invalid for {model.type} type model')
            
            if post:
                if at_par == 'best': y = model.best_phtspec(E, T).astype(float)
                elif at_par == 'best-ci': y = model.best_ci_phtspec(E, T).astype(float)
                elif at_par == 'median': y = model.median_phtspec(E, T).astype(float)
                elif at_par == 'mean': y = model.mean_phtspec(E, T).astype(float)
                else: raise ValueError(f'unsupported at_par argument: {at_par}')
                y_sample = model.phtspec_sample(E, T)
                y_ci = y_sample['Isigma'].astype(float)
            else:
                y = model.phtspec(E, T).astype(float)
                
        elif self.style == 'Fv' or self.style == 'ENE':
            if model.type not in ['add', 'tinv']:
                raise AttributeError(f'{self.style} is invalid for {model.type} type model')
            
            if post:
                if at_par == 'best': y = model.best_flxspec(E, T).astype(float)
                elif at_par == 'best-ci': y = model.best_ci_flxspec(E, T).astype(float)
                elif at_par == 'median': y = model.median_flxspec(E, T).astype(float)
                elif at_par == 'mean': y = model.mean_flxspec(E, T).astype(float)
                else: raise ValueError(f'unsupported at_par argument: {at_par}')
                y_sample = model.flxspec_sample(E, T)
                y_ci = y_sample['Isigma'].astype(float)
            else:
                y = model.flxspec(E, T).astype(float)
                
        elif self.style == 'vFv' or self.style == 'EENE':
            if model.type not in ['add', 'tinv', 'math']:
                raise AttributeError(f'{self.style} is invalid for {model.type} type model')
            
            if post:
                if at_par == 'best': y = model.best_ergspec(E, T).astype(float)
                elif at_par == 'best-ci': y = model.best_ci_ergspec(E, T).astype(float)
                elif at_par == 'median': y = model.median_ergspec(E, T).astype(float)
                elif at_par == 'mean': y = model.mean_ergspec(E, T).astype(float)
                else: raise ValueError(f'unsupported at_par argument: {at_par}')
                y_sample = model.ergspec_sample(E, T)
                y_ci = y_sample['Isigma'].astype(float)
            else:
                y = model.ergspec(E, T).astype(float)

        elif self.style == 'NoU':
            if model.type not in ['mul', 'math']:
                raise AttributeError(f'{self.style} is invalid for {model.type} type model')
            
            if post:
                if at_par == 'best': y = model.best_nouspec(E).astype(float)
                elif at_par == 'best-ci': y = model.best_ci_nouspec(E).astype(float)
                elif at_par == 'median': y = model.median_nouspec(E).astype(float)
                elif at_par == 'mean': y = model.mean_nouspec(E).astype(float)
                else: raise ValueError(f'unsupported at_par argument: {at_par}')
                y_sample = model.nouspec_sample(E)
                y_ci = y_sample['Isigma'].astype(float)
            else:
                y = model.nouspec(E).astype(float)
                
        else:
            raise ValueError(f'unsupported style argument: {self.style}')
        
        if self.ploter == 'plotly':
            mo = go.Scatter(
                x=x, 
                y=y, 
                mode='lines', 
                name=model.expr, 
                showlegend=True, 
                line=dict(width=2, color=ModelPlot.colors[self.model_index]))
            self.fig.add_trace(mo)
            
            if post:
                low = go.Scatter(
                    x=x, 
                    y=y_ci[0], 
                    mode='lines', 
                    name=f'{model.expr} lower', 
                    fill=None, 
                    line_color='rgba(0,0,0,0)', 
                    showlegend=False)
                self.fig.add_trace(low)
                
                upp = go.Scatter(
                    x=x, 
                    y=y_ci[1], 
                    mode='lines', 
                    name=f'{model.expr} CI', 
                    fill='tonexty', 
                    line_color='rgba(0,0,0,0)', 
                    fillcolor=ModelPlot.get_rgb(ModelPlot.colors[self.model_index], 0.5), 
                    showlegend=True)
                self.fig.add_trace(upp)

        elif self.ploter == 'matplotlib':
            self.ax.plot(x, y, lw=1.0, color=ModelPlot.colors[self.model_index], label=model.expr)
            if post: 
                self.ax.fill_between(x, y_ci[0], y_ci[1], fc=ModelPlot.colors[self.model_index], 
                                     alpha=0.5, label=f'{model.expr} CI')
            self.ax.legend(frameon=True)

        if post:
            self.fig_data[model.expr] = {'x': x, 'y': y, 'y_ci': y_ci}
        else:
            self.fig_data[model.expr] = {'x': x, 'y': y}


    def get_fig(self):

        return Figure(self.fig, self.fig_data, self.ploter)



class Figure(object):
    
    def __init__(self, fig, fig_data, plotter):
        
        self.fig = fig
        self.fig_data = fig_data
        self.plotter = plotter
        
        if self.is_notebook() and self.plotter == 'plotly':
            self.fig.show()


    @staticmethod
    def is_notebook():
        return 'ipykernel' in sys.modules


    def save(self, fname):
        
        if self.fig_data is not None:
            json_dump(self.fig_data, f'{fname}.json')
        
        if self.plotter == 'plotly':
            self.fig.write_html(f'{fname}.html')
        elif self.plotter == 'matplotlib':
            self.fig.savefig(f'{fname}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close(self.fig)
        elif self.plotter == 'cornerpy':
            self.fig.savefig(f'{fname}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close(self.fig)
        elif self.plotter == 'getdist':
            self.fig.export(f'{fname}.pdf')
        else:
            raise ValueError(f'unsupported plotter: {self.plotter}')
