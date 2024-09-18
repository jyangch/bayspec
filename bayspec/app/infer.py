import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO
from bayspec.util.plot import Plot
from threading import current_thread
from contextlib import contextmanager
from bayspec.infer.infer import Infer
from bayspec.util.tools import init_session_state
from streamlit.runtime.scriptrunner.script_run_context import SCRIPT_RUN_CONTEXT_ATTR_NAME


css='''
<style>
    section.main > div {max-width:75rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

init_session_state()

@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME, None):
                buffer.write(b + '')
                output_func(buffer.getvalue() + '')
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write

@contextmanager
def st_stdout(dst):
    "this will show the prints"
    with st_redirect(sys.stdout, dst):
        yield

@contextmanager
def st_stderr(dst):
    "This will show the logging"
    with st_redirect(sys.stderr, dst):
        yield

def set_ini(key, ini=None):
    if key not in st.session_state.infer_state:
        st.session_state.infer_state[key] = ini

def get_val(key):
    if key in st.session_state:
        st.session_state.infer_state[key] = st.session_state[key]
    return st.session_state.infer_state[key]

def get_data(key):
    if key in st.session_state:
        for row, edited in st.session_state[key]['edited_rows'].items():
            for col, value in edited.items():
                st.session_state.infer_state[key].loc[int(row), col] = value
    return st.session_state.infer_state[key]

def get_idx(key, options):
    if key in st.session_state:
        st.session_state.infer_state[key] = st.session_state[key]
    value = st.session_state.infer_state[key]
    if (value is None) or (value not in options):
        return None
    else:
        return options.index(value)

st.session_state.infer = None
st.session_state.infer_state['infer_pair_flag'] = False
st.session_state.infer_state['run_state'] = False

all_pairs = {}
for data_key in st.session_state.data.keys():
    model_key = st.session_state.data_state[f'{data_key}_model']
    if model_key is not None:
        if st.session_state.model_state[f'{model_key}_data'] == data_key:
            all_pairs[f'{data_key}🔗{model_key}'] = [data_key, model_key]

with st.expander('***Set fitting pairs***', expanded=False):
    
    key = 'infer_pairs'; ini = list(all_pairs.keys()); set_ini(key, ini)
    options = list(all_pairs.keys())
    pairs = st.multiselect('Select infer pairs', 
                           options=options, 
                           default=get_val(key), 
                           key=key)

    if len(pairs) > 0:
        st.session_state.infer_state['infer_pair_flag'] = True
        
        pair_list = list()
        for pair in pairs:
            data_key, model_key = all_pairs[pair]
            pair_list.append([st.session_state.data[data_key], st.session_state.model[model_key]])
            
        infer = Infer(pair_list) 
        st.session_state.infer = infer
    
    cfg_col, _, par_col = st.columns([4.9, 0.2, 4.9])
    
    with cfg_col:
        with st.popover("Configurations", use_container_width=True):
            if not st.session_state.infer_state['infer_pair_flag']:
                st.warning('No infer pair!', icon="⚠️")
            else:
                cfg_df = pd.DataFrame(infer.cfg_info.data_dict)
                key = 'infer_cfg'
                cfg_df = st.data_editor(cfg_df,
                                        use_container_width=True, 
                                        num_rows='fixed', 
                                        disabled=True, 
                                        hide_index=True, 
                                        key=key)
                
    with par_col:
        with st.popover("Parameters", use_container_width=True):
            if not st.session_state.infer_state['infer_pair_flag']:
                st.warning('No infer pair!', icon="⚠️")
            else:
                key = 'infer_nlink'; ini = 'min'; set_ini(key, ini)
                nlink = st.number_input('Input the number of linking parameters', 
                                        min_value=0, 
                                        value=ini, 
                                        key=key)

                for idx in range(nlink):
                    key = 'infer_link_%d' % idx; ini = None; set_ini(key, ini)
                    options = [f'par#{p}' for p in list(infer.par.keys())]
                    pids = st.multiselect('Select the parameters to link', 
                                          options=options, 
                                          default=ini, 
                                          key=key)
                    if len(pids) > 1:
                        infer.link([int(pi[4:]) for pi in pids])
                
                par_df = pd.DataFrame(infer.par_info.data_dict)
                key = 'infer_par'
                par_df = st.data_editor(par_df, 
                                        use_container_width=True, 
                                        num_rows='fixed', 
                                        disabled=True, 
                                        hide_index=True, 
                                        key=key)
                
                free_par_df = pd.DataFrame(infer.free_par_info.data_dict)
                key = 'infer_free_par'
                free_par_df = st.data_editor(free_par_df, 
                                             use_container_width=True, 
                                             num_rows='fixed', 
                                             disabled=True, 
                                             hide_index=True, 
                                             key=key)
        
with st.expander('***Manual fitting***', expanded=False):
    
    if not st.session_state.infer_state['infer_pair_flag']:
        st.warning('No infer pair!', icon="⚠️")
    else:
        free_par_df = pd.DataFrame(infer.free_par_info.data_dict)
        key = 'manual_free_par'
        free_par_df = st.data_editor(free_par_df, 
                                     use_container_width=True, 
                                     num_rows='fixed', 
                                     disabled=['par#', 
                                               'Expression', 
                                               'Component', 
                                               'Parameter', 
                                               'Prior'], 
                                     hide_index=True, 
                                     key=key)
        now_par = list()
        for _, row in free_par_df.to_dict('index').items():
            now_par.append(row['Value'])
        infer.at_par(now_par)
    
    stat_col, _, plot_col = st.columns([4.9, 0.2, 4.9])
    
    if st.session_state.infer_state['infer_pair_flag']:

        with stat_col:
            stat_df = pd.DataFrame(infer.stat_info.data_dict)
            key = 'manual_stat'
            stat_df = st.data_editor(stat_df, 
                                     use_container_width=True, 
                                     num_rows='fixed', 
                                     disabled=True, 
                                     hide_index=True, 
                                     key=key)
        
        with plot_col:
            fig = Plot.infer_ctsspec(infer, style='CE', show=False)
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            
with st.expander('***Bayesian inference***', expanded=False):
    
    run_col, _, post_col = st.columns([4.9, 0.2, 4.9])
    
    with run_col:
        with st.popover("Sampler settings", use_container_width=True):
            key = 'infer_sampler'; ini = 'multinest'; set_ini(key, ini)
            options = ['multinest', 'emcee']
            sampler = st.selectbox('Choose bayesian sampler', 
                                    options, 
                                    index=get_idx(key, options), 
                                    key=key)

            key = 'infer_resume'; ini = 'Yes'; set_ini(key, ini)
            options = ['Yes', 'No']
            resume = st.selectbox('Choose to resume or not', 
                                  options, 
                                  index=get_idx(key, options), 
                                  key=key)
            if resume == 'Yes': resume = True
            if resume == 'No': resume = False

            if sampler == 'multinest':
                key = 'infer_multinest_nlive'; ini = 300; set_ini(key, ini)
                multinest_nlive = st.slider('Select the number of live point', 
                                            50, 1000, 
                                            value=get_val(key), 
                                            step=50, 
                                            key=key)

            if sampler == 'emcee':
                key = 'infer_emcee_nstep'; ini = 2000; set_ini(key, ini)
                emcee_nstep = st.slider('Select the number of steps', 
                                        0, 10000, 
                                        value=get_val(key), 
                                        step=1000, 
                                        key=key)
                
                key = 'infer_emcee_discard'; ini = 100; set_ini(key, ini)
                emcee_discard = st.slider('Select the discard steps', 
                                          0, 2000, 
                                          value=get_val(key), 
                                          step=100, 
                                          key=key)
                
        key = 'infer_savepath'; ini = None; set_ini(key, ini)
        savepath = st.text_input('Input folder name of results', 
                                 value=get_val(key), 
                                 placeholder='bsp', 
                                 key=key)
        if savepath == '' or savepath is None: 
            savepath = 'bsp_%d' % (np.random.uniform() * 1e10)
        savepath = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))) \
                + '/results/' + savepath
        if os.path.exists(savepath):
            st.info('Note: the folder of results has already existed!')

        key = 'infer_run'
        run = st.button(':rainbow[**RUN**]', 
                        key=key, 
                        help=None, 
                        use_container_width=True)
        
        if run:
            if not st.session_state.infer_state['infer_pair_flag']:
                st.warning('No infer pair!', icon="⚠️")
            else:
                with st.sidebar.status('Running...', expanded=True) as status:
                    st.write('Start: %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    with st_stdout("info"):
                        if sampler == 'multinest':
                            post = infer.multinest(nlive=multinest_nlive, 
                                                    resume=resume, 
                                                    savepath=savepath)
                        if sampler == 'emcee':
                            post = infer.emcee(nstep=emcee_nstep, 
                                                resume=resume, 
                                                savepath=savepath)
                    st.write('Stop: %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    st.session_state.infer_state['run_state'] = True
                    status.update(label="Run complete!", state="complete", expanded=False)
                    
    with post_col:
        
        with st.popover("Posterior analyse", use_container_width=True):
            if not st.session_state.infer_state['run_state']:
                st.warning('Please run Bayesian inference!', icon="⚠️")
            else:
                free_par_df = pd.DataFrame(post.free_par_info.data_dict)
                key = 'post_free_par'
                free_par_df = st.data_editor(free_par_df,
                                             use_container_width=True, 
                                             num_rows='fixed', 
                                             disabled=True, 
                                             hide_index=True, 
                                             key=key)
                
                stat_df = pd.DataFrame(post.stat_info.data_dict)
                key = 'post_stat'
                stat_df = st.data_editor(stat_df,
                                         use_container_width=True, 
                                         num_rows='fixed', 
                                         disabled=True, 
                                         hide_index=True, 
                                         key=key)
                
                IC_df = pd.DataFrame(post.IC_info.data_dict)
                key = 'post_IC'
                IC_df = st.data_editor(IC_df,
                                       use_container_width=True, 
                                       num_rows='fixed', 
                                       disabled=True, 
                                       hide_index=True, 
                                       key=key)
        
        with st.popover("Posterior corner plot", use_container_width=True):
            if not st.session_state.infer_state['run_state']:
                st.warning('Please run Bayesian inference!', icon="⚠️")
            else:
                fig = Plot.post_corner(post, show=False)
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                
        with st.popover("Counts spectra plot", use_container_width=True):
            if not st.session_state.infer_state['run_state']:
                st.warning('Please run Bayesian inference!', icon="⚠️")
            else:
                fig = Plot.infer_ctsspec(post, style='CE', show=False)
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                
        with st.popover("Model spectra plot", use_container_width=True):       
            key = 'post_model_style'; ini = None; set_ini(key, ini)
            options = ['Fv', 'NE', 'vFv', 'NoU']
            style = st.selectbox('Select spectral style to display', 
                                 options, 
                                 index=ini, 
                                 key=key)
            
            all_comps = dict()
            for cdict in st.session_state.model_component.values():
                all_comps.update(cdict)
                
            nou_comps = dict()
            you_comps = dict()
            for key, comp in all_comps.items():
                if comp.type in ['mul', 'math']:
                    nou_comps[key] = comp
                if comp.type in ['add', 'tinv']:
                    you_comps[key] = comp
            
            if style in ['Fv', 'NE', 'vFv']:
                options = list(you_comps.keys())
            elif style in ['NoU']:
                options = list(nou_comps.keys())
            else:
                options = []
            
            key = f'post_model_comps'; ini = None; set_ini(key, ini)
            comp_keys = st.multiselect('Select model components to display', 
                                       options=options, 
                                       default=ini, 
                                       key=key)

            if len(comp_keys) > 0:

                modelplot = Plot.model(style=style, CI=True)
                
                comp_tabs = st.tabs([r'%s' % comp for comp in comp_keys])
                for comp_key, comp_tab in zip(comp_keys, comp_tabs):
                    comp = all_comps[comp_key]
                    with comp_tab:
                        key = f'post_{comp_key}_erange'; ini = (0, 4); set_ini(key, ini)
                        erange = st.slider('Select energy range in logspace', 
                                           -1, 5, 
                                           value=(0, 4), 
                                           key=key)
                        earr = np.logspace(erange[0], erange[1], 300)
                        
                        if comp.type == 'tinv':
                            key = f'post_{comp_key}_epoch'; ini = None; set_ini(key, ini)
                            epoch = st.text_input('Input spectral time', 
                                                  value=ini, 
                                                  placeholder='defaults to 0', 
                                                  key=key)
                            if epoch == '' or epoch is None: epoch = 0.0
                            try: 
                                epoch = float(epoch)
                            except: 
                                st.error('The input value should be int or float!', icon="🚨")
                            else: 
                                tarr = epoch * np.ones_like(earr)
                        else:
                            tarr = None
                            
                    fig = modelplot.add_model(comp, earr, tarr, show=False)

                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
