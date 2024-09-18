import pandas as pd
import streamlit as st
from bayspec.util.plot import Plot
from bayspec.data.data import DataUnit, Data
from bayspec.util.tools import init_session_state


css='''
<style>
    section.main > div {max-width:75rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

init_session_state()

def set_ini(key, ini=None):
    if key not in st.session_state.data_state:
        st.session_state.data_state[key] = ini

def get_val(key):
    if key in st.session_state:
        st.session_state.data_state[key] = st.session_state[key]
    return st.session_state.data_state[key]

def get_idx(key, options):
    if key in st.session_state:
        st.session_state.data_state[key] = st.session_state[key]
    value = st.session_state.data_state[key]
    if (value is None) or (value not in options):
        return None
    else:
        return options.index(value)

def get_file(key, accept_multiple_files=False):
    if accept_multiple_files:
        if st.session_state[key] != []:
            st.session_state.data_state[key] = st.session_state[key]
        else:
            if st.session_state.data_state[key] != []:
                for file in st.session_state.data_state[key]:
                    st.write('📄', file.name)
    else:
        if st.session_state[key] is not None:
            st.session_state.data_state[key] = st.session_state[key]
        else:
            if st.session_state.data_state[key] is not None:
                st.write('📄', st.session_state.data_state[key].name)
    return st.session_state.data_state[key]

def reset_data():
    st.session_state.data = {}

key = 'ndata'; ini = 'min'; set_ini(key, ini)
ndata = st.sidebar.number_input('**Input the number of Data**', 
                                min_value=1, 
                                value=get_val(key), 
                                key=key, 
                                on_change=reset_data)

for i in range(ndata): 
    st.session_state.data[f'Data{i+1}'] = Data()

for di, data_key in enumerate(st.session_state.data.keys()):
    with st.expander(f'***Configure the {data_key}***', expanded=False):
        
        nunit_col, _, fit_col = st.columns([4.9, 0.2, 4.9])
        
        with nunit_col:
            key = f'{data_key}_nunit'; ini = 'min'; set_ini(key, ini)
            nunit = st.number_input('Input the number of units of Data', 
                                    min_value=1, 
                                    value=get_val(key), 
                                    key=key)
        with fit_col:
            key = f'{data_key}_model'; ini = None; set_ini(key, ini)
            options = list(st.session_state.model.keys())
            model_key = st.selectbox('Choose a Model to fit this Data', 
                                     options, 
                                     index=get_idx(key, options), 
                                     key=key)
            st.session_state.model_state[f'{model_key}_data'] = data_key

        unit_keys = [f'unit{di+1}-{i+1}' for i in range(nunit)]
        unit_tabs = st.tabs(unit_keys)
        
        for ui, (unit_key, unit_tab) in enumerate(zip(unit_keys, unit_tabs)):
            with unit_tab:
                set_col, _, info_col = st.columns([4.9, 0.2, 4.9])
                
                with set_col:    
                    key = f'{data_key}_{unit_key}_expr'; ini = unit_key; set_ini(key, ini)
                    expr = st.text_input('Input dataunit expression', 
                                         value=get_val(key), 
                                         placeholder=unit_key, 
                                         key=key)
                    if expr is None or expr == '': expr = unit_key
                    
                    if expr in st.session_state.data[data_key]:
                        st.warning('Sorry for prohibiting the use of the same dataunit name', icon="⚠️")
                        
                    src = bkg = rsp = rmf = arf = None
                        
                    key = f'{data_key}_{unit_key}_spec'; ini = []; set_ini(key, ini)
                    spec_files = st.file_uploader('Upload spectral files: src, bkg, rsp (or rmf and arf)', 
                                                  accept_multiple_files=True, key=key)
                    spec_files = get_file(key, True)
                    if spec_files is not None:
                        for speci in spec_files:
                            if 'src' in speci.name or 'pha' in speci.name: src = speci 
                            if 'bkg' in speci.name or 'bak' in speci.name: bkg = speci
                            if 'rsp' in speci.name or 'resp' in speci.name: rsp = speci
                            if 'rmf' in speci.name: rmf = speci
                            if 'arf' in speci.name: arf = speci
                    
                    key = f'{data_key}_{unit_key}_src'; ini = None; set_ini(key, ini)
                    _ = st.file_uploader('Upload source spectrum: src', key=key)
                    if get_file(key) is not None: src = get_file(key)

                    key = f'{data_key}_{unit_key}_bkg'; ini = None; set_ini(key, ini)
                    _ = st.file_uploader('Upload background spectrum: bkg', key=key)
                    if get_file(key) is not None: bkg = get_file(key)

                    key = f'{data_key}_{unit_key}_rsp'; ini = None; set_ini(key, ini)
                    _ = st.file_uploader('Upload response matrix: rsp', key=key)
                    if get_file(key) is not None: rsp = get_file(key)

                    key = f'{data_key}_{unit_key}_rmf'; ini = None; set_ini(key, ini)
                    _ = st.file_uploader('Upload redistribution matrix: rmf', key=key)
                    if get_file(key) is not None: rmf = get_file(key)

                    key = f'{data_key}_{unit_key}_arf'; ini = None; set_ini(key, ini)
                    _ = st.file_uploader('Upload auxiliary response matrix: arf', key=key)
                    if get_file(key) is not None: arf = get_file(key)

                    key = f'{data_key}_{unit_key}_stat'; ini = 'pgstat'; set_ini(key, ini)
                    options = ['gstat', 'chi2', 'pstat', 'ppstat', 'cstat', 
                               'pgstat', 'Xppstat', 'Xcstat', 'Xpgstat', 'ULppstat', 'ULpgstat']
                    stat = st.selectbox('Choose fitting statistic metric: stat', 
                                        options, 
                                        index=get_idx(key, options), 
                                        key=key)

                    key = f'{data_key}_{unit_key}_notc'; ini = None; set_ini(key, ini)
                    notc = st.text_input('Input notice energy: notc', 
                                         value=get_val(key), 
                                         placeholder='8-30;40-1000 (defaults to None)', 
                                         key=key)
                    if notc == '': notc = None
                    if notc is not None:
                        notc_list = notc.split(';')
                        notc = []
                        for notc_str in notc_list:
                            notc_range = notc_str.split('-')
                            if len(notc_range) == 2: 
                                try: 
                                    nt1 = float(notc_range[0].strip())
                                except: 
                                    st.error('The input value should be int or float!', icon="🚨")
                                try: 
                                    nt2 = float(notc_range[1].strip())
                                except: 
                                    st.error('The input value should be int or float!', icon="🚨")
                                notc.append([nt1, nt2])
                            else: st.error('The input value is in a wrong format!', icon="🚨")

                    key = f'{data_key}_{unit_key}_grpg_evt'; ini = None; set_ini(key, ini)
                    grpg_min_evt = st.text_input('Input grouping minimum events: grpg_min_evt', 
                                                 value=get_val(key), 
                                                 placeholder='5 (defaults to None)', 
                                                 key=key)
                    if grpg_min_evt == '': grpg_min_evt = None
                    if grpg_min_evt is not None:
                        try: 
                            grpg_min_evt = int(grpg_min_evt)
                        except: 
                            st.error('The input value should be int!', icon="🚨")

                    key = f'{data_key}_{unit_key}_grpg_sig'; ini = None; set_ini(key, ini)
                    grpg_min_sigma = st.text_input('Set grouping minimum sigma: grpg_min_sigma', 
                                                   value=get_val(key), 
                                                   placeholder='3 (defaults to None)', 
                                                   key=key)
                    if grpg_min_sigma == '': grpg_min_sigma = None
                    if grpg_min_sigma is not None:
                        try: 
                            grpg_min_sigma = float(grpg_min_sigma)
                        except: 
                            st.error('The input value should be int or float!', icon="🚨")

                    key = f'{data_key}_{unit_key}_grpg_bin'; ini = None; set_ini(key, ini)
                    grpg_max_bin = st.text_input('Input grouping maximum bins: grpg_max_bin', 
                                                 value=get_val(key), 
                                                 placeholder='20 (defaults to None)', 
                                                 key=key)
                    if grpg_max_bin == '': grpg_max_bin = None
                    if grpg_max_bin is not None:
                        try: 
                            grpg_max_bin = int(grpg_max_bin)
                        except: 
                            st.error('The input value should be int!', icon="🚨")

                    if grpg_min_evt is None and grpg_min_sigma is None and grpg_max_bin is None: 
                        grpg = None
                    else: 
                        grpg = {'min_evt': grpg_min_evt, 
                                'min_sigma': grpg_min_sigma, 
                                'max_bin': grpg_max_bin}

                    key = f'{data_key}_{unit_key}_time'; ini = None; set_ini(key, ini)
                    time = st.text_input('Input spectral time: time', 
                                         value=get_val(key), 
                                         placeholder='1.0 (defaults to None)', 
                                         key=key)
                    if time == '': time = None
                    if time is not None:
                        try: 
                            time = float(time)
                        except: 
                            st.error('The input value should be int or float!', icon="🚨")

                    if src is not None:
                        dataunit = DataUnit(src=src, bkg=bkg, rsp=rsp, rmf=rmf, arf=arf, 
                                            stat=stat, notc=notc, grpg=grpg, time=time)
                        if dataunit.completeness:
                            st.session_state.data[data_key][expr] = dataunit

                with info_col: 
                    st.write(''); st.write('')
                    
                    key = f'{data_key}_{unit_key}_info'; ini = False; set_ini(key, ini)
                    if st.checkbox('Show dataunit infomation', value=ini, key=key):
                        if src is None:
                            st.warning('dataunit does not exist!', icon="⚠️")
                        else:
                            info = dict()
                            info['property'] = list(dataunit.info.data_dict['property'])[:7] \
                                + ['grpg_min_evt', 'grpg_min_sigma', 'grpg_max_bin', 'time']
                            info[expr] = list(dataunit.info.data_dict[expr])[:7] \
                                + [grpg_min_evt, grpg_min_sigma, grpg_max_bin, time]
                            st.dataframe(pd.DataFrame(info), use_container_width=True, hide_index=True)
                    
                    with st.popover('Display dataunit counts spectra', use_container_width=True):
                        if src is None:
                            st.warning('dataunit does not exist!', icon="⚠️")
                        else:
                            if not dataunit.completeness:
                                st.warning('dataunit is not complete!', icon="⚠️")
                            else:
                                fig = Plot.dataunit(dataunit, style='CE', show=False)
                                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
