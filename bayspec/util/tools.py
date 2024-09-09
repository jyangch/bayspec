import json
import numpy as np
from io import BytesIO
from .param import Par
import streamlit as st
from collections import OrderedDict



class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Par):
            return obj.todict()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, BytesIO):
            return obj.name
        else:
            return super(JsonEncoder, self).default(obj)
        
        

class SuperDict(OrderedDict):
    
    def __getitem__(self, key):
        
        if isinstance(key, int):
            if key < 1 or key > len(self):
                raise IndexError("index out of range")
            key = list(self.keys())[key - 1]
            
        return super().__getitem__(key)
    
    
def init_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = {}
    if 'data_state' not in st.session_state:
        st.session_state.data_state = {}
    if 'model' not in st.session_state:
        st.session_state.model = {}
    if 'model_component' not in st.session_state:
        st.session_state.model_component = {}
    if 'model_state' not in st.session_state:
        st.session_state.model_state = {}
    if 'infer' not in st.session_state:
        st.session_state.infer = None
    if 'infer_state' not in st.session_state:
        st.session_state.infer_state = {}
