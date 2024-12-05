import json
import numpy as np
from io import BytesIO
from .param import Par
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
        
        
def json_dump(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, cls=JsonEncoder)
        

class SuperDict(OrderedDict):
    
    def __getitem__(self, key):
        
        if isinstance(key, int):
            if key < 1 or key > len(self):
                raise IndexError("index out of range")
            key = list(self.keys())[key - 1]
            
        return super().__getitem__(key)
