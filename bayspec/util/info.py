import pandas as pd
from tabulate import tabulate



class Info(object):
    
    def __init__(self, data_dict):
        
        self.data_dict = data_dict
        
        
    @property
    def data_dict(self):
        
        return self._data_dict
    
    
    @data_dict.setter
    def data_dict(self, new_data_dict):
        
        if not isinstance(new_data_dict, dict):
            raise TypeError('expected an instance of dict')

        normalized_dict = {}
        for key, value in new_data_dict.items():
            if isinstance(value, list):
                normalized_dict[key] = value
            else:
                normalized_dict[key] = [value]
        
        self._data_dict = normalized_dict
        
        
    @property
    def data_list(self):
        
        return Info.dict_to_list(self.data_dict)
    
    
    @property
    def data_list_dict(self):
        
        return Info.dict_to_list_dict(self.data_dict)
    
    
    @property
    def data_frame(self):
        
        return pd.DataFrame(self.data_dict)
    
    
    @property
    def sanitized_data_dict(self):

        sanitized_data_dict = {}
        
        for key, values in self.data_dict.items():
            new_col = []
            for v in values:
                if isinstance(v, bool):
                    new_col.append(str(v))

                elif isinstance(v, float):
                    new_col.append(f'{v:.3f}')

                elif v is None:
                    new_col.append('None')

                else:
                    new_col.append(str(v))
            
            sanitized_data_dict[key] = new_col
            
        return sanitized_data_dict
        
        
    @classmethod
    def from_dict(cls, data_dict):
        
        if not isinstance(data_dict, dict):
            raise TypeError('expected an instance of dict')
        
        return cls(data_dict)
    
    
    @classmethod
    def from_list(cls, data_list):
        
        if not isinstance(data_list, list):
            raise TypeError('expected an instance of list')
        
        data_dict = Info.list_to_dict(data_list)
        
        return cls(data_dict)
    
    
    @classmethod
    def from_list_dict(cls, list_dict):
        
        if not isinstance(list_dict, list):
            raise TypeError('expected an instance of list')
        
        if not isinstance(list_dict[0], dict):
            raise TypeError('expected an instance of dict')
        
        data_dict = Info.list_dict_to_dict(list_dict)
        
        return cls(data_dict)


    @staticmethod
    def dict_to_list(data_dict):
        
        keys = list(data_dict.keys())
        values = list(zip(*data_dict.values()))
        
        return [keys] + [list(item) for item in values]


    @staticmethod
    def dict_to_list_dict(data_dict):
        
        return [dict(zip(data_dict.keys(), values)) for values in zip(*data_dict.values())]


    @staticmethod
    def list_to_dict(data_list):
        
        keys = data_list[0]
        values = list(zip(*data_list[1:]))
        
        return {keys[i]: list(values[i]) for i in range(len(keys))}


    @staticmethod
    def list_to_list_dict(data_list):
        
        keys = data_list[0]
        
        return [dict(zip(keys, item)) for item in data_list[1:]]


    @staticmethod
    def list_dict_to_dict(list_dict):
        
        keys = list(list_dict[0].keys())
        values = [[item[key] for item in list_dict] for key in keys]
        
        return dict(zip(keys, values))
    
    
    @staticmethod
    def list_dict_to_list(list_dict):
        
        keys = list(list_dict[0].keys())
        
        values = [list(item.values()) for item in list_dict]
        
        return [keys] + values


    @property
    def text_table(self):
        
        return tabulate(self.sanitized_data_dict, 
                        headers='keys', 
                        tablefmt='fancy_grid', 
                        missingval='None', 
                        numalign='center', 
                        stralign='center', 
                        disable_numparse=True)


    @property
    def html_style(self):
        
        return """
            <style>
            .my-table {
                border-collapse: collapse;
                font-family: sans-serif;
            }
            .my-table th, .my-table td {
                padding-left: 12px;
                padding-right: 12px;

                padding-top: 8px;
                padding-bottom: 8px;
                
                text-align: center
                border: none;
            }
            </style>
            """


    @property
    def html_table(self):
        
        return tabulate(self.sanitized_data_dict, 
                        headers='keys', 
                        tablefmt='html', 
                        missingval='None', 
                        numalign='center', 
                        stralign='center', 
                        disable_numparse=True
                        ).replace('<table>', '<table class="my-table">')
        
        
    def __str__(self):
        
        return f'{self.text_table}'
    
    
    def __repr__(self):
        
        return self.__str__()
    
    
    def _repr_html_(self):
        
        return f'{self.html_table}'
