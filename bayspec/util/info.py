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
        
        self._data_dict = new_data_dict
        
        
    @property
    def data_list(self):
        
        return Info.dict_to_list(self.data_dict)
    
    
    @property
    def data_list_dict(self):
        
        return Info.dict_to_list_dict(self.data_dict)
        
        
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
    def table(self):
        
        return tabulate(self.data_dict, 
                        headers='keys', 
                        tablefmt='fancy_grid', 
                        numalign='center', 
                        stralign='center')

    def __str__(self):
        
        print(self.table)

        return ''
