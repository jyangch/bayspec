"""Tabular ``Info`` wrapper used to format structured metadata.

Normalizes between three equivalent shapes — a column-major ``dict`` of
lists, a header-plus-rows ``list``, and a list of row dicts — and renders
the result as text or HTML for CLI and notebook display.
"""

import pandas as pd
from tabulate import tabulate



class Info(object):
    """Tabular view over a column-major dictionary of metadata.

    Accepts scalar values by broadcasting them to single-element columns,
    and exposes common conversions (list, list-of-dicts, DataFrame) plus
    formatted text and HTML tables.

    Attributes:
        data_dict: Column-major dictionary with list-valued columns.
    """

    def __init__(self, data_dict):
        """Store ``data_dict`` after normalizing scalar columns to lists.

        Args:
            data_dict: Column-major dictionary. Scalar values are wrapped
                into single-element lists.
        """

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
        """Header-plus-rows list representation of the table."""

        return Info.dict_to_list(self.data_dict)


    @property
    def data_list_dict(self):
        """List of row dictionaries, one entry per row."""

        return Info.dict_to_list_dict(self.data_dict)


    @property
    def data_frame(self):
        """Return the data as a ``pandas.DataFrame``."""

        return pd.DataFrame(self.data_dict)


    @property
    def sanitized_data_dict(self):
        """Column-major dict with every value coerced to a display string.

        Floats are formatted with three decimal places and ``None`` values
        become the literal string ``'None'``.
        """

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
        """Build an ``Info`` from a column-major dictionary.

        Raises:
            TypeError: If ``data_dict`` is not a ``dict``.
        """

        if not isinstance(data_dict, dict):
            raise TypeError('expected an instance of dict')

        return cls(data_dict)


    @classmethod
    def from_list(cls, data_list):
        """Build an ``Info`` from a header-plus-rows list.

        Raises:
            TypeError: If ``data_list`` is not a ``list``.
        """

        if not isinstance(data_list, list):
            raise TypeError('expected an instance of list')

        data_dict = Info.list_to_dict(data_list)

        return cls(data_dict)


    @classmethod
    def from_list_dict(cls, list_dict):
        """Build an ``Info`` from a list of row dictionaries.

        Raises:
            TypeError: If ``list_dict`` is not a non-empty list of dicts.
        """

        if not isinstance(list_dict, list):
            raise TypeError('expected an instance of list')

        if not isinstance(list_dict[0], dict):
            raise TypeError('expected an instance of dict')

        data_dict = Info.list_dict_to_dict(list_dict)

        return cls(data_dict)


    @staticmethod
    def dict_to_list(data_dict):
        """Convert a column-major dict into a header-plus-rows list."""

        keys = list(data_dict.keys())
        values = list(zip(*data_dict.values()))

        return [keys] + [list(item) for item in values]


    @staticmethod
    def dict_to_list_dict(data_dict):
        """Convert a column-major dict into a list of row dictionaries."""

        return [dict(zip(data_dict.keys(), values)) for values in zip(*data_dict.values())]


    @staticmethod
    def list_to_dict(data_list):
        """Convert a header-plus-rows list into a column-major dict."""

        keys = data_list[0]
        values = list(zip(*data_list[1:]))

        return {keys[i]: list(values[i]) for i in range(len(keys))}


    @staticmethod
    def list_to_list_dict(data_list):
        """Convert a header-plus-rows list into a list of row dictionaries."""

        keys = data_list[0]

        return [dict(zip(keys, item)) for item in data_list[1:]]


    @staticmethod
    def list_dict_to_dict(list_dict):
        """Convert a list of row dictionaries into a column-major dict."""

        keys = list(list_dict[0].keys())
        values = [[item[key] for item in list_dict] for key in keys]

        return dict(zip(keys, values))


    @staticmethod
    def list_dict_to_list(list_dict):
        """Convert a list of row dictionaries into a header-plus-rows list."""

        keys = list(list_dict[0].keys())

        values = [list(item.values()) for item in list_dict]

        return [keys] + values


    @property
    def text_table(self):
        """Fancy-grid text rendering of the table suitable for the CLI."""

        return tabulate(self.sanitized_data_dict,
                        headers='keys',
                        tablefmt='fancy_grid',
                        missingval='None',
                        numalign='center',
                        stralign='center',
                        disable_numparse=True)


    @property
    def html_style(self):
        """Inline CSS paired with ``html_table`` for notebook rendering."""

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
        """HTML rendering of the table tagged with the ``my-table`` class."""

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
