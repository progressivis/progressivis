import pandas as pd
import numpy as np

def remove_nan(d):
    if isinstance(d,float) and np.isnan(d):
        return None
    if isinstance(d, list):
        for i, v in enumerate(d):
            if isinstance(v,float) and np.isnan(v):
                d[i] = None
            else:
                remove_nan(v)
    elif isinstance(d, dict):
        for k, v in d.iteritems():
            if isinstance(v,float) and np.isnan(v):
                d[k] = None
            else:
                remove_nan(v)
    return d

def empty_typed_dataframe(columns, types=None):
    d = {}
    cols = []
    if not isinstance(types,list):
        itr = columns
    else:
        itr = zip(columns, types)
    for vals in itr: # cannot unpack since columns can have 3 values
        d[vals[0]] = pd.Series([], dtype=vals[1])
        cols.append(vals[0])
    return pd.DataFrame(d, columns=cols)

def typed_dataframe(columns, types=None, values=None):
    d = {}
    cols = []
    if types is None:
        itr = columns
    else:
        itr = zip(columns, types, values)
    for (name, dtype, val) in itr:
        d[name] = pd.Series([val], dtype=dtype)
        cols.append(name)
    return pd.DataFrame(d, columns=cols)

class AttributeDict(object):
    def __init__(self, d):
        self.d = d
        
    def __getattr__(self, attr):
        return self.__dict__['d'][attr]

    def __getitem__(self, key):
        return self.__getattribute__('d')[key]

    def __dir__(self):
        return self.__getattribute__('d').keys()

class DataFrameAsDict(object):
    def __init__(self, df):
        super(DataFrameAsDict, self).__setattr__('df', df)
        
    def __getitem__(self, key):
        df = super(DataFrameAsDict, self).__getattribute__('df')
        return df.at[df.index[-1], key]

    def __setitem__(self, key, value):
        df = super(DataFrameAsDict, self).__getattribute__('df')
        df.at[df.index[-1], key] = value

    def __getattr__(self, attr):
        return self[attr]
    
    def __setattr__(self, attr, value):
        self[attr] = value

    def __dir__(self):
        return list(self.__dict__['df'].columns)
