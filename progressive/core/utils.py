import pandas as pd

def empty_typed_dataframe(columns, types):
    d = {}
    cols = []
    for (name, dtype) in zip(columns, types):
        d[name] = pd.Series([], dtype=dtype)
        cols.append(name)
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

class DataFrameAsDict(object):
    def __init__(self, df):
        super(DataFrameAsDict, self).__setattr__('df', df)
        
    def __getattr__(self, attr):
        return super(DataFrameAsDict, self).__getattribute__('df').at[0, attr]
    
    def __setattr__(self, attr, value):
        super(DataFrameAsDict, self).__getattribute__('df').at[0, attr] = value
