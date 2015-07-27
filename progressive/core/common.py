import pandas as pd

class ProgressiveError(Exception):
    pass

def empty_typed_dataframe(columns, types):
    d = {}
    cols = []
    for (name, dtype) in zip(columns, types):
        d[name] = pd.Series([], dtype=dtype)
        cols.append(name)
    return pd.DataFrame(d, columns=cols)

def typed_dataframe(columns, types, values):
    d = {}
    cols = []
    for (name, dtype, val) in zip(columns, types, values):
        d[name] = pd.Series([val], dtype=dtype)
        cols.append(name)
    return pd.DataFrame(d, columns=cols)

