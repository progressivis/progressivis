import datashape as ds
import pandas as pd
import numpy as np
from progressivis.core.utils import integer_types, gen_columns

def dshape_print(dshape):
    return ds.pprint(dshape, 1000000)

def dshape_fields(dshape):
    return dshape[0].fields    

def dshape_table_check(dshape):
    return len(dshape)==1 and isinstance(dshape[0], ds.Record)

def dshape_create(x):
    "Create a datashape, maybe check later to limit to known types."
    return ds.dshape(x)

def dshape_comp_to_shape(x, var):
    if isinstance(x, ds.Var):
        return var
    else:
        return int(x)

def dshape_to_shape(dshape, var=None):
    return [ dshape_comp_to_shape(x, var) for x in dshape.shape ]

OBJECT = np.dtype('O')
VSTRING = OBJECT

def dshape_to_h5py(dshape):
    dtype = dshape.measure.to_numpy_dtype()
    if dtype == OBJECT:
        return VSTRING
    return dtype.str

def dshape_from_dtype(dtype):
    if dtype is str:
        return "string"
    if dtype is object:
        return "object"
    if dtype is bool:
        return "bool"
    if dtype is int:
        return "int64"
    return ds.CType.from_numpy_dtype(dtype)

def dshape_extract(data, columns=None):
    if data is None:
        return None
    if hasattr(data, 'dshape'):
        return data.dshape
    if isinstance(data, np.ndarray):
        dshape = dshape_from_dtype(data.dtype)
        if columns is None:
            columns = gen_columns(len(data))
        dshapes = [ "%s: %s"%(column, dshape) for column in columns ]
        return "{" + ", ".join(dshapes)+"}"
    if isinstance(data, pd.DataFrame):
        return dshape_from_dataframe(data)
    if isinstance(data, dict):
        return dshape_from_dict(data)
    return None

def dataframe_dshape(dtype):
    if dtype == OBJECT:
        return "string"
    else:
        return dtype

def np_dshape(v, skip=1):
    dshape = None
    if isinstance(v, np.ndarray):
        if v.dtype == OBJECT:
            dshape = "string"
        else:
            dshape = v.dtype.name
            shape = v.shape
            for d in shape[skip:]:
                dshape = "%d * %s"%(d,dshape)
    elif isinstance(v, list):
        e = v[0]
        if isinstance(e, str):
            dshape = "string"
        elif isinstance(e, integer_types):
            dshape = "int"
        elif isinstance(e, float):
            dshape = "float64"
        elif isinstance(e, np.ndarray):
            dshape = np_dshape(e, skip=0) # recursive call
        else:
            raise ValueError('unknown dshape for %s'%v)
    return dshape


def dshape_from_dict(d):
    shape = ",".join(["%s: %s"%(c, np_dshape(d[c])) for c in d])
    return ds.dshape("{"+shape+"}")

def dshape_from_pytable(pt):
    shape = ",".join(["{}: {}".format(c, pt.coltypes[c]) for c in pt.colnames])
    return ds.dshape("{"+shape+"}")


def dshape_from_dataframe(df):
    columns=df.columns
    if columns.dtype==np.int64:
        shape = ",".join(["_%s:%s"%(df[c].name, dataframe_dshape(df[c].dtype)) for c in df])
    else:
        shape = ",".join(["%s:%s"%(df[c].name, dataframe_dshape(df[c].dtype)) for c in df])
    return ds.dshape("{"+shape+"}")

#myds = dshape("{a: int, b: float32, c: string, d:string, e:string, f:int32, g:float32}")
#get_projection_dshape(myds, [2,4,6])
def get_projection_dshape(dshape_, projection_ix):
    shape = ",".join(["{arg[0]}:{arg[1]}".format(arg=dshape_[0].fields[elt]) for elt in  projection_ix])
    return ds.dshape("{"+shape+"}")

#get_projection_dshape_with_keys(myds, ['c','e','g'])
def get_projection_dshape_with_keys(dshape_, projection_keys):
    dict_ = {k:ix for ix, (k,_) in enumerate(dshape_[0].fields)}
    return get_projection_dshape(dshape_,[dict_[key] for key in projection_keys])

    
def dshape_compatible(ds1, ds2):
    #TODO fixme
    assert isinstance(ds1, ds.DataShape) and  isinstance(ds2, ds.DataShape)
    return True

#
# left = ds.dshape("{a: int, b: float32, c: string, d:string, e:string, f:int32, g:float32}")
# right = ds.dshape("{x: int, y: float32, z: string}")
# dshape_join(left, right)
# right2 = ds.dshape("{x: int, y: float32, c:int32,  z: string}")
# dshape_join(left, right2, lsuffix='_l', rsuffix='_r')
# left2 = ds.dshape("{a: int, b: float32, c: string, d:string, y:float32,  e:string, f:int32, g:float32}")
# dshape_join(left2, right, lsuffix='_l')
# dshape_join(left2, right2)
def dshape_join(left, right, lsuffix='', rsuffix=''):
    res = []
    rename = {'left': {}, 'right': {}}
    suffix = {'left': lsuffix, 'right': rsuffix}    
    left_cols = left[0].fields
    left_keys, _ = zip(*left_cols)
    left_keys = set(left_keys)
    right_cols = right[0].fields
    right_keys, _ = zip(*right_cols)
    right_keys = set(right_keys)
    inter_keys = left_keys.intersection(right_keys)
    if inter_keys and not lsuffix and not rsuffix:
        raise ValueError("columns overlapped in join without left/right suffixes")
    len_left = len(left_keys)
    all_cols = left_cols + right_cols
    for i, (cname, ctype) in enumerate(all_cols):
        side = 'left' if i < len_left else 'right'
        if cname in inter_keys and suffix[side]:
            alias = cname + suffix[side]
            rename[side][cname] = alias
        else:
            alias = cname
        res.append((alias, ctype))
    res = '{'+",".join(["{}: {}".format(f, t) for f, t in res])+'}'    
    return ds.dshape(res), rename

def dshape_union(left, right):
    res = []
    left_dict = dict(left[0].fields)
    left_keys = set(left_dict.keys())
    right_dict = dict(right[0].fields)
    right_keys = set(right_dict.keys())
    union_keys = sorted(left_keys.union(right_keys))
    for key in union_keys:
        #ctype = left_dict.get(key, right_dict[key]) # nice bug!
        ctype = left_dict[key] if key in left_dict else right_dict[key]
        res.append((key, ctype))
    return '{'+",".join(["{}: {}".format(f, t) for f, t in res])+'}'
