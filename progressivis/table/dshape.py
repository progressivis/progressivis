import datashape as ds
from datashape import DataShape as DataShape
import pandas as pd
import numpy as np
from progressivis.core.utils import integer_types, gen_columns

from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    List,
    Optional,
    Sequence,
    Type,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .table_base import BaseTable


def dshape_print(dshape: Union[ds.Mono, str]) -> str:
    return ds.pprint(dshape, 1000000)


def dshape_fields(dshape: DataShape) -> Tuple[Tuple[str, ds.Mono], ...]:
    return dshape[0].fields


def dshape_table_check(dshape: DataShape) -> bool:
    return len(dshape) == 1 and isinstance(dshape[0], ds.Record)


def dshape_create(x: Union[DataShape, str, ds.Mono, Sequence[Any]]) -> DataShape:
    "Create a datashape, maybe check later to limit to known types."
    return ds.dshape(x)


def dshape_comp_to_shape(x: Any, var: Any) -> Any:
    if isinstance(x, ds.Var):
        return var
    else:
        return int(x)


def dshape_to_shape(dshape: ds.Mono, var: Optional[Any] = None) -> List[Any]:
    return [dshape_comp_to_shape(x, var) for x in dshape.shape]


OBJECT = np.dtype("O")
VSTRING = OBJECT


def dshape_to_h5py(dshape: DataShape) -> str:
    dtype = dshape.measure.to_numpy_dtype()
    if dtype == OBJECT:
        return VSTRING.str
    return dtype.str


def dshape_from_dtype(dtype: Union[np.dtype[Any], Type[Any]]) -> str:
    if dtype is str:
        return "string"
    if dtype is object:
        return "object"
    if dtype is bool:
        return "bool"
    if dtype is int:
        return "int64"
    assert isinstance(dtype, np.dtype)
    return str(ds.CType.from_numpy_dtype(dtype))


def dshape_extract(
    data: Any, columns: Optional[List[str]] = None
) -> Optional[DataShape]:
    if data is None:
        return None
    if hasattr(data, "dshape"):
        return data.dshape  # type: ignore
    if isinstance(data, np.ndarray):
        dshape = dshape_from_dtype(data.dtype)
        if columns is None:
            columns = gen_columns(len(data))
        dshapes = ["%s: %s" % (column, dshape) for column in columns]
        return ds.dshape("{" + ", ".join(dshapes) + "}")
    if isinstance(data, pd.DataFrame):
        return dshape_from_dataframe(data)
    if isinstance(data, dict):
        return dshape_from_dict(data)
    return None


def dshape_projection(
    table: BaseTable,
    columns: Optional[List[str]] = None,
    names: Optional[List[str]] = None,
) -> DataShape:
    if columns is None and names is None:
        return table.dshape
    dshapes: List[str] = []
    if names is None:
        names = columns
    assert columns is not None and names is not None
    assert len(columns) == len(names)
    for colname, newname in zip(columns, names):
        col = table._column(colname)
        if len(col.shape) > 1:
            dshapes.append("%s: %d * %s" % (newname, col.shape[1], col.dshape))
        else:
            dshapes.append("%s: %s" % (newname, col.dshape))
    return ds.dshape("{" + ",".join(dshapes) + "}")


def dshape_from_columns(table: BaseTable, columns: List[str], dshape: Any) -> DataShape:
    dshapes: List[str] = []
    for colname in columns:
        col = table._column(colname)
        if len(col.shape) > 1:
            dshapes.append("%s: %d * %s" % (col.name, col.shape[1], dshape))
        else:
            dshapes.append("%s: %s" % (col.name, dshape))
    return ds.dshape("{" + ",".join(dshapes) + "}")


def dataframe_dshape(dtype: np.dtype[Any]) -> str:
    if dtype == OBJECT:
        return "string"
    else:
        return str(dtype)


def np_dshape(v: Any, skip: int = 1) -> str:
    dshape: str
    if isinstance(v, np.ndarray):
        if v.dtype == OBJECT:
            dshape = "string"
        else:
            dshape = v.dtype.name
            shape = v.shape
            for d in shape[skip:]:
                dshape = "%d * %s" % (d, dshape)
    elif isinstance(v, list):
        e = v[0]
        if isinstance(e, str):
            dshape = "string"
        elif isinstance(e, integer_types):
            dshape = "int"
        elif isinstance(e, float):
            dshape = "float64"
        elif isinstance(e, np.ndarray):
            dshape = np_dshape(e, skip=0)  # recursive call
        else:
            raise ValueError("unknown dshape for %s" % v)
    return dshape


def dshape_from_dict(d: Dict[str, Any]) -> DataShape:
    shape = ",".join(["%s: %s" % (c, np_dshape(d[c])) for c in d])
    return ds.dshape("{" + shape + "}")


# def dshape_from_pytable(pt) -> DataShape:
#     shape = ",".join(["{}: {}".format(c, pt.coltypes[c]) for c in pt.colnames])
#     return ds.dshape("{" + shape + "}")


def dshape_from_dataframe(df: pd.DataFrame) -> DataShape:
    columns = df.columns
    if columns.dtype == np.int64:
        shape = ",".join(
            ["_%s:%s" % (df[c].name, dataframe_dshape(df[c].dtype)) for c in df]
        )
    else:
        shape = ",".join(
            ["%s:%s" % (df[c].name, dataframe_dshape(df[c].dtype)) for c in df]
        )
    return ds.dshape("{" + shape + "}")


def array_dshape(
    df: Union[np.ndarray[Any, Any], BaseTable, pd.DataFrame], array_col: str
) -> DataShape:
    if isinstance(df, np.ndarray):
        shape = dataframe_dshape(df.dtype)
        length = df.shape[1]
    else:
        col_dshapes = set([dataframe_dshape(df[c].dtype) for c in df])
        if len(col_dshapes) != 1:
            raise ValueError("All column must have the same data type")
        shape = col_dshapes.pop()
        length = len(df.columns)
    if length == 1:
        return ds.dshape(f"{{{array_col}: {shape}}}")
    else:
        return ds.dshape(f"{{{array_col}: {length} * {shape}}}")


# myds = dshape("{a: int, b: float32, c: string, d:string, e:string, f:int32, g:float32}")
# get_projection_dshape(myds, [2,4,6])
def get_projection_dshape(dshape_: DataShape, projection_ix: List[int]) -> DataShape:
    shape = ",".join(
        [
            "{arg[0]}:{arg[1]}".format(arg=dshape_[0].fields[elt])
            for elt in projection_ix
        ]
    )
    return ds.dshape("{" + shape + "}")


# get_projection_dshape_with_keys(myds, ['c','e','g'])
def get_projection_dshape_with_keys(
    dshape_: DataShape, projection_keys: List[str]
) -> DataShape:
    dict_ = {k: ix for ix, (k, _) in enumerate(dshape_[0].fields)}
    return get_projection_dshape(dshape_, [dict_[key] for key in projection_keys])


def dshape_compatible(ds1: Optional[DataShape], ds2: DataShape) -> bool:
    if ds1 is None:
        return False
    assert isinstance(ds1, DataShape) and isinstance(ds2, DataShape)
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


def dshape_join(
    left: DataShape, right: DataShape, lsuffix: str = "", rsuffix: str = ""
) -> Tuple[DataShape, Dict[str, Dict[str, str]]]:
    res = []
    rename: Dict[str, Dict[str, str]] = {"left": {}, "right": {}}
    suffix = {"left": lsuffix, "right": rsuffix}
    left_cols = left[0].fields
    keys, _ = zip(*left_cols)
    left_keys = set(keys)
    right_cols = right[0].fields
    keys, _ = zip(*right_cols)
    right_keys = set(keys)
    inter_keys = left_keys.intersection(right_keys)
    if inter_keys and not lsuffix and not rsuffix:
        raise ValueError("columns overlap in join without left/right suffixes")
    len_left = len(left_keys)
    all_cols = left_cols + right_cols
    for i, (cname, ctype) in enumerate(all_cols):
        side = "left" if i < len_left else "right"
        if cname in inter_keys and suffix[side]:
            alias = cname + suffix[side]
            rename[side][cname] = alias
        else:
            alias = cname
        res.append((alias, ctype))
    ret = "{" + ",".join(["{}: {}".format(f, t) for f, t in res]) + "}"
    return ds.dshape(ret), rename


def dshape_union(left: DataShape, right: DataShape) -> DataShape:
    res = []
    left_dict = dict(left[0].fields)
    left_keys = set(left_dict.keys())
    right_dict = dict(right[0].fields)
    right_keys = set(right_dict.keys())
    union_keys = sorted(left_keys.union(right_keys))
    for key in union_keys:
        # ctype = left_dict.get(key, right_dict[key]) # nice bug!
        ctype = left_dict[key] if key in left_dict else right_dict[key]
        res.append((key, ctype))
    return ds.dshape("{" + ",".join(["{}: {}".format(f, t) for f, t in res]) + "}")


def dshape_all_dtype(columns: List[str], dtype: np.dtype[Any]) -> DataShape:
    dshape = dshape_from_dtype(dtype)
    dshapes = ["%s: %s" % (column, dshape) for column in columns]
    return ds.dshape("{" + ", ".join(dshapes) + "}")


EMPTY_DSHAPE = ds.dshape("{}")
