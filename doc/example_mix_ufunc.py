import numpy as np
from progressivis.linalg.mixufunc import MixUfuncABC
from progressivis import PTable, def_input, def_output
from typing import Any


@def_input("first", type=PTable)
@def_input("second", type=PTable)
@def_output("result", type=PTable, required=False, datashape={"first": ["_1", "_2"]})
class MixUfuncSample(MixUfuncABC):
    """
    Explanation: the result table has two columns "_1" and "_2" which are calculated
    with the underlying expressions
    NB: Here, columns in first and second table are supposed to be _1, _2, ...
    """

    expr = {"_1": (np.add, "first._2", "second._3"), "_2": (np.log, "second._3")}


@def_input("first", type=PTable)
@def_input("second", type=PTable)
@def_output("result", type=PTable, required=False)
class MixUfuncSample2(MixUfuncABC):
    """
    The output types can be coerced if necessary
    """

    expr = {
        "_1:float64": (np.add, "first._2", "second._3"),
        "_2:float64": (np.log, "second._3"),
    }


def custom_unary(x: float) -> float:
    return (x + np.sin(x)) / (x + np.cos(x))  # type: ignore


custom_unary_ufunc: Any = np.frompyfunc(custom_unary, 1, 1)


@def_input("first", type=PTable)
@def_input("second", type=PTable)
@def_output("result", type=PTable, required=False)
class MixUfuncCustomUnary(MixUfuncABC):
    """
    Module using a custom unary function
    """

    expr = {
        "_1:float64": (np.add, "first._2", "second._3"),
        "_2:float64": (custom_unary_ufunc, "second._3"),
    }


def custom_binary(x: float, y: float) -> float:
    return (x + np.sin(y)) / (x + np.cos(y))  # type: ignore


custom_binary_ufunc: Any = np.frompyfunc(custom_binary, 2, 1)


@def_input("first", type=PTable)
@def_input("second", type=PTable)
@def_output("result", type=PTable, required=False)
class MixUfuncCustomBinary(MixUfuncABC):
    """
    Module using a custom unary function
    """

    expr = {
        "_1:float64": (custom_binary_ufunc, "first._2", "second._3"),
        "_2:float64": (np.log, "second._3"),
    }
