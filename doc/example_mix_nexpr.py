from progressivis.linalg.nexpr import NumExprABC
from progressivis import PTable, def_input, def_output


@def_input("first", type=PTable)
@def_input("second", type=PTable)
@def_output("result", type=PTable, required=False, datashape={"first": ["_1", "_2"]})
class NumExprSample(NumExprABC):
    """
    NB: Here, columns in first and second table are supposed to be _1, _2, ...
    """

    expr = {"_1": "{first._2}+2*{second._3}", "_2": "{first._3}-5*{second._2}"}


@def_input("first", type=PTable)
@def_input("second", type=PTable)
@def_output("result", type=PTable, required=False)
class NumExprSample2(NumExprABC):
    """
    The output types can be coerced if necessary
    """

    expr = {
        "_1:float64": "{first._2}+2*{second._3}",
        "_2:float64": "{first._3}-5*{second._2}",
    }
