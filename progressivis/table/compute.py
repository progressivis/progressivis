import datetime
import calendar

from datashape import DataShape
from dataclasses import dataclass, field
import numpy as np
from .column_selected import PColumnComputedView
from .column_expr import PColumnExpr
from .column_vfunc import PColumnVFunc

from typing import Any, Union, List, Dict, Tuple, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .table_base import BasePTable


def week_day_int(vec: Tuple[int, ...]) -> int:
    return datetime.datetime(*vec).weekday()  # type: ignore


def week_day(vec: Tuple[int, ...]) -> str:
    return calendar.day_name[week_day_int(vec)]


def ymd_string(vec: Tuple[int, ...]) -> str:
    y, m, d, *_ = vec
    return f"{y}-{m}-{d}"


def is_weekend(vec: Tuple[int, ...]) -> bool:
    return week_day_int(vec) >= 5


class _Unchanged:
    pass


def _null_op(a: Any, b: Any) -> Any:
    pass


UNCHANGED = _Unchanged()
UBool = Union[_Unchanged, bool]
Op = Callable[[Any, Any], Any]


def make_if_else(
    op_: Op, test_val: Any, if_true: UBool = UNCHANGED, if_false: UBool = UNCHANGED
) -> Callable[[Any], Any]:
    assert if_true != UNCHANGED or if_false != UNCHANGED

    def _fun(x: Any) -> Any:
        return if_true if op_(x, test_val) else if_false

    def _fun_if_true_unchanged(x: Any) -> Any:
        return x if op_(x, test_val) else if_false

    def _fun_if_false_unchanged(x: Any) -> Any:
        return if_true if op_(x, test_val) else x

    if if_true is UNCHANGED:
        return _fun_if_true_unchanged
    if if_false is UNCHANGED:
        return _fun_if_false_unchanged
    return _fun


ComputedColumn = Union[PColumnComputedView, PColumnExpr, PColumnVFunc]


# @dataclass(kw_only=True)
@dataclass
class ColFunc:
    _computed_col: Optional[ComputedColumn] = field(default=None, init=False)
    base: Union[str, List[str]]  #: column(s) to be provided as input(s)
    dtype: Optional[np.dtype[Any]] = None  #: column datatype
    #: column shape excluding the first axis (axis=0).
    #: Useful only when column elements are multidimensional
    xshape: Tuple[int, ...] = ()
    dshape: Optional[DataShape] = None  #: column datashape as specified by the `datashape` library


# @dataclass(kw_only=True)
@dataclass
class SingleColFunc(ColFunc):
    """
    This class instances supply the information for constructing a computed table
    column. This column is build over another (stored or computed) column
    using an universal function (:class:`numpy.ufunc`) a vectorized function
    :class:`numpy.vectorize` or a custom function
    compatible with :func:`numpy.apply_along_axis`.

    Example
    -------

    **Computed column using an universal function**

    >>> from progressivis.table import PTable
    >>> from progressivis.table.compute import SingleColFunc
    >>> import numpy as np
    >>> t = PTable("t", dshape="{a: int, b: float32}", create=True)
    >>> t.resize(5)
    >>> np.random.seed(42)
    >>> t["a"] = np.random.randint(100, size=5)
    >>> fvalues = np.array(np.random.rand(20), np.float32)
    >>> t["b"] = np.random.rand(5)
    >>> t
    PTable("t", dshape="{a: int32, b: float32}")[5]
       Index    |     a      |     b      |
               0|          51|  0.23277134|
               1|          92| 0.090606436|
               2|          14|  0.61838603|
               3|          71|    0.382462|
               4|          60|   0.9832309|
    >>> colfunc = SingleColFunc(func=np.arcsin, base="b")
    >>> t.computed["arcsin_b"] = colfunc
    >>> t.loc[:, :]
    BasePTable("anonymous", dshape="{a: int32, b: float32, arcsin_b: float32}")[5]
       Index    |     a      |     b      |  arcsin_b  |
               0|          51|  0.23277134|  0.23492633|
               1|          92| 0.090606436|  0.09073087|
               2|          14|  0.61838603|   0.6666873|
               3|          71|    0.382462|  0.39245942|
               4|          60|   0.9832309|    1.387405|
    >>>

    """
    #: input column (existing column that will be passed as an argument to the function)
    base: str = ""  #: column(s) to be provided as input(s)
    #: function to be applied to the elements of the input column.
    func: Op = _null_op

    def _make_computed(self, index: Any, name: str, table_: "BasePTable") -> PColumnComputedView:
        from .column_base import BasePColumn
        base: Union[BasePColumn, PColumnComputedView, PColumnExpr, PColumnVFunc]
        if self.base in table_.computed:
            x = table_.computed[self.base]._computed_col
            assert x is not None  # columns must be ordered TODO: avoid it
            base = x
        else:
            base = table_._columns[table_._columndict[self.base]]
        self._computed_col = PColumnComputedView(
            base=base,
            index=index,
            aka=name,
            func=self.func,
            dtype=self.dtype,
            xshape=self.xshape,
        )
        return self._computed_col


# @dataclass(kw_only=True)
@dataclass
class MultiColFunc(ColFunc):
    """
    This class instances supply the information for constructing a computed table
    column based on two or many other columns.

    Example
    -------

    >>> from progressivis.table import PTable
    >>> from progressivis.table.compute import MultiColFunc
    >>> import numpy as np
    >>> from typing import Any, Dict
    >>> t = PTable("t", dshape="{a: int, b: float32}", create=True)
    >>> t.resize(5)
    >>> np.random.seed(42)
    >>> t["a"] = np.random.randint(100, size=5)
    >>> fvalues = np.array(np.random.rand(20), np.float32)
    >>> t["b"] = np.random.rand(5)
    >>> t
    PTable("t", dshape="{a: int32, b: float32}")[5]
       Index    |     a      |     b      |
               0|          51|  0.23277134|
               1|          92| 0.090606436|
               2|          14|  0.61838603|
               3|          71|    0.382462|
               4|          60|   0.9832309|
    >>> def _axb(index, local_dict: Dict[str, Any]) -> Any:
    ...     return local_dict["a"] * local_dict["b"]
    ...
    >>> colfunc = MultiColFunc(func=_axb, base=["a", "b"], dtype=np.dtype("float32"))
    >>> t.computed["a_x_b"] = colfunc
    >>> t.loc[:, :]
    BasePTable("anonymous", dshape="{a: int32, b: float32, a_x_b: float32}")[5]
       Index    |     a      |     b      |   a_x_b    |
               0|          51|  0.23277134|[11.8713381.|
               1|          92| 0.090606436|[8.33579212]|
               2|          14|  0.61838603|[8.65740442]|
               3|          71|    0.382462|[27.1548016.|
               4|          60|   0.9832309|[58.9938533.|
    >>>

    """
    base: list[str]  #: columns to be provided as inputs
    #: function reference
    #: the function must have the following signature:
    #:
    #: ``def some_function(index: Any, local_dict: Dict[str, Any]) -> Any``
    #: where:
    #:
    #: * ``index`` is the index of the column
    #: * ``local_dict`` contains the input columns (the keys are the column names)
    func: Callable[[Any, Any], Dict[str, Any]] = _null_op

    def _make_computed(self, index: Any, name: str, table_: "BasePTable") -> PColumnVFunc:
        self._computed_col = PColumnVFunc(
            name=name,
            table=table_,
            index=index,
            func=self.func,
            cols=self.base,
            dtype=self.dtype,
            xshape=self.xshape,
            dshape=self.dshape,
        )
        return self._computed_col


# @dataclass(kw_only=True)
@dataclass
class MultiColExpr(ColFunc):
    """
    This class instances supply the information for constructing a computed table
    column based on two or many other columns. Instead of a ``python`` function it
    uses a numexpr expression

    Example
    -------

    >>> from progressivis.table import PTable
    >>> from progressivis.table.compute import MultiColExpr
    >>> import numpy as np
    >>> from typing import Any, Dict
    >>> t = PTable("t", dshape="{a: int, b: float32}", create=True)
    >>> t.resize(5)
    >>> np.random.seed(42)
    >>> t["a"] = np.random.randint(100, size=5)
    >>> fvalues = np.array(np.random.rand(20), np.float32)
    >>> t["b"] = np.random.rand(5)
    >>> t
    PTable("t", dshape="{a: int32, b: float32}")[5]
       Index    |     a      |     b      |
               0|          51|  0.23277134|
               1|          92| 0.090606436|
               2|          14|  0.61838603|
               3|          71|    0.382462|
               4|          60|   0.9832309|
    >>> colexpr = MultiColExpr(expr="a*b", base=["a", "b"], dtype=np.dtype("float32"))
    >>> t.computed["a_x_b"] = colexpr
    >>> t.loc[:, :]
    BasePTable("anonymous", dshape="{a: int32, b: float32, a_x_b: float32}")[5]
       Index    |     a      |     b      |   a_x_b    |
               0|          51|  0.23277134| [11.871338]|
               1|          92| 0.090606436|  [8.335793]|
               2|          14|  0.61838603|  [8.657404]|
               3|          71|    0.382462| [27.154802]|
               4|          60|   0.9832309| [58.993855]|
    >>>

    """
    base: list[str] = []  #: columns to be provided as inputs
    #: numexpr expression
    expr: str = ""

    def _make_computed(self, index: Any, name: str, table_: "BasePTable") -> PColumnExpr:
        assert self.dtype is not None
        self._computed_col = PColumnExpr(
            name=name,
            table=table_,
            index=index,
            expr=self.expr,
            cols=self.base,
            dtype=self.dtype,
            xshape=self.xshape,
            dshape=self.dshape,
        )
        return self._computed_col


Computed = Union[SingleColFunc, MultiColFunc, MultiColExpr]
