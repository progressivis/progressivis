import datetime
import calendar
from datashape import DataShape
from dataclasses import dataclass, KW_ONLY, field
import numpy as np
from .column_selected import PColumnComputedView
from .column_expr import PColumnExpr
from .column_vfunc import PColumnVFunc

from typing import Any, Union, Dict, Tuple, Callable, Optional


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


@dataclass
class ColFunc:
    _: KW_ONLY
    _computed_col: Optional[ComputedColumn] = field(default=None, init=False)
    base: str | list[str]  #: column(s) to be provided as input(s)
    dtype: np.dtype[Any] | None = None  #: column datatype
    #: column shape excluding the first axis (axis=0).
    #: Useful only when column elements are multidimensional
    xshape: Tuple[int, ...] = ()
    dshape: DataShape | None = None  #: column datashape as specified by the `datashape` library


@dataclass
class SingleColFunc(ColFunc):
    """
    This class instances supply the information for constructing a computed table
    column. This column is build over another (stored or computed) column
    using a universal function (:class:`numpy.ufunc`) or a custom function
    compatible with :func:`numpy.apply_along_axis`.
    """
    from .table_base import BasePTable
    _: KW_ONLY
    _col_class: Any = field(default=PColumnComputedView, init=False)
    #: input column (existing column that will be passed as an argument to the function)
    base: str  #: column(s) to be provided as input(s)
    func: Callable[
        [Any], Any
    ]  #: function to be applied to the elements of the input column.

    def _make_computed(self, index: Any, name: str, table_: BasePTable) -> PColumnComputedView:
        from .column_base import BasePColumn
        base: BasePColumn | PColumnComputedView | PColumnExpr | PColumnVFunc
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


@dataclass
class MultiColFunc(ColFunc):
    """
    This class instances supply the information for constructing a computed table
    column based on two or many other columns.
    """
    from .table_base import BasePTable
    _: KW_ONLY
    base: list[str]  #: columns to be provided as inputs
    #: function reference
    #: the function must have the following signature:
    #:
    #: ``def some_function(index: Any, local_dict: Dict[str, Any]) -> Any``
    #: where:
    #:
    #: * ``index`` is the index of the column
    #: * ``local_dict`` contains the input columns (the keys are the column names)
    func: Callable[[Any, Any], Dict[str, Any]]

    def _make_computed(self, index: Any, name: str, table_: BasePTable) -> PColumnVFunc:
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


@dataclass
class MultiColExpr(ColFunc):
    """
    This class instances supply the information for constructing a computed table
    column based on two or many other columns. Instead of a ``python`` function it
    uses a numexpr expression
    """
    from .table_base import BasePTable
    _: KW_ONLY
    base: list[str]  #: columns to be provided as inputs
    expr: str  #: numexpr expression

    def _make_computed(self, index: Any, name: str, table_: BasePTable) -> PColumnExpr:
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


Computed = SingleColFunc | MultiColFunc | MultiColExpr
