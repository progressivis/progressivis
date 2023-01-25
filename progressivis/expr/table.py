from __future__ import annotations

from .core import Expr
from ..core.psdict import PDict
from ..table.constant import Constant
from progressivis.core.module import Module
from progressivis.table.module import PTableModule, PDictModule

from typing import (
    Optional,
    Type,
    Tuple,
    Any,
    Dict,
    List,
    Union,
    cast,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from progressivis.table.table_base import BasePTable


class PDataExpr(Expr):
    def __init__(
        self,
        module_class: Type[PTableModule],
        args: Any,
        kwds: Dict[str, Any],
        module: Optional[PTableModule] = None,
        output_slot: Optional[str] = None,
        output_slot_table: str = "table",
    ):
        super().__init__(
            module_class, args, kwds, module=module, output_slot=output_slot
        )
        self._output_slot_table = output_slot_table

    @property
    def result(self) -> Union[BasePTable, PDict]:
        assert isinstance(self._module, (PTableModule, PDictModule))
        return self._module.result

    def select(self, columns: List[str]) -> PDataExpr:
        return PDataExpr(Constant, self.result.loc[:, columns], kwds={})

    # def filter(self, predicate):
    #     return PTableExpr(Filter, self.module, predicate)


class Pipeable:
    def __init__(
        self,
        expr_class: Type[Expr],
        module_class: Type[Module],
        args: Tuple[Any, ...] = (),
        kwds: Dict[str, Any] = {},
        repipe: Optional[str] = None,
        out: Optional[str] = None,
    ):
        self._expr_class = expr_class
        self._module_class = module_class
        self._args = args
        self._kwds = kwds
        self._repipe: Optional[str] = repipe
        self._repipe_out = out

    def __call__(self, *args: Any, **kwds: Any) -> Union[Expr, Pipeable]:
        _kw = dict(**kwds)
        rp_: Optional[str] = cast(str, _kw.pop("repipe", None))
        if len(args) > 0:
            ret = self._expr_class(self._module_class, args, _kw)
            if rp_ is not None:
                return ret.repipe(rp_)
            return ret
        # raise ValueError("At least one non-keyword arg is required")
        return Pipeable(self._expr_class, self._module_class, args, _kw, rp_)

    def __or__(self, other: Pipeable) -> Expr:
        expr = self._expr_class(self._module_class, self._args, self._kwds)
        return expr | other

    def tee(self, lambda1: Any, lambda2: Any) -> Any:
        lambda1(self)
        return lambda2(self)

    def repipe(self, mod_name: str, out: Any = None) -> Pipeable:
        self._repipe = mod_name
        self._repipe_out = out
        return self


class PipedInput(object):
    def __init__(self, obj: Any):
        self._obj = obj

    def __or__(self, other: Pipeable) -> Expr:
        return other._expr_class(
            other._module_class, (self._obj,) + other._args, other._kwds
        )
