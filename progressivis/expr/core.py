from __future__ import annotations

import logging
from progressivis.table.module import TableModule, Module

from typing import (
    Type,
    Tuple,
    Any,
    Dict,
    Optional,
    List,
    TYPE_CHECKING,
    Union,
    Iterable,
)

if TYPE_CHECKING:
    from progressivis.core.scheduler import Scheduler
    from .table import Pipeable

logger = logging.getLogger(__name__)


class ValidationError(RuntimeError):
    pass


def filter_underscore(lst: Iterable[str]):
    return [elt for elt in lst if not elt.startswith("_")]


class Expr:
    def __init__(
        self,
        module_class: Type[Module],
        args: Tuple[Any, ...],
        kwds: Dict[str, Any],
        output_slot: str = None,
        module: Module = None,
    ):
        self._module_class = module_class
        lazy = kwds.pop("lazy", False)
        self._args = args
        self._kwds = kwds
        self._module = module
        self._output_slot = output_slot
        self._valid: Optional[bool] = (module is not None) or None
        self._expr_args: Tuple[Expr, ...] = ()
        self._non_expr_args: Tuple[Any, ...] = ()
        self._expr_kwds: Dict[str, Expr] = {}
        self._non_expr_kwds: Dict[str, Any] = {}
        self._repipe: Optional[str] = None
        if not lazy:
            self.validate()

    @property
    def module(self) -> Optional[Module]:
        return self._module

    @property
    def output_slot(self) -> Optional[str]:
        return self._output_slot

    def get_data(self, name: str) -> Any:
        if self.module is None:
            return None
        return self.module.get_data(name)

    def __getitem__(self, output_slot: str) -> Expr:
        assert self._module is not None
        self._module.get_output_slot(
            output_slot
        )  # raise an error if output_slot does not exist
        return Expr(
            self._module_class,
            self._non_expr_args,
            dict(lazy=True, **self._non_expr_kwds),
            output_slot=output_slot,
            module=self._module,
        )

    def tee(self, lambda1, lambda2):
        lambda1(self)
        return lambda2(self)

    def _validate_args(self) -> None:
        modules: List[Expr] = []
        non_modules: List[Any] = []
        for a in self._args:
            if isinstance(a, Expr):
                a.validate()
                modules.append(a)
            else:
                non_modules.append(a)
        self._expr_args = tuple(modules)
        self._non_expr_args = tuple(non_modules)

    def _validate_kwds(self) -> None:
        modules: Dict[str, Expr] = {}
        non_modules: Dict[str, Any] = {}
        for (k, a) in self._kwds.items():
            if isinstance(a, Expr):
                a.validate()
                modules[k] = a
            else:
                non_modules[k] = a
        self._expr_kwds = modules
        self._non_expr_kwds = non_modules

    def _connect(self, module: Module, expr: Expr, input_slot: str = None):
        input_module = expr.module
        assert input_module
        output_slot = expr.output_slot
        if output_slot is None:
            output_slots = filter_underscore(input_module.output_slot_names())
            if len(output_slots) == 0:
                raise ValueError(
                    "Cannot extract output slot from module %s", input_module
                )
            output_slot = output_slots[0]  # take the first one
        if input_slot is None:
            input_slots = filter_underscore(module.input_slot_names())
            for inp in input_slots:
                if not module.has_input_slot(inp):  # no input slot connected yet
                    input_slot = inp
                    break
            if input_slot is None:
                raise ValueError("Cannot extract input slot from module %s", module)
        input_module.connect_output(output_slot, module, input_slot)

    def _instanciate_module(self):
        module = self._module_class(*self._non_expr_args, **self._non_expr_kwds)
        for expr in self._expr_args:
            self._connect(module, expr, None)

        for (input_slot, expr) in self._expr_kwds.items():
            self._connect(module, expr, input_slot)

        self._module = module

    def validate(self) -> Module:
        if self._valid is None:
            try:
                self._validate_args()
                self._validate_kwds()
                self._instanciate_module()
                self._valid = True
            except Exception:
                self._valid = False
                raise
        if self._valid is False:
            raise ValidationError("Module not valid")
        assert self._module is not None
        return self._module

    def invalidate(self) -> None:
        self._valid = None

    def scheduler(self) -> Scheduler:
        assert self._module
        return self._module.scheduler()

    def repipe(self, mod_name: str, out: str = None) -> Expr:
        mod_ = self.scheduler()[mod_name]
        if isinstance(mod_, TableModule):
            from .table import TableExpr

            return TableExpr(
                type(mod_), (), dict(lazy=True), module=mod_, output_slot=out
            )
        return Expr(type(mod_), (), dict(lazy=True), module=mod_, output_slot=out)

    def fetch(
        self, mod_name: str, out: str = None
    ) -> Expr:  # a simple alias for repipe
        return self.repipe(mod_name, out)

    def __or__(self, other: Union[None, Expr, Pipeable]) -> Expr:
        if other is None:
            return self
        if isinstance(other, Expr):
            return other
        # assert isinstance(other, Pipeable)
        ret: Expr = other._expr_class(
            other._module_class, (self,) + other._args, other._kwds
        )
        if other._repipe is not None:
            return ret.repipe(other._repipe, out=other._repipe_out)
        return ret
