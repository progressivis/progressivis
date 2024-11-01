from __future__ import annotations

import sys

import numpy as np

from ..core.module import ReturnRunStep, def_input, def_output
from ..core.utils import indices_len, fix_loc
from ..core.module import Module
from ..table.api import PTable, BasePTable
from ..core.decorators import process_slot, run_if_any
from ..utils.psdict import PDict
from ..table.dshape import dshape_projection
from ..core.slot_join import SlotJoin
from collections import OrderedDict
from typeguard import check_type
from ..core.docstrings import INPUT_SEL
from typing import List, Any, Optional, Dict, Union, Callable, Sequence, Tuple

Cols = Union[List[str], Dict[str, List[str]]]

ModuleMeta = type

UFunc = Union[np.ufunc, Callable[..., Any]]

not_tested_unaries = (
    "isnat",  # input array with datetime or timedelta data type.
    "modf",  # two outputs
    "frexp",  # two outputs
    "bitwise_not",
)
other_tested_unaries = ("arccosh", "invert", "bitwise_not")
unary_except = not_tested_unaries + other_tested_unaries

not_tested_binaries = ("divmod", "matmul")  # two outputs  # ...
other_tested_binaries = (
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "gcd",
    "lcm",
    "ldexp",
    "left_shift",
    "right_shift",
)
binary_except = not_tested_binaries + other_tested_binaries


unary_dict_all = {
    v.__name__: v
    for (k, v) in np.__dict__.items()
    if isinstance(v, np.ufunc) and v.nin == 1
}

binary_dict_all = {
    k: v
    for (k, v) in np.__dict__.items()
    if isinstance(v, np.ufunc) and v.nin == 2 and k != "matmul"
}

unary_dict_gen_tst = {
    k: v for (k, v) in unary_dict_all.items() if k not in unary_except
}

binary_dict_gen_tst = {
    k: v for (k, v) in binary_dict_all.items() if k not in binary_except
}

binary_dict_int_tst = {
    k: v for (k, v) in binary_dict_all.items() if k in other_tested_binaries
}

unary_modules: List[Unary] = []
binary_modules: List[Binary] = []
reduce_modules: List[Reduce] = []


def info() -> None:
    print("unary dict", unary_dict_all)
    print("*************************************************")
    print("binary dict", binary_dict_all)


@def_input("table", type=PTable, required=True, hint_type=Sequence[str], doc=INPUT_SEL)
@def_output("result", type=PTable, required=False,
            datashape={"table": "#columns"},
            doc="The output table follows the structure of ``table``"
            )
class Unary(Module):
    def __init__(self, ufunc: UFunc, **kwds: Any) -> None:
        super().__init__(**kwds)
        self._ufunc: UFunc = ufunc
        self._kwds = {}

    def reset(self) -> None:
        if self.result is not None:
            assert isinstance(self.result, PTable)
            self.result.resize(0)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        slot = self.get_input_slot("table")
        data_in = slot.data()
        if not data_in:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self.result is None:
            dshape_ = self.get_output_datashape("result")
            self.result = PTable(
                self.generate_table_name(f"unary_{self._ufunc.__name__}"),
                dshape=dshape_,
                create=True,
            )
        cols = slot.hint or data_in.columns
        if len(cols) == 0:
            # return self._return_run_step(self.state_blocked, steps_run=0)
            raise ValueError("Empty list of columns")
        steps = 0
        steps_todo = step_size
        if slot.deleted.any():
            indices = slot.deleted.next(length=steps_todo, as_slice=False)
            del self.result.loc[indices]
            steps += indices_len(indices)
            steps_todo -= indices_len(indices)
            if steps_todo <= 0:
                return self._return_run_step(self.next_state(slot), steps_run=steps)
        if slot.updated.any():
            indices = slot.updated.next(length=steps_todo, as_slice=False)
            vec = self.filter_slot_columns(slot, fix_loc(indices)).raw_unary(
                self._ufunc, **self._kwds
            )
            self.result.loc[indices, cols] = vec
            steps += indices_len(indices)
            steps_todo -= indices_len(indices)
            if steps_todo <= 0:
                return self._return_run_step(self.next_state(slot), steps_run=steps)
        if not slot.created.any():
            return self._return_run_step(self.next_state(slot), steps_run=steps)
        indices = slot.created.next(length=step_size, as_slice=False)
        steps += indices_len(indices)
        vec = self.filter_slot_columns(slot, fix_loc(indices)).raw_unary(
            self._ufunc, **self._kwds
        )
        assert isinstance(self.result, PTable)
        self.result.append(vec, indices=indices)
        return self._return_run_step(self.next_state(slot), steps_run=steps)


def make_subclass(super_: ModuleMeta, cname: str, ufunc: UFunc) -> ModuleMeta:
    def _init_func(self_: ModuleMeta, *args: Any, **kwds: Any) -> None:
        super_.__init__(self_, ufunc, *args, **kwds)  # type: ignore

    # cls = type(cname, (super_,), {})
    cls = ModuleMeta(cname, (super_,), {})
    cls.__module__ = globals()["__name__"]  # avoids cls to be part of abc module ...
    cls.__init__ = _init_func  # type: ignore
    return cls


_g = globals()


def func2class_name(s: str) -> str:
    return "".join([e.capitalize() for e in s.split("_")])


for k, v in unary_dict_all.items():
    name: str = func2class_name(k)
    # _g[name] = make_subclass(Unary, name, v)
    # unary_modules.append(_g[name])


def _simple_binary(
    tbl: BasePTable,
    op: UFunc,
    cols1: List[str],
    cols2: List[str],
    cols_out: List[str],
    **kwargs: Any,
) -> Dict[str, Any]:
    axis = kwargs.pop("axis", 0)
    assert axis == 0
    res = OrderedDict()
    for cn1, cn2, co in zip(cols1, cols2, cols_out):
        col1 = tbl[cn1]
        col2 = tbl[cn2]
        value = op(col1.value, col2.value)
        res[co] = value
    return res


@def_input("table", type=PTable, required=True,
           hint_type=Tuple[Sequence[str], Sequence[str]],
           doc=("The two items of the hint are lists of comumns used"
                " to select the operands"))
@def_output("result", type=PTable, required=False)
class ColsBinary(Module):
    def __init__(
        self,
        ufunc: UFunc,
        cols_out: Optional[List[str]] = None,
        **kwds: Any,
    ) -> None:
        super().__init__(**kwds)
        self._ufunc = ufunc
        self._first: List[str] = []
        self._second: List[str] = []
        self._cols_out = cols_out
        self._kwds = {}

    def reset(self) -> None:
        if self.result is not None:
            assert isinstance(self.result, PTable)
            self.result.resize(0)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        slot = self.get_input_slot("table")
        data_in = slot.data()
        if not data_in:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if not self._first:
            hint = slot.hint
            self._first, self._second = check_type(hint, Tuple[Sequence[str], Sequence[str]])
            assert self._first and len(self._first) == len(self._second)
        if self._cols_out is None:
            self._cols_out = self._first
        if self.result is None:
            dshape_ = dshape_projection(data_in, self._first, self._cols_out)
            self.result = PTable(
                self.generate_table_name(f"simple_binary_{self._ufunc.__name__}"),
                dshape=dshape_,
                create=True,
            )
        steps = 0
        steps_todo = step_size
        if slot.deleted.any():
            indices = slot.deleted.next(length=steps_todo, as_slice=False)
            del self.result.loc[indices]
            steps += indices_len(indices)
            steps_todo -= indices_len(indices)
            if steps_todo <= 0:
                return self._return_run_step(self.next_state(slot), steps_run=steps)
        if slot.updated.any():
            indices = slot.updated.next(length=steps_todo, as_slice=False)
            view = data_in.loc[fix_loc(indices)]
            vec = _simple_binary(
                view,
                self._ufunc,
                self._first,
                self._second,
                self._cols_out,
                **self._kwds,
            )
            self.result.loc[indices, self._cols_out] = vec
            steps += indices_len(indices)
            steps_todo -= indices_len(indices)
            if steps_todo <= 0:
                return self._return_run_step(self.next_state(slot), steps_run=steps)
        if not slot.created.any():
            return self._return_run_step(self.next_state(slot), steps_run=steps)
        indices = slot.created.next(length=step_size, as_slice=False)
        steps = indices_len(indices)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        view = data_in.loc[fix_loc(indices)]
        vec = _simple_binary(
            view, self._ufunc, self._first, self._second, self._cols_out, **self._kwds
        )
        assert isinstance(self.result, PTable)
        self.result.append(vec, indices=indices)
        return self._return_run_step(self.next_state(slot), steps_run=steps)


def _binary(
    tbl: BasePTable,
    op: UFunc,
    other: BasePTable,
    other_cols: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    if other_cols is None:
        other_cols = tbl.columns
    axis = kwargs.pop("axis", 0)
    assert axis == 0
    res = OrderedDict()
    isscalar = isinstance(other, dict)

    def _value(c: Any) -> Any:
        if isscalar:
            return c
        return c.value

    for i, col in enumerate(tbl._columns):
        name = col.name
        name2 = other_cols[i]
        value = op(col.value, _value(other[name2]))
        res[name] = value
    return res


for k, v in binary_dict_all.items():
    name = f"Cols{func2class_name(k)}"
    # _g[name] = make_subclass(ColsBinary, name, v)
    # binary_modules.append(_g[name])


@def_input("first", type=PTable, required=True, hint_type=Sequence[str], doc=INPUT_SEL)
@def_input("second", type=PTable, required=True, hint_type=Sequence[str], doc="Similar to ``first``")
@def_output("result", PTable, required=False, datashape={"first": "#columns"}, doc="The output table follows the structure of ``first``")
class Binary(Module):
    def __init__(self, ufunc: UFunc, **kwds: Any):
        """
        Args:
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(**kwds)
        self._ufunc = ufunc
        self._kwds = {}
        self._join: Optional[SlotJoin] = None

    def reset(self) -> None:
        if self.result is not None:
            assert isinstance(self.result, PTable)
            self.result.resize(0)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        first = self.get_input_slot("first")
        second = self.get_input_slot("second")
        data = first.data()
        data2 = second.data()
        if not (data and data2):
            return self._return_run_step(self.state_blocked, steps_run=0)
        _t2t = isinstance(data2, BasePTable)
        if not _t2t and second.has_buffered():  # i.e. second is a dict
            first.reset()
            second.reset()
            self.reset()
            first.update(run_number)
            second.update(run_number)
            second.clear_buffers()
        steps = 0
        steps_todo = step_size
        if self.result is None:
            dshape_ = self.get_output_datashape("result")
            self.result = PTable(
                self.generate_table_name(f"binary_{self._ufunc.__name__}"),
                dshape=dshape_,
                create=True,
            )
        if self._join is None:
            slots_ = (first, second) if isinstance(data2, BasePTable) else (first,)
            self._join = self.make_slot_join(*slots_)
        cols_ii = second.hint or (data2.columns if _t2t else list(data2.keys()))
        with self._join as join:
            if join.has_deleted():
                indices = join.next_deleted(steps_todo)
                del self.result.loc[indices]
                steps += indices_len(indices)
                steps_todo -= indices_len(indices)
                if steps_todo <= 0:
                    return self._return_run_step(
                        self.next_state(first), steps_run=steps
                    )
            if join.has_updated():
                indices = join.next_updated(steps_todo)
                other = (
                    self.filter_slot_columns(second, fix_loc(indices))
                    if _t2t
                    else data2
                )
                vec = _binary(
                    self.filter_slot_columns(first, fix_loc(indices)),
                    self._ufunc,
                    other,
                    cols_ii,
                    **self._kwds,
                )
                self.result.loc[indices, :] = vec
                steps += indices_len(indices)
                steps_todo -= indices_len(indices)
                if steps_todo <= 0:
                    return self._return_run_step(
                        self.next_state(first), steps_run=steps
                    )
            if (not join.has_created()) or steps_todo <= 0:
                return self._return_run_step(self.next_state(first), steps_run=steps)
            indices = join.next_created(steps_todo)
            steps += indices_len(indices)
            other = (
                self.filter_slot_columns(second, fix_loc(indices))
                if _t2t
                else data2
            )
            vec = _binary(
                self.filter_slot_columns(first, fix_loc(indices)),
                self._ufunc,
                other,
                cols_ii,
                **self._kwds,
            )
            assert isinstance(self.result, PTable)
            self.result.append(vec, indices=indices)
            return self._return_run_step(self.next_state(first), steps_run=steps)


for k, v in binary_dict_all.items():
    name = func2class_name(k)
    # _g[name] = make_subclass(Binary, name, v)
    # binary_modules.append(_g[name])


def _reduce(tbl: BasePTable, op: UFunc, initial: Any, **kwargs: Any) -> Dict[str, Any]:
    res = {}
    for col in tbl._columns:
        cn = col.name
        res[cn] = op(col.values, initial=initial.get(cn), **kwargs)
    return res


@def_input("table", type=PTable, required=True, hint_type=Sequence[str], doc=INPUT_SEL)
@def_output("result", PDict, doc="The key's names follow the input table columns")
class Reduce(Module):
    def __init__(
        self, ufunc: np.ufunc, **kwds: Any
    ) -> None:
        assert ufunc.nin == 2
        super().__init__(**kwds)
        self._ufunc = getattr(ufunc, "reduce")
        self._kwds = kwds

    def reset(self) -> None:
        if self.result is not None:
            self.result.clear()

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            data_in = ctx.table.data()
            cols = ctx.table.hint or data_in.columns
            pdict = self.result
            if pdict is None:
                pdict = PDict()
                self.result = pdict
            else:
                assert self.result is not None
                pdict = self.result
            if len(cols) == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            indices = ctx.table.created.next(length=step_size)
            steps = indices_len(indices)
            rdict = _reduce(
                self.filter_slot_columns(ctx.table, fix_loc(indices)),
                self._ufunc,
                pdict,
                **({"dtype": self._kwds["dtype"]} if "dtype" in self._kwds else {}),
            )
            pdict.update(rdict)
            return self._return_run_step(self.next_state(ctx.table), steps_run=steps)


for k, v in binary_dict_all.items():
    name = f"{func2class_name(k)}Reduce"
    # _g[name] = make_subclass(Reduce, name, v)
    # reduce_modules.append(_g[name])


def make_unary(func: UFunc, name: Optional[str] = None) -> ModuleMeta:
    if not isinstance(func, np.ufunc):
        if name is None:
            name = func2class_name(func.__name__)
        func = np.frompyfunc(func, 1, 1)
    else:
        assert name is not None
    return make_subclass(Unary, name, func)


def unary_module(func: UFunc) -> ModuleMeta:
    name = func.__name__
    if isinstance(func, np.ufunc):  # it should never happen
        raise ValueError(
            "Universal functions (numpy.ufunc) cannot "
            "be decorated. Use make_unary() instead"
        )
    else:
        func = np.frompyfunc(func, 1, 1)
    return make_subclass(Unary, name, func)


def make_binary(func: UFunc, name: Optional[str] = None) -> ModuleMeta:
    if not isinstance(func, np.ufunc):
        if name is None:
            name = func2class_name(func.__name__)
        func = np.frompyfunc(func, 2, 1)
    else:
        assert name is not None
    return make_subclass(Binary, name, func)


def binary_module(func: UFunc) -> ModuleMeta:
    name = func.__name__
    if isinstance(func, np.ufunc):  # it should never happen
        raise ValueError(
            "Universal functions (numpy.ufunc) cannot "
            "be decorated. Use make_binary() instead"
        )
    else:
        func = np.frompyfunc(func, 2, 1)
    return make_subclass(Binary, name, func)


def make_reduce(func: UFunc, name: Optional[str] = None) -> ModuleMeta:
    if not isinstance(func, np.ufunc):
        if name is None:
            name = f"{func2class_name(func.__name__)}Reduce"
        func = np.frompyfunc(func, 2, 1)
    else:
        assert name is not None
    return make_subclass(Reduce, name, func)


def reduce_module(func: UFunc) -> ModuleMeta:
    name = func.__name__
    if isinstance(func, np.ufunc):  # it should never happen
        raise ValueError(
            "Universal functions (numpy.ufunc) cannot "
            "be decorated. Use make_reduce() instead"
        )
    else:
        func = np.frompyfunc(func, 2, 1)
    return make_subclass(Reduce, name, func)


def generate_stubs(out: Any = sys.stdout) -> None:
    decls: List[str] = []

    super = "Unary"
    for k, v in unary_dict_all.items():
        name = func2class_name(k)
        decls.append(name)
        print(
            f"""class {name}({super}):
    def __init__(self, *args: Any, **kwds: Any):
        super().__init__(np.{k}, **kwds)

""",
            file=out,
        )
    super = "Binary"
    for k, v in binary_dict_all.items():
        name = func2class_name(k)
        decls.append(name)
        print(
            f"""class {name}({super}):
    def __init__(self, *args: Any, **kwds: Any):
        super().__init__(np.{k}, **kwds)

""",
            file=out,
        )
    super = "ColsBinary"
    for k, v in binary_dict_all.items():
        name = f"Cols{func2class_name(k)}"
        decls.append(name)
        print(
            f"""class {name}({super}):
    def __init__(self, *args: Any, **kwds: Any):
        super().__init__(np.{k}, **kwds)

""",
            file=out,
        )
    super = "Reduce"
    for k, v in binary_dict_all.items():
        name = f"{func2class_name(k)}Reduce"
        decls.append(name)
        print(
            f"""class {name}({super}):
    def __init__(self, *args: Any, **kwds: Any):
        super().__init__(np.{k}, **kwds)

""",
            file=out,
        )
    print("from progressivis.linalg._elementwise import (", file=out)
    for decl in decls:
        print(f"    {decl},", file=out)
    print(")", file=out)


def generate_csv(
    header: str, row: str, dict_: dict[str, Any], out: Any = sys.stdout
) -> None:
    import progressivis.linalg._elementwise as _elementwise

    if isinstance(out, str):
        out = open(out, "w")
    print(header, file=out)
    for k, v in sorted(dict_.items()):
        name = func2class_name(k)
        if name not in _elementwise.__dict__:
            continue
        print(row.format(module=name, func=k), file=out)


def generate_unary_csv(out: Any = sys.stdout) -> None:
    header = "Module name, underlying :term:`Universal Function <ufunc>`"
    row = "{module}, `{func} <https://numpy.org/doc/stable/reference/generated/numpy.{func}.html>`_"
    generate_csv(header, row, unary_dict_all, out)


def generate_binary_csv(out: Any = sys.stdout) -> None:
    header = "Module name, underlying :term:`Universal Function <ufunc>`"
    row = "{module} / Cols{module} / {module}Reduce, `{func} <https://numpy.org/doc/stable/reference/generated/numpy.{func}.html>`_"
    generate_csv(header, row, binary_dict_all, out)
