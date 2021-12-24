from __future__ import annotations

import sys

import numpy as np

from ..core.utils import indices_len, fix_loc
from ..table.module import TableModule, ReturnRunStep
from ..table import Table, BaseTable
from ..core.decorators import process_slot, run_if_any
from .. import SlotDescriptor
from ..utils.psdict import PsDict
from ..core.module import ModuleMeta
from ..table.dshape import dshape_projection
from collections import OrderedDict

from typing import List, cast

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
    k: v for (k, v) in np.__dict__.items() if isinstance(v, np.ufunc) and v.nin == 1
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


def info():
    print("unary dict", unary_dict_all)
    print("*************************************************")
    print("binary dict", binary_dict_all)


class Unary(TableModule):
    inputs = [SlotDescriptor("table", type=Table, required=True)]
    outputs = [
        SlotDescriptor(
            "result", type=Table, required=False, datashape={"table": "#columns"}
        )
    ]

    def __init__(self, ufunc: np.ufunc, **kwds):
        super(Unary, self).__init__(**kwds)
        self._ufunc: np.ufunc = ufunc
        self._kwds = {}

    def reset(self) -> None:
        if self.result is not None:
            self.table.resize(0)

    def run_step(self,
                 run_number: int,
                 step_size: int,
                 howlong: float) -> ReturnRunStep:
        slot = self.get_input_slot("table")
        data_in = slot.data()
        if not data_in:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self.result is None:
            dshape_ = self.get_output_datashape("result")
            self.result = Table(
                self.generate_table_name(f"unary_{self._ufunc.__name__}"),
                dshape=dshape_,
                create=True,
            )
        cols = self.get_columns(data_in)
        if len(cols) == 0:
            # return self._return_run_step(self.state_blocked, steps_run=0)
            raise ValueError("Empty list of columns")
        steps = 0
        steps_todo = step_size
        table = self.table
        if slot.deleted.any():
            indices = slot.deleted.next(steps_todo, as_slice=False)
            del table.loc[indices]
            steps += indices_len(indices)
            steps_todo -= indices_len(indices)
            if steps_todo <= 0:
                return self._return_run_step(self.next_state(slot), steps_run=steps)
        if slot.updated.any():
            indices = slot.updated.next(steps_todo, as_slice=False)
            vec = self.filter_columns(data_in, fix_loc(indices)).raw_unary(
                self._ufunc, **self._kwds
            )
            table.loc[indices, cols] = vec
            steps += indices_len(indices)
            steps_todo -= indices_len(indices)
            if steps_todo <= 0:
                return self._return_run_step(self.next_state(slot), steps_run=steps)
        if not slot.created.any():
            return self._return_run_step(self.next_state(slot), steps_run=steps)
        indices = slot.created.next(step_size)
        steps += indices_len(indices)
        vec = self.filter_columns(data_in, fix_loc(indices)).raw_unary(
            self._ufunc, **self._kwds
        )
        table.append(vec, indices=indices)
        return self._return_run_step(self.next_state(slot), steps_run=steps)


def make_subclass(super_: ModuleMeta, cname: str, ufunc: np.ufunc) -> ModuleMeta:
    def _init_func(self_, *args, **kwds):
        super_.__init__(self_, ufunc, *args, **kwds)

    # cls = type(cname, (super_,), {})
    cls = ModuleMeta(cname, (super_,), {})
    cls.__module__ = globals()["__name__"]  # avoids cls to be part of abc module ...
    cls.__init__ = _init_func  # type: ignore
    return cls


_g = globals()


def func2class_name(s):
    return "".join([e.capitalize() for e in s.split("_")])


for k, v in unary_dict_all.items():
    name = func2class_name(k)
    # _g[name] = make_subclass(Unary, name, v)
    # unary_modules.append(_g[name])


def _simple_binary(tbl, op, cols1, cols2, cols_out, **kwargs):
    axis = kwargs.pop("axis", 0)
    assert axis == 0
    res = OrderedDict()
    for cn1, cn2, co in zip(cols1, cols2, cols_out):
        col1 = tbl[cn1]
        col2 = tbl[cn2]
        value = op(col1.value, col2.value)
        res[co] = value
    return res


class ColsBinary(TableModule):
    inputs = [SlotDescriptor("table", type=Table, required=True)]
    outputs = [SlotDescriptor("result", type=Table, required=False)]

    def __init__(self, ufunc: np.ufunc, first, second, cols_out=None, **kwds):
        super(ColsBinary, self).__init__(**kwds)
        self._ufunc = ufunc
        self._first = first
        self._second = second
        self._cols_out = cols_out
        self._kwds = {}
        if self._columns is None:
            self._columns = first + second

    def reset(self) -> None:
        if self.result is not None:
            table = self.table
            table.resize(0)

    def run_step(self,
                 run_number: int,
                 step_size: int,
                 howlong: float) -> ReturnRunStep:
        slot = self.get_input_slot("table")
        data_in = slot.data()
        if not data_in:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self._cols_out is None:
            self._cols_out = self._first
        if self.result is None:
            dshape_ = dshape_projection(data_in, self._first, self._cols_out)
            self.result = Table(
                self.generate_table_name(f"simple_binary_{self._ufunc.__name__}"),
                dshape=dshape_,
                create=True,
            )
        steps = 0
        steps_todo = step_size
        table = self.table
        if slot.deleted.any():
            indices = slot.deleted.next(steps_todo, as_slice=False)
            del table.loc[indices]
            steps += indices_len(indices)
            steps_todo -= indices_len(indices)
            if steps_todo <= 0:
                return self._return_run_step(self.next_state(slot), steps_run=steps)
        if slot.updated.any():
            indices = slot.updated.next(steps_todo, as_slice=False)
            view = data_in.loc[fix_loc(indices)]
            vec = _simple_binary(
                view,
                self._ufunc,
                self._first,
                self._second,
                self._cols_out,
                **self._kwds,
            )
            table.loc[indices, self._cols_out] = vec
            steps += indices_len(indices)
            steps_todo -= indices_len(indices)
            if steps_todo <= 0:
                return self._return_run_step(self.next_state(slot), steps_run=steps)
        if not slot.created.any():
            return self._return_run_step(self.next_state(slot), steps_run=steps)
        indices = slot.created.next(step_size)
        steps = indices_len(indices)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        view = data_in.loc[fix_loc(indices)]
        vec = _simple_binary(
            view, self._ufunc, self._first, self._second, self._cols_out, **self._kwds
        )
        table.append(vec, indices=indices)
        return self._return_run_step(self.next_state(slot), steps_run=steps)


def _binary(tbl, op, other, other_cols=None, **kwargs):
    if other_cols is None:
        other_cols = tbl.columns
    axis = kwargs.pop("axis", 0)
    assert axis == 0
    res = OrderedDict()
    isscalar = isinstance(other, dict)

    def _value(c):
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


class Binary(TableModule):
    inputs = [
        SlotDescriptor("first", type=Table, required=True),
        SlotDescriptor("second", type=(Table, PsDict), required=True),
    ]
    outputs = [
        SlotDescriptor(
            "result", type=Table, required=False, datashape={"first": "#columns"}
        )
    ]

    def __init__(self, ufunc, **kwds):
        super(Binary, self).__init__(**kwds)
        self._ufunc = ufunc
        self._kwds = {}
        _assert = self._columns is None or (
            "first" in self._columns_dict and "second" in self._columns_dict
        )
        assert _assert
        self._join = None

    def reset(self):
        if self.result is not None:
            self.table.resize(0)

    def run_step(self,
                 run_number: int,
                 step_size: int,
                 howlong: float) -> ReturnRunStep:
        first = self.get_input_slot("first")
        second = self.get_input_slot("second")
        data = first.data()
        data2 = second.data()
        if not (data and data2):
            return self._return_run_step(self.state_blocked, steps_run=0)
        _t2t = isinstance(data2, BaseTable)
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
            self.result = Table(
                self.generate_table_name(f"binary_{self._ufunc.__name__}"),
                dshape=dshape_,
                create=True,
            )
        if self._join is None:
            slots_ = (first, second) if isinstance(data2, BaseTable) else (first,)
            self._join = self.make_slot_join(*slots_)
        table = self.table
        with self._join as join:
            if join.has_deleted():
                indices = join.next_deleted(steps_todo)
                del table.loc[indices]
                steps += indices_len(indices)
                steps_todo -= indices_len(indices)
                if steps_todo <= 0:
                    return self._return_run_step(
                        self.next_state(first), steps_run=steps
                    )
            if join.has_updated():
                indices = join.next_updated(steps_todo)
                other = (
                    self.filter_columns(data2, fix_loc(indices), "second")
                    if _t2t
                    else data2
                )
                vec = _binary(
                    self.filter_columns(data, fix_loc(indices), "first"),
                    self._ufunc,
                    other,
                    self.get_columns(data2, "second"),
                    **self._kwds,
                )
                table.loc[indices, :] = vec
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
                self.filter_columns(data2, fix_loc(indices), "second")
                if _t2t
                else data2
            )
            vec = _binary(
                self.filter_columns(data, fix_loc(indices), "first"),
                self._ufunc,
                other,
                self.get_columns(data2, "second"),
                **self._kwds,
            )
            table.append(vec, indices=indices)
            return self._return_run_step(self.next_state(first), steps_run=steps)


for k, v in binary_dict_all.items():
    name = func2class_name(k)
    # _g[name] = make_subclass(Binary, name, v)
    # binary_modules.append(_g[name])


def _reduce(tbl, op, initial, **kwargs):
    res = {}
    for col in tbl._columns:
        cn = col.name
        res[cn] = op(col.values, initial=initial.get(cn), **kwargs)
    return res


class Reduce(TableModule):
    inputs = [SlotDescriptor("table", type=Table, required=True)]

    def __init__(self, ufunc: np.ufunc, columns: List[str] = None, **kwds):
        assert ufunc.nin == 2
        super(Reduce, self).__init__(**kwds)
        self._ufunc = getattr(ufunc, "reduce")
        self._columns = columns
        self._kwds = {}

    def reset(self) -> None:
        if self.result is not None:
            cast(PsDict, self.result).clear()  # is a PsDict

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(self,
                 run_number: int,
                 step_size: int,
                 howlong: float) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            data_in = ctx.table.data()
            psdict = self.result
            if psdict is None:
                psdict = PsDict()
                self.result = psdict
            else:
                psdict = cast(PsDict, self.result)
            cols = self.get_columns(data_in)
            if len(cols) == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            indices = ctx.table.created.next(step_size)
            steps = indices_len(indices)
            rdict = _reduce(
                self.filter_columns(data_in, fix_loc(indices)),
                self._ufunc,
                psdict,
                **self._kwds,
            )
            psdict.update(rdict)
            return self._return_run_step(self.next_state(ctx.table), steps_run=steps)


for k, v in binary_dict_all.items():
    name = f"{func2class_name(k)}Reduce"
    # _g[name] = make_subclass(Reduce, name, v)
    # reduce_modules.append(_g[name])


def make_unary(func, name=None):
    if not isinstance(func, np.ufunc):
        if name is None:
            name = func2class_name(func.__name__)
        func = np.frompyfunc(func, 1, 1)
    else:
        assert name is not None
    return make_subclass(Unary, name, func)


def unary_module(func):
    name = func.__name__
    if isinstance(func, np.ufunc):  # it should never happen
        raise ValueError(
            "Universal functions (numpy.ufunc) cannot "
            "be decorated. Use make_unary() instead"
        )
    else:
        func = np.frompyfunc(func, 1, 1)
    return make_subclass(Unary, name, func)


def make_binary(func, name=None):
    if not isinstance(func, np.ufunc):
        if name is None:
            name = func2class_name(func.__name__)
        func = np.frompyfunc(func, 2, 1)
    else:
        assert name is not None
    return make_subclass(Binary, name, func)


def binary_module(func):
    name = func.__name__
    if isinstance(func, np.ufunc):  # it should never happen
        raise ValueError(
            "Universal functions (numpy.ufunc) cannot "
            "be decorated. Use make_binary() instead"
        )
    else:
        func = np.frompyfunc(func, 2, 1)
    return make_subclass(Binary, name, func)


def make_reduce(func, name=None):
    if not isinstance(func, np.ufunc):
        if name is None:
            name = f"{func2class_name(func.__name__)}Reduce"
        func = np.frompyfunc(func, 2, 1)
    else:
        assert name is not None
    return make_subclass(Reduce, name, func)


def reduce_module(func):
    name = func.__name__
    if isinstance(func, np.ufunc):  # it should never happen
        raise ValueError(
            "Universal functions (numpy.ufunc) cannot "
            "be decorated. Use make_reduce() instead"
        )
    else:
        func = np.frompyfunc(func, 2, 1)
    return make_subclass(Reduce, name, func)


def generate_stubs(out=sys.stdout):
    decls: List[str] = []

    super = "Unary"
    for k, v in unary_dict_all.items():
        name = func2class_name(k)
        decls.append(name)
        print(f"""class {name}({super}):
    def __init__(self, *args, **kwds):
        super({name}, self).__init__(np.{k}, **kwds)

""", file=out)
    super = "Binary"
    for k, v in binary_dict_all.items():
        name = func2class_name(k)
        decls.append(name)
        print(f"""class {name}({super}):
    def __init__(self, *args, **kwds):
        super({name}, self).__init__(np.{k}, **kwds)

""", file=out)
    super = "ColsBinary"
    for k, v in binary_dict_all.items():
        name = f"Cols{func2class_name(k)}"
        decls.append(name)
        print(f"""class {name}({super}):
    def __init__(self, *args, **kwds):
        super({name}, self).__init__(np.{k}, **kwds)

""", file=out)
    super = "Reduce"
    for k, v in binary_dict_all.items():
        name = f"{func2class_name(k)}Reduce"
        decls.append(name)
        print(f"""class {name}({super}):
    def __init__(self, *args, **kwds):
        super({name}, self).__init__(np.{k}, **kwds)

""", file=out)
    print("from progressivis.linalg._elementwise import (", file=out)
    for decl in decls:
        print(f"    {decl},", file=out)
    print(")", file=out)
