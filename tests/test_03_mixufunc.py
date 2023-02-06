from __future__ import annotations

import numpy as np

from . import ProgressiveTest

from progressivis.core import aio
from progressivis import Print
from progressivis.linalg.mixufunc import MixUfuncABC
from progressivis.stats import RandomPTable, RandomDict
from progressivis.table.table import PTable
from progressivis.core.module import def_input, def_output
from typing import Any, Type


@def_input("first", type=PTable)
@def_input("second", type=PTable)
@def_output("result", type=PTable, required=False, datashape={"first": ["_1", "_2"]})
class MixUfuncSample(MixUfuncABC):
    """ """

    expr = {"_1": (np.add, "first._2", "second._3"), "_2": (np.log, "second._3")}


@def_input("first", type=PTable)
@def_input("second", type=PTable)
@def_output("result", type=PTable, required=False)
class MixUfuncSample2(MixUfuncABC):
    """ """

    expr = {
        "_1:float64": (np.add, "first._2", "second._3"),
        "_2:float64": (np.log, "second._3"),
    }


# https://stackoverflow.com/questions/6768245/difference-between-frompyfunc-and-vectorize-in-numpy
def custom_unary(x: float) -> float:
    return (x + np.sin(x)) / (x + np.cos(x))  # type: ignore


custom_unary_ufunc: Any = np.frompyfunc(custom_unary, 1, 1)  # type: ignore


@def_input("first", type=PTable)
@def_input("second", type=PTable)
@def_output("result", type=PTable, required=False)
class MixUfuncCustomUnary(MixUfuncABC):
    """ """

    expr = {
        "_1:float64": (np.add, "first._2", "second._3"),
        "_2:float64": (custom_unary_ufunc, "second._3"),
    }


def custom_binary(x: float, y: float) -> float:
    return (x + np.sin(y)) / (x + np.cos(y))  # type: ignore


custom_binary_ufunc: Any = np.frompyfunc(custom_binary, 2, 1)  # type: ignore


@def_input("first", type=PTable)
@def_input("second", type=PTable)
@def_output("result", type=PTable, required=False)
class MixUfuncCustomBinary(MixUfuncABC):
    """ """

    expr = {
        "_1:float64": (custom_binary_ufunc, "first._2", "second._3"),
        "_2:float64": (np.log, "second._3"),
    }


class TestMixUfunc(ProgressiveTest):
    def t_mix_ufunc_impl(
        self,
        cls: Type[MixUfuncABC],
        ufunc1: np.ufunc = np.log,
        ufunc2: np.ufunc = np.add,
    ) -> None:
        s = self.scheduler()
        random1 = RandomPTable(10, rows=100000, scheduler=s)
        random2 = RandomPTable(10, rows=100000, scheduler=s)
        module = cls(
            columns={"first": ["_1", "_2", "_3"], "second": ["_1", "_2", "_3"]},
            scheduler=s,
        )

        module.input.first = random1.output.result
        module.input.second = random2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        first = random1.result.to_array()
        first_2 = first[:, 1]
        _ = first[:, 2]
        second = random2.result.to_array()
        _ = second[:, 1]
        second_3 = second[:, 2]
        ne_1 = ufunc2(first_2, second_3).astype("float64")
        ne_2 = ufunc1(second_3).astype("float64")
        res = module.result.to_array()
        self.assertTrue(np.allclose(res[:, 0], ne_1, equal_nan=True))
        self.assertTrue(np.allclose(res[:, 1], ne_2, equal_nan=True))

    def t_mix_ufunc_table_dict_impl(self, cls: Type[MixUfuncABC]) -> None:
        s = self.scheduler()
        random1 = RandomDict(10, scheduler=s)
        random2 = RandomPTable(10, rows=100000, scheduler=s)
        module = cls(
            columns={"first": ["_1", "_2", "_3"], "second": ["_1", "_2", "_3"]},
            scheduler=s,
        )

        module.input.first = random1.output.result
        module.input.second = random2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        first = list(random1.result.values())
        first_2 = first[1]
        _ = first[2]
        second = random2.result.to_array()
        _ = second[:, 1]
        second_3 = second[:, 2]
        ne_1 = np.add(first_2, second_3)
        ne_2 = np.log(second_3)
        res = module.result.to_array()
        self.assertTrue(np.allclose(res[:, 0], ne_1, equal_nan=True))
        self.assertTrue(np.allclose(res[:, 1], ne_2, equal_nan=True))

    def test_mix_ufunc(self) -> None:
        return self.t_mix_ufunc_impl(MixUfuncSample)

    def test_mix_ufunc2(self) -> None:
        return self.t_mix_ufunc_impl(MixUfuncSample2)

    def test_mix_custom1(self) -> None:
        return self.t_mix_ufunc_impl(MixUfuncCustomUnary, ufunc1=custom_unary_ufunc)

    def test_mix_custom2(self) -> None:
        return self.t_mix_ufunc_impl(MixUfuncCustomBinary, ufunc2=custom_binary_ufunc)

    def test_mix_ufunc3(self) -> None:
        return self.t_mix_ufunc_table_dict_impl(MixUfuncSample2)
