from __future__ import annotations

from . import ProgressiveTest
# from progressivis.storage import init_temp_dir_if, cleanup_temp_dir
from progressivis.core import aio
from progressivis import Print
from progressivis.linalg import (
    Unary,
    Binary,
    Reduce,
    func2class_name,
    binary_dict_gen_tst,
)
import progressivis.linalg as arr
from progressivis.stats import RandomPTable
import numpy as np

from typing import Any, Type


class SubUnary(Unary):  # hack for mypy
    def __init__(*args: Any, **kw: Any) -> None:
        pass


class SubBinary(Binary):  # hack for mypy
    def __init__(*args: Any, **kw: Any) -> None:
        pass


class SubReduce(Reduce):  # hack for mypy
    def __init__(*args: Any, **kw: Any) -> None:
        pass


class TestReduce(ProgressiveTest):
    def _t_impl(self, cls: Type[SubReduce], ufunc: np.ufunc, mod_name: str) -> None:
        print("Testing", mod_name)
        dtype = (
            "float64"
            if ("ff->f" in ufunc.types or "gg->g" in ufunc.types)
            else bool
            if "ff->?" in ufunc.types
            else object
        )
        s = self.scheduler()
        random = RandomPTable(10, rows=10_000, scheduler=s)
        module = cls(scheduler=s, dtype=dtype)
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        assert module.result is not None
        assert random.result is not None
        res1 = getattr(ufunc, "reduce")(random.result.to_array(), dtype=dtype)
        res2 = np.array(list(module.result.values()))
        self.assertTrue(module.name.startswith(mod_name))
        assert np.allclose(res1, res2, equal_nan=True)
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))


def add_reduce_tst(c: Type[TestReduce], k: str, ufunc: np.ufunc) -> None:
    cls = f"{func2class_name(k)}Reduce"
    if cls not in arr.__dict__:
        print(f"Class {cls} not implemented")
        return
    mod_name = f"{k}_reduce_"
    if mod_name != "arctan2_reduce_":
        return

    def _f(self_: TestReduce) -> None:
        c._t_impl(self_, arr.__dict__[cls], ufunc, mod_name)

    setattr(c, f"test_{k}", _f)


for k, ufunc in binary_dict_gen_tst.items():
    add_reduce_tst(TestReduce, k, ufunc)
