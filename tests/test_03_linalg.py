from __future__ import annotations

from . import ProgressiveTest
from progressivis.storage import init_temp_dir_if, cleanup_temp_dir
from progressivis.core import aio
from progressivis import Print
from progressivis.table.stirrer import Stirrer
from progressivis.linalg import (
    Unary,
    Binary,
    ColsBinary,
    Reduce,
    func2class_name,
    unary_module,
    make_unary,
    binary_module,
    make_binary,
    reduce_module,
    make_reduce,
    binary_dict_int_tst,
    unary_dict_gen_tst,
    binary_dict_gen_tst,
)
from progressivis.linalg._elementwise import (
    Invert,
    BitwiseNot,
    ColsLdexp,
    Ldexp,
    Arccosh,
)
import progressivis.linalg as arr
from progressivis.core.pintset import PIntSet
from progressivis.stats import RandomPTable, RandomDict
import numpy as np

from typing import Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis.core.module import Module


class TestUnary(ProgressiveTest):
    def test_unary(self) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=100_000, scheduler=s)
        module = Unary(np.log, scheduler=s)
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.log(random.result.to_array())
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith("unary_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_unary2(self) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=100_000, scheduler=s)
        module = Unary(np.log, columns=["_3", "_5", "_7"], scheduler=s)
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.log(random.result.to_array()[:, [2, 4, 6]])
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith("unary_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def _t_stirred_unary(self, **kw: Any) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=100_000, scheduler=s)
        stirrer = Stirrer(update_column="_3", fixed_step_size=1000, scheduler=s, **kw)
        stirrer.input[0] = random.output.result
        module = Unary(np.log, columns=["_3", "_5", "_7"], scheduler=s)
        module.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.log(stirrer.result.to_array()[:, [2, 4, 6]])
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith("unary_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_unary3(self) -> None:
        self._t_stirred_unary(delete_rows=5)

    def test_unary4(self) -> None:
        self._t_stirred_unary(update_rows=5)

    def _t_impl(self, cls: Type[Module], ufunc: np.ufunc, mod_name: str) -> None:
        print("Testing", mod_name)
        s = self.scheduler()
        random = RandomPTable(10, rows=10_000, scheduler=s)
        module = cls(scheduler=s)
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = ufunc(random.result.to_array())
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith(mod_name))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))


def add_un_tst(k: str, ufunc: np.ufunc) -> None:
    cls = func2class_name(k)
    if cls not in arr.__dict__:
        print(f"Class {cls} not implemented")
        return
    mod_name = k + "_"

    def _f(self_: TestUnary) -> None:
        init_temp_dir_if()
        TestUnary._t_impl(self_, arr.__dict__[cls], ufunc, mod_name)
        cleanup_temp_dir()

    setattr(TestUnary, "test_" + k, _f)


for k, ufunc in unary_dict_gen_tst.items():
    add_un_tst(k, ufunc)


class TestOtherUnaries(ProgressiveTest):
    def test_arccosh(self) -> None:
        module_name = "arccosh_"
        print("Testing", module_name)
        s = self.scheduler()
        random = RandomPTable(
            10, random=lambda x: np.random.rand(x) * 10000.0, rows=100_000, scheduler=s
        )
        module = Arccosh(scheduler=s)
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.arccosh(random.result.to_array())
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith(module_name))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_invert(self) -> None:
        module_name = "invert_"
        print("Testing", module_name)
        s = self.scheduler()
        random = RandomPTable(
            10,
            random=lambda x: np.random.randint(100_000, size=x),  # type: ignore
            dtype="int64",
            rows=100_000,
            scheduler=s,
        )
        module = Invert(scheduler=s)
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.invert(random.result.to_array())
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith(module_name))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_bitwise_not(self) -> None:
        module_name = "bitwise_not_"
        print("Testing", module_name)
        s = self.scheduler()
        random = RandomPTable(
            10,
            random=lambda x: np.random.randint(100_000, size=x),  # type: ignore
            dtype="int64",
            rows=100_000,
            scheduler=s,
        )
        module = BitwiseNot(scheduler=s)
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.bitwise_not(random.result.to_array())
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith(module_name))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))


class TestColsBinary(ProgressiveTest):
    def test_cols_binary(self) -> None:
        s = self.scheduler()
        cols = 10
        random = RandomPTable(cols, rows=100_000, scheduler=s)
        module = ColsBinary(
            np.add, first=["_3", "_5", "_7"], second=["_4", "_6", "_8"], scheduler=s
        )
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        self.assertListEqual(module.result.columns, ["_3", "_5", "_7"])
        arr = random.result.to_array()
        res1 = np.add(arr[:, [2, 4, 6]], arr[:, [3, 5, 7]])
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith("cols_binary_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_cols_binary2(self) -> None:
        s = self.scheduler()
        cols = 10
        random = RandomPTable(cols, rows=100, scheduler=s)
        module = ColsBinary(
            np.add,
            first=["_3", "_5", "_7"],
            second=["_4", "_6", "_8"],
            cols_out=["x", "y", "z"],
            scheduler=s,
        )
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        self.assertListEqual(module.result.columns, ["x", "y", "z"])

    def t_stirred_cols_binary(self, **kw: Any) -> None:
        s = self.scheduler()
        cols = 10
        random = RandomPTable(cols, rows=10_000, scheduler=s)
        stirrer = Stirrer(update_column="_3", fixed_step_size=1000, scheduler=s, **kw)
        stirrer.input[0] = random.output.result
        module = ColsBinary(
            np.add, first=["_3", "_5", "_7"], second=["_4", "_6", "_8"], scheduler=s
        )
        module.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        self.assertListEqual(module.result.columns, ["_3", "_5", "_7"])
        arr = stirrer.result.to_array()
        res1 = np.add(arr[:, [2, 4, 6]], arr[:, [3, 5, 7]])
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith("cols_binary_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_cols_binary3(self) -> None:
        self.t_stirred_cols_binary(delete_rows=5)

    def test_cols_binary4(self) -> None:
        self.t_stirred_cols_binary(update_rows=5)

    def _t_impl(self, cls: Type[Module], ufunc: np.ufunc, mod_name: str) -> None:
        print("Testing", mod_name)
        s = self.scheduler()
        random = RandomPTable(10, rows=10_000, scheduler=s)
        module = cls(
            first=["_3", "_5", "_7"],
            second=["_4", "_6", "_8"],
            cols_out=["x", "y", "z"],
            scheduler=s,
        )
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        self.assertListEqual(module.result.columns, ["x", "y", "z"])
        arr = random.result.to_array()
        res1 = ufunc(arr[:, [2, 4, 6]], arr[:, [3, 5, 7]])
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith(mod_name))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))


def add_cols_bin_tst(c: Type[TestColsBinary], k: str, ufunc: np.ufunc) -> None:
    cls = f"Cols{func2class_name(k)}"
    if cls not in arr.__dict__:
        print(f"Class {cls} not implemented")
        return
    mod_name = f"cols_{k}_"

    def _f(self_: TestColsBinary) -> None:
        c._t_impl(self_, arr.__dict__[cls], ufunc, mod_name)

    setattr(c, "test_" + k, _f)


for k, ufunc in binary_dict_gen_tst.items():
    add_cols_bin_tst(TestColsBinary, k, ufunc)


class TestOtherColsBinaries(ProgressiveTest):
    def _t_impl(self, cls: Type[Module], ufunc: np.ufunc, mod_name: str) -> None:
        print("Testing", mod_name)
        s = self.scheduler()
        cols = 10
        random = RandomPTable(
            cols,
            rows=10_000,
            scheduler=s,
            random=lambda x: np.random.randint(10, size=x),  # type: ignore
            dtype="int64",
        )
        module = cls(
            first=["_3", "_5", "_7"],
            second=["_4", "_6", "_8"],
            cols_out=["x", "y", "z"],
            scheduler=s,
        )
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        self.assertListEqual(module.result.columns, ["x", "y", "z"])
        arr = random.result.to_array()
        res1 = ufunc(arr[:, [2, 4, 6]], arr[:, [3, 5, 7]])
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith(mod_name))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_ldexp(self) -> None:
        cls, ufunc, mod_name = ColsLdexp, np.ldexp, "cols_ldexp_"
        print("Testing", mod_name)
        s = self.scheduler()
        cols = 10
        random = RandomPTable(
            cols,
            rows=10_000,
            scheduler=s,
            random=lambda x: np.random.randint(10, size=x),  # type: ignore
            dtype="int64",
        )
        module = cls(
            first=["_3", "_5", "_7"],
            second=["_4", "_6", "_8"],
            cols_out=["x", "y", "z"],
            scheduler=s,
        )
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        self.assertListEqual(module.result.columns, ["x", "y", "z"])
        arr = random.result.to_array()
        res1 = ufunc(arr[:, [2, 4, 6]], arr[:, [3, 5, 7]])
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith(mod_name))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))


def add_other_cols_bin_tst(
    c: Type[TestOtherColsBinaries], k: str, ufunc: np.ufunc
) -> None:
    cls = f"Cols{func2class_name(k)}"
    if cls not in arr.__dict__:
        print(f"Class {cls} not implemented")
        return
    mod_name = f"cols_{k}_"

    def _f(self_: TestOtherColsBinaries) -> None:
        c._t_impl(self_, arr.__dict__[cls], ufunc, mod_name)

    setattr(c, f"test_cols_{k}", _f)


for k, ufunc in binary_dict_int_tst.items():
    if k == "ldexp":
        continue
    add_other_cols_bin_tst(TestOtherColsBinaries, k, ufunc)


class TestBin(ProgressiveTest):
    def _t_impl(self, cls: Type[Module], ufunc: np.ufunc, mod_name: str) -> None:
        pass


class TestBinary(TestBin):
    def test_binary(self) -> None:
        s = self.scheduler()
        random1 = RandomPTable(3, rows=100_000, scheduler=s)
        random2 = RandomPTable(3, rows=100_000, scheduler=s)
        module = Binary(np.add, scheduler=s)
        module.input.first = random1.output.result
        module.input.second = random2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.add(random1.result.to_array(), random2.result.to_array())
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith("binary_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_binary2(self) -> None:
        s = self.scheduler()
        cols = 10
        _ = RandomPTable(cols, rows=100_000, scheduler=s)
        _ = RandomPTable(cols, rows=100_000, scheduler=s)
        with self.assertRaises(AssertionError):
            _ = Binary(np.add, columns=["_3", "_5", "_7"], scheduler=s)

    def test_binary3(self) -> None:
        s = self.scheduler()
        random1 = RandomPTable(10, rows=100_000, scheduler=s)
        random2 = RandomPTable(10, rows=100_000, scheduler=s)
        module = Binary(
            np.add,
            columns={"first": ["_3", "_5", "_7"], "second": ["_4", "_6", "_8"]},
            scheduler=s,
        )
        module.input.first = random1.output.result
        module.input.second = random2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.add(
            random1.result.to_array()[:, [2, 4, 6]],
            random2.result.to_array()[:, [3, 5, 7]],
        )
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith("binary_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def _t_stirred_binary(self, **kw: Any) -> None:
        s = self.scheduler()
        random1 = RandomPTable(10, rows=100000, scheduler=s)
        random2 = RandomPTable(10, rows=100000, scheduler=s)
        stirrer1 = Stirrer(update_column="_3", fixed_step_size=1000, scheduler=s, **kw)
        stirrer1.input[0] = random1.output.result
        stirrer2 = Stirrer(update_column="_3", fixed_step_size=1000, scheduler=s, **kw)
        stirrer2.input[0] = random2.output.result
        module = Binary(
            np.add,
            columns={"first": ["_3", "_5", "_7"], "second": ["_4", "_6", "_8"]},
            scheduler=s,
        )
        module.input.first = stirrer1.output.result
        module.input.second = stirrer2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        idx1 = stirrer1.result.index.to_array()
        idx2 = stirrer2.result.index.to_array()
        common = PIntSet(idx1) & PIntSet(idx2)
        bt1 = stirrer1.result.loc[common, :]
        bt2 = stirrer2.result.loc[common, :]
        assert bt1 is not None and bt2 is not None
        t1 = bt1.to_array()[:, [2, 4, 6]]
        t2 = bt2.to_array()[:, [3, 5, 7]]
        res1 = np.add(t1, t2)
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith("binary_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_stirred_binary1(self) -> None:
        self._t_stirred_binary(delete_rows=5)

    def test_stirred_binary2(self) -> None:
        self._t_stirred_binary(update_rows=5)

    def _t_impl(self, cls: Type[Module], ufunc: np.ufunc, mod_name: str) -> None:
        print("Testing", mod_name)
        s = self.scheduler()
        random1 = RandomPTable(3, rows=10_000, scheduler=s)
        random2 = RandomPTable(3, rows=10_000, scheduler=s)
        module = cls(scheduler=s)
        module.input.first = random1.output.result
        module.input.second = random2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = ufunc(random1.result.to_array(), random2.result.to_array())
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith(mod_name))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))


def add_bin_tst(c: Type[TestBin], k: str, ufunc: np.ufunc) -> None:
    cls = func2class_name(k)
    if cls not in arr.__dict__:
        print(f"Class {cls} not implemented")
        return
    mod_name = k + "_"

    def _f(self_: TestBinary) -> None:
        c._t_impl(self_, arr.__dict__[cls], ufunc, mod_name)

    setattr(c, "test_" + k, _f)


for k, ufunc in binary_dict_gen_tst.items():
    add_bin_tst(TestBinary, k, ufunc)


class TestBinaryTD(TestBin):
    def test_binary(self) -> None:
        s = self.scheduler()
        cols = 3
        random1 = RandomPTable(cols, rows=100000, scheduler=s)
        random2 = RandomDict(cols, scheduler=s)
        module = Binary(np.add, scheduler=s)
        module.input.first = random1.output.result
        module.input.second = random2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.add(
            random1.result.to_array(), np.array(list(random2.result.values()))
        )
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith("binary_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_binary2(self) -> None:
        s = self.scheduler()
        cols = 10
        _ = RandomPTable(cols, rows=100_000, scheduler=s)
        _ = RandomDict(cols, scheduler=s)
        with self.assertRaises(AssertionError):
            _ = Binary(np.add, columns=["_3", "_5", "_7"], scheduler=s)

    def test_binary3(self) -> None:
        s = self.scheduler()
        cols = 10
        random1 = RandomPTable(cols, rows=100_000, scheduler=s)
        random2 = RandomDict(cols, scheduler=s)
        module = Binary(
            np.add,
            columns={"first": ["_3", "_5", "_7"], "second": ["_4", "_6", "_8"]},
            scheduler=s,
        )
        module.input.first = random1.output.result
        module.input.second = random2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.add(
            random1.result.to_array()[:, [2, 4, 6]],
            np.array(list(random2.result.values()))[[3, 5, 7]],
        )
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith("binary_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def _t_impl(self, cls: Type[Module], ufunc: np.ufunc, mod_name: str) -> None:
        print("Testing", mod_name)
        s = self.scheduler()
        cols = 3
        random1 = RandomPTable(3, rows=10_000, scheduler=s)
        random2 = RandomDict(cols, scheduler=s)
        module = cls(scheduler=s)
        module.input.first = random1.output.result
        module.input.second = random2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = ufunc(random1.result.to_array(), np.array(list(random2.result.values())))
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith(mod_name))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))


for k, ufunc in binary_dict_gen_tst.items():
    add_bin_tst(TestBinaryTD, k, ufunc)


class TestOtherBinaries(ProgressiveTest):
    def _t_impl(self, cls: Type[Module], ufunc: np.ufunc, mod_name: str) -> None:
        print("Testing", mod_name)
        s = self.scheduler()
        random1 = RandomPTable(
            3,
            rows=100_000,
            scheduler=s,
            random=lambda x: np.random.randint(10, size=x),  # type: ignore
            dtype="int64",
        )
        random2 = RandomPTable(
            3,
            rows=100_000,
            scheduler=s,
            random=lambda x: np.random.randint(10, size=x),  # type: ignore
            dtype="int64",
        )
        module = cls(scheduler=s)
        module.input.first = random1.output.result
        module.input.second = random2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = ufunc(random1.result.to_array(), random2.result.to_array())
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith(mod_name))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_ldexp(self) -> None:
        cls, ufunc, mod_name = Ldexp, np.ldexp, "ldexp_"
        print("Testing", mod_name)
        s = self.scheduler()
        random1 = RandomPTable(3, rows=100_000, scheduler=s)
        random2 = RandomPTable(
            3,
            rows=100_000,
            scheduler=s,
            random=lambda x: np.random.randint(10, size=x),  # type: ignore
            dtype="int64",
        )
        module = cls(scheduler=s)
        module.input.first = random1.output.result
        module.input.second = random2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = ufunc(random1.result.to_array(), random2.result.to_array())
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith(mod_name))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))


def add_other_bin_tst(c: Type[TestOtherBinaries], k: str, ufunc: np.ufunc) -> None:
    cls = func2class_name(k)
    if cls not in arr.__dict__:
        print(f"Class {cls} not implemented")
        return
    mod_name = k + "_"

    def _f(self_: TestOtherBinaries) -> None:
        c._t_impl(self_, arr.__dict__[cls], ufunc, mod_name)

    setattr(c, "test_" + k, _f)


for k, ufunc in binary_dict_int_tst.items():
    if k == "ldexp":
        continue
    add_other_bin_tst(TestOtherBinaries, k, ufunc)


class TestReduce(ProgressiveTest):
    def test_reduce(self) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=100_000, scheduler=s)
        module = Reduce(np.add, scheduler=s)
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.add.reduce(random.result.to_array())
        res2 = np.array(list(module.result.values()))
        self.assertTrue(module.name.startswith("reduce_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_reduce2(self) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=100_000, scheduler=s)
        module = Reduce(np.add, columns=["_3", "_5", "_7"], scheduler=s)
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.add.reduce(random.result.to_array()[:, [2, 4, 6]])
        res2 = np.array(list(module.result.values()))
        self.assertTrue(module.name.startswith("reduce_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def _t_impl(self, cls: Type[Module], ufunc: np.ufunc, mod_name: str) -> None:
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
        res1 = getattr(ufunc, "reduce")(random.result.to_array(), dtype=dtype)
        res2 = np.array(list(module.result.values()))
        self.assertTrue(module.name.startswith(mod_name))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))


def add_reduce_tst(c: Type[TestReduce], k: str, ufunc: np.ufunc) -> None:
    cls = f"{func2class_name(k)}Reduce"
    if cls not in arr.__dict__:
        print(f"Class {cls} not implemented")
        return
    mod_name = f"{k}_reduce_"

    def _f(self_: TestReduce) -> None:
        c._t_impl(self_, arr.__dict__[cls], ufunc, mod_name)

    setattr(c, f"test_{k}", _f)


for k, ufunc in binary_dict_gen_tst.items():
    add_reduce_tst(TestReduce, k, ufunc)


class TestCustomFunctions(ProgressiveTest):
    def test_custom_unary(self) -> None:
        def custom_unary(x: float) -> float:
            return (x + np.sin(x)) / (x + np.cos(x))  # type: ignore

        CustomUnary = make_unary(custom_unary)
        s = self.scheduler()
        random = RandomPTable(10, rows=100_000, scheduler=s)
        module = CustomUnary(scheduler=s)
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.array(module._ufunc(random.result.to_array()), dtype="float64")
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith("custom_unary_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_custom_binary(self) -> None:
        def custom_binary(x: float, y: float) -> float:
            return (x + np.sin(y)) / (x + np.cos(y))  # type: ignore

        CustomBinary = make_binary(custom_binary)
        s = self.scheduler()
        random1 = RandomPTable(3, rows=100_000, scheduler=s)
        random2 = RandomPTable(3, rows=100_000, scheduler=s)
        module = CustomBinary(scheduler=s)
        module.input.first = random1.output.result
        module.input.second = random2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.array(
            module._ufunc(random1.result.to_array(), random2.result.to_array()),
            dtype="float64",
        )
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith("custom_binary_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_custom_reduce(self) -> None:
        def custom_binary(x: float, y: float) -> float:
            return (x + np.sin(y)) / (x + np.cos(y))  # type: ignore

        CustomBinaryReduce = make_reduce(custom_binary)
        s = self.scheduler()
        random = RandomPTable(10, rows=100_000, scheduler=s)
        module = CustomBinaryReduce(scheduler=s)
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.array(module._ufunc(random.result.to_array()), dtype="float64")
        res2 = np.array(list(module.result.values()))
        self.assertTrue(module.name.startswith("custom_binary_reduce_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))


class TestOtherReduces(ProgressiveTest):
    def _t_impl(self, cls: Type[Module], ufunc: np.ufunc, mod_name: str) -> None:
        print("Testing", mod_name)
        s = self.scheduler()
        random = RandomPTable(
            3,
            rows=10_000,
            scheduler=s,
            random=lambda x: np.random.randint(10, size=x),  # type: ignore
            dtype="int64",
        )
        module = cls(scheduler=s)
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = getattr(ufunc, "reduce")(random.result.to_array())
        res2 = np.array(list(module.result.values()))
        self.assertTrue(module.name.startswith(mod_name))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))


def add_other_reduce_tst(c: Type[TestOtherReduces], k: str, ufunc: np.ufunc) -> None:
    cls = f"{func2class_name(k)}Reduce"
    if cls not in arr.__dict__:
        print(f"Class {cls} not implemented")
        return
    mod_name = f"{k}_reduce_"

    def _f(self_: TestOtherReduces) -> None:
        c._t_impl(self_, arr.__dict__[cls], ufunc, mod_name)

    setattr(c, f"test_{k}", _f)


for k, ufunc in binary_dict_int_tst.items():
    if k == "ldexp":
        continue
    add_other_reduce_tst(TestOtherReduces, k, ufunc)


class TestDecorators(ProgressiveTest):
    def test_decorator_unary(self) -> None:
        @unary_module
        def CustomUnary(x: float) -> float:
            return (x + np.sin(x)) / (x + np.cos(x))  # type: ignore

        s = self.scheduler()
        random = RandomPTable(10, rows=100_000, scheduler=s)
        module = CustomUnary(scheduler=s)
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.array(module._ufunc(random.result.to_array()), dtype="float64")
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith("custom_unary_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_decorator_binary(self) -> None:
        @binary_module
        def CustomBinary(x: float, y: float) -> float:
            return (x + np.sin(y)) / (x + np.cos(y))  # type: ignore

        s = self.scheduler()
        random1 = RandomPTable(3, rows=100_000, scheduler=s)
        random2 = RandomPTable(3, rows=100_000, scheduler=s)
        module = CustomBinary(scheduler=s)
        module.input.first = random1.output.result
        module.input.second = random2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.array(
            module._ufunc(random1.result.to_array(), random2.result.to_array()),
            dtype="float64",
        )
        res2 = module.result.to_array()
        self.assertTrue(module.name.startswith("custom_binary_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_decorator_reduce(self) -> None:
        @reduce_module
        def CustomBinaryReduce(x: float, y: float) -> float:
            return (x + np.sin(y)) / (x + np.cos(y))  # type: ignore

        s = self.scheduler()
        random = RandomPTable(10, rows=100_000, scheduler=s)
        module = CustomBinaryReduce(scheduler=s)
        module.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.array(module._ufunc(random.result.to_array()), dtype="float64")
        res2 = np.array(list(module.result.values()))
        self.assertTrue(module.name.startswith("custom_binary_reduce_"))
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))
