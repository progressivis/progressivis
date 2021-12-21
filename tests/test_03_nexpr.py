from . import ProgressiveTest

from progressivis.core import aio
from progressivis import Print
from progressivis.linalg import NumExprABC
import numpy as np
from progressivis.stats import RandomTable
from progressivis.table.table import Table
from progressivis.core import SlotDescriptor
import numexpr as ne


class NumExprSample(NumExprABC):
    inputs = [
        SlotDescriptor("first", type=Table, required=True),
        SlotDescriptor("second", type=Table, required=True),
    ]
    outputs = [
        SlotDescriptor(
            "result", type=Table, required=False, datashape={"first": ["_1", "_2"]}
        )
    ]
    expr = {"_1": "{first._2}+2*{second._3}", "_2": "{first._3}-5*{second._2}"}


class NumExprSample2(NumExprABC):
    inputs = [
        SlotDescriptor("first", type=Table, required=True),
        SlotDescriptor("second", type=Table, required=True),
    ]
    outputs = [SlotDescriptor("table", type=Table, required=False)]
    expr = {
        "_1:float64": "{first._2}+2*{second._3}",
        "_2:float64": "{first._3}-5*{second._2}",
    }


class TestNumExpr(ProgressiveTest):
    def t_num_expr_impl(self, cls):
        s = self.scheduler()
        random1 = RandomTable(10, rows=100000, scheduler=s)
        random2 = RandomTable(10, rows=100000, scheduler=s)
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
        first_3 = first[:, 2]
        second = random2.result.to_array()
        second_2 = second[:, 1]
        second_3 = second[:, 2]
        ne_1 = ne.evaluate("first_2+2*second_3")
        ne_2 = ne.evaluate("first_3-5*second_2")
        res = module.result.to_array()
        self.assertTrue(np.allclose(res[:, 0], ne_1, equal_nan=True))
        self.assertTrue(np.allclose(res[:, 1], ne_2, equal_nan=True))
        return first_2, first_3, second_2, second_3

    def test_num_expr(self):
        return self.t_num_expr_impl(NumExprSample)

    def test_num_expr2(self):
        return self.t_num_expr_impl(NumExprSample2)
