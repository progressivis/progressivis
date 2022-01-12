from . import ProgressiveTest
from progressivis.table.paging_helper import PagingHelper
from progressivis.table.table import Table
from progressivis.core.bitmap import bitmap

import numpy as np


class TestPagingHelper(ProgressiveTest):
    def test_paging_helper_t(self) -> None:
        t = Table("table_for_paging", dshape="{a: int, b: float32}", create=True)
        t.resize(200)
        _ = np.arange(200)
        ivalues = np.random.randint(100, size=200)
        t["a"] = ivalues
        fvalues = np.array(np.random.rand(200), np.float32)
        t["b"] = fvalues
        # import pdb; pdb.set_trace()
        ph_t = PagingHelper(t)
        page = ph_t.get_page(0, 10)
        self.assertEqual(page[0][0], 0)
        self.assertEqual(page[-1][0], 9)
        del t.loc[5]
        ph_t = PagingHelper(t)
        page = ph_t.get_page(0, 10)
        self.assertEqual(page[0][0], 0)
        self.assertEqual(page[-1][0], 10)
        sel = bitmap(range(10, 75, 2))
        print(sel)
        view = t.loc[sel, :]
        self.assertTrue(view is not None)
        assert view is not None
        ph_t = PagingHelper(view)
        page = ph_t.get_page(10, 20)
        self.assertEqual(page[0][0], 30)
        self.assertEqual(page[-1][0], 48)
        print(page)
