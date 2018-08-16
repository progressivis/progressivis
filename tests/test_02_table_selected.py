from . import ProgressiveTest

from progressivis import Scheduler
from progressivis.table.table import Table
from progressivis.table.table_selected import TableSelectedView
from progressivis.core.bitmap import bitmap

import numpy as np


class TestTableSelected(ProgressiveTest):
    def setUp(self):
        super(TestTableSelected, self).setUp()        
        self.scheduler = Scheduler.default

    def test_loc_table_selection(self):
        t = Table('table_for_sel', dshape="{a: int, b: float32}", create=True)
        t.resize(10)
        ivalues = np.random.randint(100,size=20)
        t['a'] = ivalues[:10]
        fvalues = np.array(np.random.rand(20), np.float32)
        t['b'] = fvalues[:10]
        t.append({'a': ivalues[10:], 'b': fvalues[10:]})
        sel = bitmap(range(5,8))
        view = TableSelectedView(t, sel, None)
        self.assertEqual(type(view), TableSelectedView)
        self.assertTrue(np.array_equal(view[0].value, ivalues[5:8]))
        self.assertEqual(view.at[6, 'a'], ivalues[6])
        self.assertEqual(view.at[7, 'b'], fvalues[7])
        with self.assertRaises(KeyError):
            self.assertEqual(view.at[4, 'a'], ivalues[4])
        with self.assertRaises(KeyError):
            self.assertEqual(view.at[8, 'a'], ivalues[8])
        
