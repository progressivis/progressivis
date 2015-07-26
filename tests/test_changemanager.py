import unittest
import pandas as pd
from timeit import default_timer

import os
import numpy as np
from pprint import pprint

from progressive import *
from progressive.core.changemanager import ChangeManager, NIL

class TestChangeManager(unittest.TestCase):
    def test_changemanager(self):
        cm = ChangeManager()
        self.assertEqual(cm.last_time, None)
        self.assertEqual(len(cm.created), 0)
        self.assertEqual(len(cm.deleted), 0)

        df = pd.DataFrame({'a': [ 1, 2, 3],
                           Module.UPDATE_COLUMN: [ 0, 0, 0 ]})
        now = default_timer()
        cm.update(now, df)
        self.assertEqual(cm.last_time, now)
        self.assertTrue((cm.created==np.array([0, 1, 2])).all())
        self.assertEqual(len(cm.deleted), 0)

        df = df.append(pd.DataFrame({'a': [ 4], Module.UPDATE_COLUMN: [ now ]}),
                       ignore_index=True)
        now = default_timer()
        cm.update(now, df)
        self.assertEqual(cm.last_time, now)
        self.assertTrue((cm.created==np.array([3])).all())
        self.assertEqual(len(cm.deleted), 0)
        
        df = df.append(pd.DataFrame({'a': [ 5], Module.UPDATE_COLUMN: [ now ]}),
                       ignore_index=True)
        now = default_timer()
        cm.update(now, df)
        self.assertEqual(cm.last_time, now)
        self.assertTrue((cm.created==np.array([4])).all())
        self.assertEqual(len(cm.deleted), 0)
        
        df2 = df[df.index != 2] # remove index==2 
        now = default_timer()
        cm.update(now, df)
        self.assertEqual(cm.last_time, now)
        self.assertEqual(len(cm.created), 0)
        self.assertTrue((cm.deleted==np.array([2])).all())


if __name__ == '__main__':
    unittest.main()
