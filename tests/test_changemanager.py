import unittest

import pandas as pd
import numpy as np

from progressivis import *
from progressivis.core.changemanager import ChangeManager, NIL

class TestChangeManager(unittest.TestCase):
    def test_changemanager(self):
        cm = ChangeManager()
        self.assertEqual(cm.last_run, None)
        self.assertEqual(len(cm._created), 0)
        self.assertEqual(len(cm._deleted), 0)

        df = pd.DataFrame({'a': [ 1, 2, 3],
                           Module.UPDATE_COLUMN: [ 1, 1, 1 ]})
        now = 1
        cm.update(now, df)
        self.assertEqual(cm.last_run, now)
        self.assertTrue((cm._created==np.array([0, 1, 2])).all())
        self.assertEqual(len(cm._deleted), 0)

        now = 2
        df = df.append(pd.DataFrame({'a': [ 4], Module.UPDATE_COLUMN: [ now ]}),
                       ignore_index=True)
        cm.update(now, df)
        self.assertEqual(cm.last_run, now)
        self.assertTrue((cm._created==np.array([3])).all())
        self.assertEqual(len(cm._deleted), 0)
        
        now = 3
        df = df.append(pd.DataFrame({'a': [ 5], Module.UPDATE_COLUMN: [ now ]}),
                       ignore_index=True)
        cm.update(now, df)
        self.assertEqual(cm.last_run, now)
        self.assertTrue((cm._created==np.array([4])).all())
        self.assertEqual(len(cm._deleted), 0)
        
        now = 4
        df2 = df[df.index != 2] # remove index==2 
        cm.update(now, df)
        self.assertEqual(cm.last_run, now)
        self.assertEqual(len(cm._created), 0)
        self.assertTrue((cm._deleted==np.array([2])).all())


if __name__ == '__main__':
    unittest.main()
