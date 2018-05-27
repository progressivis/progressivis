from . import ProgressiveTest

from progressivis.storage.base import StorageEngine, Group, Dataset #, Attribute
from progressivis.table.table import Table

import numpy as np

class TestStorageEngine(ProgressiveTest):
    def test_storage_engine(self):
        e = StorageEngine.default
        self.assertIsNotNone(e)
        se = StorageEngine.lookup(e)
        g = se['/']
        self.assertIsNotNone(g)
        self.assertIsInstance(g, Group)
        
        g2 = g.create_group('g2')
        self.assertIsNotNone(g2)
        self.assertIsInstance(g2, Group)
        d1 = g.create_dataset('d1', shape=(10,), dtype=np.int32)
        self.assertIsNotNone(d1)
        self.assertIsInstance(d1, Dataset)
        
    def test_storage_engines(self):
        print('Engines detected: ', list(StorageEngine.engines().keys()))
        for e in StorageEngine.engines():
            s = StorageEngine.lookup(e)
            self.assertIsNotNone(s)
            g = s['/']
            self.assertIsNotNone(g)
            self.assertIsInstance(g, Group)
        
            g2 = g.create_group('g_'+e)
            self.assertIsNotNone(g2)
            self.assertIsInstance(g2, Group)
            d1 = g.create_dataset('d_'+e, shape=(10,), dtype=np.int32)
            self.assertIsNotNone(d1)
            self.assertIsInstance(d1, Dataset)
            arr = d1[:]
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(len(arr), 10)
            self.assertEqual(arr.dtype, np.int32)
            s = StorageEngine.lookup(e)
            group = s.require_group('table')
            t = self._create_table(e, group)
            self.assertEqual(t.storagegroup, group)

        # for e in StorageEngine.engines():
        #     with StorageEngine.default_engine(e) as _:
        #         t = self._create_table(None)
        #         self.assertEqual(t.storagegroup, e)


    def _create_table(self, storageengine, group):
        if storageengine == "mmap":
            t = Table('table_'+str(storageengine),
                      dshape='{a: int64, b: real}',
                      data={'a': [1,2,3], 'b': [0.1, 0.2, 0.3]},
                      storagegroup=group)
        else:
            t = Table('table_'+str(storageengine),
                      dshape='{a: int64, b: real, c: string}',
                      data={'a': [1,2,3], 'b': [0.1, 0.2, 0.3], 'c': [u'one', u'two', u'three']},
                      storagegroup=group)
        self.assertEqual(len(t), 3)
        return t
        
