"Test for mmap-based strorage engine"
import os.path
import shutil

import numpy as np

from progressivis.storage.mmap import MMapGroup
from progressivis.storage.base import Group, Dataset
from progressivis.table.table import Table
from . import ProgressiveTest, skip
import pandas as pd


class TestMMap(ProgressiveTest):
    "Test mmap-based file storage"
    tmp = 'test_mmap'

    def tearDown(self):
        self._rmtree()
        super(TestMMap, self).tearDown()
    def _rmtree(self):
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)

    def test_mmap(self):
        "Actual test"
        self._rmtree()
        group = MMapGroup(self.tmp)
        self.assertIsNotNone(group)
        self.assertIsInstance(group, Group)
        group2 = group.create_group('g_'+self.tmp)
        self.assertIsNotNone(group2)
        self.assertIsInstance(group2, Group)
        dataset1 = group.create_dataset('d_'+self.tmp, shape=(10,), dtype=np.int32)
        self.assertIsNotNone(dataset1)
        self.assertIsInstance(dataset1, Dataset)
        arr = dataset1[:]
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(len(arr), 10)
        self.assertEqual(arr.dtype, np.int32)
        group3 = group.require_group('table')
        table = self._create_table(group3)
        self.assertEqual(table.storagegroup, group3)
        self.assertEqual(table['a'][1], 2)
        self.assertTrue(np.array_equal(table['a'][0:2], [1,2]))
        self.assertTrue(np.array_equal(table['a'][0:3], [1,2,3]))
        self.assertEqual(table['c'][1], 'two')
        self.assertTrue(np.array_equal(table['c'][0:2], ['one','two']))
        self.assertTrue(np.array_equal(table['c'][0,1,2], ['one','two','three']))
        self.assertTrue(np.array_equal(table['c'][1:], ['two','three']))
        self._rmtree()

    def test_mmap2(self):
        self._rmtree()
        group = MMapGroup(self.tmp)
        group3 = group.require_group('table')
        table = self._create_table2(group3)
        self.assertEqual(table.storagegroup, group3)
        self._rmtree()

    def test_mmap3(self):
        #pylint: disable=protected-access
        #self.scheduler._run_number = 1
        self._rmtree()
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3], 'c': ['a', 'b', 'cd']})
        t = Table('table_2', data=df)
        self.assertEqual(len(t),len(df))
        for colname in df:
            coldf = df[colname]
            colt = t[colname]
            self.assertEqual(len(coldf), len(colt))
            self.assertTrue(np.all(coldf.values==colt.values))
        t.append(df)
        self.assertEqual(len(t),2*len(df))
        self._rmtree()
    def test_mmap4(self):
        #pylint: disable=protected-access
        #self.scheduler._run_number = 1
        self._rmtree()
        print("test_mmap4")
        df = pd.DataFrame({'a': np.arange(10000), 'b': np.random.rand(10000)})
        t = Table('table_4', data=df)
        np.min(t['b'])
        self.assertEqual(len(t),len(df))
        self._rmtree()
        
    def _create_table(self, group):
        table = Table('table',
                      dshape='{a: int64, b: real, c: string, d: 10*int}',
                      data={'a': [1,2,3], 'b': [0.1, 0.2, 0.3], 'c': ['one', 'two', 'three'], 'd': [np.arange(10)]*3},
                      storagegroup=group)
        self.assertEqual(len(table), 3)
        return table
    
    def _create_table2(self, group):
        t = Table('table',
                  dshape="{a: int, b: float32, c: string, d: 10*int}", create=True, storagegroup=group)
        self.assertTrue(t is not None)
        self.assertEqual(t.ncol, 4)
        col1 = t['a']
        col2 = t[0]
        self.assertTrue(col1 is col2)
        return t
