"Test for mmap-based strorage engine"
import os.path
import shutil

import numpy as np

from progressivis.storage.mmap import MMapGroup
from progressivis.storage.mmap_enc import MAX_SHORT
from progressivis.storage.base import Group, Dataset
from progressivis.table.table import Table
from . import ProgressiveTest, skip, skipIf
import pandas as pd
from progressivis.storage import IS_PERSISTENT as MMAP

LONG_SIZE = MAX_SHORT *2

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

    def test_mmap5(self):
        #pylint: disable=protected-access
        self._rmtree()
        t = Table('table_mmap_5', dshape='{anint: int, atext: string}')
        for i in range(100):
            t.add(dict(anint=i, atext="abc"))
            t.add(dict(anint=i, atext="xyz"))
        nb_str = len(set(t._column("atext").storagegroup["atext"].view))
        self.assertEqual(nb_str, 2)

    @skipIf(not MMAP, "storage is not mmap, test skipped")
    def test_mmap6(self):
        #pylint: disable=protected-access
        long_text = "a"*LONG_SIZE
        self._rmtree()
        t = Table('table_mmap_6', dshape='{anint: int, atext: string}')
        for i in range(100):
            t.add(dict(anint=i, atext=long_text))
        nb_str = len(set(t._column("atext").storagegroup["atext"].view))
        self.assertEqual(nb_str, 100)

    def _tst_impl_mmap_strings(self, t_name, initial, stride):
        #pylint: disable=protected-access
        long_text = initial
        cnt = len(initial)
        self._rmtree()
        t = Table(t_name, dshape='{anint: int, atext: string}')
        t.add(dict(anint=1, atext=long_text))
        for i in range(10):
            if stride >= 0:
                long_text += stride*chr(ord("a")+i%26+1)
            else:
                long_text = long_text[:stride]
            t.loc[0, "atext"] = long_text
            cnt+=len(long_text)
        return t, cnt

    @skipIf(not MMAP, "storage is not mmap, test skipped")
    def test_mmap_strings_all_miss(self):
        t, cnt = self._tst_impl_mmap_strings(t_name='table_mmap_all_miss', initial=LONG_SIZE*"a", stride=10)
        offset = t._column("atext").storagegroup["atext"]._strings.sizes[0]*4
        self.assertGreater(offset, cnt)
        self.assertAlmostEqual(offset/float(cnt), 1, delta=0.2)

    @skipIf(not MMAP, "storage is not mmap, test skipped")
    def test_mmap_strings_all_hit(self):
        len_init = LONG_SIZE * 100
        initial = len_init*"a"
        t, cnt = self._tst_impl_mmap_strings(t_name='table_mmap_all_hit', initial=initial, stride=-1)
        offset = t._column("atext").storagegroup["atext"]._strings.sizes[0]*4
        self.assertGreater(offset, len_init)
        self.assertAlmostEqual(offset/float(len_init), 1, delta=0.2)

    @skipIf(not MMAP, "storage is not mmap, test skipped")
    def test_mmap_drop(self):
        def _free_chunk_nb(t):
            return sum([len(e) for e in t._column("atext").storagegroup["atext"]._strings._freelist])
        self._rmtree()
        t = Table('table_mmap_drop', dshape='{anint: int, atext: string}')
        for i in range(100):
            t.add(dict(anint=i, atext="a"*LONG_SIZE))
        del t.loc[[1,3,5]]
        self.assertEqual(len(t), 97)
        self.assertEqual(_free_chunk_nb(t), 3)
        del t.loc[30:40] # 40 is deleted too
        self.assertEqual(len(t), 87)
        self.assertEqual(_free_chunk_nb(t), 14)
        del t.loc[80:]
        self.assertEqual(len(t), 67)
        self.assertEqual(_free_chunk_nb(t), 34)

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
