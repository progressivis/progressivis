"Test for mmap-based strorage engine"
import os.path
import shutil

import numpy as np

from progressivis.storage.mmap import MMapGroup
from progressivis.storage.base import Group, Dataset
from progressivis.table.table import Table
from . import ProgressiveTest, skip


class TestMMap(ProgressiveTest):
    "Test mmap-based file storage"
    tmp = 'test_mmap'

    def tearDown(self):
        self._rmtree()

    def _rmtree(self):
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)

    @skip("Not ready yet")
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
        self._rmtree()

    def _create_table(self, group):
        table = Table('table',
                      #dshape='{a: int64, b: real, c: string}',
                      dshape='{a: int64, b: real}',
                      #data={'a': [1,2,3], 'b': [0.1, 0.2, 0.3], 'c': ['one', 'two', 'three']},
                      data={'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]},
                      storagegroup=group)
        self.assertEqual(len(table), 3)
        return table
