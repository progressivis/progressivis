from . import ProgressiveTest

from progressivis.table.column import Column

from progressivis.table.table_base import IndexTable
from progressivis.core.bitmap import bitmap

import numpy as np


class TestColumn(ProgressiveTest):
    def test_column(self) -> None:
        index = IndexTable()
        self.assertTrue(index.is_identity)
        self.assertEqual(index.last_id, -1)
        self.assertEqual(index.changes, None)

        col1 = Column("col1", index)
        self.assertEqual(col1.name, "col1")
        self.assertIs(col1.index, index)
        self.assertIsNone(col1.base)
        self.assertIsNone(col1.dataset)
        self.assertIsNotNone(col1.storagegroup)
        a = np.arange(10)
        col2 = Column("col2", None, data=a)
        self.assertEqual(col2.name, "col2")
        self.assertIsInstance(col2.index, IndexTable)
        self.assertEqual(col2.size, 10)
        self.assertEqual(col2.shape, (10,))
        assert col2.dataset is not None
        self.assertTrue(np.array_equal(col2.dataset[:], a))
        self.assertEqual(str(col2.index), 'IndexTable("anonymous", dshape="{}")[10]')
        # self.assertEqual(repr(col2.index), 'IdColumn("_ID", dshape=int64)[10][IDENTITY]')

        with self.assertRaises(ValueError):
            col2 = Column("col2.1", None, data=a, indices=[1, 2, 3, 4])
        with self.assertRaises(ValueError):
            col2 = Column("col2.2", None)
        with self.assertRaises(ValueError):
            col2 = Column("col2.3", None, data=[1, 2, 3])  # not handled yet
        col2a = Column("col2a", None, data=col2)
        self.assertIsInstance(col2a.index, IndexTable)
        self.assertTrue(col2a.index.is_identity)
        self.assertEqual(col2a.size, 10)
        self.assertEqual(col2a.shape, (10,))
        assert col2a.dataset is not None
        self.assertTrue(np.array_equal(col2a.dataset[:], a))
        with self.assertRaises(ValueError):
            col2a.add(10, 9)  # id 9 is already allocated
        # check nothing has changed inside
        self.assertTrue(col2a.index.is_identity)
        self.assertEqual(col2a.size, 10)
        self.assertEqual(col2a.shape, (10,))
        assert col2a.dataset is not None
        self.assertTrue(np.array_equal(col2a.dataset[:], a))

        del col2a[10:10]  # Should do nothing
        self.assertTrue(col2a.index.is_identity)
        self.assertEqual(col2a.size, 10)
        self.assertEqual(col2a.shape, (10,))
        assert col2a.dataset is not None
        self.assertTrue(np.array_equal(col2a.dataset[:], a))

        del col2a[8:8]  # Should do nothing
        self.assertTrue(col2a.index.is_identity)
        self.assertEqual(col2a.size, 10)
        self.assertEqual(col2a.shape, (10,))
        assert col2a.dataset is not None
        self.assertTrue(np.array_equal(col2a.dataset[:], a))

        with self.assertRaises(ValueError):
            del col2a[9:11]
        del col2a[9]  # delete end, remain identity mapping
        self.assertTrue(col2a.index.is_identity)
        self.assertEqual(len(col2a), 9)
        assert col2a.dataset is not None
        self.assertTrue(np.array_equal(col2a.dataset[:], np.arange(9)))

        del col2a[8:9]  # delete end, remain identity mapping
        self.assertTrue(col2a.index.is_identity)
        self.assertEqual(len(col2a), 8)
        assert col2a.dataset is not None
        self.assertTrue(np.array_equal(col2a.dataset[:], np.arange(8)))

        del col2a[bitmap([])]  # should do nothing
        self.assertTrue(col2a.index.is_identity)
        self.assertEqual(len(col2a), 8)
        assert col2a.dataset is not None
        self.assertTrue(np.array_equal(col2a.dataset[:], np.arange(8)))

        del col2a[bitmap([7])]
        self.assertTrue(col2a.index.is_identity)
        self.assertEqual(len(col2a), 7)
        assert col2a.dataset is not None
        self.assertTrue(np.array_equal(col2a.dataset[:], np.arange(7)))

        with self.assertRaises(ValueError):
            del col2a[bitmap([1, 2, 9, 10])]

        # del col2a[np.array([])]  column.__del__ does not work with []
        self.assertTrue(col2a.index.is_identity)
        self.assertEqual(len(col2a), 7)
        self.assertTrue(np.array_equal(col2a.dataset[:], np.arange(7)))
        with self.assertRaises(OverflowError):
            del col2a[np.array([-1, 2])]
        with self.assertRaises(ValueError):
            del col2a[np.array([0, 2, 20])]

        del col2a[np.array([6])]
        self.assertTrue(col2a.index.is_identity)
        self.assertEqual(len(col2a), 6)
        self.assertTrue(np.array_equal(col2a.dataset[:], np.arange(6)))

        del col2a[np.array([3])]
        self.assertFalse(col2a.index.is_identity)
        self.assertEqual(len(col2a), 5)  # len might be != than size

        col2a.add(3)
        self.assertFalse(col2a.index.is_identity)
        self.assertEqual(len(col2a), 6)  # len might be != than size
        col2a.append([7, 8, 9, 10], [7, 8, 9, 20])  # add non-id mapping
        self.assertFalse(col2a.index.is_identity)
        self.assertEqual(len(col2a), 10)  # len might be != than size

        col2a.append([42] * 20)
        self.assertTrue("..." in repr(col2a.index))
        # with self.assertRaises(RuntimeError):
        #    col2a.index[4] = 4
        del col2a[5:11]
        self.assertEqual(len(col2a), 25)  # was 24 in ColumnId variant

        col3 = Column("col3", None, dshape="float64", fillvalue=0)
        self.assertEqual(col3.name, "col3")
        self.assertIsInstance(col3.index, IndexTable)
        self.assertEqual(col3.shape, (0,))
        self.assertIsInstance(col3.chunks, tuple)

        col3.append(None)
        with self.assertRaises(ValueError):
            col3.append([1, 2], indices=[2, 3, 4])  # Bad indices length
        with self.assertRaises(ValueError):
            col3.set_shape((3, 2))
        self.assertEqual(col3.maxshape, (0,))
