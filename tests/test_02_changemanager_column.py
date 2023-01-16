from . import ProgressiveTest

import numpy as np

from progressivis.table.column import PColumn
from progressivis.table.changemanager_column import PColumnChangeManager
from progressivis.table.tablechanges import PTableChanges
from progressivis.table.changemanager_table_selected import FakeSlot


class TestPColumnChangeManager(ProgressiveTest):
    def setUp(self) -> None:
        super(TestPColumnChangeManager, self).setUp()
        self.scheduler_ = self.scheduler()

    def test_columnchangemanager(self) -> None:
        # pylint: disable=protected-access
        column = PColumn("test_changemanager_column", None, data=np.array([1, 2, 3]))
        s = self.scheduler_
        column.changes = PTableChanges()
        s._run_number = 1
        last = s._run_number
        slot = FakeSlot(column)

        mid1 = "m1"
        cm = PColumnChangeManager(slot, buffer_updated=True, buffer_deleted=True)
        self.assertEqual(cm.last_update(), 0)
        self.assertEqual(cm.created.length(), 0)
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 0)

        mid2 = "m2"
        cm2 = PColumnChangeManager(slot, buffer_updated=True, buffer_deleted=True)
        self.assertEqual(cm2.last_update(), 0)
        self.assertEqual(cm2.created.length(), 0)
        self.assertEqual(cm2.updated.length(), 0)
        self.assertEqual(cm2.deleted.length(), 0)

        # mid3 = 3
        cm3 = PColumnChangeManager(slot, buffer_updated=True, buffer_deleted=True)
        self.assertEqual(cm3.last_update(), 0)
        self.assertEqual(cm3.created.length(), 0)
        self.assertEqual(cm3.updated.length(), 0)
        self.assertEqual(cm3.deleted.length(), 0)

        cm.update(last, column, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        self.assertEqual(cm.created.next(), slice(0, 3))
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 0)

        s._run_number += 1
        last = s._run_number
        column.append(np.array([4]))
        cm.update(last, column, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        self.assertEqual(cm.created.next(), slice(3, 4))
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 0)

        s._run_number += 1
        last = s._run_number
        column.append(np.array([5]))
        cm.update(last, column, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        self.assertEqual(cm.created.next(), slice(4, 5))
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 0)

        s._run_number += 1
        column[3] = 42
        column[4] = 52
        last = s._run_number
        cm.update(last, column, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        self.assertEqual(cm.created.length(), 0)
        self.assertEqual(cm.updated.next(), slice(3, 5))
        self.assertEqual(cm.deleted.length(), 0)

        s._run_number += 1
        last = s._run_number
        cm.update(last, column, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        self.assertEqual(cm.created.length(), 0)
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 0)

        s._run_number += 1
        last2 = 0
        column[2] = 22
        column[1] = 0.12

        last2 = s._run_number
        cm2.update(last2, column, mid=mid2)
        self.assertEqual(cm2.last_update(), last2)
        self.assertEqual(cm2.created.next(), slice(0, 5))
        self.assertEqual(cm2.updated.length(), 0)
        self.assertEqual(cm2.deleted.length(), 0)

        s._run_number += 1
        column[0] = 11
        column[2] = 32
        column.append(np.array([6]))

        # tv = column.loc[1:2]
        # last3 = s._run_number
        # cm3.update(last3, tv, mid=mid3)
        # self.assertEqual(cm3.created.next(), slice(1, 3)) # test ids, not indices
        # self.assertEqual(cm2.updated.length(), 0)
        # self.assertEqual(cm2.deleted.length(), 0)

        s._run_number += 1
        last = s._run_number
        # with self.assertRaises(ValueError):
        #     cm.update(last+1, column, mid=mid1)
        cm.update(last, column, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        self.assertEqual(cm.created.next(), slice(5, 6))
        self.assertEqual(cm.updated.next(), slice(0, 3))
        self.assertEqual(cm.deleted.length(), 0)

        s._run_number += 1
        last2 = s._run_number
        cm2.update(last2, column, mid=mid2)
        self.assertEqual(cm2.last_update(), last2)
        self.assertEqual(cm2.created.next(), slice(5, 6))
        self.assertEqual(list(cm2.updated.next(as_slice=False)), [0, 2])
        self.assertEqual(cm2.deleted.length(), 0)

        # s._run_number += 1
        # column[0] = 1
        # column[2] = 22
        # last3 = s._run_number
        # cm3.update(last3, tv, mid=mid3)
        # self.assertEqual(cm3.last_update(), last3)
        # self.assertEqual(cm3.created.length(), 0)
        # self.assertEqual(cm3.updated.next(), slice(2,3))
        # self.assertEqual(cm3.deleted.length(), 0)

        # test deletes
        s._run_number += 1
        del column.loc[2]
        last = s._run_number
        cm.update(last, column, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        self.assertEqual(cm.created.length(), 0)
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.next(), slice(2, 3))
        #        self.assertTrue(np.all(column[:]==np.array([1,2,a.fillvalue,42,5,6])))
        #        self.assertTrue(np.all(b[:]==np.array([0.11,0.12,a.fillvalue,0.42,.52,0.6])))

        s._run_number += 1
        del column.loc[4]
        column.append(np.array([7, 8]))
        column[5] = 55
        last2 = s._run_number
        cm2.update(last2, column, mid=mid2)
        self.assertEqual(cm2.last_update(), last2)
        self.assertEqual(cm2.created.next(), slice(6, 8))
        self.assertEqual(cm2.updated.next(), slice(5, 6))
        self.assertEqual(list(cm2.deleted.next(as_slice=False)), [2, 4])

        # TODO test reset
        cm.reset(mid=mid1)
        self.assertEqual(cm.last_update(), 0)


if __name__ == "__main__":
    ProgressiveTest.main()
