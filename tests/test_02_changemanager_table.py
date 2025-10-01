"test PTableChangeManager"

from progressivis.table.table import PTable
from progressivis.table.changemanager_table import PTableChangeManager
from progressivis.table.changemanager_table_selected import FakeSlot
from progressivis.table.tablechanges import PTableChanges
from progressivis.core.pintset import PIntSet

from . import ProgressiveTest


class TestPTableChangeManager(ProgressiveTest):
    "Test case for PTableChangeManager"

    def setUp(self) -> None:
        super(TestPTableChangeManager, self).setUp()
        self._scheduler = self.scheduler

    def test_tablechangemanager(self) -> None:
        "main test"
        # pylint: disable=protected-access,too-many-locals,too-many-statements
        table = PTable(
            "test_changemanager_table", data={"a": [1, 2, 3], "b": [10.1, 0.2, 0.3]}
        )
        col_a = table["a"]
        col_b = table["b"]
        s = self.scheduler
        table.changes = PTableChanges()
        s._run_number = 1
        last = s._run_number
        slot = FakeSlot(table)

        mid1 = "m1"
        changemanager = PTableChangeManager(
            slot, buffer_updated=True, buffer_deleted=True
        )
        self.assertEqual(changemanager.last_update(), 0)
        self.assertEqual(changemanager.created.length(), 0)
        self.assertEqual(changemanager.updated.length(), 0)
        self.assertEqual(changemanager.deleted.length(), 0)

        mid2 = "m2"
        cm2 = PTableChangeManager(slot, buffer_updated=True, buffer_deleted=True)
        self.assertEqual(cm2.last_update(), 0)
        self.assertEqual(cm2.created.length(), 0)
        self.assertEqual(cm2.updated.length(), 0)
        self.assertEqual(cm2.deleted.length(), 0)

        cm3 = PTableChangeManager(slot, buffer_updated=True, buffer_deleted=True)
        self.assertEqual(cm3.last_update(), 0)
        self.assertEqual(cm3.created.length(), 0)
        self.assertEqual(cm3.updated.length(), 0)
        self.assertEqual(cm3.deleted.length(), 0)

        changemanager.update(last, table, mid=mid1)
        self.assertEqual(changemanager.last_update(), last)
        self.assertEqual(changemanager.created.next(), slice(0, 3))
        self.assertEqual(changemanager.updated.length(), 0)
        self.assertEqual(changemanager.deleted.length(), 0)

        s._run_number += 1
        last = s._run_number
        table.append({"a": [4], "b": [0.5]})
        changemanager.update(last, table, mid=mid1)
        self.assertEqual(changemanager.last_update(), last)
        self.assertEqual(changemanager.created.next(), slice(3, 4))
        self.assertEqual(changemanager.updated.length(), 0)
        self.assertEqual(changemanager.deleted.length(), 0)

        s._run_number += 1
        last = s._run_number
        table.append({"a": [5], "b": [0.5]})
        changemanager.update(last, table, mid=mid1)
        self.assertEqual(changemanager.last_update(), last)
        self.assertEqual(changemanager.created.next(), slice(4, 5))
        self.assertEqual(changemanager.updated.length(), 0)
        self.assertEqual(changemanager.deleted.length(), 0)

        s._run_number += 1
        col_a[3] = 42
        col_b[3] = 0.42
        col_b[4] = 0.52
        last = s._run_number
        changemanager.update(last, table, mid=mid1)
        self.assertEqual(changemanager.last_update(), last)
        self.assertEqual(changemanager.created.length(), 0)
        self.assertEqual(changemanager.updated.next(), slice(3, 5))
        self.assertEqual(changemanager.deleted.length(), 0)

        s._run_number += 1
        last = s._run_number
        changemanager.update(last, table, mid=mid1)
        self.assertEqual(changemanager.last_update(), last)
        self.assertEqual(changemanager.created.length(), 0)
        self.assertEqual(changemanager.updated.length(), 0)
        self.assertEqual(changemanager.deleted.length(), 0)

        s._run_number += 1
        last2 = 0
        col_a[2] = 22
        col_b[2] = 0.22
        col_b[1] = 0.12

        last2 = s._run_number
        cm2.update(last2, table, mid=mid2)
        self.assertEqual(cm2.last_update(), last2)
        self.assertEqual(cm2.created.next(), slice(0, 5))
        self.assertEqual(cm2.updated.length(), 0)
        self.assertEqual(cm2.deleted.length(), 0)

        s._run_number += 1
        col_a[0] = 11
        col_b[0] = 0.11
        col_b[2] = 0.32
        table.append({"a": [6], "b": [0.6]})

        s._run_number += 1
        last = s._run_number
        changemanager.update(last, table, mid=mid1)
        self.assertEqual(changemanager.last_update(), last)
        self.assertEqual(changemanager.created.next(), slice(5, 6))
        self.assertEqual(changemanager.updated.next(), slice(0, 3))
        self.assertEqual(changemanager.deleted.length(), 0)
        s._run_number += 1
        last2 = s._run_number
        cm2.update(last2, table, mid=mid2)
        self.assertEqual(cm2.last_update(), last2)
        self.assertEqual(cm2.created.next(), slice(5, 6))
        self.assertEqual(list(cm2.updated.next(as_slice=False)), [0, 2])
        self.assertEqual(cm2.deleted.length(), 0)

        s._run_number += 1
        col_a[0] = 1
        col_b[0] = 0.11
        col_b[2] = 0.22

        # test deletes
        s._run_number += 1
        del table.loc[2]
        last = s._run_number
        changemanager.update(last, table, mid=mid1)
        self.assertEqual(changemanager.last_update(), last)
        self.assertEqual(changemanager.created.length(), 0)
        # new behaviour prev. 0
        self.assertEqual(changemanager.updated.length(), 1)
        self.assertEqual(changemanager.deleted.next(), slice(2, 3))
        with self.assertRaises(KeyError):
            table.loc[2]
        # Not sure we want to specify what happens inside a deleted slot?
        # self.assertTrue(np.all(a[:]==np.array([1,2,a.fillvalue,42,5,6])))
        # self.assertTrue(np.all(b[:]==np.array([0.11,0.12,a.fillvalue,0.42,.52,0.6])))

        s._run_number += 1
        del table.loc[4]
        table.append({"a": [7, 8], "b": [0.7, 0.8]})
        col_a[5] = 0.55
        last2 = s._run_number
        cm2.update(last2, table, mid=mid2)
        self.assertEqual(cm2.last_update(), last2)
        self.assertEqual(cm2.created.next(), slice(6, 8))
        # new behaviour, prev. slice(5, 6)
        self.assertEqual(cm2.updated.next(), PIntSet([0, 5]))
        self.assertEqual(list(cm2.deleted.next(as_slice=False)), [2, 4])

        # TODO test reset
        changemanager.reset(mid1)
        self.assertEqual(changemanager.last_update(), 0)


if __name__ == "__main__":
    ProgressiveTest.main()
