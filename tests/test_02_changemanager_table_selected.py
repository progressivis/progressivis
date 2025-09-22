from . import ProgressiveTest
from progressivis.table.table_base import PTableSelectedView
from progressivis.table.table import PTable
from progressivis.core.pintset import PIntSet
from progressivis.table.changemanager_table_selected import (
    PTableSelectedChangeManager,
    FakeSlot,
)


class TestPTableSelectedChangeManager(ProgressiveTest):
    def setUp(self) -> None:
        super(TestPTableSelectedChangeManager, self).setUp()
        self.s = self.scheduler

    def test_tablechangemanager(self) -> None:
        # pylint: disable=protected-access
        table = PTable(
            "test_changemanager_table_selected",
            data={"a": [1, 2, 3], "b": [10.1, 0.2, 0.3]},
        )
        selection = PIntSet([1, 2])
        table_selected: PTableSelectedView = PTableSelectedView(table, selection)

        s = self.s
        s._run_number = 1
        last = s._run_number
        slot = FakeSlot(table_selected)

        mid1 = "m1"
        cm = PTableSelectedChangeManager(
            slot,
            buffer_exposed=True,
            buffer_updated=True,
            buffer_deleted=True,
            buffer_masked=True,
        )
        self.assertEqual(cm.last_update(), 0)
        self.assertEqual(cm.created.length(), 0)
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 0)

        # mid2 = 2
        cm2 = PTableSelectedChangeManager(
            slot,
            buffer_exposed=True,
            buffer_updated=True,
            buffer_deleted=True,
            buffer_masked=True,
        )
        self.assertEqual(cm2.last_update(), 0)
        self.assertEqual(cm2.created.length(), 0)
        self.assertEqual(cm2.updated.length(), 0)
        self.assertEqual(cm2.deleted.length(), 0)

        # mid3 = 3
        cm3 = PTableSelectedChangeManager(
            slot,
            buffer_exposed=True,
            buffer_updated=True,
            buffer_deleted=True,
            buffer_masked=True,
        )
        self.assertEqual(cm3.last_update(), 0)
        self.assertEqual(cm3.created.length(), 0)
        self.assertEqual(cm3.updated.length(), 0)
        self.assertEqual(cm3.deleted.length(), 0)
        cm.update(last, table_selected, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        self.assertEqual(cm.created.next(), slice(1, 3))  # without the mask
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 0)

        s._run_number += 1
        last = s._run_number
        table.append({"a": [4], "b": [0.5]})  # invisible since id=3
        cm.update(last, table_selected, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        self.assertEqual(cm.created.length(), 0)
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 0)

        s._run_number += 1
        last = s._run_number
        table.append({"a": [5, 6, 7, 8], "b": [0.5, 0.6, 0.7, 0.8]})
        table_selected.selection = PIntSet(range(1, 8))
        cm.update(last, table_selected, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        self.assertEqual(cm.created.next(), slice(3, 8))
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 0)

        s._run_number += 1
        last = s._run_number
        del table.loc[[1, 2, 3]]
        table_selected.selection = PIntSet(
            [3, 4]
        )  # i.e 1,2,5,6,7 were deleted in selection
        cm.update(last, table_selected, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        self.assertEqual(cm.created.length(), 0)
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.base.deleted.length(), 3)  # 1, 2, 3
        self.assertEqual(
            cm.selection.deleted.length(), 6
        )  # 1, 2, 5, 6, 7[+3 removed because it was perm.deleted]
        self.assertEqual(cm.deleted.length(), 6)  # 1, 2, 3, 5, 6, 7
        cm.base.deleted.next()
        cm.selection.deleted.next()
        s._run_number += 1
        last = s._run_number
        table.append({"a": [15, 16, 17, 18], "b": [0.51, 0.61, 0.71, 0.81]})
        table_selected._selection = slice(5, None)
        cm.update(last, table_selected, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        self.assertEqual(cm.base.created.changes, PIntSet([8, 9, 10, 11]))
        self.assertEqual(cm.selection.created.changes, PIntSet([5, 6, 7, 8, 9, 10, 11]))
        self.assertEqual(cm.selection.deleted.changes, PIntSet([4]))
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.base.deleted.length(), 0)
        self.assertEqual(cm.deleted.length(), 1)
        cm.deleted.next()
        self.assertEqual(cm.deleted.length(), 0)
        cm.created.next()
        self.assertEqual(cm.base.created.length(), 0)
        self.assertEqual(cm.selection.created.length(), 0)
        # s._run_number += 1
        # a[3] = 42
        # b[3] = 0.42
        # b[4] = 0.52
        # last = s._run_number
        # cm.update(last, table, mid=mid1)
        # self.assertEqual(cm.last_update(), last)
        # self.assertEqual(cm.created.length(), 0)
        # self.assertEqual(cm.updated.next(), slice(3,5))
        # self.assertEqual(cm.deleted.length(), 0)

        # s._run_number += 1
        # last = s._run_number
        # cm.update(last, table, mid=mid1)
        # self.assertEqual(cm.last_update(), last)
        # self.assertEqual(cm.created.length(), 0)
        # self.assertEqual(cm.updated.length(), 0)
        # self.assertEqual(cm.deleted.length(), 0)

        # s._run_number += 1
        # last2 = 0
        # a[2] = 22
        # b[2] = 0.22
        # b[1] = 0.12

        # last2 = s._run_number
        # cm2.update(last2, table, mid=mid2)
        # self.assertEqual(cm2.last_update(), last2)
        # self.assertEqual(cm2.created.next(), slice(0, 5))
        # self.assertEqual(cm2.updated.length(), 0)
        # self.assertEqual(cm2.deleted.length(), 0)

        # s._run_number += 1
        # a[0] = 11
        # b[0] = 0.11
        # b[2] = 0.32
        # table.append({'a': [ 6], 'b': [0.6] })

        # tv = table.loc[1:2]
        # last3 = s._run_number
        # cm3.update(last3, tv, mid=mid3)
        # self.assertEqual(cm3.created.next(), slice(1, 3)) # test ids, not indices
        # self.assertEqual(cm2.updated.length(), 0)
        # self.assertEqual(cm2.deleted.length(), 0)

        # s._run_number += 1
        # last = s._run_number
        # # with self.assertRaises(ValueError):
        # #     cm.update(last+1, table, mid=mid1)
        # cm.update(last, table, mid=mid1)
        # self.assertEqual(cm.last_update(), last)
        # self.assertEqual(cm.created.next(), slice(5,6))
        # self.assertEqual(cm.updated.next(), slice(0,3))
        # self.assertEqual(cm.deleted.length(), 0)

        # s._run_number += 1
        # last2 = s._run_number
        # cm2.update(last2, table, mid=mid2)
        # self.assertEqual(cm2.last_update(), last2)
        # self.assertEqual(cm2.created.next(), slice(5,6))
        # self.assertEqual(list(cm2.updated.next()), [0,2])
        # self.assertEqual(cm2.deleted.length(), 0)

        # s._run_number += 1
        # a[0] = 1
        # b[0] = 0.11
        # b[2] = 0.22
        # last3 = s._run_number
        # cm3.update(last3, tv, mid=mid3)
        # self.assertEqual(cm3.last_update(), last3)
        # self.assertEqual(cm3.created.length(), 0)
        # self.assertEqual(cm3.updated.next(), slice(2,3))
        # self.assertEqual(cm3.deleted.length(), 0)

        # # test deletes
        # s._run_number += 1
        # del table.loc[2]
        # last = s._run_number
        # cm.update(last, table, mid=mid1)
        # self.assertEqual(cm.last_update(), last)
        # self.assertEqual(cm.created.length(), 0)
        # self.assertEqual(cm.updated.length(), 0)
        # self.assertEqual(cm.deleted.next(), slice(2,3))
        # self.assertTrue(np.all(a[:]==np.array([1,2,a.fillvalue,42,5,6])))
        # self.assertTrue(np.all(b[:]==np.array([0.11,0.12,a.fillvalue,0.42,.52,0.6])))

        # s._run_number += 1
        # del table.loc[4]
        # table.append({'a': [ 7,8], 'b': [0.7,0.8] })
        # a[5] = 0.55
        # last2 = s._run_number
        # cm2.update(last2, table, mid=mid2)
        # self.assertEqual(cm2.last_update(), last2)
        # self.assertEqual(cm2.created.next(), slice(6,8))
        # self.assertEqual(cm2.updated.next(), slice(5,6))
        # self.assertEqual(list(cm2.deleted.next()), [2,4])

        # TODO test reset
        cm.reset(mid=mid1)
        self.assertEqual(cm.last_update(), 0)


if __name__ == "__main__":
    ProgressiveTest.main()
