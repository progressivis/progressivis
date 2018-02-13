from . import ProgressiveTest

from progressivis.table.table_selected import TableSelectedView
from progressivis.table.table import Table
from progressivis.core.bitmap import bitmap
from progressivis.table.changemanager_table_selected import TableSelectedChangeManager
from progressivis.table.tablechanges import TableChanges
import numpy as np

class FakeSlot(object):
    def __init__(self, scheduler, table):
        self._scheduler = scheduler
        self.table = table

    def scheduler(self):
        return self._scheduler

    def data(self):
        return self.table

class TestTableSelectedChangeManager(ProgressiveTest):
    def setUp(self):
        super(TestTableSelectedChangeManager, self).setUp()
        self.s = self.scheduler()
    
    def test_tablechangemanager(self):
        #pylint: disable=protected-access
        table = Table('test_changemanager_table_selected',
                      data={'a': [ 1, 2, 3], 'b': [10.1, 0.2, 0.3]})
        selection = bitmap([1,2])
        table_selected = TableSelectedView(table, selection)
        
        s = self.s
        s._run_number = 1
        last = s._run_number
        slot = FakeSlot(self.s, table_selected)

        mid1 = 1
        cm = TableSelectedChangeManager(slot,
                                        buffer_updated=True,
                                        buffer_deleted=True)
        self.assertEqual(cm.last_update(), 0)
        self.assertEqual(cm.created.length(), 0)
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 0)

        mid2 = 2
        cm2 = TableSelectedChangeManager(slot,
                                         buffer_updated=True,
                                         buffer_deleted=True)
        self.assertEqual(cm2.last_update(), 0)
        self.assertEqual(cm2.created.length(), 0)
        self.assertEqual(cm2.updated.length(), 0)
        self.assertEqual(cm2.deleted.length(), 0)

        mid3 = 3
        cm3 = TableSelectedChangeManager(slot,
                                         buffer_updated=True,
                                         buffer_deleted=True)
        self.assertEqual(cm3.last_update(), 0)
        self.assertEqual(cm3.created.length(), 0)
        self.assertEqual(cm3.updated.length(), 0)
        self.assertEqual(cm3.deleted.length(), 0)

        cm.update(last, table_selected, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        #self.assertEqual(cm.created.next(),slice(0, 3)) without the mask
        self.assertEqual(cm.created.next(),slice(1, 3))
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 0)

        s._run_number += 1
        last = s._run_number
        table.append({'a': [ 4], 'b': [0.5]}) # invisible since id=3
        cm.update(last, table_selected, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        self.assertEqual(cm.created.length(), 0)
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 0)

        s._run_number += 1
        last = s._run_number
        table.append({'a': [ 5, 6, 7, 8], 'b': [0.5, 0.6, 0.7, 0.8] })
        table_selected.selection = bitmap(range(1,8))
        cm.update(last, table_selected, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        self.assertEqual(cm.created.next(),slice(3, 8))
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 0)

        s._run_number += 1
        last = s._run_number
        del table.loc[[1,2,3]]
        table_selected.selection = bitmap([3,4]) # i.e 1,2,5,6,7 were deleted in selection
        cm.update(last, table_selected, mid=mid1)
        self.assertEqual(cm.last_update(), last)
        self.assertEqual(cm.created.length(), 0)
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 6) # 1, 2, 3, 5, 6, 7

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

        #TODO test reset
        cm.reset()
        self.assertEqual(cm.last_update(), 0)
        
        
        



if __name__ == '__main__':
    ProgressiveTest.main()
