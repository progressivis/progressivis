from progressivis.core.pintset import PIntSet
from progressivis.core.changemanager_pintset import PIntSetChangeManager
from progressivis.table.changemanager_table_selected import FakeSlot

from . import ProgressiveTest


class TestBitmapChangeManager(ProgressiveTest):
    def test_PIntSetchangemanager(self) -> None:
        mid1 = "m1"
        bm = PIntSet([1, 2, 3])
        slot = FakeSlot(bm)

        cm = PIntSetChangeManager(slot)
        self.assertEqual(cm.last_update(), 0)
        self.assertEqual(cm.created.length(), 0)
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 0)

        cm.update(1, bm, mid1)
        self.assertEqual(cm.last_update(), 1)
        self.assertEqual(cm.created.length(), 3)
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 0)

        bm = PIntSet([2, 3, 4])
        cm.update(2, bm, mid1)
        self.assertEqual(cm.last_update(), 2)
        # 1 should be removed because deleted at ts=2
        self.assertEqual(cm.created.next(), slice(2, 5))
        self.assertEqual(cm.updated.length(), 0)
        # 0 has been created then deleted before it got consumed
        self.assertEqual(cm.deleted.length(), 0)

        bm = PIntSet([3, 4, 5])
        cm.update(3, bm, mid1)
        self.assertEqual(cm.last_update(), 3)
        self.assertEqual(cm.created.next(), slice(5, 6))
        self.assertEqual(cm.updated.length(), 0)
        self.assertEqual(cm.deleted.length(), 1)  # 2 is deleted but buffered

        bm = PIntSet([2, 3, 4])
        cm.update(4, bm, mid1)
        self.assertEqual(cm.last_update(), 4)
        # 2 has been created before it was consumed so it becomes updated
        self.assertEqual(cm.created.length(), 0)
        self.assertEqual(cm.created.length(), len(cm.created))
        self.assertEqual(cm.updated.length(), 0)  # updates are ignored by default
        # 2 should be removed because added at ts=4
        self.assertEqual(cm.deleted.next(), slice(5, 6))

        cm.created.clear()
        self.assertEqual(cm.created.length(), 0)
        cm.created.set_buffered(False)
        self.assertIsNone(cm.created.next())


if __name__ == "__main__":
    ProgressiveTest.main()
