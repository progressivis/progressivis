from . import ProgressiveTest

from progressivis.core.index_update import IndexUpdate


class TestIndexUpdate(ProgressiveTest):
    def test_index_update(self) -> None:
        iu = IndexUpdate(created=None, updated=None, deleted=None)
        self.assertTrue(iu.test())
        iu2 = iu.copy()
        self.assertEqual(iu, iu2)
        self.assertEqual(
            repr(iu),
            "IndexUpdate(created=PIntSet([]),"
            "updated=PIntSet([]),"
            "deleted=PIntSet([]))",
        )
        iu2.created.update([1, 2, 3])
        self.assertTrue(iu != iu2)
        self.assertNotEqual(iu, iu2)
