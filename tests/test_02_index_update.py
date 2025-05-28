from . import ProgressiveTest

from progressivis.core.delta import Delta


class TestIndexUpdate(ProgressiveTest):
    def test_index_update(self) -> None:
        iu = Delta(created=None, updated=None, deleted=None)
        self.assertTrue(iu.test())
        iu2 = iu.copy()
        self.assertEqual(iu, iu2)
        self.assertEqual(
            repr(iu),
            "Delta(created=PIntSet([]),"
            "updated=PIntSet([]),"
            "deleted=PIntSet([]))",
        )
        iu2.created.update([1, 2, 3])
        self.assertTrue(iu != iu2)
        self.assertNotEqual(iu, iu2)
