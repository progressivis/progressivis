from . import ProgressiveTest

from progressivis.core.index_update import IndexUpdate


class TestIndexUpdate(ProgressiveTest):
    def test_index_update(self):
        iu = IndexUpdate(created=None,updated=None,deleted=None)
        self.assertTrue(iu.test())
        iu2 = iu.copy()
        self.assertEqual(iu, iu2)
        self.assertEqual(repr(iu),
                         "IndexUpdate(created=bitmap([]),updated=bitmap([]),deleted=bitmap([]))")
        iu2.created.update([1,2,3])
        self.assertTrue(iu!=iu2)

