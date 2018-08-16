from progressivis.core.bitmap import bitmap
from . import ProgressiveTest


class TestBitmap(ProgressiveTest):
    def test_bitmap(self):
        bm = bitmap([0, 1, 2, 3])
        self.assertEqual(repr(bm), 'bitmap([0, 1, 2, 3])')
        bm2 = bitmap(slice(0, 4))
        self.assertEqual(bm, bm2)
        self.assertEqual(2, bm[2])
        b2 = bm[2:4]
        self.assertEqual(type(b2), bitmap)
        self.assertEqual(bitmap([2, 3]), b2)
        self.assertEqual(repr(b2), "bitmap([2, 3])")
        res = bitmap([1, 2, 3, 4])
        self.assertTrue(1 in res)
        self.assertTrue([1, 2] in res)
        self.assertTrue([1, 2, 4] in res)
        self.assertFalse([1, 2, 5] in res)
        b3 = bitmap(slice(0, 100))
        self.assertEqual(len(b3), 100)
        b4 = b3.pop(10)
        self.assertEqual(b4, bitmap(slice(0, 10)))
        self.assertEqual(b3, bitmap(slice(10, 100)))
        b4 = b3.pop(10)
        self.assertEqual(b4, bitmap(slice(10, 20)))
        self.assertEqual(b3, bitmap(slice(20, 100)))
        b4 = b3.pop(80)
        self.assertEqual(b4, bitmap(slice(20, 100)))
        self.assertEqual(b3, bitmap())
        b4 = b3.pop(10)
        self.assertEqual(b4, bitmap())
        self.assertEqual(b3, bitmap())
        res = bitmap([1, 2, 3, 4])
        b3.update(res)
        b4 = b3.pop(10)
        self.assertEqual(b4, res)
        self.assertEqual(b3, bitmap())
        res = bitmap([1, 2, 3, 4])
        self.assertEqual(res.to_slice_maybe(), slice(1, 5))
        res = bitmap([1, 2, 4])
        self.assertEqual(res.to_slice_maybe(), res)
        bm = bitmap(range(1000))
        self.assertEqual(repr(bm), 'bitmap([0, 1, 2, 3, 4, 5...(1000)...999)])')
        self.assertEqual(bm | None, bm)
        bm |= None
        self.assertEqual(len(bm), 1000)
        bm.update(None)
        self.assertEqual(len(bm), 1000)
        bm.update(slice(1000, 1001))
        self.assertEqual(len(bm), 1001)
        with self.assertRaises(TypeError):
            bm = bm & 10
        with self.assertRaises(TypeError):
            bm = bm + "hello"

if __name__ == '__main__':
    ProgressiveTest.main()
