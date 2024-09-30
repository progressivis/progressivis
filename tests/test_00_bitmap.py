from progressivis import PIntSet
from . import ProgressiveTest


class TestBitmap(ProgressiveTest):
    def test_PIntSet(self) -> None:
        bm = PIntSet([0, 1, 2, 3])
        self.assertEqual(repr(bm), "PIntSet([0, 1, 2, 3])")
        bm2 = PIntSet(slice(0, 4))
        self.assertEqual(bm, bm2)
        self.assertEqual(2, bm[2])
        b2 = bm[2:4]
        self.assertEqual(type(b2), PIntSet)
        self.assertEqual(PIntSet([2, 3]), b2)
        self.assertEqual(repr(b2), "PIntSet([2, 3])")
        res = PIntSet([1, 2, 3, 4])
        self.assertTrue(1 in res)
        self.assertTrue([1, 2] in res)
        self.assertTrue([1, 2, 4] in res)
        self.assertFalse([1, 2, 5] in res)
        b3 = PIntSet(slice(0, 100))
        self.assertEqual(len(b3), 100)
        b4 = b3.pop(10)
        self.assertEqual(b4, PIntSet(slice(0, 10)))
        self.assertEqual(b3, PIntSet(slice(10, 100)))
        b4 = b3.pop(10)
        self.assertEqual(b4, PIntSet(slice(10, 20)))
        self.assertEqual(b3, PIntSet(slice(20, 100)))
        b4 = b3.pop(80)
        self.assertEqual(b4, PIntSet(slice(20, 100)))
        self.assertEqual(b3, PIntSet())
        b4 = b3.pop(10)
        self.assertEqual(b4, PIntSet())
        self.assertEqual(b3, PIntSet())
        res = PIntSet([1, 2, 3, 4])
        b3.update(res)
        b4 = b3.pop(10)
        self.assertEqual(b4, res)
        self.assertEqual(b3, PIntSet())
        res = PIntSet([1, 2, 3, 4])
        self.assertEqual(res.to_slice_maybe(), slice(1, 5))
        res = PIntSet([1, 2, 4])
        self.assertEqual(res.to_slice_maybe(), res)
        bm = PIntSet(range(1000))
        self.assertEqual(repr(bm), "PIntSet([0, 1, 2, 3, 4, 5...(1000)...999)])")
        self.assertEqual(bm | None, bm)
        bm |= None
        self.assertEqual(len(bm), 1000)
        bm.update(None)
        self.assertEqual(len(bm), 1000)
        bm.update(slice(1000, 1001))
        self.assertEqual(len(bm), 1001)
        with self.assertRaises(TypeError):
            bm = bm & 10  # type: ignore
        with self.assertRaises(TypeError):
            bm = bm + "hello"  # type: ignore
        with self.assertRaises(TypeError):
            bm.update(10)
        bm = PIntSet(range(100))
        self.assertEqual(bm & None, bm & PIntSet())
        self.assertEqual(bm ^ None, bm ^ PIntSet())
        self.assertEqual(bm - None, bm - PIntSet())
        bm2 = bm.copy()
        bm2 &= None
        self.assertEqual(bm & None, bm2)
        bm2 = bm.copy()
        bm2 ^= None
        self.assertEqual(bm ^ None, bm2)
        bm2 = bm.copy()
        bm2 -= None
        self.assertEqual(bm - None, bm2)
        self.assertEqual(PIntSet([1, 2]).flip(0, 4), PIntSet([0, 3]))

    def test_PIntSet_fancy(self) -> None:
        bm = PIntSet(range(100))
        fancy = [10, 20, 30]
        self.assertEqual(bm[fancy], PIntSet(fancy))
        bm -= PIntSet([0, 1, 2])
        self.assertEqual(bm[fancy], PIntSet([13, 23, 33]))


if __name__ == "__main__":
    ProgressiveTest.main()
