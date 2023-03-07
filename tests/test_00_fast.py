from . import ProgressiveTest, skipIf

import random
from itertools import chain
import numpy as np


IERR = False
try:
    from progressivis.utils.fast import (
        check_contiguity,
        PROP_START_AT_0,
        PROP_MONOTONIC_INC,
        PROP_CONTIGUOUS,
        PROP_IDENTITY,
        indices_to_slice,
        indices_to_slice_iterator,
    )
except ImportError:
    IERR = True


@skipIf(IERR, "running without fast module")
class TestFast(ProgressiveTest):
    def tearDown(self) -> None:
        pass

    def test_check_contiguity(self) -> None:
        a = np.arange(10, dtype=np.uint32)
        with self.assertRaises(ValueError):
            check_contiguity(a)
        a = np.arange(100, dtype=np.uint32).reshape(10, 10)
        with self.assertRaises(ValueError):
            check_contiguity(a)
        a = np.arange(100, dtype=np.uint32)
        c = check_contiguity(a)
        self.assertEqual(c, PROP_IDENTITY)
        a[50:] += np.uint32(1)
        c = check_contiguity(a)
        self.assertEqual(c, PROP_START_AT_0 | PROP_MONOTONIC_INC)
        a[:50] += np.uint32(1)
        c = check_contiguity(a)
        self.assertEqual(c, PROP_MONOTONIC_INC | PROP_CONTIGUOUS)

        np.random.seed(42)
        np.random.shuffle(a)
        a = a[:50]
        a.sort()
        c = check_contiguity(a)
        self.assertEqual(c, PROP_MONOTONIC_INC)

        np.random.shuffle(a)
        c = check_contiguity(a)
        self.assertEqual(c, 0)

    def test_indices_to_slice(self) -> None:
        s = indices_to_slice([])
        self.assertEqual(s, slice(0, 0))
        s = indices_to_slice([0, 1, 2, 3])
        self.assertEqual(s, slice(0, 4))
        # not sure the following are desirable
        # s = indices_to_slice([0, 1, 1, 2, 3])
        # self.assertEqual(s, slice(0,4))
        s = indices_to_slice([1, 2, 3])
        self.assertEqual(s, slice(1, 4))
        s = indices_to_slice([1, 2, 3, 5])
        self.assertEqual(s, [1, 2, 3, 5])
        s = indices_to_slice(np.array([0, 1, 2, 3]))
        self.assertEqual(s, slice(0, 4))
        s = indices_to_slice(np.array([1, 2, 3, 4]))
        self.assertEqual(s, slice(1, 5))

    def test_indices_to_slice_iterator(self) -> None:
        s = indices_to_slice_iterator([])
        self.assertEqual(list(s), [])
        s = indices_to_slice_iterator([0, 1, 2, 3])
        self.assertEqual(list(s), [slice(0, 4)])
        s = indices_to_slice_iterator([1, 2, 3])
        self.assertEqual(list(s), [slice(1, 4)])
        s = indices_to_slice_iterator([1, 2, 3, 5])
        self.assertEqual(list(s), [slice(1, 4), slice(5, 6)])
        s = indices_to_slice_iterator(np.array([0, 1, 2, 3]))
        self.assertEqual(list(s), [slice(0, 4)])
        random.seed(42)
        for _ in range(100):
            ln = random.sample(range(100), 80)
            sl = indices_to_slice_iterator(ln)
            self.assertEqual(
                ln, list(chain.from_iterable([range(*i.indices(i.stop)) for i in sl]))
            )
