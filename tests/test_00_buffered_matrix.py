from . import ProgressiveTest

import numpy as np

from progressivis.table.buffered_matrix import BufferedMatrix


class TestBufferedMatrix(ProgressiveTest):
    def test_buffered_matrix(self) -> None:
        buf = BufferedMatrix()
        omat = buf.matrix()
        self.assertEqual(len(buf), 0)
        self.assertEqual(buf.allocated_size(), 0)
        for i in range(10, 100, 10):
            mat = buf.resize(i)
            self.assertEqual(len(buf), i)
            self.assertIs(mat.base, buf._base)
            if omat is not None:
                self.assertTrue(
                    (omat == mat[0 : omat.shape[0], 0 : omat.shape[1]]).all()
                )
            mat[:, :] = np.random.rand(i, i)
            omat = mat
