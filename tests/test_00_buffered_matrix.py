from . import ProgressiveTest

import pandas as pd
import numpy as np

from progressivis.table.buffered_matrix import BufferedMatrix

import logging
import sys

class TestBufferedMatrix(ProgressiveTest):
    def setUpNO(self):
        super(TestBufferedMatrix, self).setUp()
        self.logger = logging.getLogger('progressivis.core.buffered_matrix')
        self.saved=self.logger.getEffectiveLevel()
        self.logger.setLevel(logging.DEBUG)
        self.sh = logging.StreamHandler(stream=sys.stdout)
        self.logger.addHandler(self.sh)

    def tearDownNO(self):
        self.logger.setLevel(self.saved)
        self.logger.removeHandler(self.sh)

    def test_buffered_matrix(self):
        buf = BufferedMatrix()
        omat = buf.matrix()
        self.assertEqual(len(buf), 0)
        self.assertEqual(buf.allocated_size(), 0)
        for i in range(10,100,10):
            mat = buf.resize(i)
            self.assertEqual(len(buf), i)
            self.assertIs(mat.base,buf._base)
            if omat is not None:
                self.assertTrue((omat==mat[0:omat.shape[0],0:omat.shape[1]]).all())
            mat[:,:] = np.random.rand(i,i)
            omat = mat
