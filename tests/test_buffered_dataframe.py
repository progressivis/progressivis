import unittest

import pandas as pd
import numpy as np

from progressivis.core.buffered_dataframe import BufferedDataFrame

class TestBufferedDataFrame(unittest.TestCase):
    def test_buffered_dataframe(self):
        buf = BufferedDataFrame()

        for i in range(0,100,10):
            df = pd.DataFrame({'a': range(i,i+10), 'b': range(i,i+10), 'c': range(i,i+10)})
            buf.append(df)
            print 'Iteration %d'%i,
            self.assertEquals(len(buf.df()), i+10)
            self.assertTrue((buf.df().loc[i:i+9,'a']==pd.Series(range(i,i+10))).all())
            print 'OK, size=%d'%len(buf._base)
            # Test if _df is shared with _base
            self.assertIs(buf._df['a'].base,buf._base['a'].base)
