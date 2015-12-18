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
            
        row = {'a': 1010, 'b': 2020, 'c': 3030}
        buf.append_row(row)
        self.assertIs(buf._df['a'].base,buf._base['a'].base)
        self.assertEquals(len(buf.df()), 101)
        for k in row:
            self.assertEquals(buf.df().loc[100,k], row[k])
        row = pd.Series(np.random.rand(3), index=['a','b','c'])
        buf.append_row(row)
        self.assertEquals(len(buf.df()), 102)
        for k in row.index:
            self.assertEquals(buf.df().loc[101,k], row[k])
        
